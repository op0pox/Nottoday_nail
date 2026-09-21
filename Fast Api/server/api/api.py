import os

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Tuple

from classification import contour_compare, xor_compare
from classification.compare_pipeline import (
    CATALOG_TEMPLATES,
    extract_shape_code,
    nearest_template,
)
from segmentation.nail_segmentation import YoloNailBackend, measure_nail_from_mask

SQUARES_X = int(os.getenv("SQUARES_X"))
SQUARES_Y = int(os.getenv("SQUARES_Y"))
SQUARE_MM = float(os.getenv("SQUARE_MM"))
MARKER_MM = float(os.getenv("MARKER_MM"))
CAMERA_HEIGHT_MM = float(os.getenv("CAMERA_HEIGHT_MM"))
NAIL_HEIGHT_MM = float(os.getenv("NAIL_HEIGHT_MM"))

router = APIRouter(prefix="/api")
DEFAULT_SAMPLES = 160
COMPARE_METRICS = {
    "chamfer": contour_compare,
    "xor": xor_compare,
}

backend = YoloNailBackend(conf=0.25)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, aruco_dict)

charuco_params = cv2.aruco.CharucoParameters()
# 보드 바깥이 잘려 마커가 한쪽만 보여도 코너를 보간한다 (기본값은 인접 마커 2개)
if hasattr(charuco_params, "minMarkers"):
    charuco_params.minMarkers = 1

detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector_params.minMarkerPerimeterRate = 0.02
detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


def format_contour(contour):
    if contour is None or len(contour) < 3:
        return []
    return [[float(x), float(y)] for x, y in np.asarray(contour, dtype=np.float32).reshape(-1, 2)]


def classify_contour(contour, metric_name):
    metric = COMPARE_METRICS[metric_name]
    result = nearest_template(
        np.asarray(contour, dtype=np.float32).reshape(-1, 2),
        CATALOG_TEMPLATES,
        DEFAULT_SAMPLES,
        "cuticle",
        metric,
    )
    return extract_shape_code(result["predicted_shape"]), float(result["distance"])


# 손톱 한 개의 측정 결과
class MeasurementResult(BaseModel):
    length_mm: float
    width_mm: Optional[float] = None
    shape: Optional[str] = None
    shape_score: Optional[float] = None
    metric: Optional[str] = None
    contours: Optional[List[List[Tuple[float, float]]]] = None


# 사진 한 장에서 손톱 길이·폭·형태 측정
@router.post("/measure", response_model=List[MeasurementResult])
async def measure_nails(
    file: UploadFile = File(...),
    metric: str = Form("chamfer"),
):
    metric_name = (metric or "chamfer").strip().lower()
    if metric_name not in COMPARE_METRICS:
        raise HTTPException(status_code=400, detail="metric은 chamfer 또는 xor 이어야 합니다.")
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        # 코너가 모자라면 국소 대비를 올려 한 번 더 시도한다
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        charuco_corners, charuco_ids = detector.detectBoard(clahe.apply(gray))[:2]

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        raise HTTPException(status_code=400, detail="ChArUco failed")

    # 코너 ID는 (SQUARES_X-1)열 격자를 행 우선으로 센 번호라, 몫·나머지로 격자 위치를 되돌린다
    cols = SQUARES_X - 1
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    mm_points = np.array(
        [
            ((int(cid) % cols + 1) * SQUARE_MM, (int(cid) // cols + 1) * SQUARE_MM)
            for cid in charuco_ids.flatten()
        ],
        dtype=np.float32,
    )

    # 픽셀 좌표를 보드 평면의 mm 좌표로 옮기는 행렬
    H, _ = cv2.findHomography(image_points, mm_points, cv2.RANSAC, 2.0)
    if H is None:
        raise HTTPException(status_code=400, detail="Homography failed")

    nail_masks = backend.segment(image)
    if not nail_masks:
        raise HTTPException(status_code=400, detail="Nail detection failed")

    results = []
    for nail_mask in nail_masks:
        best_shape = None
        min_dist = None
        formatted_contours = []

        if nail_mask.contour is not None and len(nail_mask.contour) >= 3:
            formatted_contours = [format_contour(nail_mask.contour)]
            try:
                best_shape, min_dist = classify_contour(nail_mask.contour, metric_name)
            except ValueError:
                best_shape = None
                min_dist = None

        measured = measure_nail_from_mask(
            nail_mask.mask,
            H,
            camera_height_mm=CAMERA_HEIGHT_MM,
            nail_height_mm=NAIL_HEIGHT_MM,
        )

        if measured:
            results.append(MeasurementResult(
                length_mm=round(measured["length_mm"], 2),
                width_mm=round(measured["width_mm"], 2) if measured.get("width_mm") is not None else None,
                shape=best_shape,
                shape_score=round(min_dist, 4) if min_dist is not None else None,
                metric=metric_name,
                contours=formatted_contours,
            ))

    return results
