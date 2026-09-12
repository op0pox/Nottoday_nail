import os

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple

from segmentation.nail_segmentation import YoloNailBackend, measure_nail_from_mask
from classification.nail_classification import classify_nail_shape

SQUARES_X = int(os.getenv("SQUARES_X"))
SQUARES_Y = int(os.getenv("SQUARES_Y"))
SQUARE_MM = float(os.getenv("SQUARE_MM"))
MARKER_MM = float(os.getenv("MARKER_MM"))
CAMERA_HEIGHT_MM = float(os.getenv("CAMERA_HEIGHT_MM"))
NAIL_HEIGHT_MM = float(os.getenv("NAIL_HEIGHT_MM"))

router = APIRouter(prefix="/api")

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


# 마스크 외곽선을 [[x, y], ...] 목록으로
def format_contours(mask):
    display_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    formatted = []
    for cnt in display_contours:
        if cnt is None or len(cnt) == 0:
            continue
        cnt_list = cnt.squeeze().tolist()
        if not cnt_list:
            continue
        # 점이 하나면 squeeze가 [x, y]로 납작해지므로 다시 감싼다
        if not isinstance(cnt_list[0], list):
            cnt_list = [cnt_list]
        formatted.append(cnt_list)
    return formatted


# 손톱 한 개의 측정 결과
class MeasurementResult(BaseModel):
    length_mm: float
    width_mm: Optional[float] = None
    shape: Optional[str] = None
    shape_score: Optional[float] = None
    contours: Optional[List[List[Tuple[int, int]]]] = None


# 사진 한 장에서 손톱 길이·폭·형태 측정
@router.post("/measure", response_model=List[MeasurementResult])
async def measure_nails(
    file: UploadFile = File(...),
):
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
        contours, _ = cv2.findContours(nail_mask.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        best_shape = None
        min_dist = None
        formatted_contours = []

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            best_shape, min_dist = classify_nail_shape(main_contour)
            formatted_contours = format_contours(nail_mask.mask)

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
                shape_score=round(min_dist, 4) if min_dist else None,
                contours=formatted_contours
            ))

    return results
