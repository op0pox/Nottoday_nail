import base64
import os

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Tuple

from classification.nail_preprocess import prepare_nail_gray
from classification.shape_model import ShapeClassifier
from segmentation.nail_segmentation import YoloNailBackend, measure_nail_from_mask, model_path

SQUARES_X = int(os.getenv("SQUARES_X"))
SQUARES_Y = int(os.getenv("SQUARES_Y"))
SQUARE_MM = float(os.getenv("SQUARE_MM"))
MARKER_MM = float(os.getenv("MARKER_MM"))
CAMERA_HEIGHT_MM = float(os.getenv("CAMERA_HEIGHT_MM"))
NAIL_HEIGHT_MM = float(os.getenv("NAIL_HEIGHT_MM"))

router = APIRouter(prefix="/api")
SCALE_MODES = ("board", "hardware")
# board는 체커보드 사진용 세그, hardware는 흰 배경용 세그
SEG_BACKENDS = {
    "board": YoloNailBackend(model_path("seg_checkerboard.pt")),
    "hardware": YoloNailBackend(model_path("seg_white.pt")),
}
FINGER_GROUPS = {
    "thumb": ShapeClassifier(model_path("cls_thumb.pt")),
    "other": ShapeClassifier(model_path("cls_other.pt")),
}
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


# 손톱 한 개의 측정 결과. shape 는 정면·측면이 둘 다 있을 때만 채워진다.
class NailView(BaseModel):
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    shape: Optional[str] = None
    contours: Optional[List[List[Tuple[float, float]]]] = None
    preview: Optional[str] = None


class MeasureResponse(BaseModel):
    front: List[NailView]
    side: Optional[List[NailView]] = None


def homography_from_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        charuco_corners, charuco_ids = detector.detectBoard(clahe.apply(gray))[:2]

    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < 4:
        raise HTTPException(status_code=400, detail="ChArUco failed")

    cols = SQUARES_X - 1
    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    mm_points = np.array(
        [
            ((int(cid) % cols + 1) * SQUARE_MM, (int(cid) // cols + 1) * SQUARE_MM)
            for cid in charuco_ids.flatten()
        ],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(image_points, mm_points, cv2.RANSAC, 2.0)
    if homography is None:
        raise HTTPException(status_code=400, detail="Homography failed")
    return homography


def decode_image(contents):
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image


def jpeg_data_url(gray):
    ok, encoded = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def analyze_view(image, scale_name):
    homography = homography_from_board(image) if scale_name == "board" else None
    nail_masks = SEG_BACKENDS[scale_name].segment(image)
    if not nail_masks:
        raise HTTPException(status_code=400, detail="Nail detection failed")

    items = []
    for nail_mask in nail_masks:
        contour = nail_mask.contour
        gray = None
        contours = []
        order_x = 0.0
        if contour is not None and len(contour) >= 3:
            points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
            order_x = float(points[:, 0].mean())
            contours = [format_contour(contour)]
            gray = prepare_nail_gray(image, contour)

        length_mm = None
        width_mm = None
        if scale_name == "board":
            measured = measure_nail_from_mask(
                nail_mask.mask,
                homography,
                camera_height_mm=CAMERA_HEIGHT_MM,
                nail_height_mm=NAIL_HEIGHT_MM,
            )
            if not measured:
                continue
            length_mm = round(measured["length_mm"], 2)
            width_mm = round(measured["width_mm"], 2) if measured.get("width_mm") is not None else None
        # else:
        #     하드웨어 mm는 아직 없다. 세그·전처리·형태만 반환한다.

        items.append({
            "x": order_x,
            "gray": gray,
            "result": NailView(
                length_mm=length_mm,
                width_mm=width_mm,
                contours=contours,
                preview=jpeg_data_url(gray) if gray is not None else None,
            ),
        })

    if not items:
        raise HTTPException(status_code=400, detail="Nail measurement failed")
    items.sort(key=lambda item: item["x"])
    return items


def assign_shapes(front_items, side_items, classifier):
    for front_item, side_item in zip(front_items, side_items):
        if front_item["gray"] is None or side_item["gray"] is None:
            continue
        shape = classifier.predict(front_item["gray"], side_item["gray"])
        for item in (front_item, side_item):
            item["result"].shape = shape


@router.post("/measure", response_model=MeasureResponse)
async def measure_nails(
    file: UploadFile = File(...),
    side: Optional[UploadFile] = File(None),
    scale: str = Form("board"),
    group: str = Form("other"),
):
    scale_name = (scale or "board").strip().lower()
    group_name = (group or "other").strip().lower()
    if scale_name not in SCALE_MODES:
        raise HTTPException(status_code=400, detail="scale은 board 또는 hardware 이어야 합니다.")
    if group_name not in FINGER_GROUPS:
        raise HTTPException(status_code=400, detail="group은 thumb 또는 other 이어야 합니다.")

    front_image = decode_image(await file.read())
    front_items = analyze_view(front_image, scale_name)
    side_items = None
    if side is not None and side.filename:
        side_image = decode_image(await side.read())
        side_items = analyze_view(side_image, scale_name)
        assign_shapes(front_items, side_items, FINGER_GROUPS[group_name])

    return MeasureResponse(
        front=[item["result"] for item in front_items],
        side=[item["result"] for item in side_items] if side_items is not None else None,
    )
