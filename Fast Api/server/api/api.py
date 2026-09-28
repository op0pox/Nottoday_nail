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

# 체커보드(board) 방식은 지금 쓰지 않아 임시로 끈다. 세그 모델과 ChArUco 보드를 만들지 않고,
# .env 의 체커보드 값도 읽지 않는다. 다시 쓰려면 True 로 바꾸고 .env 체커보드 값을 채운다.
BOARD_ENABLED = False

# feat/nano-capture 의 측면 식. 1920x1080은 크롭이라 초점(px)을 나누지 않는다.
FOCAL_PX_FULL = 3.04 * 1000.0 / 1.12
SENSOR_FOCALS = {
    (3264, 2464): FOCAL_PX_FULL,
    (1920, 1080): FOCAL_PX_FULL,
    (1640, 1232): FOCAL_PX_FULL / 2,
    (1280, 720): FOCAL_PX_FULL / 2,
}


def env_float(name, default):
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


if BOARD_ENABLED:
    SQUARES_X = int(os.getenv("SQUARES_X"))
    SQUARES_Y = int(os.getenv("SQUARES_Y"))
    SQUARE_MM = float(os.getenv("SQUARE_MM"))
    MARKER_MM = float(os.getenv("MARKER_MM"))
CAMERA_HEIGHT_MM = env_float("CAMERA_HEIGHT_MM", 0.0)
NAIL_HEIGHT_MM = env_float("NAIL_HEIGHT_MM", 0.0)

# 측면은 렌즈~손톱 7.0cm. 정면은 아래 실측선으로 배율을 정한다.
SIDE_DISTANCE_CM = env_float("SIDE_DISTANCE_CM", 7.0)

# 정면 원본 1920x1080을 1024x576으로 줄인 사진에서 잰 선.
# 세로(빨강) 131px = 14mm, 가로(파랑) 87px = 9.5mm.
FRONT_REF_SIZE = (1024, 576)
FRONT_LENGTH_PX = 131.0
FRONT_LENGTH_MM = 14.0
FRONT_WIDTH_PX = 87.0
FRONT_WIDTH_MM = 9.5

# 실측으로 맞춘 영점(mm). 길이, 폭 순. env offset은 이 위에 더한다.
ZERO_MM = {
    "front": (4.0, 2.3),
    "side": (4.7, 1.4),
}

# 식 뒤에 더하는 mm. 비우면 0.
HARDWARE_OFFSET_MM = {
    "front": (env_float("FRONT_LENGTH_OFFSET_MM", 0.0), env_float("FRONT_WIDTH_OFFSET_MM", 0.0)),
    "side": (env_float("SIDE_LENGTH_OFFSET_MM", 0.0), env_float("SIDE_WIDTH_OFFSET_MM", 0.0)),
}

router = APIRouter(prefix="/api")
SCALE_MODES = ("board", "hardware") if BOARD_ENABLED else ("hardware",)
# board는 체커보드 사진용 세그, hardware는 흰 배경용 세그
SEG_BACKENDS = {
    "hardware": YoloNailBackend(model_path("seg_white.pt")),
}
if BOARD_ENABLED:
    SEG_BACKENDS["board"] = YoloNailBackend(model_path("seg_checkerboard.pt"))
FINGER_GROUPS = {
    "thumb": ShapeClassifier(model_path("cls_thumb.pt")),
    "other": ShapeClassifier(model_path("cls_other.pt")),
}
detector = None
if BOARD_ENABLED:
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


def homography_from_side(image, distance_cm):
    if distance_cm <= 0:
        raise HTTPException(status_code=400, detail="하드웨어 거리는 0보다 커야 합니다.")
    height, width = image.shape[:2]
    focal_px = SENSOR_FOCALS.get((width, height)) or SENSOR_FOCALS.get((height, width))
    if focal_px is None:
        supported = ", ".join("%dx%d" % (w, h) for w, h in SENSOR_FOCALS)
        raise HTTPException(status_code=400, detail="지원하지 않는 해상도입니다: %dx%d (%s)" % (width, height, supported))
    scale = (distance_cm * 10.0) / focal_px
    return np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def homography_from_front(image):
    height, width = image.shape[:2]
    ref_w, ref_h = FRONT_REF_SIZE
    scale_x = FRONT_WIDTH_MM / (FRONT_WIDTH_PX * width / ref_w)
    scale_y = FRONT_LENGTH_MM / (FRONT_LENGTH_PX * height / ref_h)
    return np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


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


def analyze_view(image, scale_name, camera):
    if scale_name == "board":
        homography = homography_from_board(image)
        camera_height_mm = CAMERA_HEIGHT_MM
        nail_height_mm = NAIL_HEIGHT_MM
    else:
        # 거리는 손톱 표면까지라서 플레이트 시차 보정은 더하지 않는다.
        homography = homography_from_front(image) if camera == "front" else homography_from_side(image, SIDE_DISTANCE_CM)
        camera_height_mm = 0.0
        nail_height_mm = 0.0
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

        measured = measure_nail_from_mask(
            nail_mask.mask,
            homography,
            camera_height_mm=camera_height_mm,
            nail_height_mm=nail_height_mm,
        )
        if not measured:
            continue
        if scale_name == "hardware":
            length_bias, width_bias = ZERO_MM[camera]
            length_offset, width_offset = HARDWARE_OFFSET_MM[camera]
        else:
            length_bias, width_bias = 0.0, 0.0
            length_offset, width_offset = 0.0, 0.0
        length_mm = round(measured["length_mm"] + length_bias + length_offset, 2)
        width_mm = round(measured["width_mm"] + width_bias + width_offset, 2) if measured.get("width_mm") is not None else None

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
    scale: str = Form("hardware"),
    group: str = Form("other"),
):
    scale_name = (scale or "hardware").strip().lower()
    group_name = (group or "other").strip().lower()
    if scale_name not in SCALE_MODES:
        raise HTTPException(status_code=400, detail="scale은 %s 중 하나여야 합니다. (체커보드는 임시 비활성)" % ", ".join(SCALE_MODES))
    if group_name not in FINGER_GROUPS:
        raise HTTPException(status_code=400, detail="group은 thumb 또는 other 이어야 합니다.")

    # file(왼쪽)은 측면, side(오른쪽)은 정면. 응답 키는 화면 순서 그대로다.
    left_image = decode_image(await file.read())
    left_items = analyze_view(left_image, scale_name, "side")
    right_items = None
    if side is not None and side.filename:
        right_image = decode_image(await side.read())
        right_items = analyze_view(right_image, scale_name, "front")
        assign_shapes(right_items, left_items, FINGER_GROUPS[group_name])

    return MeasureResponse(
        front=[item["result"] for item in left_items],
        side=[item["result"] for item in right_items] if right_items is not None else None,
    )
