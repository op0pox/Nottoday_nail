import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional, Tuple

from segmentation.nail_segmentation import YoloNailBackend, measure_nail_from_mask

router = APIRouter(prefix="/api")

# 체커보드 규격 하드코딩
SQUARES_X = 18
SQUARES_Y = 26
SQUARE_MM = 10.0
MARKER_MM = 7.0
CAMERA_HEIGHT_MM = 295.0 # 피사체로부터 카메라의 거리
NAIL_HEIGHT_MM = 0.0 # 바닥에서부터 손톱까지의 높이(원근 오차 보정)

# 세그멘테이션 모델파일과 체커보드 보정 변수 선언
backend = YoloNailBackend(conf=0.25)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

def get_mm_point(charuco_id):
    cols = SQUARES_X - 1
    col = charuco_id % cols
    row = charuco_id // cols
    return (col + 1) * SQUARE_MM, (row + 1) * SQUARE_MM

class MeasurementResult(BaseModel):
    length_mm: float
    width_mm: Optional[float] = None
    contours: Optional[List[List[Tuple[int, int]]]] = None 
    
@router.post("/measure", response_model=List[MeasurementResult])
async def measure_nails(
    file: UploadFile = File(...),
    hand: str = Form("right")
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < 4:
        raise HTTPException(status_code=400, detail="ChArUco failed")

    image_points = charuco_corners.reshape(-1, 2).astype(np.float32)
    mm_points = np.array(
        [get_mm_point(int(cid)) for cid in charuco_ids.flatten()],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(image_points, mm_points, cv2.RANSAC, 2.0)
    if H is None:
        raise HTTPException(status_code=400, detail="Homography failed")

    nail_masks = backend.segment(image)
    if not nail_masks:
        raise HTTPException(status_code=400, detail="Nail detection failed")

    results = []
    for nm in nail_masks:
        contours, _ = cv2.findContours(nm.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        formatted_contours = []
        for cnt in contours:
            cnt_list = cnt.squeeze().tolist()
            if not isinstance(cnt_list[0], list):
                 cnt_list = [cnt_list]
            formatted_contours.append(cnt_list)

        measured = measure_nail_from_mask(
            nm.mask,
            H,
            camera_height_mm=CAMERA_HEIGHT_MM,
            nail_height_mm=NAIL_HEIGHT_MM,
        )
        
        if measured:
            results.append(MeasurementResult(
                length_mm=round(measured["length_mm"], 2),
                width_mm=round(measured["width_mm"], 2) if measured.get("width_mm") is not None else None,
                contours=formatted_contours  
            ))

    return results