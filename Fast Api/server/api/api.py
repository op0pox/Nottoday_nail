import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional, Tuple

from segmentation.nail_segmentation import YoloNailBackend, measure_nail_from_mask
from classification.nail_classification import classify_nail_shape

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
    shape: Optional[str] = None          # 쉐입 분류 결과
    shape_score: Optional[float] = None  # 오차 점수 (낮을수록 일치율 높음)
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
    for nail_mask in nail_masks:
        # [수정] 챔퍼 거리 계산을 위해 CHAIN_APPROX_NONE 사용
        contours, _ = cv2.findContours(nail_mask.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        best_shape = None
        min_dist = None
        formatted_contours = []
        
        if contours:
            # 1) 쉐입 분류 로직 실행 (가장 큰 외곽선 기준)
            main_contour = max(contours, key=cv2.contourArea)
            best_shape, min_dist = classify_nail_shape(main_contour, "shape_templates.json")
            
            # 2) 프론트엔드 반환을 위한 윤곽선 리스트 포맷팅
            # (프론트 전송용으로는 너무 무거울 수 있으므로, 여기서만 다시 SIMPLE을 적용해 윤곽선을 줄여서 보내는 것도 좋은 최적화 방법입니다)
            for cnt in contours:
                cnt_list = cnt.squeeze().tolist()
                if not isinstance(cnt_list[0], list):
                    cnt_list = [cnt_list]
                formatted_contours.append(cnt_list)

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