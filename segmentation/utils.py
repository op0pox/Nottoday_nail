"""
utils.py
========
segmentation 프로젝트의 공용 도구 상자.
직접 실행하는 파일이 아니라 다른 파일에서
    from utils import ...
형태로 불러와서 사용한다.

여기 있는 함수들:
- ensure_dir()            : 폴더가 없으면 자동 생성
- timestamp_filename()    : 날짜/시간이 들어간 파일명 생성
- check_gui_available()   : SSH 환경에서 화면(GUI) 창을 띄울 수 있는지 미리 확인
- load_json() / save_json(): JSON 파일 읽기/쓰기
- transform_points_to_mm(), parallax_factor(), measure_length_mm()
                          : ChArUco 호모그래피 기반 mm 변환
- FINGERS, FINGER_LABELS_KO : 5개 손가락 이름 상수
"""

import os
import sys
import json
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 손가락 관련 상수
# ---------------------------------------------------------------------------
# 내부적으로(코드, 결과 JSON 등) 사용하는 영문 이름은 이 순서를 그대로 따른다.
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# 화면에 출력할 때 사용하는 한글 이름
FINGER_LABELS_KO = {
    "thumb": "엄지",
    "index": "검지",
    "middle": "중지",
    "ring": "약지",
    "pinky": "소지",
}


# ---------------------------------------------------------------------------
# 폴더 / 파일 유틸
# ---------------------------------------------------------------------------
def ensure_dir(path):
    """
    path가 파일 경로든 폴더 경로든 상관없이,
    필요한 상위 폴더가 없으면 자동으로 만들어준다.

    예: ensure_dir("results/2026_07_02.jpg")
        -> "results" 폴더가 없으면 생성한다.
    """
    folder = path
    # 확장자가 있는 "파일 경로"로 보이면 상위 폴더만 추출
    if os.path.splitext(path)[1]:
        folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def timestamp_filename(prefix="capture", ext="jpg"):
    """
    날짜와 시간이 포함된 파일명을 만들어 준다.
    예: capture_20260702_193045.jpg
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}.{ext}"


def load_json(path):
    """JSON 파일을 읽어서 dict로 반환한다. 파일이 없으면 None을 반환한다."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """dict를 JSON 파일로 저장한다. 상위 폴더가 없으면 자동으로 만든다."""
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_gui_available():
    """
    현재 환경에서 GUI 창을 띄울 수 있는지 간단히 확인한다.

    SSH로 접속만 한 상태(X11 포워딩 없음)에서는 화면이 없기 때문에
    창을 띄우려 하면 "Could not connect to display" 류의 오류가 난다.
    완벽한 검증은 아니지만, 흔한 실수를 미리 막기 위해
    DISPLAY 환경변수가 비어있는 리눅스 환경이면 경고를 출력한다.
    """
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            print(
                "\n"
                "[경고] 현재 화면(DISPLAY) 정보가 없습니다.\n"
                "일반 SSH 접속 상태라면 GUI 창이 뜨지 않습니다.\n"
                "  1) X11 포워딩으로 SSH 접속: ssh -X user@ip\n"
                "  2) 모니터/키보드를 직접 연결해서 그 화면에서 실행\n"
            )
            return False
    return True


# ---------------------------------------------------------------------------
# ChArUco 호모그래피 기반 mm 변환
# ---------------------------------------------------------------------------
def transform_points_to_mm(homography, points):
    """
    이미지 픽셀 좌표들을 호모그래피(homography) 행렬을 이용해
    보드 평면의 mm 좌표로 변환한다.

    homography: 3x3 행렬 (list 또는 np.ndarray)
    points: [(x1, y1), (x2, y2), ...] 형태의 픽셀 좌표 리스트

    반환값: (N, 2) 크기의 mm 좌표 numpy 배열
    """
    H = np.asarray(homography, dtype=np.float64)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def parallax_factor(camera_height_mm, nail_height_mm):
    """
    손톱이 모눈판(보드) 평면보다 손가락 두께만큼 카메라 쪽으로 떠 있을 때
    생기는 시차(원근) 오차를 보정하는 배율을 계산한다.

    camera_height_mm: 카메라 렌즈 ~ 보드 평면까지의 높이 (mm)
    nail_height_mm: 보드 평면 ~ 손톱까지의 높이 (mm, 기본 0이면 보정 없음)
    """
    if not camera_height_mm:
        return 1.0
    return (camera_height_mm - nail_height_mm) / camera_height_mm


def measure_length_mm(homography, point_a, point_b, camera_height_mm=295.0, nail_height_mm=0.0):
    """
    이미지 픽셀 좌표 두 점(point_a, point_b) 사이의 실제 거리를
    호모그래피로 mm 평면에 옮긴 뒤, 시차 보정까지 적용해서 mm 단위로 계산한다.
    """
    mm_pts = transform_points_to_mm(homography, [point_a, point_b])
    raw_length_mm = float(np.linalg.norm(mm_pts[0] - mm_pts[1]))
    return raw_length_mm * parallax_factor(camera_height_mm, nail_height_mm)
