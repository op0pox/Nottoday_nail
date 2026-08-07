# -*- coding: utf-8 -*-
"""
segment.py
============
자동 세그멘테이션 + 너비/길이 추출.

정면(위→아래 촬영)/측면(옆에서 촬영) 모두 **손가락 1개씩** 사진을 찍는다.
그룹사진 한 장에서 색상만으로 5개 손톱을 자동으로 구분하는 방식도 시도했지만,
실제 조명에서 손가락 중간 마디의 하이라이트가 손톱보다 더 "손톱스러운" 색상
후보로 잡혀서 라벨이 자꾸 엉키는 걸 확인하고 포기했다 — 사진 1장당 손가락이
하나뿐이면애초에 "이 마스크가 어느 손가락이냐"는 모호함 자체가 없어져서
훨씬 안정적이다.

두 측정 모두 "마스크 안에서 최댓값을 찾는" 방식 대신 "고정된 기준 위치(중앙)에서
재는" 방식을 쓴다 — 사진 속 손가락이 살짝 기울어 있을 때 탐색 방식은 엉뚱한
위치(대각선 방향 등)를 최댓값으로 고를 수 있어서, 항상 같은 기준(중앙)에서
재는 쪽이 더 예측 가능하고 안정적이다. mask_width_points/mask_height_points가
이 "고정 기준" 로직이고, 정면/측면 양쪽에서 축만 바꿔 재사용한다
(nail_measure의 find_horizontal_width_endpoints / find_vertical_axis_endpoints와 동일한 방식).

정면 손톱 마스크는 사용자가 별도로 학습시킨 YOLOv8-seg 모델(yolo_nail_mask)로
찾는다 — 색상/위치 휴리스틱보다 실측에서 훨씬 정확했다 (경계가 정확히 손톱
모양을 따라간다). 모델을 못 쓰는 경우에만 손가락 전체 실루엣(피부색 기반,
_skin_mask)에서 위쪽 절반을 잘라 쓰는 근사치로 폴백한다.

측면은 아직 이 모델을 안 쓴다 (기존 색상 기반 방식이 이미 잘 동작해서 그대로
둠). 손가락 실루엣은 피부색 기반(_skin_mask)으로 찾는다 — 사진에 보드(체커
무늬)가 항상 같이 찍히므로, 적응형 이진화 같은 밝기 기반 방식은 보드 패턴을
손가락으로 잘못 집을 수 있는데 보드는 피부색이 아니므로 자연히 제외된다.
"""

import cv2
import numpy as np

from board_calib import mm_distance

# 정면 손톱 검출용 YOLOv8-seg 모델 (사용자가 다른 폴더에서 직접 학습시킨 것).
# 이 저장소 밖에 있는 파일이라 절대경로로 참조한다 — 다른 컴퓨터에서 돌리려면
# 이 경로를 실제 모델 위치로 바꿀 것.
YOLO_MODEL_PATH = "C:/src/nail_segmentation/YOLOv8/nails_seg_s_yolov8_v1.pt"
YOLO_CONF_THRESHOLD = 0.5

_yolo_model = None
_yolo_load_error = None


def _get_yolo_model():
    """YOLO 모델을 최초 1회만 로드해서 재사용한다 (로딩에 몇 초 걸림)."""
    global _yolo_model, _yolo_load_error
    if _yolo_model is not None or _yolo_load_error is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - 모델 로딩 실패 원인이 다양해서 광범위하게 잡음
        _yolo_load_error = str(exc)
    return _yolo_model

FINGER_LABELS_KO = {
    "thumb": "엄지", "index": "검지", "middle": "중지", "ring": "약지", "pinky": "소지",
}


class SegmentationError(RuntimeError):
    pass


WIDTH_MIN_MM = 1.0
WIDTH_MAX_MM = 25.0
LENGTH_MIN_MM = 3.0
LENGTH_MAX_MM = 40.0  # 정면은 손가락 전체 마스크를 쓰므로(손톱보다 조금 더 잡힘) 여유를 좀 더 둠


SKIN_Y_MIN = 120  # 이 밝기 미만은 손 그림자로 보고 제외 (아래 설명 참고)


def _skin_mask(image_bgr):
    """
    YCrCb 색공간에서 Cr/Cb(색상)만으로 피부색을 판정하고 Y(밝기)는 원래
    제한이 없었는데, 그러면 손가락이 보드에 드리운 그림자도 "색상은 피부와
    같고 밝기만 어두운" 상태라서 같은 피부 덩어리로 잡혀버리는 문제가 실측에서
    확인됐다 (정면 사진에서 그림자 진 옆 체커보드 칸까지 손가락으로 잘못
    잡히는 원인). Y 하한(SKIN_Y_MIN)을 둬서 그림자를 제외한다.
    """
    h, w = image_bgr.shape[:2]
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (SKIN_Y_MIN, 133, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hand_contour = max(contours, key=cv2.contourArea)
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hand_mask, [hand_contour], -1, 255, thickness=cv2.FILLED)
    return hand_mask


def single_finger_mask(image_bgr):
    """손가락 1개짜리 사진(정면/측면 공용)에서 손가락 실루엣 마스크를 찾는다."""
    skin_mask = _skin_mask(image_bgr)
    if skin_mask is None:
        return None
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


FRONT_TIP_RATIO = 0.5  # 정면: 손가락 마스크 세로 범위 중 위쪽(손톱 끝 방향) 이 비율만 손톱으로 간주


def tip_region_mask(mask, ratio):
    """
    마스크의 세로 범위 중 위쪽(손톱 끝 방향) ratio만큼만 남긴다 (색상 필터
    없이 위치로만 제한). 정면 사진에서 손가락 전체 마스크를 그대로 쓰면
    손톱 아래 손가락 살까지 포함돼 길이가 실측보다 훨씬 크게 나오는 문제가
    있었고, 반대로 색상(채도) 필터로 손톱만 추리려 하면 사진 대부분이 이미
    손톱이라 오히려 손톱 안의 하이라이트 한 조각만 남는 문제가 있었다.
    "위쪽 절반만 쓴다"는 단순 위치 기준이 그 중간에서 실측으로 가장 무난했다.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    tip_boundary = y_min + int((y_max - y_min) * ratio)
    result = mask.copy()
    result[tip_boundary:, :] = 0
    return result


NAIL_TIP_SEARCH_RATIO = 0.45  # 측면: 손가락 마스크 세로 범위 중 위쪽(끝 방향) 이 비율만 손톱 후보로 탐색
MIN_NAIL_AREA_PX = 80


def _nail_candidate_mask(image_bgr, region_mask):
    """region_mask 안에서 채도 낮고 밝은(손톱 특유의 광택) 픽셀만 골라낸다."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    region_sat_values = sat[region_mask > 0]
    if region_sat_values.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    sat_threshold = float(np.percentile(region_sat_values, 35))
    candidate = np.zeros((h, w), dtype=np.uint8)
    candidate[(region_mask > 0) & (sat < sat_threshold) & (val > 80)] = 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return candidate


def nail_submask(image_bgr, finger_mask):
    """
    손가락 실루엣(finger_mask) 안에서 손톱만 골라낸다.

    "채도 낮고 밝은 영역=손톱"이라는 색상 기준만 손가락 전체에 적용하면, 손가락
    밑동 쪽 하이라이트가 손톱보다 더 크게 잡혀서 엉뚱한 곳을 고르는 문제가
    실측에서 확인됐다 (side_thumb 실측 사진 참고). 그래서 "손톱 끝은 항상 사진
    위쪽을 향하게 촬영한다"는 촬영 규칙(finger.html 안내)으로 탐색 범위 자체를
    마스크 위쪽 NAIL_TIP_SEARCH_RATIO 구간으로 미리 좁힌 뒤에 색상 기준을 적용한다.
    후보를 못 찾으면 None을 반환한다 (호출부가 손가락 전체 마스크로 폴백해야 함).
    """
    ys, xs = np.nonzero(finger_mask)
    if len(ys) == 0:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    tip_boundary = y_min + int((y_max - y_min) * NAIL_TIP_SEARCH_RATIO)
    tip_region = finger_mask.copy()
    tip_region[tip_boundary:, :] = 0

    nail_candidate = _nail_candidate_mask(image_bgr, tip_region)
    contours, _ = cv2.findContours(nail_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_NAIL_AREA_PX:
        return None
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def best_nail_mask(image_bgr):
    """손톱만 골라낸 마스크를 우선 시도하고, 실패하면 손가락 전체로 폴백한다."""
    finger_mask = single_finger_mask(image_bgr)
    if finger_mask is None:
        return None
    nail_mask = nail_submask(image_bgr, finger_mask)
    return nail_mask if nail_mask is not None else finger_mask


def mask_width_points(mask):
    """
    마스크의 세로 중앙 높이(y 범위 중간)에서 좌/우 끝점을 찾는다.
    (nail_measure의 find_horizontal_width_endpoints와 동일한 방식 — "가로")
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y_mid = int(round((ys.min() + ys.max()) / 2.0))
    row_ys = np.unique(ys)
    if y_mid not in row_ys:
        y_mid = int(row_ys[np.argmin(np.abs(row_ys - y_mid))])
    row_xs = xs[ys == y_mid]
    return (float(row_xs.min()), float(y_mid)), (float(row_xs.max()), float(y_mid))


def mask_height_points(mask):
    """
    마스크의 가로 중앙(x 범위 중간)에서 위/아래 끝점을 찾는다. mask_width_points와
    가로/세로 축만 바꾼 대칭 버전이다.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x_mid = int(round((xs.min() + xs.max()) / 2.0))
    col_xs = np.unique(xs)
    if x_mid not in col_xs:
        x_mid = int(col_xs[np.argmin(np.abs(col_xs - x_mid))])
    col_ys = ys[xs == x_mid]
    return (float(x_mid), float(col_ys.min())), (float(x_mid), float(col_ys.max()))


def yolo_nail_mask(image_bgr):
    """
    학습된 YOLOv8-seg 모델로 손톱 마스크를 찾는다 (정면 전용). 색상/위치 휴리스틱
    (채도 필터, 위쪽 절반 자르기 등)보다 실측에서 훨씬 정확했다 — 진짜 손톱
    모양을 인식하는 모델이라 경계가 삐뚤빼뚤하거나 하이라이트에 낚이는 문제가
    없다. 여러 개 검출되면 신뢰도(conf)가 가장 높은 것 하나를 쓴다.
    모델 로딩 실패/검출 실패 시 None을 반환한다.
    """
    model = _get_yolo_model()
    if model is None:
        return None
    result = model(image_bgr, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None

    confs = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    if confs[best_idx] < YOLO_CONF_THRESHOLD:
        return None

    h, w = image_bgr.shape[:2]
    m = result.masks.data[best_idx].cpu().numpy()
    m = cv2.resize(m, (w, h))
    return (m > 0.5).astype(np.uint8) * 255


def measure_front_finger(image_bgr, homography):
    """
    손가락 1개 정면사진(위→아래 촬영) -> front_width_mm(가로) + front_length_mm(세로).

    손톱 마스크는 학습된 YOLOv8-seg 모델(yolo_nail_mask)로 찾는다. 색상/위치
    휴리스틱(채도 필터로 하이라이트만 남거나, 위쪽 절반을 잘라 손가락 살까지
    포함되는 등)보다 실측에서 훨씬 정확했다. 모델이 없거나 이 사진에서 손톱을
    못 찾으면 위쪽 절반 근사(tip_region_mask)로 폴백한다.
    """
    mask = yolo_nail_mask(image_bgr)
    if mask is None:
        finger_mask = single_finger_mask(image_bgr)
        if finger_mask is None:
            raise SegmentationError("손(피부색) 영역을 찾지 못했습니다. 조명/배경을 확인하고 다시 촬영해주세요.")
        mask = tip_region_mask(finger_mask, FRONT_TIP_RATIO)
        if mask is None:
            raise SegmentationError("측정 지점을 찾지 못했습니다.")

    w_pts = mask_width_points(mask)
    l_pts = mask_height_points(mask)
    if w_pts is None or l_pts is None:
        raise SegmentationError("측정 지점을 찾지 못했습니다.")

    p1, p2 = w_pts
    lp1, lp2 = l_pts
    width_mm = mm_distance(homography, p1, p2)
    length_mm = mm_distance(homography, lp1, lp2)

    if not (WIDTH_MIN_MM <= width_mm <= WIDTH_MAX_MM):
        raise SegmentationError(
            "가로너비 계산값(%.1fmm)이 정상 범위(%.0f~%.0fmm)를 벗어났습니다. "
            "조명/배경/보드 겹침을 확인하고 재촬영해주세요." % (width_mm, WIDTH_MIN_MM, WIDTH_MAX_MM)
        )
    if not (LENGTH_MIN_MM <= length_mm <= LENGTH_MAX_MM):
        raise SegmentationError(
            "세로길이 계산값(%.1fmm)이 정상 범위(%.0f~%.0fmm)를 벗어났습니다. "
            "조명/배경/보드 겹침을 확인하고 재촬영해주세요." % (length_mm, LENGTH_MIN_MM, LENGTH_MAX_MM)
        )

    return {
        "front_width_mm": width_mm,
        "front_length_mm": length_mm,
        "point_a_px": p1,
        "point_b_px": p2,
        "length_point_a_px": lp1,
        "length_point_b_px": lp2,
        "mask": mask,
    }


def measure_side_width(image_bgr, homography):
    """
    측면사진(손가락 1개) -> side_width_mm.
    측면사진도 정면과 같은 세로(손가락 길이가 위아래) 방향으로 찍는 것을
    전제로 하므로, 가로 방향(mask_width_points, 세로 중앙 고정)으로 재야
    손톱 두께가 나온다 (세로로 재면 손가락 length를 재는 꼴이 된다).
    """
    mask = best_nail_mask(image_bgr)
    if mask is None:
        raise SegmentationError("손(피부색) 영역을 찾지 못했습니다. 조명/배경을 확인하고 다시 촬영해주세요.")
    pts = mask_width_points(mask)
    if pts is None:
        raise SegmentationError("측면 너비 지점을 찾지 못했습니다.")
    p_left, p_right = pts
    width_mm = mm_distance(homography, p_left, p_right)
    if not (WIDTH_MIN_MM <= width_mm <= WIDTH_MAX_MM):
        raise SegmentationError(
            "측면 너비 계산값(%.1fmm)이 정상 범위(%.0f~%.0fmm)를 벗어났습니다. "
            "보드 무늬나 배경이 손가락과 함께 잡혔을 수 있습니다. 손가락만 크고 또렷하게 "
            "다시 촬영해주세요." % (width_mm, WIDTH_MIN_MM, WIDTH_MAX_MM)
        )
    return {
        "side_width_mm": width_mm,
        "mask": mask,
        "point_left_px": p_left,
        "point_right_px": p_right,
    }
