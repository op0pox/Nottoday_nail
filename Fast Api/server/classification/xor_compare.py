import cv2
import numpy as np

from classification.compare_pipeline import encode_png, fit_to_canvas

RASTER_SIZE = 512  # 면적을 셀 래스터 한 변
RASTER_PAD = 4  # 도형이 잘리지 않을 만큼만 여백
OVERLAY_TITLE = "XOR 면적"


# 정렬된 두 윤곽을 같은 캔버스에 채운 마스크
def _fill_masks(prepared, size=RASTER_SIZE, pad=RASTER_PAD):
    a = prepared["full_a"]
    b = prepared["full_b"]
    cut_y = prepared["cut_y"]

    # 같은 변환으로 찍어야 두 면적이 같은 자로 잰 값이 된다
    to_px, mn, scale = fit_to_canvas(np.vstack([a, b]), size, pad)
    mask_a = np.zeros((size, size), np.uint8)
    mask_b = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask_a, [to_px(a)], 255)
    cv2.fillPoly(mask_b, [to_px(b)], 255)

    if cut_y is not None:
        # 정렬 좌표는 뿌리가 y=0이고 팁이 음수라, 픽셀에서는 뿌리가 아래쪽 행이다.
        # 절단선보다 위(팁 쪽) 행을 지우면 뿌리 띠만 남는다.
        # 닫힌 윤곽을 채운 뒤 자르므로 점 순서가 감겨도 도형이 안 꼬인다
        row = int(round((cut_y - mn[1]) * scale)) + pad
        row = max(0, min(size, row))
        mask_a[:row] = 0
        mask_b[:row] = 0

    return mask_a, mask_b


# 겹치지 않는 면적의 비율
def score(prepared):
    mask_a, mask_b = _fill_masks(prepared)
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    mean_area = (area_a + area_b) / 2.0
    if mean_area <= 0:
        raise ValueError("XOR 면적을 잴 도형이 없습니다")

    # 합집합에서 교집합을 뺀 부분.
    # 픽셀 수를 그대로 쓰면 쌍마다 래스터 배율이 달라 비교가 안 되므로 평균 면적으로 나눈다
    xor_area = int(np.count_nonzero(cv2.bitwise_xor(mask_a, mask_b)))
    return xor_area / mean_area


# 교집합과 어긋난 부분을 색으로 구분한 PNG
def overlay_png(prepared):
    mask_a, mask_b = _fill_masks(prepared)
    both = cv2.bitwise_and(mask_a, mask_b)
    only_a = cv2.bitwise_and(mask_a, cv2.bitwise_not(mask_b))
    only_b = cv2.bitwise_and(mask_b, cv2.bitwise_not(mask_a))

    img = np.full((mask_a.shape[0], mask_a.shape[1], 3), 255, np.uint8)
    img[both > 0] = (225, 225, 225)  # 겹친 곳은 회색으로 깔고
    img[only_a > 0] = (0, 180, 0)  # 1번째에만 있는 곳은 초록
    img[only_b > 0] = (0, 0, 220)  # 2번째에만 있는 곳은 빨강
    return encode_png(img)
