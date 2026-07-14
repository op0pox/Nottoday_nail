# -*- coding: utf-8 -*-
"""
detection/contour.py
======================
[역할]
    1. ThresholdNailSegmenter: 고전 CV(CLAHE -> 블러 -> 적응형 이진화/Canny ->
       모폴로지 -> 최대 컨투어)로 손톱(정면) / 손가락 실루엣(측면) 마스크를 얻는다.
       detection/base.py의 NailSegmenter 인터페이스 구현체이며, 나중에
       detection/dl_segmenter.py의 AI 모델로 교체 가능하다.
    2. extract_front_keypoints() / extract_side_keypoints(): 마스크 하나로부터
       9(front)/7(side)개 키포인트를 계산하는 공용 로직. 마스크가 고전 CV에서
       왔든 AI 모델에서 왔든 이 함수들은 그대로 재사용된다 (마스크 출처 무관).
    3. manual_adjust_keypoints(): 자동 검출 결과를 화면에서 마우스로 드래그해서
       수정하는 수동 보정 UI (공식 검증 단계에서 자동 검출 실패를 보완하기 위해 필수).

[좌표계 규약] 이 파일이 다루는 키포인트는 전부 "픽셀" 좌표다 (mm 변환은
calibration.py가 담당). 이미지 좌표계와 동일하게 y는 아래로 갈수록 커진다.
"""

import math

import cv2
import numpy as np

from detection.base import NailSegmenter

FRONT_KEYPOINT_IDS = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
SIDE_KEYPOINT_IDS = ("1", "2", "3", "4", "5", "9", "10")


# ---------------------------------------------------------------------------
# 1. 마스크 획득 (고전 CV)
# ---------------------------------------------------------------------------
class ThresholdNailSegmenter(NailSegmenter):
    """CLAHE -> 블러 -> 적응형 이진화/Canny -> 모폴로지 -> 최대 컨투어로 마스크를 얻는다."""

    name = "classical"

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.debug_images = {}  # segment() 호출 후 채워짐 (디버그 모드일 때만)

    def segment(self, image_bgr, roi=None):
        cfg = self.config["detection"]
        img_h, img_w = image_bgr.shape[:2]
        if roi is None:
            x, y, w, h = 0, 0, img_w, img_h
        else:
            x, y, w, h = roi
        crop = image_bgr[y : y + h, x : x + w]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=cfg["clahe_clip_limit"], tileGridSize=tuple(cfg["clahe_tile_grid_size"])
        )
        enhanced = clahe.apply(gray)

        blur_kernel = tuple(cfg["blur_kernel"])
        blurred = cv2.GaussianBlur(enhanced, blur_kernel, 0)

        if cfg["use_adaptive_threshold"]:
            block_size = int(cfg["adaptive_block_size"])
            if block_size % 2 == 0:
                block_size += 1  # adaptiveThreshold는 홀수 블록 크기만 허용
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                cfg["adaptive_c"],
            )
        else:
            binary = cv2.Canny(blurred, cfg["canny_low"], cfg["canny_high"])

        morph_kernel = np.ones(tuple(cfg["morph_kernel"]), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel, iterations=cfg["morph_iterations"])
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, morph_kernel, iterations=1)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        crop_area = float(crop.shape[0] * crop.shape[1])
        min_area = cfg["min_contour_area_ratio"] * crop_area
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if self.debug:
            self.debug_images = {
                "1_gray": gray,
                "2_clahe": enhanced,
                "3_blurred": blurred,
                "4_binary": binary,
                "5_morph": opened,
            }

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        mask_crop = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_crop, [largest], -1, 255, thickness=cv2.FILLED)

        mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_full[y : y + h, x : x + w] = mask_crop

        if self.debug:
            self.debug_images["6_mask"] = mask_full

        return mask_full


def save_debug_images(debug_images, output_dir, prefix):
    """ThresholdNailSegmenter.debug_images를 파일로 저장한다."""
    import os

    if not debug_images:
        return []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    saved = []
    for name, img in debug_images.items():
        path = os.path.join(output_dir, "%s_%s.jpg" % (prefix, name))
        cv2.imwrite(path, img)
        saved.append(path)
    return saved


# ---------------------------------------------------------------------------
# 공용 헬퍼: 컨투어 / 곡률
# ---------------------------------------------------------------------------
def _largest_contour_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64)


def _discrete_curvature_closed(points_xy, smoothing_window):
    """
    닫힌 컨투어(폐곡선)에 대한 이산 곡률 k = |x'y''-y'x''| / (x'^2+y'^2)^1.5 을
    계산한다. np.roll로 순환 경계를 처리해서 폐곡선 양 끝단에서도 값이 튀지 않게 한다.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    n = len(pts)
    if n < 5:
        return np.zeros(n)

    if smoothing_window and smoothing_window > 1:
        w = int(smoothing_window)
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w) / float(w)
        xs = _circular_convolve(pts[:, 0], kernel)
        ys = _circular_convolve(pts[:, 1], kernel)
    else:
        xs, ys = pts[:, 0], pts[:, 1]

    dx = (np.roll(xs, -1) - np.roll(xs, 1)) / 2.0
    dy = (np.roll(ys, -1) - np.roll(ys, 1)) / 2.0
    ddx = np.roll(xs, -1) - 2.0 * xs + np.roll(xs, 1)
    ddy = np.roll(ys, -1) - 2.0 * ys + np.roll(ys, 1)

    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom < 1e-9] = 1e-9
    curvature = np.abs(dx * ddy - dy * ddx) / denom
    return curvature


def _circular_convolve(values, kernel):
    """폐곡선용 순환 컨볼루션(경계에서 값이 끊기지 않게)."""
    n = len(values)
    pad = len(kernel) // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]]) if pad > 0 else values
    result = np.convolve(padded, kernel, mode="valid")
    return result[:n]


def _avg_point_near_extreme(points_xy, axis, extreme, tol_px=2.0):
    """
    points_xy에서 axis(0=x, 1=y) 기준 극값(extreme="min"/"max")에 tol_px 이내로
    가까운 점들의 평균 좌표를 반환한다 (직선 변에서 대표점 하나를 뽑기 위함).
    """
    values = points_xy[:, axis]
    target = values.min() if extreme == "min" else values.max()
    near = points_xy[np.abs(values - target) <= tol_px]
    if len(near) == 0:
        near = points_xy[values == target]
    return near.mean(axis=0)


def _nearest_point(points_xy, target_xy):
    diffs = points_xy - np.asarray(target_xy)
    dists = np.einsum("ij,ij->i", diffs, diffs)
    return points_xy[np.argmin(dists)]


def _exclude_near(points_xy, indices, excluded_points, margin_px):
    """indices 중 excluded_points 어느 하나와도 margin_px 이내로 가까운 점은 제외."""
    if not len(indices):
        return indices
    keep = []
    for i in indices:
        p = points_xy[i]
        too_close = False
        for ep in excluded_points:
            if math.hypot(p[0] - ep[0], p[1] - ep[1]) < margin_px:
                too_close = True
                break
        if not too_close:
            keep.append(i)
    return np.array(keep, dtype=np.int64) if keep else np.array(indices)


# ---------------------------------------------------------------------------
# 2. 정면 뷰 키포인트 추출 (9개)
# ---------------------------------------------------------------------------
def extract_front_keypoints(mask, config):
    """
    정면 뷰 9개 키포인트를 마스크에서 계산한다.
        1: 하단(큐티클) 중앙   5: 상단(프리엣지) 중앙
        3: 좌측 최대너비        7: 우측 최대너비
        9: 중심점
        2,4,6,8: 좌하/좌상/우상/우하 곡률 최대점
    반환: {"1": (x,y), ..., "9": (x,y)} (픽셀)
    """
    pts = _largest_contour_points(mask)
    if pts is None or len(pts) < 10:
        raise RuntimeError("정면 마스크에서 유효한 컨투어를 찾지 못했습니다 (점 개수 부족).")

    det_cfg = config["detection"]

    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    else:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    point9 = (cx, cy)

    point1 = tuple(_avg_point_near_extreme(pts, axis=1, extreme="max"))  # 아래(큐티클)
    point5 = tuple(_avg_point_near_extreme(pts, axis=1, extreme="min"))  # 위(프리엣지)
    point3 = tuple(_avg_point_near_extreme(pts, axis=0, extreme="min"))  # 왼쪽
    point7 = tuple(_avg_point_near_extreme(pts, axis=0, extreme="max"))  # 오른쪽

    bbox_diag = math.hypot(*[np.ptp(pts[:, 0]), np.ptp(pts[:, 1])])
    margin_px = det_cfg["curvature_search_margin_ratio"] * bbox_diag

    curvature = _discrete_curvature_closed(pts, det_cfg["curvature_smoothing_window"])
    excluded = [point1, point3, point5, point7]

    idx_all = np.arange(len(pts))
    left_lower = idx_all[(pts[:, 0] < cx) & (pts[:, 1] > cy)]
    left_upper = idx_all[(pts[:, 0] < cx) & (pts[:, 1] <= cy)]
    right_upper = idx_all[(pts[:, 0] >= cx) & (pts[:, 1] <= cy)]
    right_lower = idx_all[(pts[:, 0] >= cx) & (pts[:, 1] > cy)]

    def _pick_max_curvature(indices, fallback):
        indices = _exclude_near(pts, indices, excluded, margin_px)
        if len(indices) == 0:
            return fallback
        best = indices[np.argmax(curvature[indices])]
        return tuple(pts[best])

    point2 = _pick_max_curvature(left_lower, point3)
    point4 = _pick_max_curvature(left_upper, point3)
    point6 = _pick_max_curvature(right_upper, point7)
    point8 = _pick_max_curvature(right_lower, point7)

    return {
        "1": point1,
        "2": point2,
        "3": point3,
        "4": point4,
        "5": point5,
        "6": point6,
        "7": point7,
        "8": point8,
        "9": point9,
    }


# ---------------------------------------------------------------------------
# 3. 측면 뷰 키포인트 추출
# ---------------------------------------------------------------------------
def extract_side_keypoints(mask, config):
    """
    측면 뷰 키포인트를 마스크(손가락 실루엣)에서 계산한다.
        1: 손톱 후단(큐티클 쪽)          5: 손톱 끝점 (윗 프로파일 tip)
        10: 손가락 살 끝점 (아랫 프로파일 tip)
        3, 9: 최대 너비 지점 (위/아래)
        2, 4: 최대 곡률점 (아래쪽/위쪽 프로파일)

    "손톱 끝(5)"과 "살 끝(10)"은 마스크가 손톱+손가락 살을 합친 실루엣이라는
    전제 하에, 끝쪽(tip) 영역을 위/아래 프로파일로 나눠서 각각의 최돌출점으로
    근사한다. 이 근사는 조명/각도에 따라 부정확할 수 있어 수동 보정이 특히
    중요한 부분이다 (README 참고).
    """
    pts = _largest_contour_points(mask)
    if pts is None or len(pts) < 10:
        raise RuntimeError("측면 마스크에서 유효한 컨투어를 찾지 못했습니다 (점 개수 부족).")

    det_cfg = config["detection"]
    side_cfg = config["side_view"]
    tip_direction = side_cfg.get("tip_direction", "right")
    tip_region_ratio = side_cfg.get("tip_region_ratio", 0.35)

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    width = x_max - x_min
    cy = pts[:, 1].mean()

    if tip_direction == "right":
        tip_x_bound = x_max - width * tip_region_ratio
        in_tip_region = pts[:, 0] >= tip_x_bound
        cuticle_extreme = "min"
    else:
        tip_x_bound = x_min + width * tip_region_ratio
        in_tip_region = pts[:, 0] <= tip_x_bound
        cuticle_extreme = "max"

    point1 = tuple(_avg_point_near_extreme(pts, axis=0, extreme=cuticle_extreme))

    upper_mask = pts[:, 1] <= cy
    lower_mask = pts[:, 1] > cy

    tip_upper = pts[in_tip_region & upper_mask]
    tip_lower = pts[in_tip_region & lower_mask]

    axis_extreme = "max" if tip_direction == "right" else "min"
    if len(tip_upper) > 0:
        point5 = tuple(_avg_point_near_extreme(tip_upper, axis=0, extreme=axis_extreme))
    else:
        point5 = tuple(_avg_point_near_extreme(pts, axis=0, extreme=axis_extreme))

    if len(tip_lower) > 0:
        point10 = tuple(_avg_point_near_extreme(tip_lower, axis=0, extreme=axis_extreme))
    else:
        point10 = point5

    # 최대 너비: 각 x열에서 (아래y - 위y) 두께가 가장 큰 위치
    order = np.argsort(pts[:, 0])
    xs_sorted = pts[order, 0]
    best_x = None
    best_thickness = -1.0
    best_top = None
    best_bottom = None
    unique_xs = np.unique(np.round(xs_sorted))
    for ux in unique_xs:
        col = pts[np.abs(pts[:, 0] - ux) <= 1.0]
        if len(col) < 2:
            continue
        top = col[np.argmin(col[:, 1])]
        bottom = col[np.argmax(col[:, 1])]
        thickness = bottom[1] - top[1]
        if thickness > best_thickness:
            best_thickness = thickness
            best_top = top
            best_bottom = bottom
            best_x = ux
    if best_top is None:
        best_top = pts[np.argmin(pts[:, 1])]
        best_bottom = pts[np.argmax(pts[:, 1])]

    point3 = tuple(best_top)
    point9 = tuple(best_bottom)

    # 곡률 최대점: 위쪽 절반/아래쪽 절반 각각에서, 극값점들과 너무 가깝지 않은 곳
    bbox_diag = math.hypot(width, np.ptp(pts[:, 1]))
    margin_px = det_cfg["curvature_search_margin_ratio"] * bbox_diag
    curvature = _discrete_curvature_closed(pts, det_cfg["curvature_smoothing_window"])
    excluded = [point1, point5, point10, point3, point9]

    idx_all = np.arange(len(pts))
    upper_idx = _exclude_near(pts, idx_all[upper_mask], excluded, margin_px)
    lower_idx = _exclude_near(pts, idx_all[lower_mask], excluded, margin_px)

    point4 = tuple(pts[upper_idx[np.argmax(curvature[upper_idx])]]) if len(upper_idx) else point3
    point2 = tuple(pts[lower_idx[np.argmax(curvature[lower_idx])]]) if len(lower_idx) else point9

    return {
        "1": point1,
        "2": point2,
        "3": point3,
        "4": point4,
        "5": point5,
        "9": point9,
        "10": point10,
    }


# ---------------------------------------------------------------------------
# 4. 수동 보정 UI
# ---------------------------------------------------------------------------
def manual_adjust_keypoints(image_bgr, keypoints_px, view, window_name="Manual Adjust", select_radius=18):
    """
    자동 검출된 키포인트를 화면에 표시하고, 마우스로 드래그해서 위치를 수정한다.

    조작:
        - 점 근처를 눌러서 드래그 -> 놓으면 그 위치로 이동
        - 's' 또는 Enter: 확정하고 종료
        - 'q': 취소 (원본 keypoints_px 그대로 반환)

    반환: 수정된(또는 취소 시 원본) keypoints_px dict
    """
    ids = FRONT_KEYPOINT_IDS if view == "front" else SIDE_KEYPOINT_IDS
    points = {k: list(v) for k, v in keypoints_px.items() if k in ids}
    state = {"dragging": None}

    def _redraw():
        disp = image_bgr.copy()
        for kid, (x, y) in points.items():
            cv2.circle(disp, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(
                disp, kid, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        cv2.putText(
            disp,
            "drag: move point | s/Enter: confirm | q: cancel",
            (10, disp.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        return disp

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest_id, nearest_dist = None, float("inf")
            for kid, (px, py) in points.items():
                d = math.hypot(px - x, py - y)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_id = kid
            if nearest_id is not None and nearest_dist <= select_radius:
                state["dragging"] = nearest_id
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] is not None:
            points[state["dragging"]] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    cancelled = False
    while True:
        cv2.imshow(window_name, _redraw())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), 13):  # 's' 또는 Enter
            break
        if key == ord("q"):
            cancelled = True
            break

    cv2.destroyWindow(window_name)

    if cancelled:
        return {k: tuple(v) for k, v in keypoints_px.items()}
    return {k: tuple(v) for k, v in points.items()}
