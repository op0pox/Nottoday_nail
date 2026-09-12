import os

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_DIR = os.getenv("MODEL_DIR")
CAMERA_HEIGHT_MM = float(os.getenv("CAMERA_HEIGHT_MM"))
NAIL_HEIGHT_MM = float(os.getenv("NAIL_HEIGHT_MM"))


# 픽셀 두 점 사이 거리를 mm로 환산
def measure_length_mm(homography, point_a, point_b, camera_height_mm=CAMERA_HEIGHT_MM, nail_height_mm=NAIL_HEIGHT_MM):
    H = np.asarray(homography, dtype=np.float64)
    pts = np.array([point_a, point_b], dtype=np.float32).reshape(-1, 1, 2)
    mm_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    raw_length_mm = float(np.linalg.norm(mm_pts[0] - mm_pts[1]))

    if not camera_height_mm:
        return raw_length_mm
    # 손톱이 플레이트에서 떠 있으면 카메라에 그만큼 가까워 실제보다 크게 찍힌다.
    # 높이 비율을 곱해 플레이트에 붙어 있을 때의 크기로 되돌린다
    return raw_length_mm * (camera_height_mm - nail_height_mm) / camera_height_mm


# 손톱 한 개의 마스크
class NailMask:
    def __init__(self, mask, confidence=1.0, bbox=None):
        self.mask = mask
        self.confidence = confidence
        self.bbox = bbox


# YOLO 세그멘테이션으로 손톱 마스크 추출
class YoloNailBackend:
    def __init__(self, conf=0.25, min_area_ratio=0.0003):
        self.conf = conf
        self.min_area_ratio = min_area_ratio
        if not MODEL_DIR:
            raise RuntimeError("MODEL_DIR is not set")
        if not MODEL_DIR.startswith(("http://", "https://")) and not os.path.isfile(MODEL_DIR):
            raise FileNotFoundError(f"YOLO weights not found: {MODEL_DIR}")
        self.model = YOLO(MODEL_DIR)

    # 이미지 한 장 -> NailMask 목록 (신뢰도 상위 5개)
    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks.xy) == 0:
            return []

        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(result.masks.xy))
        min_area = self.min_area_ratio * w * h
        candidates = []

        for i, seg in enumerate(result.masks.xy):
            pts = np.asarray(seg, dtype=np.int32).reshape(-1, 1, 2)
            if len(pts) < 3:
                continue
            binary = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(binary, [pts], 255)
            # 화면 대비 너무 작은 조각은 손톱이 아니라고 보고 버린다
            area = int(np.count_nonzero(binary))
            if area < min_area:
                continue
            candidates.append((binary, float(confs[i]), area))

        if len(candidates) > 5:
            candidates.sort(key=lambda c: c[1], reverse=True)
            candidates = candidates[:5]

        nail_masks = []
        for binary, conf, _area in candidates:
            x, y, bw, bh = cv2.boundingRect(binary)
            nail_masks.append(NailMask(mask=binary, confidence=conf, bbox=(x, y, bw, bh)))
        return nail_masks


# 마스크의 세로(길이) 또는 가로(폭) 양 끝점
def find_endpoints(mask, y_mid=0, flag="vertical"):
    if flag == "vertical":
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        # 가로 무게중심을 세로 중심선으로 잡고 위아래 끝을 쓴다
        cx = float(xs.mean())
        y_min = float(ys.min())
        y_max = float(ys.max())
        return np.array([cx, y_min]), np.array([cx, y_max])
    elif flag == "horizontal":
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        # 중간 높이에 마스크 픽셀이 없으면 가장 가까운 행으로 당긴다
        y_mid_int = int(round(y_mid))
        row_ys = np.unique(ys)
        if y_mid_int not in row_ys:
            y_mid_int = int(row_ys[np.argmin(np.abs(row_ys - y_mid_int))])

        row_xs = xs[ys == y_mid_int]
        x_left = float(row_xs.min())
        x_right = float(row_xs.max())
        return np.array([x_left, float(y_mid_int)]), np.array([x_right, float(y_mid_int)])
    else:
        print(f"현재 flag변수 = {flag} => 잘못된 변수값")


# 마스크 하나에서 길이·폭(mm)
def measure_nail_from_mask(mask, homography, camera_height_mm=CAMERA_HEIGHT_MM, nail_height_mm=NAIL_HEIGHT_MM):
    endpoints = find_endpoints(mask, flag="vertical")
    if endpoints is None:
        return None
    p1, p2 = endpoints

    length_mm = measure_length_mm(
        homography, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])),
        camera_height_mm=camera_height_mm, nail_height_mm=nail_height_mm
    )

    # 폭은 길이 양 끝점의 중간 높이에서 가로로 잰다
    y_mid = (float(p1[1]) + float(p2[1])) / 2.0
    width_endpoints = find_endpoints(mask, y_mid, "horizontal")

    width_mm = None
    if width_endpoints is not None:
        w1, w2 = width_endpoints
        width_mm = measure_length_mm(
            homography, (float(w1[0]), float(w1[1])), (float(w2[0]), float(w2[1])),
            camera_height_mm=camera_height_mm, nail_height_mm=nail_height_mm
        )

    return {
        "length_mm": length_mm,
        "width_mm": width_mm,
    }
