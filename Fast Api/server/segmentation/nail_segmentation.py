import os

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_DIR = os.getenv("MODEL_DIR")
CAMERA_HEIGHT_MM = float(os.getenv("CAMERA_HEIGHT_MM"))
NAIL_HEIGHT_MM = float(os.getenv("NAIL_HEIGHT_MM"))

# 픽셀 좌표를 호모그래피 행렬을 통해 mm 평면 좌표로 변환
def transform_points_to_mm(homography, points):
    H = np.asarray(homography, dtype=np.float64)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)

# 손톱이 플레이트에서 얼마나 떠있는지를 구해 원근오차보정을 진행
# 손톱이 올라와있다 => 실제보다 조금더 가깝게있다 => 리턴값으로 원근오차 보정(0.1...같은 소수점)값을 보냄
def nail_height_Calibration(camera_height_mm, nail_height_mm):
    if not camera_height_mm:
        return 1.0
    return (camera_height_mm - nail_height_mm) / camera_height_mm

# 픽셀 두 점 사이의 거리를 mm로 환산하고 원근오차보정
def measure_length_mm(homography, point_a, point_b, camera_height_mm=CAMERA_HEIGHT_MM, nail_height_mm=NAIL_HEIGHT_MM):
    mm_pts = transform_points_to_mm(homography, [point_a, point_b])
    raw_length_mm = float(np.linalg.norm(mm_pts[0] - mm_pts[1]))
    return raw_length_mm * nail_height_Calibration(camera_height_mm, nail_height_mm) # 위에서 구한 원근오차 보정값을 곱해 실제값(손톱이 플레이트에 딱 붙어있을경우)을 구함

# 마스크
class NailMask:
    def __init__(self, mask, confidence=1.0, bbox=None):
        self.mask = mask
        self.confidence = confidence
        self.bbox = bbox

class YoloNailBackend:
    def __init__(self, conf=0.25, min_area_ratio=0.0003):
        self.conf = conf
        self.min_area_ratio = min_area_ratio
        if not MODEL_DIR:
            raise RuntimeError("MODEL_DIR is not set")
        if not MODEL_DIR.startswith(("http://", "https://")) and not os.path.isfile(MODEL_DIR):
            raise FileNotFoundError(f"YOLO weights not found: {MODEL_DIR}")
        self.model = YOLO(MODEL_DIR)

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

def find_endpoints(mask, y_mid=0, flag="vertical"):
    if flag == "vertical":
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        cx = float(xs.mean())
        y_min = float(ys.min())
        y_max = float(ys.max())
        return np.array([cx, y_min]), np.array([cx, y_max])
    elif flag == "horizontal":
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        
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


def measure_nail_from_mask(mask, homography, camera_height_mm=CAMERA_HEIGHT_MM, nail_height_mm=NAIL_HEIGHT_MM):
    endpoints = find_endpoints(mask, flag="vertical")
    if endpoints is None:
        return None
    p1, p2 = endpoints

    length_mm = measure_length_mm(
        homography, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])),
        camera_height_mm=camera_height_mm, nail_height_mm=nail_height_mm
    )

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
