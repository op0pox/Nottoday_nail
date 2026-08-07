import os
import urllib.request
import cv2
import numpy as np

# ==========================================
# 1. 거리 측정 및 좌표 변환 로직 (utils.py 병합 부분)
# ==========================================
def transform_points_to_mm(homography, points):
    """픽셀 좌표를 호모그래피 행렬을 통해 mm 평면 좌표로 변환합니다."""
    H = np.asarray(homography, dtype=np.float64)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)

def parallax_factor(camera_height_mm, nail_height_mm):
    """손톱이 보드 평면보다 떠 있을 때의 원근 오차를 보정하는 배율을 계산합니다."""
    if not camera_height_mm:
        return 1.0
    return (camera_height_mm - nail_height_mm) / camera_height_mm

def measure_length_mm(homography, point_a, point_b, camera_height_mm=295.0, nail_height_mm=0.0):
    """픽셀 두 점 사이의 거리를 mm로 환산하고 시차 보정을 적용합니다."""
    mm_pts = transform_points_to_mm(homography, [point_a, point_b])
    raw_length_mm = float(np.linalg.norm(mm_pts[0] - mm_pts[1]))
    return raw_length_mm * parallax_factor(camera_height_mm, nail_height_mm)

# ==========================================
# 2. 공통 데이터 클래스
# ==========================================
class NailMask:
    def __init__(self, mask, finger=None, confidence=1.0, bbox=None):
        self.mask = mask
        self.finger = finger
        self.confidence = confidence
        self.bbox = bbox

# ==========================================
# 3. YOLO 객체 탐지 엔진
# ==========================================
MODEL_URL = "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_FILENAME = "nails_seg_s_yolov8_v1.pt"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def ensure_model_downloaded(model_path=None):
    model_path = model_path or DEFAULT_MODEL_PATH
    if os.path.exists(model_path):
        return model_path

    folder = os.path.dirname(model_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as e:
        raise RuntimeError(f"모델 다운로드 실패: {e}")
    return model_path

class YoloNailBackend:
    def __init__(self, model_path=None, conf=0.25, min_area_ratio=0.0003):
        self.conf = conf
        self.min_area_ratio = min_area_ratio

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"ultralytics 패키지가 필요합니다: {e}")

        weight_path = ensure_model_downloaded(model_path)
        self.model = YOLO(weight_path)

    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks.data) == 0:
            return []

        mask_data = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(mask_data.shape[0])

        candidates = []
        min_area = self.min_area_ratio * w * h
        for i in range(mask_data.shape[0]):
            resized = cv2.resize(mask_data[i], (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (resized > 0.5).astype(np.uint8) * 255
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

# ==========================================
# 4. 실측값 계산 및 라벨링 모듈
# ==========================================
def find_vertical_axis_endpoints(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx = float(xs.mean())
    y_min = float(ys.min())
    y_max = float(ys.max())
    return np.array([cx, y_min]), np.array([cx, y_max])

def find_horizontal_width_endpoints(mask, y_mid):
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

def measure_nail_from_mask(mask, homography, camera_height_mm=295.0, nail_height_mm=0.0, finger=None):
    endpoints = find_vertical_axis_endpoints(mask)
    if endpoints is None:
        return None
    p1, p2 = endpoints

    length_mm = measure_length_mm(
        homography, (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])),
        camera_height_mm=camera_height_mm, nail_height_mm=nail_height_mm
    )

    y_mid = (float(p1[1]) + float(p2[1])) / 2.0
    width_endpoints = find_horizontal_width_endpoints(mask, y_mid)
    
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

def label_fingers_by_x_order(nail_masks, hand="right"):
    if len(nail_masks) != 5:
        return None
    
    centroids = []
    for nm in nail_masks:
        ys, xs = np.nonzero(nm.mask)
        if len(xs) == 0:
            return None
        centroids.append((float(xs.mean()), float(ys.mean())))

    order = sorted(range(len(centroids)), key=lambda i: centroids[i][0])
    finger_order = ["thumb", "index", "middle", "ring", "pinky"] if hand == "right" else ["pinky", "ring", "middle", "index", "thumb"]

    labels = [None] * len(centroids)
    for rank, idx in enumerate(order):
        labels[idx] = finger_order[rank]
    return labels