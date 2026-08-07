"""
backends/measure_from_mask.py
==============================
[역할]
    세그멘테이션 백엔드가 만든 바이너리 마스크(손톱 하나) 하나하나에서
    "뿌리 <-> 끝" 두 후보점을 PCA(주성분분석)로 찾고, ChArUco 호모그래피로
    mm 길이를 계산한다. 또한 검출된 여러 마스크에 엄지~소지 라벨을
    자동으로 붙이는 기능도 제공한다.

    이 파일은 직접 실행하지 않는다. seg_gui.py에서
        from backends.measure_from_mask import measure_nail_from_mask, label_fingers
    형태로 불러와서 사용한다.
"""

import cv2
import numpy as np

from utils import measure_length_mm


def find_long_axis_endpoints(mask):
    """
    바이너리 마스크(HxW, 0/255)에서 가장 큰 컨투어를 찾고, PCA로 장축
    방향을 구한 뒤 그 방향으로 가장 멀리 떨어진 윤곽선 위의 두 점을 반환한다.
    (손톱은 대체로 길쭉한 타원형이므로, 장축의 양 끝이 뿌리/끝에 해당한다)

    반환값: (point_a, point_b) 각각 (x, y) numpy 배열, 마스크가 비었으면 None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        return None

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]

    projections = centered @ major_axis
    idx_min = int(np.argmin(projections))
    idx_max = int(np.argmax(projections))
    return pts[idx_min], pts[idx_max]


def find_vertical_axis_endpoints(mask):
    """
    바이너리 마스크(HxW, 0/255)에서 마스크의 세로(y) 범위를 그대로 뿌리/끝으로 쓴다.
    x좌표는 마스크 중심의 x로 고정해서, 결과적으로 완전히 수직인 직선 길이를 잰다.
    (엄지를 제외한 4개 손가락은 사진에서 항상 세로로 곧게 나오므로, PCA 장축보다
    이 방식이 마스크 일부 결손에 더 안정적이다)

    반환값: (point_a, point_b) 각각 (x, y) numpy 배열, 마스크가 비었으면 None
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    cx = float(xs.mean())
    y_min = float(ys.min())
    y_max = float(ys.max())
    return np.array([cx, y_min]), np.array([cx, y_max])


def find_horizontal_width_endpoints(mask, y_mid):
    """
    마스크에서 y_mid 행(정수로 반올림)의 좌우 끝점을 찾아 가로 폭의 두 점을
    반환한다. (뿌리/끝 세로선의 중간 높이에서 가로로 선을 그어 폭을 잰다)
    y_mid 행에 마스크 픽셀이 하나도 없으면 가장 가까운 행으로 대체한다.

    반환값: (point_a, point_b) 각각 (x, y) numpy 배열, 마스크가 비었으면 None
    """
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
    """
    마스크 하나에 대해 뿌리/끝(세로 길이)과 폭(가로, 세로선 중간 높이 기준) 후보점과
    mm 길이를 계산한다.
    finger가 주어지면(5개 손가락 전부) 항상 수직(세로) 직선 길이를 잰다.
    finger가 None(하위호환, 손가락 라벨 없이 호출된 경우)이면 PCA 장축 방식을 쓴다
    (이 경우 가로 폭은 세로선의 중간 y를 기준으로 동일하게 계산한다).
    반환: {"point_a", "point_b", "pixel_distance", "length_mm",
           "width_point_a", "width_point_b", "width_pixel_distance", "width_mm"}
    또는 None(마스크가 비었을 때)
    """
    if finger is None:
        endpoints = find_long_axis_endpoints(mask)
    else:
        endpoints = find_vertical_axis_endpoints(mask)
    if endpoints is None:
        return None
    p1, p2 = endpoints

    pixel_distance = float(np.linalg.norm(p2 - p1))
    length_mm = measure_length_mm(
        homography,
        (float(p1[0]), float(p1[1])),
        (float(p2[0]), float(p2[1])),
        camera_height_mm=camera_height_mm,
        nail_height_mm=nail_height_mm,
    )

    y_mid = (float(p1[1]) + float(p2[1])) / 2.0
    width_endpoints = find_horizontal_width_endpoints(mask, y_mid)
    if width_endpoints is not None:
        w1, w2 = width_endpoints
        width_pixel_distance = float(np.linalg.norm(w2 - w1))
        width_mm = measure_length_mm(
            homography,
            (float(w1[0]), float(w1[1])),
            (float(w2[0]), float(w2[1])),
            camera_height_mm=camera_height_mm,
            nail_height_mm=nail_height_mm,
        )
        width_point_a = (float(w1[0]), float(w1[1]))
        width_point_b = (float(w2[0]), float(w2[1]))
    else:
        width_pixel_distance = None
        width_mm = None
        width_point_a = None
        width_point_b = None

    return {
        "point_a": (float(p1[0]), float(p1[1])),
        "point_b": (float(p2[0]), float(p2[1])),
        "pixel_distance": pixel_distance,
        "length_mm": length_mm,
        "width_point_a": width_point_a,
        "width_point_b": width_point_b,
        "width_pixel_distance": width_pixel_distance,
        "width_mm": width_mm,
    }


# ---------------------------------------------------------------------------
# 손가락 라벨링 (마스크 여러 개 -> 엄지~소지 배정)
# ---------------------------------------------------------------------------
def _mask_centroids(nail_masks):
    centroids = []
    for nm in nail_masks:
        ys, xs = np.nonzero(nm.mask)
        if len(xs) == 0:
            return None
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


def label_fingers_by_x_order(nail_masks, hand="right"):
    """
    마스크 중심점의 x좌표 순서로 엄지~소지를 배정하는 가장 단순한 방법.
    정확히 5개일 때만 시도한다 (그 외엔 순서만으로 어느 손가락인지 알 수 없으므로 None 반환).
    """
    if len(nail_masks) != 5:
        return None
    centroids = _mask_centroids(nail_masks)
    if centroids is None:
        return None

    order = sorted(range(len(centroids)), key=lambda i: centroids[i][0])
    # 카메라를 향해 손등을 편 상태로 수직 촬영했다고 가정.
    # 오른손: 사진 왼쪽부터 엄지->소지 순, 왼손: 사진 왼쪽부터 소지->엄지 순
    # (실측 비교로 검증됨: 기존 반대 방향 가정은 틀렸었음)
    if hand == "right":
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
    else:
        finger_order = ["pinky", "ring", "middle", "index", "thumb"]

    labels = [None] * len(centroids)
    for rank, idx in enumerate(order):
        labels[idx] = finger_order[rank]
    return labels


def label_fingers_with_mediapipe(nail_masks, debug_image):
    """
    (선택 기능) mediapipe가 설치돼 있으면 손 랜드마크의 손가락 끝 좌표와
    각 마스크 중심점을 최근접 매칭해서 라벨을 붙인다.
    mediapipe가 없거나 손을 못 찾으면 None을 반환해서 다른 방법으로 폴백하게 한다.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None
    if debug_image is None:
        return None

    centroids = _mask_centroids(nail_masks)
    if centroids is None:
        return None

    mp_hands = mp.solutions.hands
    h, w = debug_image.shape[:2]
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0].landmark
        tip_ids = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
        finger_points = {name: (lm[idx].x * w, lm[idx].y * h) for name, idx in tip_ids.items()}

        pairs = []
        for finger, (fx, fy) in finger_points.items():
            for mi, (cx, cy) in enumerate(centroids):
                dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
                pairs.append((dist, finger, mi))
        pairs.sort(key=lambda p: p[0])

        labels = [None] * len(centroids)
        used_fingers, used_masks = set(), set()
        for dist, finger, mi in pairs:
            if finger in used_fingers or mi in used_masks:
                continue
            labels[mi] = finger
            used_fingers.add(finger)
            used_masks.add(mi)

        if None in labels:
            return None
        return labels


def label_fingers(nail_masks, hand="right", use_mediapipe=False, debug_image=None):
    """
    NailMask 리스트에 .finger 라벨을 채워서 반환한다.
    우선순위: mediapipe(선택, use_mediapipe=True일 때만) -> x좌표 순서(정확히 5개일 때만)
    둘 다 실패하면 라벨을 채우지 않고(finger=None) 그대로 반환한다.
    """
    labels = None
    if use_mediapipe:
        labels = label_fingers_with_mediapipe(nail_masks, debug_image)
        if labels is not None:
            print("[INFO] 손가락 라벨링: mediapipe 손 랜드마크로 자동 매칭됨")

    if labels is None:
        labels = label_fingers_by_x_order(nail_masks, hand=hand)
        if labels is not None:
            print(f"[INFO] 손가락 라벨링: x좌표 순서로 자동 매칭됨 (hand={hand})")

    if labels is None or len(set(labels)) != len(nail_masks):
        print("[WARN] 손가락 자동 라벨링 실패 (검출 개수가 5개가 아니거나 중복). 라벨 없이 진행합니다.")
        return nail_masks

    for nm, label in zip(nail_masks, labels):
        nm.finger = label
    return nail_masks
