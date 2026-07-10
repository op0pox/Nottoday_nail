"""
segmentation/measure_from_mask.py
===================================
[역할]
    세그멘테이션 백엔드가 만든 바이너리 마스크(손톱 하나) 하나하나에서
    "뿌리 <-> 끝" 두 후보점을 PCA(주성분분석)로 찾고, ChArUco 호모그래피로
    mm 길이를 계산한다. 또한 검출된 여러 마스크에 엄지~소지 라벨을
    자동으로 붙이는 기능(실패 시 수동 입력 폴백 포함)도 제공한다.

    이 파일은 직접 실행하지 않는다. measure_auto.py에서
        from segmentation.measure_from_mask import measure_nail_from_mask, label_fingers
    형태로 불러와서 사용한다.
"""

import cv2
import numpy as np

from utils import measure_length_mm, ensure_dir, FINGERS


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


def measure_nail_from_mask(mask, homography, camera_height_mm=295.0, nail_height_mm=0.0):
    """
    마스크 하나에 대해 뿌리/끝 후보점과 mm 길이를 계산한다.
    반환: {"point_a", "point_b", "pixel_distance", "length_mm"} 또는 None(마스크가 비었을 때)
    """
    endpoints = find_long_axis_endpoints(mask)
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

    return {
        "point_a": (float(p1[0]), float(p1[1])),
        "point_b": (float(p2[0]), float(p2[1])),
        "pixel_distance": pixel_distance,
        "length_mm": length_mm,
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
    # 카메라를 향해 손바닥을 편 상태로 수직 촬영했다고 가정.
    # 오른손: 사진 왼쪽부터 소지->엄지 순, 왼손: 사진 왼쪽부터 엄지->소지 순
    if hand == "right":
        finger_order = ["pinky", "ring", "middle", "index", "thumb"]
    else:
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]

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


def label_fingers_manually(nail_masks, debug_image=None, debug_path=None):
    """
    자동 매칭이 실패했을 때 터미널 입력으로 마스크 하나하나에 손가락 이름을
    지정하게 하는 폴백. GUI가 없어도(순수 SSH여도) 동작한다.
    """
    print("[INFO] 자동 손가락 매칭에 실패했습니다. 마스크 번호를 보고 직접 지정해주세요.")

    if debug_image is not None:
        overlay = debug_image.copy()
        for i, nm in enumerate(nail_masks):
            ys, xs = np.nonzero(nm.mask)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(
                overlay, str(i + 1), (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
        path = debug_path or "data/results/debug/finger_label_help.jpg"
        ensure_dir(path)
        cv2.imwrite(path, overlay)
        print(f"[INFO] 마스크 번호 확인용 이미지 저장: {path}")
        print("       (노트북이면 이미지 뷰어로, Nano SSH면 scp로 받아서 확인하세요)")

    valid = set(FINGERS)
    labels = []
    for i in range(len(nail_masks)):
        while True:
            raw = input(f"마스크 {i + 1}번은 어느 손가락인가요? ({'/'.join(FINGERS)}): ").strip().lower()
            if raw in valid:
                labels.append(raw)
                break
            print(f"  -> {', '.join(FINGERS)} 중 하나로 입력하세요.")
    return labels


def label_fingers(nail_masks, hand="right", use_mediapipe=False, debug_image=None, debug_path=None):
    """
    NailMask 리스트에 .finger 라벨을 채워서 반환한다.
    우선순위: mediapipe(선택, --use-mediapipe일 때만) -> x좌표 순서(정확히 5개일 때만)
             -> 수동 입력 폴백(터미널)
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
        labels = label_fingers_manually(nail_masks, debug_image=debug_image, debug_path=debug_path)

    for nm, label in zip(nail_masks, labels):
        nm.finger = label
    return nail_masks
