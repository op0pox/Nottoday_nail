import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

SHAPE_PATH = Path(__file__).resolve().parents[2] / "nail_shape" / "shapes.json"
CONTOUR_SAMPLES = 160


# Nx2 좌표로 펼치기
def contour_to_xy(contour):
    return np.asarray(contour, dtype=np.float32).reshape(-1, 2)


# 외곽선 보간
def resample_closed_contour(points, n=CONTOUR_SAMPLES):
    pts = contour_to_xy(points)
    if len(pts) == 0:
        return pts
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    closed = np.vstack([pts, pts[0]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)

    # 둘레를 n등분한 지점을 꺾은선 위에서 뽑는다. 라벨 꼭짓점이 몰려 있어도 간격이 고르다
    targets = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.stack([xs, ys], axis=1)


# bbox 중심·크기로 정규화
def normalize_contour(contour):
    contour = contour.astype(np.float32)
    x, y, w, h = cv2.boundingRect(contour)

    cx = x + w / 2.0
    cy = y + h / 2.0
    contour[:, 0, 0] -= cx
    contour[:, 0, 1] -= cy

    # 긴 변을 1로 맞춘다. x·y를 같은 수로 나누므로 종횡비는 안 변한다
    max_dim = max(w, h)
    if max_dim > 0:
        contour /= max_dim

    return contour


# 양방향 평균 최근접 거리
def chamfer_distance(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)

    # a의 각 점에서 b까지, b의 각 점에서 a까지 최근접 거리를 따로 평균내 더한다.
    # 한 방향만 보면 작은 윤곽이 큰 윤곽 안에 들어갔을 때 거리가 전부 작아진다
    dist_matrix = cdist(a, b, metric="euclidean")
    dist1 = float(np.mean(np.min(dist_matrix, axis=1)))
    dist2 = float(np.mean(np.min(dist_matrix, axis=0)))
    return dist1 + dist2


with open(SHAPE_PATH, "r", encoding="utf-8") as f:
    _raw_templates = json.load(f)
if not _raw_templates:
    raise RuntimeError(f"No shape templates found in {SHAPE_PATH}")

# 템플릿 이름 -> 정규화 후 보간한 윤곽 (측정 탭 분류용)
SHAPE_TEMPLATES = {
    name: resample_closed_contour(normalize_contour(np.array(points, dtype=np.float32).reshape(-1, 1, 2)))
    for name, points in _raw_templates.items()
}


# 가장 가까운 shape 이름
def classify_nail_shape(contour):
    norm_input = normalize_contour(np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2))
    input_pts = contour_to_xy(norm_input)

    best_shape = None
    min_dist = float("inf")

    for shape_name, template_pts in SHAPE_TEMPLATES.items():
        dist = chamfer_distance(input_pts, template_pts)
        if dist < min_dist:
            min_dist = dist
            best_shape = shape_name

    return best_shape, min_dist
