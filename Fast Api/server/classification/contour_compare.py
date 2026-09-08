import numpy as np
from scipy.spatial.distance import cdist

from classification.nail_classification import contour_to_xy, resample_closed_contour

SAMPLE_CHOICES = list(range(100, 401, 20))


def points_from_labelme(data):
    shapes = data.get("shapes") or []
    polygons = [s for s in shapes if s.get("points") and len(s["points"]) >= 3]
    if not polygons:
        raise ValueError("JSON에 다각형 라벨이 없습니다")
    shape = max(polygons, key=lambda s: len(s["points"]))
    return contour_to_xy(shape["points"])


def _extrema(pts):
    left = pts[int(np.argmin(pts[:, 0]))]
    right = pts[int(np.argmax(pts[:, 0]))]
    bottom = pts[int(np.argmax(pts[:, 1]))]
    width = float(pts[:, 0].max() - pts[:, 0].min())
    return left, right, bottom, width


def _scale_xy(pts, scale):
    return pts * float(scale)


def align_width_and_bottom(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        raise ValueError("윤곽 점이 부족합니다")

    _, _, _, wa = _extrema(a)
    _, _, _, wb = _extrema(b)
    if wa <= 1e-9 or wb <= 1e-9:
        raise ValueError("윤곽 폭이 0입니다")

    if wa > wb:
        a = _scale_xy(a, wb / wa)
    elif wb > wa:
        b = _scale_xy(b, wa / wb)

    a = a.copy()
    b = b.copy()
    a[:, 0] -= a[:, 0].min()
    b[:, 0] -= b[:, 0].min()
    a[:, 1] -= a[:, 1].max()
    b[:, 1] -= b[:, 1].max()
    return a, b


def cuticle_points(pts):
    left, right, _, _ = _extrema(pts)
    cut_y = float(max(left[1], right[1]))
    kept = pts[pts[:, 1] >= cut_y]
    return kept, cut_y


def compare_cuticle_chamfer(points_a, points_b, n_samples):
    if n_samples not in SAMPLE_CHOICES:
        raise ValueError(f"n_samples must be one of {SAMPLE_CHOICES}")

    a = resample_closed_contour(points_a, n_samples)
    b = resample_closed_contour(points_b, n_samples)
    a, b = align_width_and_bottom(a, b)

    left_a, right_a, _, _ = _extrema(a)
    left_b, right_b, _, _ = _extrema(b)
    cut_y = float(max(left_a[1], right_a[1], left_b[1], right_b[1]))
    keep_a = a[a[:, 1] >= cut_y]
    keep_b = b[b[:, 1] >= cut_y]
    if len(keep_a) < 2 or len(keep_b) < 2:
        raise ValueError("절단 후 비교할 점이 부족합니다")

    dist_matrix = cdist(keep_a, keep_b, metric="euclidean")
    dist1 = float(np.mean(np.min(dist_matrix, axis=1)))
    dist2 = float(np.mean(np.min(dist_matrix, axis=0)))
    return {
        "distance": dist1 + dist2,
        "cut_y": cut_y,
        "n_samples": n_samples,
        "kept_left": int(len(keep_a)),
        "kept_right": int(len(keep_b)),
    }


def nearest_cuticle(query_points, templates, n_samples):
    if not templates:
        raise ValueError("왼쪽 라벨 JSON이 없습니다")

    ranked = []
    for name, pts in templates:
        result = compare_cuticle_chamfer(pts, query_points, n_samples)
        ranked.append({"name": name, **result})
    ranked.sort(key=lambda item: item["distance"])
    best = ranked[0]
    return {
        "predicted_shape": best["name"],
        "distance": best["distance"],
        "cut_y": best["cut_y"],
        "n_samples": n_samples,
        "kept_left": best["kept_left"],
        "kept_right": best["kept_right"],
        "ranking": [{"name": item["name"], "distance": item["distance"]} for item in ranked],
    }

