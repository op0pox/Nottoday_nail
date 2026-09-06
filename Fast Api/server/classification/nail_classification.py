import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

SHAPE_PATH = Path(__file__).resolve().parents[2] / "nail_shape" / "shapes.json"
CONTOUR_SAMPLES = 160


def contour_to_xy(contour):
    return np.asarray(contour, dtype=np.float32).reshape(-1, 2)


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

    targets = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.stack([xs, ys], axis=1)


def normalize_contour(contour):
    contour = contour.astype(np.float32)
    x, y, w, h = cv2.boundingRect(contour)

    cx = x + w / 2.0
    cy = y + h / 2.0
    contour[:, 0, 0] -= cx
    contour[:, 0, 1] -= cy

    max_dim = max(w, h)
    if max_dim > 0:
        contour /= max_dim

    return contour


def prepare_template(points):
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return resample_closed_contour(normalize_contour(contour))


def load_shape_templates(path=SHAPE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    if not templates:
        raise RuntimeError(f"No shape templates found in {path}")
    return {name: prepare_template(points) for name, points in templates.items()}


SHAPE_TEMPLATES = load_shape_templates()


def calculate_chamfer_distance(contour1, contour2):
    pts1 = contour1.reshape(-1, 2)
    pts2 = contour2.reshape(-1, 2)

    dist_matrix = cdist(pts1, pts2, metric="euclidean")
    dist1 = np.mean(np.min(dist_matrix, axis=1))
    dist2 = np.mean(np.min(dist_matrix, axis=0))

    return dist1 + dist2


def classify_nail_shape(contour):
    norm_input = normalize_contour(np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2))
    input_pts = contour_to_xy(norm_input)

    best_shape = None
    min_dist = float("inf")

    for shape_name, template_pts in SHAPE_TEMPLATES.items():
        dist = calculate_chamfer_distance(input_pts, template_pts)
        if dist < min_dist:
            min_dist = dist
            best_shape = shape_name

    return best_shape, min_dist
