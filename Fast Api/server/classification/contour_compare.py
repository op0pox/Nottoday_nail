import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np

from classification.nail_classification import (
    SHAPE_PATH,
    chamfer_distance,
    contour_to_xy,
    resample_closed_contour,
)

SHAPE_CODE_RE = re.compile(r"_([A-Za-z]+)\([^)]+\)$")  # 키 끝에서 글자 코드 (SC)
STEM_RE = re.compile(r"^(\d+)_[LR]_f(\d+)$")  # 사람 ID·손가락 번호
FINGERS_DIR = Path(SHAPE_PATH).parent / "fingers"  # 라벨 JSON 폴더
SAMPLE_CHOICES = list(range(100, 401, 20))  # 허용하는 보간 점 개수


# ===== 라벨·정답 불러오기 =====

with open(SHAPE_PATH, "r", encoding="utf-8") as f:
    _catalog_raw = json.load(f)
if not _catalog_raw:
    raise RuntimeError(f"No shape templates found in {SHAPE_PATH}")

# 분류 버튼이 비교 대상으로 쓰는 카탈로그: (키, 원본 좌표)
CATALOG_TEMPLATES = [(name, contour_to_xy(points)) for name, points in _catalog_raw.items()]


# 키에서 P/S
def extract_shape_code(template_name):
    match = SHAPE_CODE_RE.search(template_name)
    if match:
        return match.group(1)[0].upper()
    return template_name


# 파일명에서 사람·손가락
def parse_stem(stem):
    match = STEM_RE.match(stem)
    if not match:
        return None
    return match.group(1), match.group(2)


# 다른 사람, 같은 손가락(좌우)만
def same_finger_catalog(stem, items):
    parsed = parse_stem(stem)
    if not parsed:
        return []
    person, finger = parsed
    catalog = []
    for other_stem, pts, _letter in items:
        other = parse_stem(other_stem)
        if not other:
            continue
        other_person, other_finger = other
        # 손가락 번호가 다르거나 자기 자신(같은 사람의 반대손 포함)이면 뺀다
        if other_finger != finger or other_person == person:
            continue
        catalog.append((other_stem, pts))
    return catalog


# LabelMe에서 가장 큰 폴리곤
def points_from_labelme(data):
    shapes = data.get("shapes") or []
    polygons = [s for s in shapes if s.get("points") and len(s["points"]) >= 3]
    if not polygons:
        raise ValueError("JSON에 다각형 라벨이 없습니다")
    shape = max(polygons, key=lambda s: len(s["points"]))
    return contour_to_xy(shape["points"])


# fingers 윤곽과 정답 글자 로드
def load_finger_items():
    with open(SHAPE_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)
    # shapes.json 키에서 파일명 -> P/S 표를 만든다
    letters = {SHAPE_CODE_RE.sub("", name): extract_shape_code(name) for name in templates}

    items = []
    skipped = []
    for path in sorted(FINGERS_DIR.glob("*.json")):
        stem = path.stem
        letter = letters.get(stem)
        # 정답 코드가 없는 라벨은 검증에서 뺀다
        if not letter:
            skipped.append(stem)
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items.append((stem, points_from_labelme(raw), letter))
    return items, skipped


# ===== 두 윤곽 맞추기 =====

# 좌우·하단·폭
def _extrema(pts):
    left = pts[int(np.argmin(pts[:, 0]))]
    right = pts[int(np.argmax(pts[:, 0]))]
    bottom = pts[int(np.argmax(pts[:, 1]))]
    width = float(pts[:, 0].max() - pts[:, 0].min())
    return left, right, bottom, width


# 균일 스케일
def _scale_xy(pts, scale):
    return pts * float(scale)


# 무게중심으로 이동
def _center_xy(pts):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return p - p.mean(axis=0)


# 폭만 같게 스케일
def scale_to_same_width(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2).copy()
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2).copy()
    _, _, _, wa = _extrema(a)
    _, _, _, wb = _extrema(b)
    if wa <= 1e-9 or wb <= 1e-9:
        raise ValueError("윤곽 폭이 0입니다")

    # 넓은 쪽만 줄인다. x·y에 같은 배수를 곱하므로 모양은 안 찌그러진다
    if wa > wb:
        a = _scale_xy(a, wb / wa)
    elif wb > wa:
        b = _scale_xy(b, wa / wb)
    return a, b


# 폭 맞추고 가로 중심·최하단 수평선
def align_width_and_bottom(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        raise ValueError("윤곽 점이 부족합니다")

    a, b = scale_to_same_width(a, b)

    # 가로는 bbox 중심을 0으로, 세로는 최하단을 0으로.
    # 이미지 좌표라 아래가 +y이므로 정렬 뒤에는 뿌리가 0이고 팁이 음수다
    a[:, 0] -= (a[:, 0].min() + a[:, 0].max()) / 2.0
    b[:, 0] -= (b[:, 0].min() + b[:, 0].max()) / 2.0
    a[:, 1] -= a[:, 1].max()
    b[:, 1] -= b[:, 1].max()
    return a, b


# 팁 자르고 뿌리 띠만 남김
def crop_cuticle_band(pts_a, pts_b):
    left_a, right_a, _, _ = _extrema(pts_a)
    left_b, right_b, _, _ = _extrema(pts_b)
    # 네 끝점 중 제일 뿌리 쪽(y가 큰 쪽)이 자르는 선이 된다
    cut_y = float(max(left_a[1], right_a[1], left_b[1], right_b[1]))
    keep_a = pts_a[pts_a[:, 1] >= cut_y]
    keep_b = pts_b[pts_b[:, 1] >= cut_y]
    if len(keep_a) < 2 or len(keep_b) < 2:
        raise ValueError("절단 후 비교할 점이 부족합니다")
    return keep_a, keep_b, cut_y


# ===== 비교 =====

# 정렬·절단 후 Chamfer
def compare_cuticle_chamfer(points_a, points_b, n_samples):
    if n_samples not in SAMPLE_CHOICES:
        raise ValueError(f"n_samples must be one of {SAMPLE_CHOICES}")

    a = resample_closed_contour(points_a, n_samples)
    b = resample_closed_contour(points_b, n_samples)
    a, b = align_width_and_bottom(a, b)
    keep_a, keep_b, cut_y = crop_cuticle_band(a, b)

    return {
        "distance": chamfer_distance(keep_a, keep_b),
        "cut_y": cut_y,
        "n_samples": n_samples,
        "kept_left": int(len(keep_a)),
        "kept_right": int(len(keep_b)),
    }


# Chamfer가 가장 작은 템플릿
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


# ===== 검증 =====

# 한 장 빼고 같은 손가락끼리만 검증
def leave_one_out_fingers(n_samples=160):
    items, skipped = load_finger_items()
    if len(items) < 2:
        raise ValueError("비교할 fingers JSON이 부족합니다")

    correct = 0
    wrong = []
    letter_by_stem = {stem: letter for stem, _pts, letter in items}

    for stem, pts, letter in items:
        others = same_finger_catalog(stem, items)
        # 짝이 될 같은 손가락이 하나도 없으면 채점하지 않는다
        if not others:
            skipped.append(stem)
            continue
        result = nearest_cuticle(pts, others, n_samples)
        matched = result["predicted_shape"]
        predicted = letter_by_stem[matched]
        if predicted == letter:
            correct += 1
        else:
            wrong.append({
                "file": stem,
                "truth": letter,
                "predicted": predicted,
                "matched": matched,
                "distance": result["distance"],
            })

    return {
        "total": correct + len(wrong),
        "correct": correct,
        "wrong": wrong,
        "skipped": skipped,
    }


# ===== 시각화 =====

# 흰 배경에 초록/빨강 윤곽
def _overlay_png(green_pts, red_pts, closed=True, size=420, pad=28):
    g = np.asarray(green_pts, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(red_pts, dtype=np.float64).reshape(-1, 2)
    all_pts = np.vstack([g, r])
    mn = all_pts.min(axis=0)
    span = np.maximum(all_pts.max(axis=0) - mn, 1e-6)
    # 두 윤곽을 한 박스에 같이 담는다. 가로세로 중 큰 쪽에 맞춰야 비율이 안 깨진다
    scale = (size - 2 * pad) / float(max(span[0], span[1]))

    # 좌표 -> 픽셀
    def to_px(pts):
        p = (pts - mn) * scale
        p[:, 0] += pad
        p[:, 1] += pad
        return np.round(p).astype(np.int32)

    img = np.full((size, size, 3), 255, np.uint8)
    gi, ri = to_px(g), to_px(r)
    if len(gi) >= 2:
        cv2.polylines(img, [gi], closed and len(gi) > 2, (0, 180, 0), 2, cv2.LINE_AA)
    if len(ri) >= 2:
        cv2.polylines(img, [ri], closed and len(ri) > 2, (0, 0, 220), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("시각화 PNG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# 비교 4단계 PNG
def visualize_compare_steps(query_pts, template_pts, n_samples=160):
    red = resample_closed_contour(query_pts, n_samples)
    green = resample_closed_contour(template_pts, n_samples)

    g2, r2 = scale_to_same_width(green, red)
    g3, r3 = align_width_and_bottom(green, red)
    g4, r4, _cut_y = crop_cuticle_band(g3, r3)

    return [
        {"step": 1, "title": "1단계 라벨링", "image": _overlay_png(_center_xy(green), _center_xy(red))},
        {"step": 2, "title": "2단계 폭 스케일", "image": _overlay_png(_center_xy(g2), _center_xy(r2))},
        {"step": 3, "title": "3단계 최하단 정렬", "image": _overlay_png(g3, r3)},
        {"step": 4, "title": "4단계 팁 절단", "image": _overlay_png(g4, r4, closed=False)},
    ]


# 폭 스케일 후 전체 윤곽 PNG
def visualize_full_shell_steps(query_pts, template_pts, n_samples=160):
    if n_samples not in SAMPLE_CHOICES:
        raise ValueError(f"n_samples must be one of {SAMPLE_CHOICES}")

    red = resample_closed_contour(query_pts, n_samples)
    green = resample_closed_contour(template_pts, n_samples)

    g2, r2 = scale_to_same_width(green, red)
    g3, r3 = align_width_and_bottom(green, red)

    return {
        # 3단계 그림에 쓴 좌표를 그대로 재서, 보이는 것과 숫자가 어긋나지 않게 한다
        "distance": chamfer_distance(g3, r3),
        "stages": [
            {"step": 1, "title": "1단계 라벨링", "image": _overlay_png(_center_xy(green), _center_xy(red))},
            {"step": 2, "title": "2단계 폭 스케일", "image": _overlay_png(_center_xy(g2), _center_xy(r2))},
            {"step": 3, "title": "3단계 전체 쉘", "image": _overlay_png(g3, r3)},
        ],
    }
