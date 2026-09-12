import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np

from classification.nail_classification import SHAPE_PATH, contour_to_xy, resample_closed_contour

SHAPE_CODE_RE = re.compile(r"_([A-Za-z]+)\([^)]+\)$")  # 키 끝에서 글자 코드 (SC)
STEM_RE = re.compile(r"^(\d+)_[LR]_f(\d+)$")  # 사람 ID·손가락 번호
FINGERS_DIR = Path(SHAPE_PATH).parent / "fingers"  # 라벨 JSON 폴더
SAMPLE_CHOICES = list(range(100, 401, 20))  # 허용하는 보간 점 개수
REGIONS = ("cuticle", "shell")  # 비교할 구역: 팁 절단 후 뿌리 띠 / 전체 윤곽
CANVAS_SIZE = 420  # 단계 그림 한 변
CANVAS_PAD = 28  # 단계 그림 여백


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


# 가로 중심·최하단을 0으로
def move_to_bottom_center(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2).copy()
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2).copy()

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


# 두 라벨을 같은 기준으로 맞춰 비교 직전 상태로
def prepare_pair(points_a, points_b, n_samples, region):
    if n_samples not in SAMPLE_CHOICES:
        raise ValueError(f"n_samples must be one of {SAMPLE_CHOICES}")
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")

    raw_a = resample_closed_contour(points_a, n_samples)
    raw_b = resample_closed_contour(points_b, n_samples)
    if len(raw_a) < 3 or len(raw_b) < 3:
        raise ValueError("윤곽 점이 부족합니다")

    width_a, width_b = scale_to_same_width(raw_a, raw_b)
    full_a, full_b = move_to_bottom_center(width_a, width_b)

    if region == "cuticle":
        band_a, band_b, cut_y = crop_cuticle_band(full_a, full_b)
    else:
        # 전체 쉘은 자르지 않으므로 비교 대상이 곧 전체 윤곽이다
        band_a, band_b, cut_y = full_a, full_b, None

    return {
        "raw_a": raw_a,  # 보간만 한 것 (1단계)
        "raw_b": raw_b,
        "width_a": width_a,  # 폭만 맞춘 것 (2단계)
        "width_b": width_b,
        "full_a": full_a,  # 정렬까지 끝난 닫힌 윤곽 (3단계, 면적 계산용)
        "full_b": full_b,
        "band_a": band_a,  # 실제로 점을 비교할 구간 (4단계)
        "band_b": band_b,
        "cut_y": cut_y,  # 전체 쉘이면 None
        "n_samples": n_samples,
        "region": region,
    }


# ===== 그리기 =====

# 두 윤곽을 한 캔버스에 담는 좌표->픽셀 변환
def fit_to_canvas(all_pts, size=CANVAS_SIZE, pad=CANVAS_PAD):
    pts = np.asarray(all_pts, dtype=np.float64).reshape(-1, 2)
    mn = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - mn, 1e-6)
    # 가로세로 중 큰 쪽에 맞춰야 두 윤곽의 비율이 안 깨진다
    scale = (size - 2 * pad) / float(max(span[0], span[1]))

    def to_px(p):
        q = (np.asarray(p, dtype=np.float64).reshape(-1, 2) - mn) * scale
        q[:, 0] += pad
        q[:, 1] += pad
        return np.round(q).astype(np.int32)

    return to_px, mn, scale


# 이미지를 base64 PNG 문자열로
def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("시각화 PNG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# 흰 배경에 초록/빨강 윤곽
def overlay_outline_png(green_pts, red_pts, closed=True, size=CANVAS_SIZE, pad=CANVAS_PAD):
    g = np.asarray(green_pts, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(red_pts, dtype=np.float64).reshape(-1, 2)
    to_px, _mn, _scale = fit_to_canvas(np.vstack([g, r]), size, pad)

    img = np.full((size, size, 3), 255, np.uint8)
    gi, ri = to_px(g), to_px(r)
    if len(gi) >= 2:
        cv2.polylines(img, [gi], closed and len(gi) > 2, (0, 180, 0), 2, cv2.LINE_AA)
    if len(ri) >= 2:
        cv2.polylines(img, [ri], closed and len(ri) > 2, (0, 0, 220), 2, cv2.LINE_AA)
    return encode_png(img)


# ===== 비교 =====

# 점수가 가장 낮은 템플릿
def nearest_template(query_points, templates, n_samples, region, metric):
    if not templates:
        raise ValueError("왼쪽 라벨 JSON이 없습니다")

    ranked = []
    for name, pts in templates:
        prepared = prepare_pair(pts, query_points, n_samples, region)
        ranked.append({
            "name": name,
            "distance": metric.score(prepared),
            "cut_y": prepared["cut_y"],
            "kept_left": int(len(prepared["band_a"])),
            "kept_right": int(len(prepared["band_b"])),
        })
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
def leave_one_out_fingers(n_samples, region, metric):
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
        result = nearest_template(pts, others, n_samples, region, metric)
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

# 맞춰가는 단계별 PNG와 최종 점수
def visualize_steps(query_pts, template_pts, n_samples, region, metric):
    # 초록이 1번째(템플릿), 빨강이 2번째(쿼리)다
    prepared = prepare_pair(template_pts, query_pts, n_samples, region)

    stages = [
        {
            "step": 1,
            "title": "1단계 라벨링",
            "image": overlay_outline_png(_center_xy(prepared["raw_a"]), _center_xy(prepared["raw_b"])),
        },
        {
            "step": 2,
            "title": "2단계 폭 스케일",
            "image": overlay_outline_png(_center_xy(prepared["width_a"]), _center_xy(prepared["width_b"])),
        },
    ]

    if region == "cuticle":
        stages.append({
            "step": 3,
            "title": "3단계 최하단 정렬",
            "image": overlay_outline_png(prepared["full_a"], prepared["full_b"]),
        })
        stages.append({
            "step": 4,
            "title": "4단계 팁 절단",
            "image": overlay_outline_png(prepared["band_a"], prepared["band_b"], closed=False),
        })
    else:
        stages.append({
            "step": 3,
            "title": "3단계 전체 쉘",
            "image": overlay_outline_png(prepared["full_a"], prepared["full_b"]),
        })

    # 계산 방식이 따로 보여줄 그림이 있으면 마지막에 한 장 더 붙인다
    extra = metric.overlay_png(prepared)
    if extra is not None:
        step = len(stages) + 1
        stages.append({"step": step, "title": f"{step}단계 {metric.OVERLAY_TITLE}", "image": extra})

    return {
        # 그림에 쓴 좌표를 그대로 재서, 보이는 것과 숫자가 어긋나지 않게 한다
        "distance": metric.score(prepared),
        "stages": stages,
    }
