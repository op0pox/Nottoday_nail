import base64
import re

import cv2
import numpy as np

from classification.nail_classification import chamfer_distance, resample_closed_contour

SAMPLE_CHOICES = list(range(100, 401, 20))
REGIONS = ("cuticle", "shell")
CANVAS_SIZE = 420
CANVAS_PAD = 28
STEM_RE = re.compile(r"^(\d+)_[LR]_f(\d+)$")


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


# 두 윤곽 폭을 똑같이 1로
def scale_to_unit_width(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    _, _, _, wa = _extrema(a)
    _, _, _, wb = _extrema(b)
    if wa <= 1e-9 or wb <= 1e-9:
        raise ValueError("윤곽 폭이 0입니다")

    # 각자 자기 폭으로 나눠 기준을 1로 고정한다. x·y를 같은 수로 나누므로 모양은 안 찌그러진다
    return _scale_xy(a, 1.0 / wa), _scale_xy(b, 1.0 / wb)


# 가로 중심·최하단을 0으로
def move_to_bottom_center(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2).copy()
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2).copy()

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

    width_a, width_b = scale_to_unit_width(raw_a, raw_b)
    full_a, full_b = move_to_bottom_center(width_a, width_b)

    if region == "cuticle":
        band_a, band_b, cut_y = crop_cuticle_band(full_a, full_b)
    else:
        band_a, band_b, cut_y = full_a, full_b, None

    return {
        "raw_a": raw_a,
        "raw_b": raw_b,
        "width_a": width_a,
        "width_b": width_b,
        "full_a": full_a,
        "full_b": full_b,
        "band_a": band_a,
        "band_b": band_b,
        "cut_y": cut_y,
        "n_samples": n_samples,
        "region": region,
    }


# 양방향 평균 최근접 거리
def score(prepared):
    return chamfer_distance(prepared["band_a"], prepared["band_b"])


# ===== 그리기 =====

# 두 윤곽을 한 캔버스에 담는 좌표->픽셀 변환
def fit_to_canvas(all_pts, size=CANVAS_SIZE, pad=CANVAS_PAD):
    pts = np.asarray(all_pts, dtype=np.float64).reshape(-1, 2)
    mn = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - mn, 1e-6)
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


# ===== 분류·검증·시각화 =====

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
        if other_finger != finger or other_person == person:
            continue
        catalog.append((other_stem, pts))
    return catalog


# 점수가 가장 낮은 템플릿
def nearest_template(query_points, templates, n_samples, region):
    if not templates:
        raise ValueError("왼쪽 라벨 JSON이 없습니다")

    ranked = []
    for name, pts in templates:
        prepared = prepare_pair(pts, query_points, n_samples, region)
        ranked.append({
            "name": name,
            "distance": score(prepared),
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


# 한 장 빼고 같은 손가락끼리만 검증
def leave_one_out_fingers(items, skipped, n_samples, region):
    if len(items) < 2:
        raise ValueError("비교할 fingers JSON이 부족합니다")

    correct = 0
    wrong = []
    letter_by_stem = {stem: letter for stem, _pts, letter in items}

    for stem, pts, letter in items:
        others = same_finger_catalog(stem, items)
        if not others:
            skipped.append(stem)
            continue
        result = nearest_template(pts, others, n_samples, region)
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


# 맞춰가는 단계별 PNG와 최종 점수
def visualize_steps(query_pts, template_pts, n_samples, region):
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

    return {
        "distance": score(prepared),
        "stages": stages,
    }
