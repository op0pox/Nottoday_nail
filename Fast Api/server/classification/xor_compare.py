import base64
import re

import cv2
import numpy as np

from classification.nail_classification import resample_closed_contour

SAMPLE_CHOICES = list(range(100, 401, 20))
REGIONS = ("cuticle", "shell")
CANVAS_SIZE = 420
CANVAS_PAD = 28
RASTER_SIZE = 512
RASTER_PAD = 4
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

    return _scale_xy(a, 1.0 / wa), _scale_xy(b, 1.0 / wb)


# 가로 중심·최하단을 0으로
def move_to_bottom_center(pts_a, pts_b):
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2).copy()
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2).copy()

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


# ===== XOR 면적 =====

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


# 정렬된 두 윤곽을 같은 캔버스에 채운 마스크
def _fill_masks(prepared, size=RASTER_SIZE, pad=RASTER_PAD):
    a = prepared["full_a"]
    b = prepared["full_b"]
    cut_y = prepared["cut_y"]

    to_px, mn, scale = fit_to_canvas(np.vstack([a, b]), size, pad)
    mask_a = np.zeros((size, size), np.uint8)
    mask_b = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask_a, [to_px(a)], 255)
    cv2.fillPoly(mask_b, [to_px(b)], 255)

    if cut_y is not None:
        # 정렬 좌표는 뿌리가 y=0이고 팁이 음수라, 픽셀에서는 뿌리가 아래쪽 행이다
        row = int(round((cut_y - mn[1]) * scale)) + pad
        row = max(0, min(size, row))
        mask_a[:row] = 0
        mask_b[:row] = 0

    return mask_a, mask_b


# 겹치지 않는 면적의 비율
def score(prepared):
    mask_a, mask_b = _fill_masks(prepared)
    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))
    mean_area = (area_a + area_b) / 2.0
    if mean_area <= 0:
        raise ValueError("XOR 면적을 잴 도형이 없습니다")

    xor_area = int(np.count_nonzero(cv2.bitwise_xor(mask_a, mask_b)))
    return xor_area / mean_area


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


# 교집합과 어긋난 부분을 색으로 구분한 PNG
def overlay_xor_png(prepared):
    mask_a, mask_b = _fill_masks(prepared)
    both = cv2.bitwise_and(mask_a, mask_b)
    only_a = cv2.bitwise_and(mask_a, cv2.bitwise_not(mask_b))
    only_b = cv2.bitwise_and(mask_b, cv2.bitwise_not(mask_a))

    img = np.full((mask_a.shape[0], mask_a.shape[1], 3), 255, np.uint8)
    img[both > 0] = (225, 225, 225)
    img[only_a > 0] = (0, 180, 0)
    img[only_b > 0] = (0, 0, 220)
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

    step = len(stages) + 1
    stages.append({"step": step, "title": f"{step}단계 XOR 면적", "image": overlay_xor_png(prepared)})

    return {
        "distance": score(prepared),
        "stages": stages,
    }
