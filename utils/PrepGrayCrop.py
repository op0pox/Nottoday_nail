# -*- coding: utf-8 -*-
"""
PrepGrayCrop.py
LabelMe json 폴리곤으로 손톱을 잘라 검은 배경·그레이로 저장한다.

전제조건:
  입력 이미지는 손이 아래에서 위를 향하도록, 세로(11자)로 반듯하게 정렬되어 있어야 한다.
  그래서 여러 손톱은 x축(왼쪽→오른쪽) 순서로만 가른다.

파일명:
  순번_손_손가락[_side]
  손: L, R, LR(양손). LF는 LR 오타로 본다.
  _side: 측면. 없으면 정면. 양손이어도 측면 규칙은 정면과 같다.
  LR: 양손을 한 장에 촬영.
    LR_4F: 양손 네 손가락. 왼쪽부터 05 04 03 02 02 03 04 05 로 잘라
            003_L_05, 003_R_02 처럼 손과 번호를 나눠 저장한다.
    LR_손가락: 같은 번호 두 장. 왼쪽 폴리곤이 왼손, 오른쪽 폴리곤이 오른손.
            폴리곤은 2개이고 003_L_01, 003_R_01 처럼 저장한다.

자를 때 왼쪽부터:
  왼손 L: 05 04 03 02
  오른손 R: 02 03 04 05
  양손 LR: 05 04 03 02 02 03 04 05
  새끼손가락=05, 검지=02, 엄지=01

출력:
  원본은 사람 폴더까지 재귀로 찾는다.
  결과는 사람별 폴더를 만들지 않고 TARGET_FOLDER_DIR 안에 파일만 모아 둔다.
  순번이 파일명에 있으므로 이름순 정렬로 사람이 모인다.
  순번_손_손가락_front_nail.jpg
  순번_손_손가락_side_nail.jpg
  이름순으로 정렬하면 같은 손가락의 정면, 측면이 붙는다.
  원본에 손가락이 빠지지 않았으면 한 사람은 정면 10장 + 측면 10장 = 20장이다.
  왼손·오른손 각각 엄지(01)와 02 03 04 05.

크롭 뒤 길쭉한 손톱은 긴 축이 세로가 되도록 한 번 더 바로잡는다.
  둥근 손톱은 축이 흔들려서 돌리지 않는다.
"""
import json
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FOLDER_DIR = r"C:\Users\USER\Downloads\파일정리"
TARGET_FOLDER_DIR = r"C:\Users\USER\Downloads\파일정리_nail"

PADDING_RATIO = 0.05
ELONGATION_MIN = 1.35
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEFT_ORDER = ("05", "04", "03", "02")
RIGHT_ORDER = ("02", "03", "04", "05")
BOTH_ORDER = (
    ("L", "05"), ("L", "04"), ("L", "03"), ("L", "02"),
    ("R", "02"), ("R", "03"), ("R", "04"), ("R", "05"),
)
STEM_RE = re.compile(r"^(?P<id>.+?)_(?P<hand>LF|LR|L|R)(?:_(?P<rest>.*))?$")


def load_image(path, json_size):
    with Image.open(path) as opened:
        raw = opened.convert("RGB")
        turned = ImageOps.exif_transpose(opened)
        oriented = raw if turned is None else turned.convert("RGB")
    if json_size == (0, 0) or oriented.size == json_size or raw.size != json_size:
        chosen = oriented
    else:
        chosen = raw
    rgb = np.asarray(chosen)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def is_image_name(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def padded_crop_box(points, image_width, image_height, pad_ratio=PADDING_RATIO):
    xs = points[:, 0]
    ys = points[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(image_width, int(round(x2 + pad_x)))
    bottom = min(image_height, int(round(y2 + pad_y)))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def crop_on_black(bgr, polygon):
    height, width = bgr.shape[:2]
    left, top, right, bottom = padded_crop_box(polygon, width, height)
    crop = bgr[top:bottom, left:right].copy()
    shifted = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    shifted[:, 0] -= left
    shifted[:, 1] -= top
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 255)
    black = np.zeros_like(crop)
    black[mask > 0] = crop[mask > 0]
    return cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)


def foreground_up_angle(gray):
    """긴 축을 세로로 세우는 OpenCV 회전각. 양수는 반시계. 점 위쪽을 유지한다."""
    fg = gray > 15
    ys, xs = np.nonzero(fg)
    if len(xs) < 30:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centered = pts - pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(centered.T))
    long_axis = vecs[:, int(np.argmax(vals))]
    if long_axis[1] > 0:
        long_axis = -long_axis
    alpha = float(np.degrees(np.arctan2(long_axis[1], long_axis[0])))
    angle = alpha + 90.0
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    ratio = float(np.sqrt(float(vals.max()) / max(float(vals.min()), 1e-9)))
    return angle, ratio


def crop_foreground(gray, pad=4):
    ys, xs = np.nonzero(gray > 15)
    if len(xs) == 0:
        return gray
    top = max(0, int(ys.min()) - pad)
    left = max(0, int(xs.min()) - pad)
    bottom = min(gray.shape[0], int(ys.max()) + 1 + pad)
    right = min(gray.shape[1], int(xs.max()) + 1 + pad)
    return gray[top:bottom, left:right]


def straighten_gray(gray):
    """길쭉한 손톱의 긴 축을 세로로 맞춘다. 둥근 손톱은 그대로 둔다."""
    found = foreground_up_angle(gray)
    if found is None:
        return gray
    angle, ratio = found
    if abs(angle) < 0.4 or ratio < ELONGATION_MIN:
        return gray
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += (new_w - width) / 2
    matrix[1, 2] += (new_h - height) / 2
    turned = cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return crop_foreground(turned)


def polygons_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polygons = []
    for shape in data.get("shapes") or []:
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        pts = np.asarray(shape.get("points") or [], dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            polygons.append(pts)
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    return polygons, (width, height)


def person_id(raw):
    if raw.isdigit() and len(raw) <= 3:
        return raw.zfill(3)
    return raw


def view_of(stem):
    return "side" if "side" in stem.lower() else "front"


def parse_stem(stem):
    match = STEM_RE.match(stem)
    if not match:
        return None
    rest = (match.group("rest") or "").strip("_")
    return person_id(match.group("id")), match.group("hand"), rest


def finger_token(rest):
    token = rest.lower().replace("side", "").strip("_")
    if token == "4f":
        return "4F"
    if "thumb" in token:
        return "01"
    match = re.match(r"0[1-5]", token)
    if match:
        return match.group(0)
    return None


def by_x(polygons):
    return sorted(polygons, key=lambda pts: float(pts[:, 0].mean()))


def nail_name(person, hand, finger, view):
    return f"{person}_{hand}_{finger}_{view}_nail"


def assign_nails(stem, polygons):
    """x축 왼쪽→오른쪽 순으로 손가락 번호를 붙인다. 개수가 규칙과 다르면 자르지 않는다."""
    view = view_of(stem)
    parsed = parse_stem(stem)
    if parsed is None:
        return [], "이름 규칙 없음"

    person, hand, rest = parsed
    if hand == "LF":
        hand = "LR"
    finger = finger_token(rest)
    ordered = by_x(polygons)
    items = []

    if finger == "4F" and hand == "LR":
        if len(ordered) != 8:
            return [], f"LR_4F 폴리곤 {len(ordered)}개 (8개 필요)"
        for (side, num), pts in zip(BOTH_ORDER, ordered):
            items.append((nail_name(person, side, num, view), pts))
        return items, "양손 4손가락"

    if finger == "4F":
        if len(ordered) != 4:
            return [], f"{hand}_4F 폴리곤 {len(ordered)}개 (4개 필요)"
        order = LEFT_ORDER if hand == "L" else RIGHT_ORDER
        for num, pts in zip(order, ordered):
            items.append((nail_name(person, hand, num, view), pts))
        return items, "한손 4손가락"

    if hand == "LR" and finger:
        if len(ordered) != 2:
            return [], f"LR_{finger} 폴리곤 {len(ordered)}개 (2개 필요, 왼쪽=왼손 오른쪽=오른손)"
        for side, pts in zip(("L", "R"), ordered):
            items.append((nail_name(person, side, finger, view), pts))
        return items, "양손 같은 손가락"

    if finger:
        if len(ordered) != 1:
            return [], f"{hand}_{finger} 폴리곤 {len(ordered)}개 (1개 필요)"
        items.append((nail_name(person, hand, finger, view), ordered[0]))
        return items, "손가락 한 장"

    return [], "손가락 번호 없음"


def save_gray(gray, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, encoded = cv2.imencode(ext, gray)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    encoded.tofile(path)


def canonical_stem(stem):
    parsed = parse_stem(stem)
    if not parsed:
        return stem
    person, hand, rest = parsed
    if hand == "LF":
        hand = "LR"
    if rest:
        return f"{person}_{hand}_{rest}"
    return f"{person}_{hand}"


def image_exists(folder, stem):
    key = canonical_stem(stem)
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS and os.path.isfile(os.path.join(folder, name)):
            if canonical_stem(base) == key:
                return True
    return False


def other_view_exists(folder, person, hand, finger, want_side):
    """같은 손가락의 반대 촬영이 있으면 참. 양손 한 장(LR)과 4F 정면도 포함한다."""
    suffix = "_side" if want_side else ""
    hands = ["LR", "L", "R"] if hand == "LR" else [hand, "LR"]
    for candidate in hands:
        if image_exists(folder, f"{person}_{candidate}_{finger}{suffix}"):
            return True
    if not want_side and finger in {"02", "03", "04", "05"}:
        if hand in {"L", "R"} and image_exists(folder, f"{person}_{hand}_4F"):
            return True
        if image_exists(folder, f"{person}_LR_4F"):
            return True
    return False


def nail_parts(nail):
    person, hand, finger, view, _nail = nail.split("_")
    return person, hand, finger, view


def iter_sources(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not is_image_name(name):
                continue
            yield os.path.join(dirpath, name)


def process_one(src_path, saved):
    name = os.path.basename(src_path)
    stem = os.path.splitext(name)[0]
    json_path = os.path.splitext(src_path)[0] + ".json"
    if not os.path.isfile(json_path):
        print(f"  {name}: json 없음")
        return 0, f"{name}: json 없음"
    polygons, json_size = polygons_from_json(json_path)
    if not polygons:
        print(f"  {name}: 폴리곤 없음")
        return 0, f"{name}: 폴리곤 없음"
    assigned, how = assign_nails(stem, polygons)
    if not assigned:
        line = f"{name}: 폴리곤 {len(polygons)}개, {how}"
        print(f"  {line}")
        return 0, line
    folder = os.path.dirname(src_path)
    is_side = stem.lower().endswith("_side")
    kept = []
    missing = []
    for nail, polygon in assigned:
        person, hand, finger, _view = nail_parts(nail)
        if other_view_exists(folder, person, hand, finger, want_side=not is_side):
            kept.append((nail, polygon))
        else:
            other = "측면" if not is_side else "정면"
            missing.append(f"{name}: 정면/측면 쌍 없음 ({other} {person}_{hand}_{finger} 없음)")
    for line in missing:
        print(f"  {line}")
    if not kept:
        return 0, "\n".join(missing)
    assigned = kept
    if missing:
        how = how + f", 짝 없는 손가락 {len(missing)}개 건너뜀"
    bgr = load_image(src_path, json_size)
    for nail, polygon in assigned:
        gray = straighten_gray(crop_on_black(bgr, polygon))
        save_gray(gray, os.path.join(TARGET_FOLDER_DIR, f"{nail}.jpg"))
        saved.append(nail)
    print(f"  {name}: {how} {len(assigned)}개")
    return len(assigned), "\n".join(missing) or None


def person_count_lines(saved_names):
    by_person = {}
    for name in saved_names:
        by_person.setdefault(name.split("_")[0], set()).add(name)
    lines = []
    for person in sorted(by_person):
        names = by_person[person]
        front = sum(name.endswith("_front_nail") for name in names)
        side = sum(name.endswith("_side_nail") for name in names)
        missing = []
        for hand in ("L", "R"):
            for finger in ("01", "02", "03", "04", "05"):
                for view in ("front", "side"):
                    nail = f"{person}_{hand}_{finger}_{view}_nail"
                    if nail not in names:
                        missing.append(f"{hand}_{finger}_{view}")
        if front == 10 and side == 10 and not missing:
            lines.append(f"{person}: 20장 (정면 10, 측면 10)")
        else:
            lines.append(f"{person}: {len(names)}장 (정면 {front}, 측면 {side}) 누락 {', '.join(missing)}")
    return lines


def main():
    if not os.path.isdir(ORIGIN_FOLDER_DIR):
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {ORIGIN_FOLDER_DIR}")
    os.makedirs(TARGET_FOLDER_DIR, exist_ok=True)
    sources = list(iter_sources(ORIGIN_FOLDER_DIR))
    print(f"입력 {len(sources)}장 → {TARGET_FOLDER_DIR}")

    total_crops = 0
    skipped = []
    saved = []
    for path in sources:
        crops, reason = process_one(path, saved)
        total_crops += crops
        if reason:
            skipped.append(reason)

    count_lines = person_count_lines(saved)
    log_path = os.path.join(TARGET_FOLDER_DIR, "polygon_skip.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        if skipped:
            fp.write("\n".join(skipped) + "\n")
        else:
            fp.write("폴리곤 개수가 규칙과 다른 파일 없음\n")
        fp.write("\n[사람별 장수] 빠짐이 없으면 정면 10 + 측면 10 = 20\n")
        fp.write("\n".join(count_lines) + "\n")

    print("\n사람별 장수 (빠짐이 없으면 정면 10 + 측면 10 = 20)")
    for line in count_lines:
        print(f"  {line}")
    print(f"\n완료: 이미지 {len(sources)}장, 크롭 {total_crops}개")
    print(f"건너뜀 {len(skipped)}장 → {log_path}")


if __name__ == "__main__":
    main()
