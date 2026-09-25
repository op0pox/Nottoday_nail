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
  _side: 측면. 없으면 정면.
  _4F: 네 손가락 정면을 한 장에 촬영. LR_4F는 양손이라 폴리곤 8개.
  LR_손가락: 양손의 같은 손가락. 폴리곤은 최대 2개.

자를 때 왼쪽부터:
  왼손 L: 05 04 03 02
  오른손 R: 02 03 04 05
  양손 LR: 05 04 03 02 02 03 04 05
  새끼손가락=05, 검지=02, 엄지=01

출력 파일명:
  순번_손_손가락_front_nail.jpg
  순번_손_손가락_side_nail.jpg
  이름순으로 정렬하면 같은 손가락의 정면, 측면이 붙는다.

크롭 뒤 손톱이 살짝 기울어 있으면, 긴 축이 세로가 되도록 한 번 더 바로잡는다.
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


def straighten_gray(gray, max_degrees=20):
    """크롭된 손톱의 긴 축을 세로로 맞춘다. 이미 11자에 가까우면 살짝만 돌린다."""
    fg = gray > 15
    ys, xs = np.nonzero(fg)
    if len(xs) < 30:
        return gray
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    _center, (rw, rh), angle = cv2.minAreaRect(pts)
    # minAreaRect 각도는 가로변 기준 [-90, 0). 긴 변이 세로가 되도록 보정각을 고른다.
    if rw < rh:
        tilt = angle + 90
    else:
        tilt = angle
    if tilt > 45:
        tilt -= 90
    elif tilt < -45:
        tilt += 90
    if abs(tilt) < 0.4 or abs(tilt) > max_degrees:
        return gray
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), tilt, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += (new_w - width) / 2
    matrix[1, 2] += (new_h - height) / 2
    return cv2.warpAffine(gray, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)


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
        if len(ordered) > 2:
            return [], f"LR_{finger} 폴리곤 {len(ordered)}개 (최대 2개)"
        if len(ordered) == 2:
            for side, pts in zip(("L", "R"), ordered):
                items.append((nail_name(person, side, finger, view), pts))
            return items, "양손 같은 손가락"
        items.append((nail_name(person, "L", finger, view), ordered[0]))
        return items, "양손인데 폴리곤 1개"

    if finger:
        if len(ordered) != 1:
            return [], f"{hand}_{finger} 폴리곤 {len(ordered)}개 (1개 필요)"
        items.append((nail_name(person, hand, finger, view), ordered[0]))
        return items, "손가락 한 장"

    return [], "손가락 번호 없음"


def save_gray(gray, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, gray)


def iter_sources(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if not is_image_name(name):
                continue
            yield os.path.join(dirpath, name)


def process_one(src_path):
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
    bgr = load_image(src_path, json_size)
    for nail, polygon in assigned:
        gray = straighten_gray(crop_on_black(bgr, polygon))
        save_gray(gray, os.path.join(TARGET_FOLDER_DIR, f"{nail}.jpg"))
    print(f"  {name}: {how} {len(assigned)}개")
    return len(assigned), None


def main():
    if not os.path.isdir(ORIGIN_FOLDER_DIR):
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {ORIGIN_FOLDER_DIR}")
    os.makedirs(TARGET_FOLDER_DIR, exist_ok=True)
    sources = list(iter_sources(ORIGIN_FOLDER_DIR))
    print(f"입력 {len(sources)}장 → {TARGET_FOLDER_DIR}")

    total_crops = 0
    skipped = []
    for path in sources:
        crops, reason = process_one(path)
        total_crops += crops
        if reason:
            skipped.append(reason)

    log_path = os.path.join(TARGET_FOLDER_DIR, "polygon_skip.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        if skipped:
            fp.write("\n".join(skipped) + "\n")
        else:
            fp.write("폴리곤 개수가 규칙과 다른 파일 없음\n")

    print(f"\n완료: 이미지 {len(sources)}장, 크롭 {total_crops}개")
    print(f"건너뜀 {len(skipped)}장 → {log_path}")


if __name__ == "__main__":
    main()
