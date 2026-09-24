import json
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FOLDER_DIR = os.path.join(SCRIPT_DIR, "..", "Training", "UprightMapped")
TARGET_FOLDER_DIR = os.path.join(SCRIPT_DIR, "..", "Training", "update_data")

PADDING_RATIO = 0.05
GROUPS = ("thumb", "other")
CLASSES = ("P", "S")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LEFT_ORDER = ("05", "04", "03", "02")
RIGHT_ORDER = ("02", "03", "04", "05")
BOTH_ORDER = (
    ("L", "05"), ("L", "04"), ("L", "03"), ("L", "02"),
    ("R", "02"), ("R", "03"), ("R", "04"), ("R", "05"),
)
STEM_RE = re.compile(r"^(?P<id>.+?)_(?P<hand>LF|LR|L|R)(?:_(?P<rest>.*))?$")
CLEAN_RE = re.compile(
    r"^(?P<id>\d+)_(?P<hand>LF|LR|L|R)_(?P<finger>0[1-5])(?P<side>_side)?$",
    re.IGNORECASE,
)


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


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        name for name in os.listdir(folder)
        if is_image_name(name) and os.path.isfile(os.path.join(folder, name))
    )


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
    if "thumb" in token:
        return "01"
    match = re.match(r"0[1-5]", token)
    if match:
        return match.group(0)
    return None


def is_bundle(rest):
    token = rest.lower().replace("side", "").strip("_")
    if token in {"", "4f", "00", "front", "front1", "front2"}:
        return True
    return "4f" in token or token.startswith("front")


def by_x(polygons):
    return sorted(polygons, key=lambda pts: float(pts[:, 0].mean()))


def split_lower(polygons, keep):
    if len(polygons) <= keep:
        return list(polygons), []
    order = sorted(range(len(polygons)), key=lambda i: float(polygons[i][:, 1].mean()))
    upper = [polygons[i] for i in order[:keep]]
    lower = [polygons[i] for i in order[keep:]]
    return upper, lower


def nail_name(person, hand, finger, view):
    return f"{person}_{hand}_{finger}_{view}_nail"


def assign_nails(stem, polygons, source_group, view=None):
    parsed = parse_stem(stem)
    if view is None:
        view = view_of(stem)
    if parsed is None:
        return [(f"{stem}_{view}_nail", source_group, pts) for pts in polygons], "이름 규칙 없음"

    person, hand, rest = parsed
    both = hand in {"LF", "LR"}
    numbered = finger_token(rest)
    items = []

    if both and len(polygons) == 2 and numbered is None and ("front" in rest.lower() or "thumb" in rest.lower()):
        for side, pts in zip(("L", "R"), by_x(polygons)):
            items.append((nail_name(person, side, "01", view), "thumb", pts))
        return items, "양손 엄지"

    if not both and len(polygons) == 1 and numbered is None and source_group == "thumb" and is_bundle(rest):
        return [(nail_name(person, hand, "01", view), "thumb", polygons[0])], "엄지 한 장"

    if both and numbered and len(polygons) == 2:
        for side, pts in zip(("L", "R"), by_x(polygons)):
            items.append((nail_name(person, side, numbered, view), "thumb" if numbered == "01" else "other", pts))
        return items, "양손 같은 손가락"

    if not both and numbered and len(polygons) >= 1:
        group = "thumb" if numbered == "01" else "other"
        chosen = max(polygons, key=lambda pts: float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())))
        return [(nail_name(person, hand, numbered, view), group, chosen)], "파일명 손가락"

    if is_bundle(rest):
        if both:
            upper, lower = split_lower(polygons, 8)
            if len(upper) == 8:
                for (side, finger), pts in zip(BOTH_ORDER, by_x(upper)):
                    items.append((nail_name(person, side, finger, view), "other", pts))
                for side, pts in zip(("L", "R"), by_x(lower)):
                    items.append((nail_name(person, side, "01", view), "thumb", pts))
                return items, "양손 묶음"
        else:
            order = LEFT_ORDER if hand == "L" else RIGHT_ORDER
            upper, lower = split_lower(polygons, 4)
            if len(upper) == 4:
                for finger, pts in zip(order, by_x(upper)):
                    items.append((nail_name(person, hand, finger, view), "other", pts))
                for pts in lower:
                    items.append((nail_name(person, hand, "01", view), "thumb", pts))
                return items, "한손 묶음"

    finger = numbered or "00"
    group = "thumb" if finger == "01" else "other"
    side = "L" if hand == "LF" else "R" if hand == "LR" else hand
    for index, pts in enumerate(by_x(polygons)):
        name = nail_name(person, side, finger, view)
        if len(polygons) > 1:
            name = f"{name}_{index}"
        items.append((name, source_group if group is None else group, pts))
    return items, "순서 불명"


def save_gray(gray, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, gray)


def clean_name(stem):
    """순번_손_손가락 또는 순번_손_손가락_side 만 처리 대상이다."""
    match = CLEAN_RE.match(stem)
    if not match:
        return None
    return person_id(match.group("id")), bool(match.group("side"))


def person_key(stem):
    parsed = parse_stem(stem)
    if parsed:
        return parsed[0]
    return person_id(stem.split("_")[0])


def collect_images():
    rows = []
    for group in GROUPS:
        for label in CLASSES:
            src_dir = os.path.join(ORIGIN_FOLDER_DIR, group, label)
            for name in list_images(src_dir):
                stem = os.path.splitext(name)[0]
                rows.append((person_key(stem), group, label, name, os.path.join(src_dir, name)))
    return rows


SKIP_PEOPLE = {"002", "005"}


def is_front_bundle(stem):
    parsed = parse_stem(stem)
    if parsed is None:
        return False
    rest = parsed[2].lower()
    return "4f" in rest or rest == "00"


def other_view(stem, bundle_person):
    if is_front_bundle(stem):
        return "front"
    if bundle_person:
        return "side"
    return "side" if "side" in stem.lower() else "front"


def process_one(group, label, src_path, view=None, dest_dir=None):
    name = os.path.basename(src_path)
    stem = os.path.splitext(name)[0]
    json_path = os.path.splitext(src_path)[0] + ".json"
    if not os.path.isfile(json_path):
        print(f"  {name}: json 없음")
        return 0, True
    polygons, json_size = polygons_from_json(json_path)
    if not polygons:
        print(f"  {name}: 폴리곤 없음")
        return 0, True
    bgr = load_image(src_path, json_size)
    assigned, how = assign_nails(stem, polygons, group, view)
    out_dir = dest_dir or os.path.join(TARGET_FOLDER_DIR, label)
    for nail, _dest_group, polygon in assigned:
        gray = crop_on_black(bgr, polygon)
        dst = os.path.join(out_dir, f"{nail}.jpg")
        copy_index = 2
        while os.path.exists(dst):
            dst = os.path.join(out_dir, f"{nail}_{copy_index}.jpg")
            copy_index += 1
        save_gray(gray, dst)
    shown = view if view is not None else view_of(stem)
    print(f"  {group}/{label}/{name}: {shown} {how} {len(assigned)}개")
    return len(assigned), False


def main():
    if not os.path.isdir(ORIGIN_FOLDER_DIR):
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {ORIGIN_FOLDER_DIR}")

    rows = [row for row in collect_images() if row[1] == "other" and row[0] not in SKIP_PEOPLE]
    bundle_people = {
        person for person, _group, _label, name, _path in rows if is_front_bundle(os.path.splitext(name)[0])
    }
    dest = os.path.join(TARGET_FOLDER_DIR, "other")
    os.makedirs(dest, exist_ok=True)
    print(f"other 처리 {len(rows)}장, 4F/00 있는 사람 {len(bundle_people)}명, 스킵 {sorted(SKIP_PEOPLE)}")

    total_crops = 0
    failed = []
    for person, group, label, name, path in rows:
        stem = os.path.splitext(name)[0]
        view = other_view(stem, person in bundle_people)
        crops, miss = process_one(group, label, path, view=view, dest_dir=dest)
        total_crops += crops
        if miss:
            failed.append(os.path.join(group, label, name))

    print(f"\n완료: 이미지 {len(rows)}장, 크롭 {total_crops}개 → {dest}")
    print(f"json/폴리곤 없음 {len(failed)}장")
    for path in failed:
        print(f"  건너뜀: {path}")


if __name__ == "__main__":
    main()
