"""제멋대로 놓인 손 사진을, 손가락이 아래쪽에서 위를 향하도록 수직으로 돌리고
LabelMe JSON에 손가락 번호를 기록한다.
크롭과 흑백 변환은 하지 않는다.
원본·결과 폴더는 아래 ORIGIN_FOLDER_DIR, TARGET_FOLDER_DIR 에 적는다.
    python RotatePrep.py
필요 패키지: pip install pillow numpy
파일명을 번호로 바꿀 때 쓰는 규칙:
- L(왼손), 왼쪽부터: 05 04 03 02
- R(오른손), 왼쪽부터: 02 03 04 05
- LF는 LR의 오타. 양손, 왼쪽부터: 05 04 03 02 02 03 04 05
- 네 손가락 정면은 4F, front, 00 으로 이름이 붙을 수 있다
- 그 줄 아래에 더 있는 손톱은 엄지로 본다
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

# 여기 두 경로만 바꾸면 된다.
SCRIPT_DIR = Path(__file__).resolve().parent
ORIGIN_FOLDER_DIR = SCRIPT_DIR.parent / "Training" / "Gray_Train_Data"
TARGET_FOLDER_DIR = SCRIPT_DIR.parent / "Training" / "UprightMapped"

LEFT_ORDER = ("05", "04", "03", "02")
RIGHT_ORDER = ("02", "03", "04", "05")
BOTH_HANDS_ORDER = LEFT_ORDER + RIGHT_ORDER
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG", ".webp", ".bmp")
STEM_RE = re.compile(r"^(?P<id>.+?)_(?P<hand>LF|LR|L|R)(?:_(?P<rest>.*))?$")
def find_image(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None
def skin_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    return (red > 85) & (green > 35) & (blue > 20) & (red > green + 6) & (red > blue)
def palm_side(rgb: np.ndarray, polygons) -> str:
    """손톱 바깥에서 피부가 이어지는 방향. 'down'이면 이미 손가락이 위를 향한다."""
    height, width = rgb.shape[:2]
    skin = skin_mask(rgb)
    votes = {"down": 0, "up": 0, "right": 0, "left": 0}
    for pts in polygons:
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        box_w = max(x2 - x1, 8.0)
        box_h = max(y2 - y1, 8.0)
        def density(xa, ya, xb, yb) -> int:
            xa, xb = int(max(0, xa)), int(min(width, xb))
            ya, yb = int(max(0, ya)), int(min(height, yb))
            if xb <= xa or yb <= ya:
                return 0
            return int(skin[ya:yb, xa:xb].mean() * 1000)
        scores = {
            "down": density(x1, y2, x2, y2 + box_h * 1.2),
            "up": density(x1, y1 - box_h * 1.2, x2, y1),
            "right": density(x2, y1, x2 + box_w * 1.2, y2),
            "left": density(x1 - box_w * 1.2, y1, x1, y2),
        }
        votes[max(scores, key=scores.get)] += 1
    return max(votes, key=votes.get)
def rotate_points(polygons, width: int, height: int, palm: str):
    """손가락 끝이 이미지 위쪽이 되도록 다각형 좌표를 회전한다."""
    if palm == "down":
        return polygons, 0
    rotated = []
    if palm == "right":
        for pts in polygons:
            nxt = np.empty_like(pts)
            nxt[:, 0] = height - 1 - pts[:, 1]
            nxt[:, 1] = pts[:, 0]
            rotated.append(nxt)
        return rotated, 90
    if palm == "left":
        for pts in polygons:
            nxt = np.empty_like(pts)
            nxt[:, 0] = pts[:, 1]
            nxt[:, 1] = width - 1 - pts[:, 0]
            rotated.append(nxt)
        return rotated, -90
    for pts in polygons:
        nxt = np.empty_like(pts)
        nxt[:, 0] = width - 1 - pts[:, 0]
        nxt[:, 1] = height - 1 - pts[:, 1]
        rotated.append(nxt)
    return rotated, 180
def rotate_image(image: Image.Image, degrees: int) -> Image.Image:
    if degrees == 90:
        return image.transpose(Image.Transpose.ROTATE_270)
    if degrees == -90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if degrees == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    return image
def centers(polygons):
    return np.array([pts.mean(axis=0) for pts in polygons])
def drop_thumbs(polygons):
    """양손이 한 장에 있을 때, 손가락 줄 아래의 엄지 두 개를 뺀다."""
    if len(polygons) <= 8:
        return polygons
    cy = centers(polygons)[:, 1]
    keep = cy <= np.median(cy) + (cy.max() - cy.min()) * 0.15
    if int(keep.sum()) != 8:
        order = np.argsort(cy)
        keep = np.zeros(len(polygons), dtype=bool)
        keep[order[:8]] = True
    return [poly for poly, flag in zip(polygons, keep) if flag]
def finger_order(token: str):
    if token in {"LR", "LF"}:
        return BOTH_HANDS_ORDER
    if token == "L":
        return LEFT_ORDER
    if token == "R":
        return RIGHT_ORDER
    return None
def finger_ids(stem: str, polygons):
    token = stem.split("_")[1] if "_" in stem else ""
    both = token in {"LR", "LF"}
    upright = drop_thumbs(polygons) if both else list(polygons)
    if not both and len(upright) > 4:
        cy = centers(upright)[:, 1]
        order = np.argsort(cy)
        upright = [upright[index] for index in np.argsort(cy)[:4]]
    upright = sorted(upright, key=lambda pts: float(pts[:, 0].mean()))
    order = finger_order(token)
    if order is None or len(upright) != len(order):
        return [(None, pts) for pts in upright]
    return list(zip(order, upright))
def parse_stem(stem: str):
    match = STEM_RE.match(stem)
    if not match:
        return stem, "", ""
    return match.group("id"), match.group("hand"), match.group("rest") or ""
def is_front_bundle(rest: str, hand: str, count: int) -> bool:
    if "side" in rest.lower():
        return False
    token = rest.lower()
    if token in {"4f", "00", "front", "front1", "front2"}:
        return True
    if "4f" in token or token.startswith("front"):
        return True
    return hand in {"LR", "LF"} and rest == "" and count >= 4


def load_polygons(data):
    shapes = []
    polygons = []
    for shape in data.get("shapes", []):
        points = shape.get("points") or []
        if shape.get("shape_type") not in (None, "polygon") or len(points) < 3:
            continue
        shapes.append(shape)
        polygons.append(np.asarray(points, dtype=np.float64))
    return shapes, polygons


def open_matched(image_path: Path, width: int, height: int) -> Image.Image:
    """JSON에 적힌 크기와 맞는 방향으로 연다. EXIF 회전이 그 크기면 그걸 따른다."""
    with Image.open(image_path) as opened:
        raw = opened.convert("RGB")
        turned = ImageOps.exif_transpose(opened)
        oriented = raw if turned is None else turned.convert("RGB")
    if oriented.size == (width, height) or raw.size != (width, height):
        return oriented
    return raw


def write_labels(stem: str, shapes, polygons) -> None:
    """회전된 좌표를 넣고, 개수가 맞을 때만 손가락 번호로 라벨을 바꾼다."""
    labeled = {id(pts): finger for finger, pts in finger_ids(stem, polygons)}
    for shape, pts in zip(shapes, polygons):
        shape["points"] = [[float(x), float(y)] for x, y in pts]
        if id(pts) not in labeled:
            shape["label"] = "thumb"
            continue
        finger = labeled[id(pts)]
        if finger is not None:
            shape["label"] = finger


def save_pair(image: Image.Image, data: dict, image_path: Path, dest_json: Path) -> None:
    dest_json.parent.mkdir(parents=True, exist_ok=True)
    dest_image = dest_json.with_suffix(image_path.suffix)
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(dest_image, quality=95)
    else:
        image.save(dest_image)
    data["imagePath"] = dest_image.name
    data["imageData"] = None
    data["imageWidth"] = image.width
    data["imageHeight"] = image.height
    dest_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_one(json_path: Path, origin: Path, target: Path) -> str:
    image_path = find_image(json_path)
    if image_path is None:
        return "이미지 없음"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes, polygons = load_polygons(data)
    if not polygons:
        return "손톱 없음"
    image = open_matched(image_path, int(data.get("imageWidth") or 0), int(data.get("imageHeight") or 0))
    palm = palm_side(np.asarray(image), polygons)
    polygons, degrees = rotate_points(polygons, image.width, image.height, palm)
    image = rotate_image(image, degrees)
    write_labels(json_path.stem, shapes, polygons)
    save_pair(image, data, image_path, target / json_path.relative_to(origin))
    return f"{degrees}°"


def main() -> None:
    origin = Path(ORIGIN_FOLDER_DIR)
    target = Path(TARGET_FOLDER_DIR)
    if not origin.is_dir():
        raise FileNotFoundError(f"원본 폴더가 없습니다: {origin}")
    json_paths = [
        path for path in origin.rglob("*.json")
        if not path.is_relative_to(target)
    ]
    done = 0
    for path in json_paths:
        status = process_one(path, origin, target)
        print(f"{path.relative_to(origin)}: {status}")
        if status.endswith("°"):
            done += 1
    print(f"완료: {done}개 → {target}")


if __name__ == "__main__":
    main()
