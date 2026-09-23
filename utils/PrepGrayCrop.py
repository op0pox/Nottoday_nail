import json
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FOLDER_DIR = os.path.join(SCRIPT_DIR, "..", "Training", "Gray_Train_Data")
TARGET_FOLDER_DIR = os.path.join(SCRIPT_DIR, "..", "Training", "Gray_Train_Data", "ClassifyPrep")
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "Training",
    "Train_model",
    "nail_segmentation_33people",
    "weights",
    "best.pt",
)

CONF = 0.25
IMGSZ = 1024
PADDING_RATIO = 0.05
VAL_RATIO = 0.2
SPLIT_SEED = 0
GROUPS = ("thumb", "other")
CLASSES = ("P", "S")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_for_predict(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image = image.convert("RGB")
        width, height = image.size
        rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, width, height


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


def crop_on_white(bgr, polygon):
    height, width = bgr.shape[:2]
    left, top, right, bottom = padded_crop_box(polygon, width, height)
    crop = bgr[top:bottom, left:right].copy()
    shifted = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
    shifted[:, 0] -= left
    shifted[:, 1] -= top
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 255)
    white = np.full_like(crop, 255)
    white[mask > 0] = crop[mask > 0]
    return cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)


def polygons_from_result(result):
    polygons = []
    if result.masks is None:
        return polygons
    for seg in result.masks.xy:
        pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        polygons.append(pts)
    return polygons


def json_path_for(src_path):
    stem, _ = os.path.splitext(src_path)
    path = stem + ".json"
    return path if os.path.isfile(path) else None


def polygons_from_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return []
    return [pts]


def polygons_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    if isinstance(data, dict) and isinstance(data.get("shapes"), list):
        for shape in data["shapes"]:
            if shape.get("shape_type", "polygon") not in ("polygon", "linestrip"):
                continue
            polygons.extend(polygons_from_points(shape.get("points") or []))
        return polygons

    if isinstance(data, list):
        return polygons_from_points(data)

    return polygons


def gray_crops(model, src_path):
    bgr, _, _ = load_image_for_predict(src_path)
    label_path = json_path_for(src_path)
    if label_path:
        polygons = polygons_from_json(label_path)
        source = "json"
    else:
        result = model.predict(bgr, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        polygons = polygons_from_result(result)
        source = "seg"
    return [crop_on_white(bgr, polygon) for polygon in polygons], source


def split_names(names):
    ordered = list(names)
    random.Random(SPLIT_SEED).shuffle(ordered)
    if len(ordered) <= 1:
        return ordered, []
    n_val = int(round(len(ordered) * VAL_RATIO))
    n_val = min(max(n_val, 1), len(ordered) - 1)
    return ordered[:-n_val], ordered[-n_val:]


def save_crops(crops, dst_dir, stem):
    os.makedirs(dst_dir, exist_ok=True)
    for index, gray in enumerate(crops):
        path = os.path.join(dst_dir, f"{stem}_n{index}.jpg")
        cv2.imwrite(path, gray)


def ensure_output_dirs():
    for group in GROUPS:
        for split in ("train", "val"):
            for label in CLASSES:
                os.makedirs(os.path.join(TARGET_FOLDER_DIR, group, split, label), exist_ok=True)


def process_class(model, group, label):
    src_dir = os.path.join(ORIGIN_FOLDER_DIR, group, label)
    names = list_images(src_dir)
    if not names:
        print(f"\n[{group}/{label}] 이미지 없음")
        return {"images": 0, "crops": 0, "skipped": []}

    train_names, val_names = split_names(names)
    split_of = {name: "train" for name in train_names}
    split_of.update({name: "val" for name in val_names})
    print(f"\n[{group}/{label}] 이미지 {len(names)}장 (train {len(train_names)}, val {len(val_names)})")

    crops_saved = 0
    skipped = []
    for name in names:
        src_path = os.path.join(src_dir, name)
        crops, source = gray_crops(model, src_path)
        if not crops:
            skipped.append(os.path.join(group, label, name))
            print(f"  {name}: {source} 검출 없음")
            continue
        stem = os.path.splitext(name)[0]
        dst_dir = os.path.join(TARGET_FOLDER_DIR, group, split_of[name], label)
        save_crops(crops, dst_dir, stem)
        crops_saved += len(crops)
        print(f"  {name}: {split_of[name]} {source} nail {len(crops)}개")

    return {"images": len(names), "crops": crops_saved, "skipped": skipped}


def main():
    if not ORIGIN_FOLDER_DIR or not TARGET_FOLDER_DIR or not MODEL_PATH:
        raise ValueError("ORIGIN_FOLDER_DIR, TARGET_FOLDER_DIR, MODEL_PATH를 지정하세요.")
    if not os.path.isdir(ORIGIN_FOLDER_DIR):
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {ORIGIN_FOLDER_DIR}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"모델이 없습니다: {MODEL_PATH}")

    ensure_output_dirs()
    model = YOLO(MODEL_PATH)
    total_images = 0
    total_crops = 0
    skipped = []
    for group in GROUPS:
        for label in CLASSES:
            stats = process_class(model, group, label)
            total_images += stats["images"]
            total_crops += stats["crops"]
            skipped.extend(stats["skipped"])

    print(f"\n완료: 이미지 {total_images}장, 크롭 {total_crops}개, 검출 실패 {len(skipped)}장")
    for path in skipped:
        print(f"  건너뜀: {path}")


if __name__ == "__main__":
    main()
