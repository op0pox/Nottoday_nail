import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_DIR = r"C:\Users\USER\Downloads\022(이태민)"
TARGET_DIR = os.path.join(SCRIPT_DIR, "..", "Training", "AutoLabel", "022")
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "Training",
    "Train_model",
    "nail_segmentation_24people",
    "weights",
    "best.pt",
)

TARGET_POINTS = 30
CONF = 0.25
IMGSZ = 1024
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def simplify_to_n(pts, n=TARGET_POINTS):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= n:
        return pts.tolist()

    contour = pts.reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return pts.tolist()

    lo, hi = 1e-4, 0.2
    best = pts
    for _ in range(25):
        mid = (lo + hi) / 2
        approx = cv2.approxPolyDP(contour, mid * peri, True).reshape(-1, 2)
        if len(approx) >= 3:
            best = approx
        if len(approx) > n:
            lo = mid
        else:
            hi = mid
    return best.reshape(-1, 2).tolist()


def make_shape(points):
    return {
        "label": "nail",
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": None,
        "description": "",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def make_labelme_json(image_filename, width, height, polygons):
    return {
        "version": "7.0.4",
        "flags": {},
        "shapes": [make_shape(poly) for poly in polygons if len(poly) >= 3],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def load_image_for_predict(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image = image.convert("RGB")
        width, height = image.size
        rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, width, height


def label_one(model, src_path, dst_dir):
    file_name = os.path.basename(src_path)
    dst_image_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_image_path)

    bgr, width, height = load_image_for_predict(src_path)
    result = model.predict(bgr, conf=CONF, imgsz=IMGSZ, verbose=False)[0]

    polygons = []
    if result.masks is not None:
        for seg in result.masks.xy:
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            polygons.append(simplify_to_n(pts, TARGET_POINTS))

    json_name = os.path.splitext(file_name)[0] + ".json"
    json_path = os.path.join(dst_dir, json_name)
    data = make_labelme_json(file_name, width, height, polygons)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(polygons)


def main():
    if not ORIGIN_DIR or not TARGET_DIR or not MODEL_PATH:
        raise ValueError("ORIGIN_DIR, TARGET_DIR, MODEL_PATH를 지정하세요.")
    if not os.path.isdir(ORIGIN_DIR):
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {ORIGIN_DIR}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"모델이 없습니다: {MODEL_PATH}")

    os.makedirs(TARGET_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    image_names = sorted(
        name for name in os.listdir(ORIGIN_DIR)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    )
    if not image_names:
        raise RuntimeError(f"이미지가 없습니다: {ORIGIN_DIR}")

    for name in image_names:
        src_path = os.path.join(ORIGIN_DIR, name)
        n_poly = label_one(model, src_path, TARGET_DIR)
        print(f"{name}: nail {n_poly}개")


if __name__ == "__main__":
    main()
