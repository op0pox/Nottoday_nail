import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FOLDER_DIR = r"C:\Users\USER\vscode-workspace\Nottoday_nail\Latest_Data\New체커보드"
TARGET_FOLDER_DIR = r"C:\Users\USER\vscode-workspace\Nottoday_nail\Latest_Data_label"
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "Training",
    "Train_model",
    "nail_segmentation_33people",
    "weights",
    "best.pt",
)

TARGET_POINTS =100
CONF = 0.25
IMGSZ = 1024
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resample_to_n(pts, n=TARGET_POINTS):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or n < 3:
        return pts.tolist()

    closed = np.vstack([pts, pts[:1]])
    dist = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(dist)])
    for i in range(1, len(cum)):
        if cum[i] <= cum[i - 1]:
            cum[i] = cum[i - 1] + 1e-6
    if cum[-1] <= 0:
        return pts.tolist()

    samples = np.linspace(0.0, cum[-1], n, endpoint=False)
    xs = np.interp(samples, cum, closed[:, 0])
    ys = np.interp(samples, cum, closed[:, 1])
    return np.stack([xs, ys], axis=1).tolist()


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
            polygons.append(resample_to_n(pts, TARGET_POINTS))

    json_name = os.path.splitext(file_name)[0] + ".json"
    json_path = os.path.join(dst_dir, json_name)
    data = make_labelme_json(file_name, width, height, polygons)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(polygons)


def is_image_name(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def list_images(folder):
    return sorted(
        name for name in os.listdir(folder)
        if is_image_name(name) and os.path.isfile(os.path.join(folder, name))
    )


def iter_image_folders(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if any(is_image_name(name) for name in filenames):
            yield dirpath


def label_folder(model, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    image_names = list_images(src_dir)
    rel_dir = os.path.relpath(src_dir, ORIGIN_FOLDER_DIR)
    print(f"\n[{rel_dir}] 이미지 {len(image_names)}장")
    for name in image_names:
        src_path = os.path.join(src_dir, name)
        n_poly = label_one(model, src_path, dst_dir)
        print(f"  {name}: nail {n_poly}개")
    return len(image_names)


def main():
    if not ORIGIN_FOLDER_DIR or not TARGET_FOLDER_DIR or not MODEL_PATH:
        raise ValueError("ORIGIN_FOLDER_DIR, TARGET_FOLDER_DIR, MODEL_PATH를 지정하세요.")
    if not os.path.isdir(ORIGIN_FOLDER_DIR):
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {ORIGIN_FOLDER_DIR}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"모델이 없습니다: {MODEL_PATH}")

    image_folders = list(iter_image_folders(ORIGIN_FOLDER_DIR))
    if not image_folders:
        raise RuntimeError(f"이미지가 없습니다: {ORIGIN_FOLDER_DIR}")

    model = YOLO(MODEL_PATH)
    total = 0
    for src_dir in image_folders:
        rel_dir = os.path.relpath(src_dir, ORIGIN_FOLDER_DIR)
        dst_dir = os.path.normpath(os.path.join(TARGET_FOLDER_DIR, rel_dir))
        total += label_folder(model, src_dir, dst_dir)
    print(f"\n완료: 폴더 {len(image_folders)}개, 이미지 {total}장")


if __name__ == "__main__":
    main()
