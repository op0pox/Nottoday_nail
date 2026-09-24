import glob
import math
import os
import datetime

import cv2
import numpy as np
from ultralytics import YOLO

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PREP_DIR = os.path.join(TRAINING_DIR, "Gray_Train_Data", "ClassifyPrep")
DATASETS = (
    ("front", "thumb"),
    ("front", "other"),
    ("side", "thumb"),
    ("side", "other"),
)

IMGSZ = 224
CELL_W = 280
CELL_H = 220
GRID_COLS = 4
OUTPUT_DIR = os.path.join(TRAINING_DIR, "results")


def imread_unicode(path):
    img_array = np.fromfile(path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")
    encoded.tofile(path)


def latest_weights(view, finger):
    pattern = os.path.join(
        TRAINING_DIR,
        "Train_model",
        f"nail_classification_{view}_{finger}_*",
        "weights",
        "best.pt",
    )
    found = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not found:
        raise FileNotFoundError(f"분류 가중치가 없습니다: {pattern}")
    return found[-1]


def list_val_images(data_dir):
    rows = []
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val 폴더가 없습니다: {val_dir}")
    for label in sorted(os.listdir(val_dir)):
        class_dir = os.path.join(val_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                rows.append((os.path.join(class_dir, name), label))
    return rows


def letterbox(image, cell_w, cell_h):
    h, w = image.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def put_label(image, text, color):
    cv2.putText(image, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def build_grid(cells):
    if not cells:
        canvas = np.full((CELL_H, CELL_W, 3), 255, dtype=np.uint8)
        put_label(canvas, "오분류 없음", (0, 128, 0))
        return canvas
    rows = math.ceil(len(cells) / GRID_COLS)
    canvas = np.full((rows * CELL_H, GRID_COLS * CELL_W, 3), 255, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r = idx // GRID_COLS
        c = idx % GRID_COLS
        canvas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cell
    return canvas


def evaluate(view, finger):
    data_dir = os.path.join(PREP_DIR, view, finger)
    weights = latest_weights(view, finger)
    samples = list_val_images(data_dir)
    if not samples:
        raise RuntimeError(f"val 이미지가 없습니다: {data_dir}")

    model = YOLO(weights)
    paths = [path for path, _label in samples]
    predictions = model.predict(paths, imgsz=IMGSZ, verbose=False)

    correct = 0
    per_class = {}
    wrong_cells = []
    for (path, label), result in zip(samples, predictions):
        pred = result.names[int(result.probs.top1)]
        per_class.setdefault(label, {"correct": 0, "total": 0})
        per_class[label]["total"] += 1
        if pred == label:
            correct += 1
            per_class[label]["correct"] += 1
            continue
        image = imread_unicode(path)
        if image is None:
            continue
        cell = letterbox(image, CELL_W, CELL_H)
        put_label(cell, f"GT {label} / Pred {pred}", (0, 0, 255))
        wrong_cells.append(cell)

    accuracy = correct / len(samples)
    print(f"{view}/{finger}  {weights}")
    print(f"  val {len(samples)}장  accuracy {accuracy:.4f}")
    for label in sorted(per_class):
        stat = per_class[label]
        print(f"  {label} {stat['correct']}/{stat['total']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"cls_{view}_{finger}_{stamp}.jpg")
    imwrite_unicode(output_path, build_grid(wrong_cells))
    print(f"  오분류 이미지 {output_path}")


if __name__ == "__main__":
    for view, finger in DATASETS:
        evaluate(view, finger)
