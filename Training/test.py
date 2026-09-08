import math
import os
import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# 선 색 (OpenCV BGR)
# 초록 (0, 255, 0) : GT 정답 폴리곤
# 파랑 (255, 0, 0) : Old 모델 예측
# 빨강 (0, 0, 255) : New 모델 예측

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

OLD_MODEL_PATH = os.path.join(TRAINING_DIR, "Train_model", "nail_segmentation_24people", "weights", "best.pt")
NEW_MODEL_PATH = os.path.join(TRAINING_DIR, "Train_model", "nail_segmentation_croped", "weights", "best.pt")

# TestDataset 하위폴더중 YOLODataset폴더중 선택
TEST_DATASET_NAME = "YOLODataset_fulldata_online"

CELL_W = 960
CELL_H = 420
GRID_COLS = 2
MAX_JPEG_DIM = 65000

OUTPUT_DIR = os.path.join(TRAINING_DIR, "results")


def imread_unicode(path):
    img_array = np.fromfile(path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        scale = min(MAX_JPEG_DIM / image.shape[0], MAX_JPEG_DIM / image.shape[1], 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path} size={image.shape}")
    encoded.tofile(path)


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


def load_gt_polygons(txt_path, width, height):
    polygons = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) > 1:
                coords = np.array(parts[1:]).reshape(-1, 2)
                coords[:, 0] *= width
                coords[:, 1] *= height
                polygons.append(coords.astype(np.int32))
    return polygons


def mask_from_polygons(shape, polygons):
    mask = np.zeros(shape, dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    return mask


def draw_predictions(image, results, color):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    drawn = image.copy()
    if results.masks is not None:
        for seg in results.masks.xy:
            pts = np.array(seg, dtype=np.int32)
            cv2.polylines(drawn, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(mask, [pts], 255)
    return drawn, mask


def iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b)
    union = np.logical_or(mask_a, mask_b)
    union_sum = np.sum(union)
    return np.sum(intersection) / union_sum if union_sum > 0 else 0.0


def short_name(file_name):
    return os.path.splitext(file_name)[0][:12]


def put_label(image, text, color):
    h, w = image.shape[:2]
    scale = max(0.7, min(h, w) / 280)
    thick = max(2, int(scale * 2))
    cv2.putText(image, text, (8, int(28 * scale) + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(image, text, (8, int(28 * scale) + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def build_legend(width):
    h = 70
    canvas = np.full((h, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    items = [
        ((0, 255, 0), "Green: 정답지"),
        ((255, 0, 0), "Blue: 기존모델"),
        ((0, 0, 255), "Red: 신규모델"),
    ]
    x = 30
    for color, text in items:
        y = 42
        cv2.line(canvas, (x, y - 8), (x + 50, y - 8), color, 6)
        cv2.putText(canvas, text, (x + 60, y), font, 1.0, (0, 0, 0), 2)
        x += 420
    return canvas


def build_grid(cells, cols, cell_w, cell_h):
    rows = math.ceil(len(cells) / cols)
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r = idx // cols
        c = idx % cols
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
    return canvas


def build_table(records, width):
    row_h = 58
    table_h = (len(records) + 3) * row_h
    canvas = np.full((table_h, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thick = 2
    col_x = [int(width * 0.04), int(width * 0.58), int(width * 0.78)]

    cv2.putText(canvas, "File", (col_x[0], row_h - 16), font, scale, (0, 0, 0), thick)
    cv2.putText(canvas, "Old IoU", (col_x[1], row_h - 16), font, scale, (0, 0, 0), thick)
    cv2.putText(canvas, "New IoU", (col_x[2], row_h - 16), font, scale, (0, 0, 0), thick)
    cv2.line(canvas, (0, row_h), (width, row_h), (0, 0, 0), 3)

    for i, (fname, i_old, i_new) in enumerate(records):
        y = (i + 2) * row_h
        cv2.putText(canvas, short_name(fname), (col_x[0], y - 16), font, scale, (0, 0, 0), thick)
        cv2.putText(canvas, f"{i_old:.4f}", (col_x[1], y - 16), font, scale, (255, 0, 0), thick)
        cv2.putText(canvas, f"{i_new:.4f}", (col_x[2], y - 16), font, scale, (0, 0, 255), thick)
        cv2.line(canvas, (0, y), (width, y), (220, 220, 220), 2)

    if records:
        mean_old = sum(r[1] for r in records) / len(records)
        mean_new = sum(r[2] for r in records) / len(records)
        y = (len(records) + 2) * row_h
        cv2.putText(canvas, f"평균 IoU : 개수={len(records)}", (col_x[0], y - 16), font, scale, (0, 0, 0), 3)
        cv2.putText(canvas, f"{mean_old:.4f}", (col_x[1], y - 16), font, scale, (255, 0, 0), 3)
        cv2.putText(canvas, f"{mean_new:.4f}", (col_x[2], y - 16), font, scale, (0, 0, 255), 3)

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_dir = os.path.join(TRAINING_DIR, "TestDataset", TEST_DATASET_NAME)
    image_dir = os.path.join(test_dir, "images", "test")
    label_dir = os.path.join(test_dir, "labels", "test")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"테스트 이미지 폴더가 없습니다: {image_dir}")
    if not os.path.isfile(OLD_MODEL_PATH):
        raise FileNotFoundError(f"old 모델이 없습니다: {OLD_MODEL_PATH}")
    if not os.path.isfile(NEW_MODEL_PATH):
        raise FileNotFoundError(f"new 모델이 없습니다: {NEW_MODEL_PATH}")

    old_model = YOLO(OLD_MODEL_PATH)
    new_model = YOLO(NEW_MODEL_PATH)

    grid_images = []
    data_records = []

    file_names = sorted(
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for file_name in file_names:
        image_path = os.path.join(image_dir, file_name)
        txt_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
        if not os.path.exists(txt_path):
            continue

        img = imread_unicode(image_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_polygons = load_gt_polygons(txt_path, w, h)
        mask_gt = mask_from_polygons((h, w), gt_polygons)

        old_results = old_model.predict(img, verbose=False)[0]
        new_results = new_model.predict(img, verbose=False)[0]

        img_old, mask_old = draw_predictions(img, old_results, (255, 0, 0))
        img_new, mask_new = draw_predictions(img, new_results, (0, 0, 255))
        cv2.polylines(img_old, gt_polygons, isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(img_new, gt_polygons, isClosed=True, color=(0, 255, 0), thickness=2)

        iou_old = iou(mask_gt, mask_old)
        iou_new = iou(mask_gt, mask_new)

        old_cell = letterbox(img_old, CELL_W // 2, CELL_H)
        new_cell = letterbox(img_new, CELL_W // 2, CELL_H)
        put_label(old_cell, f"Old {iou_old:.3f}", (255, 0, 0))
        put_label(new_cell, f"New {iou_new:.3f}", (0, 0, 255))
        cell = np.hstack((old_cell, new_cell))
        cv2.putText(cell, short_name(file_name), (8, CELL_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        grid_images.append(cell)
        data_records.append((file_name, iou_old, iou_new))

    if not data_records:
        raise RuntimeError("평가할 이미지가 없습니다.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    table = build_table(data_records, CELL_W * GRID_COLS)
    legend = build_legend(CELL_W * GRID_COLS)
    max_grid_h = MAX_JPEG_DIM - table.shape[0] - legend.shape[0] - 8
    rows_per_page = max(1, max_grid_h // CELL_H)
    cells_per_page = rows_per_page * GRID_COLS

    for page, start in enumerate(range(0, len(grid_images), cells_per_page), start=1):
        page_cells = grid_images[start:start + cells_per_page]
        grid_canvas = build_grid(page_cells, GRID_COLS, CELL_W, CELL_H)
        final_output = np.vstack((legend, grid_canvas, table))
        output_path = os.path.join(
            OUTPUT_DIR,
            f"dashboard_{TEST_DATASET_NAME}_{timestamp}_p{page}.jpg",
        )
        imwrite_unicode(output_path, final_output)


if __name__ == "__main__":
    main()
