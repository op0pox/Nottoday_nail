import os
import cv2
import numpy as np
import math
import datetime
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
old_model_path = os.path.join(PROJECT_ROOT, "Fast Api", "server", "models", "nails_seg_s_yolov8_v1.pt")
new_model_path = os.path.join(PROJECT_ROOT, "Training", "Train_model", "nail_segmentation_20260823_183100", "weights", "best.pt")

old_model = YOLO(old_model_path)
new_model = YOLO(new_model_path)

test_dir = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "Downloads", "TestDataSet", "YOLODataset")
image_dir = os.path.join(test_dir, "images", "test")
label_dir = os.path.join(test_dir, "labels", "test")
output_dir = os.path.join(PROJECT_ROOT, "Training", "results")
os.makedirs(output_dir, exist_ok=True)

grid_images = []
data_records = []

for file_name in os.listdir(image_dir):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, file_name)
    txt_name = os.path.splitext(file_name)[0] + ".txt"
    txt_path = os.path.join(label_dir, txt_name)

    if not os.path.exists(txt_path):
        continue

    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        continue

    h, w = img.shape[:2]

    gt_polygons = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) > 1:
                coords = np.array(parts[1:]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                gt_polygons.append(coords.astype(np.int32))

    mask_gt = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_gt, gt_polygons, 255)

    old_results = old_model.predict(img, verbose=False)[0]
    new_results = new_model.predict(img, verbose=False)[0]

    img_old = img.copy()
    mask_old = np.zeros((h, w), dtype=np.uint8)
    if old_results.masks is not None:
        for seg in old_results.masks.xy:
            pts = np.array(seg, dtype=np.int32)
            cv2.polylines(img_old, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
            cv2.fillPoly(mask_old, [pts], 255)

    img_new = img.copy()
    mask_new = np.zeros((h, w), dtype=np.uint8)
    if new_results.masks is not None:
        for seg in new_results.masks.xy:
            pts = np.array(seg, dtype=np.int32)
            cv2.polylines(img_new, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.fillPoly(mask_new, [pts], 255)

    cv2.polylines(img_old, gt_polygons, isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.polylines(img_new, gt_polygons, isClosed=True, color=(0, 255, 0), thickness=3)

    intersection_old = np.logical_and(mask_gt, mask_old)
    union_old = np.logical_or(mask_gt, mask_old)
    iou_old = np.sum(intersection_old) / np.sum(union_old) if np.sum(union_old) > 0 else 0.0

    intersection_new = np.logical_and(mask_gt, mask_new)
    union_new = np.logical_or(mask_gt, mask_new)
    iou_new = np.sum(intersection_new) / np.sum(union_new) if np.sum(union_new) > 0 else 0.0

    font_scale = max(2, w // 800)
    thickness = max(3, int(font_scale * 2))

    cv2.putText(img_old, f"Old IoU: {iou_old:.4f}", (50, 150 * font_scale), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)
    cv2.putText(img_new, f"New IoU: {iou_new:.4f}", (50, 150 * font_scale), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    combined = np.hstack((img_old, img_new))
    
    target_w = 2400
    target_h = int(combined.shape[0] * (target_w / combined.shape[1]))
    resized_combined = cv2.resize(combined, (target_w, target_h))
    
    cv2.putText(resized_combined, file_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
    cv2.putText(resized_combined, file_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    grid_images.append(resized_combined)
    data_records.append((file_name, iou_old, iou_new))

if grid_images:
    cols = 2 if len(grid_images) > 1 else 1
    rows = math.ceil(len(grid_images) / cols)
    
    cell_w = max(img.shape[1] for img in grid_images)
    cell_h = max(img.shape[0] for img in grid_images)
    
    grid_canvas = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    
    for idx, img in enumerate(grid_images):
        r = idx // cols
        c = idx % cols
        h_i, w_i = img.shape[:2]
        grid_canvas[r*cell_h : r*cell_h+h_i, c*cell_w : c*cell_w+w_i] = img

    table_w = grid_canvas.shape[1]
    row_h = 120
    table_h = (len(data_records) + 2) * row_h
    table_canvas = np.full((table_h, table_w, 3), 255, dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1, table_w // 1500)
    thick = max(2, int(font_scale * 1.5))
    
    col_x = [int(table_w * 0.05), int(table_w * 0.45), int(table_w * 0.75)]
    
    cv2.putText(table_canvas, "File Name", (col_x[0], row_h - 30), font, font_scale, (0, 0, 0), thick)
    cv2.putText(table_canvas, "Model 1 (Old)", (col_x[1], row_h - 30), font, font_scale, (0, 0, 0), thick)
    cv2.putText(table_canvas, "Model 2 (New)", (col_x[2], row_h - 30), font, font_scale, (0, 0, 0), thick)
    cv2.line(table_canvas, (0, row_h), (table_w, row_h), (0, 0, 0), thick * 2)
    
    for i, (fname, i_old, i_new) in enumerate(data_records):
        y = (i + 2) * row_h
        cv2.putText(table_canvas, fname, (col_x[0], y - 30), font, font_scale, (0, 0, 0), thick)
        cv2.putText(table_canvas, f"{i_old:.4f}", (col_x[1], y - 30), font, font_scale, (255, 0, 0), thick)
        cv2.putText(table_canvas, f"{i_new:.4f}", (col_x[2], y - 30), font, font_scale, (0, 0, 255), thick)
        cv2.line(table_canvas, (0, y), (table_w, y), (200, 200, 200), thick)

    final_output = np.vstack((grid_canvas, table_canvas))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"dashboard_summary_{timestamp}.jpg")
    
    result, encoded_img = cv2.imencode('.jpg', final_output)
    if result:
        encoded_img.tofile(output_path)