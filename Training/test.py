import os
import datetime
import cv2
import numpy as np
from ultralytics import YOLO

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
old_model_path = os.path.join(_PROJECT_ROOT, "Fast Api", "server", "models", "nails_seg_s_yolov8_v1.pt")
new_model_path = os.path.join(_PROJECT_ROOT, "Training", "Train_model", "nail_segmentation_20260823_183100", "weights", "best.pt")

old_model = YOLO(old_model_path)
new_model = YOLO(new_model_path)

image_path = os.path.join(_PROJECT_ROOT, "Training", "YOLODataset", "images", "val", "015_L_4F_ce7f3295.jpg")
img = cv2.imread(image_path)

old_results = old_model.predict(img, verbose=False)[0]
new_results = new_model.predict(img, verbose=False)[0]

img_old = img.copy()
if old_results.masks is not None:
    for seg in old_results.masks.xy:
        pts = np.array(seg, dtype=np.int32)
        cv2.polylines(img_old, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

img_new = img.copy()
if new_results.masks is not None:
    for seg in new_results.masks.xy:
        pts = np.array(seg, dtype=np.int32)
        cv2.polylines(img_new, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

combined = np.hstack((img_old, img_new))

output_dir = os.path.join(_PROJECT_ROOT, "Training", "results")
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"model_comparison_result_{timestamp}.jpg")

cv2.imwrite(output_path, combined)