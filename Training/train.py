from ultralytics import YOLO

MODEL_URL = "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
model = YOLO(MODEL_URL)

results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="nail_segmentation",
    name="fine_tuned_model"
)