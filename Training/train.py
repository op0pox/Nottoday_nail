from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 일단 온라인 모델 가져오기
    MODEL_URL = "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
    model = YOLO(MODEL_URL)
    
    # 현재파일 기준으로 경로설정
    target_project = os.path.dirname(os.path.abspath(__file__))
    
    results = model.train(
        data="YOLODataset/dataset.yaml", # yolo데이터의 위치
        epochs=50,
        imgsz=640,
        batch=16,
        project=os.path.join(target_project, "Train_model"), 
        name="nail_segmentation"
    )