from ultralytics import YOLO
import os
import datetime

# 폴더 이름이 클래스다. 각 경로는 train/P, train/S, val/P, val/S 를 가진다.
DATASETS = (
    ("front", "thumb"),
    ("front", "other"),
    ("side", "thumb"),
    ("side", "other"),
)

EPOCHS = 300
IMGSZ = 224
BATCH = 16

if __name__ == "__main__":
    training_dir = os.path.dirname(os.path.abspath(__file__))
    prep_dir = os.path.join(training_dir, "Gray_Train_Data", "ClassifyPrep")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for view, finger in DATASETS:
        data_dir = os.path.join(prep_dir, view, finger)
        if not os.path.isdir(os.path.join(data_dir, "train")):
            raise FileNotFoundError(f"분류 학습 폴더가 없습니다: {data_dir}")

        model = YOLO("yolov8s-cls.pt")
        model.train(
            data=data_dir,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            # 흰 배경 위 손톱 윤곽만 학습한다. 회전 외 변형은 끈다.
            degrees=10.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            fliplr=0.0,
            flipud=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            erasing=0.0,
            auto_augment=None,
            crop_fraction=1.0,
            project=os.path.join(training_dir, "Train_model"),
            name=f"nail_classification_{view}_{finger}_{stamp}",
        )
