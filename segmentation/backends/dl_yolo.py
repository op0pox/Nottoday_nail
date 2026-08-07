"""
backends/dl_yolo.py
====================
[1. 역할]
    Hugging Face에 공개된 사전학습 YOLOv8-seg 손톱 세그멘테이션 모델
    (mnemic/nails_seg_yolov8, CC BY 4.0)을 이용해 사진 속 손톱 영역을
    자동으로 검출하는 백엔드다. 직접 학습하지 않고 공개 가중치를
    그대로 사용한다.

    이 파일은 직접 실행하지 않는다. seg_gui.py에서
        from backends.dl_yolo import YoloNailBackend
    형태로 불러와서 사용한다.

[2. 실행 위치]
    노트북(리눅스 랩탑)에서만 쓴다. ultralytics/PyTorch는 무겁고
    Jetson Nano 4GB 모델에서는 설치/추론이 느리거나 불안정하므로
    이번 범위에서는 Nano에 설치하지 않는다. Nano는 촬영만 담당하고,
    사진을 노트북으로 옮긴(scp) 뒤 노트북에서 이 백엔드를 사용한다.

[3. 모델 가중치]
    최초 실행 시 models/nails_seg_s_yolov8_v1.pt (약 23.9MB)가 없으면
    아래 URL에서 자동으로 다운로드한다.
        https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt
    네트워크 문제로 자동 다운로드가 안 되면, 같은 URL을 브라우저로 열어
    받은 뒤 models/ 폴더에 직접 넣으면 된다.

[4. 오류가 발생하면 확인할 것]
    - "ultralytics 패키지가 설치되어 있지 않습니다": 노트북 터미널에서
      pip3 install -r requirements.txt
    - "모델 자동 다운로드에 실패했습니다": 인터넷 연결 확인, 방화벽/프록시
      확인, 혹은 안내된 URL로 수동 다운로드
    - 검출 개수가 5개보다 적거나 많다: --conf 값을 낮추거나(예: 0.1) 높여서
      재시도. 이 모델은 우리 촬영 환경(수직 촬영 + ChArUco 배경)으로
      학습된 게 아니라서 기본 threshold(0.25)가 안 맞을 수 있다.
"""

import os
import urllib.request

import cv2
import numpy as np

from .common import NailMask, SegmentationBackend

MODEL_URL = "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_FILENAME = "nails_seg_s_yolov8_v1.pt"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

DEFAULT_MIN_AREA_RATIO = 0.0003  # 이미지 전체 면적 대비 이 비율보다 작은 마스크는 노이즈로 간주해 버림


def ensure_model_downloaded(model_path=None):
    """
    model_path에 가중치 파일이 없으면 Hugging Face에서 자동 다운로드한다.
    다운로드에 실패하면 수동 설치 방법을 안내하는 RuntimeError를 낸다.
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    if os.path.exists(model_path):
        return model_path

    folder = os.path.dirname(model_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    print("[INFO] 손톱 세그멘테이션 모델 가중치가 없어 다운로드합니다.")
    print(f"       {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as e:
        raise RuntimeError(
            "모델 자동 다운로드에 실패했습니다.\n"
            f"원본 오류: {e}\n"
            f"아래 URL을 브라우저로 열어 직접 다운로드한 뒤 {model_path} 경로에 저장하세요.\n"
            f"{MODEL_URL}"
        )
    print(f"[OK] 모델 다운로드 완료: {model_path}")
    return model_path


class YoloNailBackend(SegmentationBackend):
    name = "yolo"

    def __init__(self, model_path=None, conf=0.25, min_area_ratio=DEFAULT_MIN_AREA_RATIO):
        self.conf = conf
        self.min_area_ratio = min_area_ratio

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics 패키지가 설치되어 있지 않습니다.\n"
                "노트북 터미널에서 다음을 실행하세요: pip3 install -r requirements.txt\n"
                f"(원본 오류: {e})"
            )

        weight_path = ensure_model_downloaded(model_path)
        self.model = YOLO(weight_path)

    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks.data) == 0:
            print(
                f"[WARN] YOLO가 마스크를 하나도 찾지 못했습니다 (conf={self.conf}). "
                "--conf 값을 낮춰서 다시 시도해보세요 (예: --conf 0.1)."
            )
            return []

        mask_data = result.masks.data.cpu().numpy()  # (N, mh, mw), 값 범위 0~1
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None
            else np.ones(mask_data.shape[0])
        )

        candidates = []
        min_area = self.min_area_ratio * w * h
        for i in range(mask_data.shape[0]):
            resized = cv2.resize(mask_data[i], (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (resized > 0.5).astype(np.uint8) * 255
            area = int(np.count_nonzero(binary))
            if area < min_area:
                continue
            candidates.append((binary, float(confs[i]), area))

        print(f"[INFO] YOLO 검출: {len(candidates)}개 (필터 전 {mask_data.shape[0]}개), conf={self.conf}")

        if len(candidates) < 5:
            print(
                f"[WARN] 손톱 5개 중 {len(candidates)}개만 검출됐습니다. "
                "--conf 값을 낮춰서 다시 시도하거나 조명/각도를 조정해보세요."
            )
        elif len(candidates) > 5:
            candidates.sort(key=lambda c: c[1], reverse=True)
            candidates = candidates[:5]
            print("[INFO] 5개 초과 검출되어 confidence 상위 5개만 사용합니다.")

        nail_masks = []
        for binary, conf, _area in candidates:
            x, y, bw, bh = cv2.boundingRect(binary)
            nail_masks.append(NailMask(mask=binary, confidence=conf, bbox=(x, y, bw, bh)))
        return nail_masks
