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

[2. YOLOv8-seg란?]
    YOLO = 실시간 객체 검출로 유명한 딥러닝 모델 계열.
    뒤에 붙은 -seg = 물체를 감싸는 사각형(box)만이 아니라
    물체의 정확한 픽셀 영역(마스크)까지 예측하는 버전이다.
    ultralytics 패키지가 모델 로드/추론을 전부 처리해준다.

[3. 모델 가중치]
    최초 실행 시 models/nails_seg_s_yolov8_v1.pt (약 23.9MB)가 없으면
    아래 URL에서 자동으로 다운로드한다.
        https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt
    네트워크 문제로 자동 다운로드가 안 되면, 같은 URL을 브라우저로 열어
    받은 뒤 models/ 폴더에 직접 넣으면 된다.
    (.pt = PyTorch 가중치 파일. 학습이 끝난 신경망의 숫자들이 담겨 있다)

[4. 오류가 발생하면 확인할 것]
    - "ultralytics 패키지가 설치되어 있지 않습니다": 터미널에서
      pip3 install -r requirements.txt
    - "모델 자동 다운로드에 실패했습니다": 인터넷 연결 확인, 방화벽/프록시
      확인, 혹은 안내된 URL로 수동 다운로드
    - 검출 개수가 기대보다 적거나 많다: conf 값을 낮추거나(예: 0.1) 높여서
      재시도. 이 모델은 우리 촬영 환경(수직 촬영 + ChArUco 배경)으로
      학습된 게 아니라서 기본 threshold(0.25)가 안 맞을 수 있다.
"""

import os
import urllib.request     # 표준 라이브러리로 파일 다운로드

import cv2
import numpy as np

from .common import NailMask, SegmentationBackend

# ── 모델 파일 경로 설정 ──────────────────────────────────────────────────────
MODEL_URL = "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
# __file__ = 이 파일의 경로. dirname을 두 번 해서 backends/의 부모(segmentation/)를 얻는다
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
    model_path = model_path or DEFAULT_MODEL_PATH   # None이면 기본 경로 사용
    if os.path.exists(model_path):                  # 이미 있으면 그대로 사용
        return model_path

    # 저장할 폴더가 없으면 생성
    folder = os.path.dirname(model_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    print("[INFO] 손톱 세그멘테이션 모델 가중치가 없어 다운로드합니다.")
    print(f"       {MODEL_URL}")
    try:
        # urlretrieve(주소, 저장경로) = 파일 다운로드
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
        """
        conf: 검출 신뢰도 임계값. 모델이 "이건 손톱이야"라고 확신하는
              정도가 이 값 이상인 것만 결과로 인정한다.
              낮추면 더 많이 검출되지만 오검출도 늘어난다.
        """
        self.conf = conf
        self.min_area_ratio = min_area_ratio

        # ultralytics는 무거워서(PyTorch 포함) 실제 사용할 때만 import.
        # 설치가 안 돼 있으면 안내 메시지와 함께 RuntimeError를 낸다.
        # (seg_gui.py는 이 오류를 잡아서 classical 백엔드로 폴백한다)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics 패키지가 설치되어 있지 않습니다.\n"
                "터미널에서 다음을 실행하세요: pip3 install -r requirements.txt\n"
                f"(원본 오류: {e})"
            )

        # 가중치 준비(없으면 다운로드) 후 모델 로드
        weight_path = ensure_model_downloaded(model_path)
        self.model = YOLO(weight_path)

    def segment(self, image_bgr):
        """common.py 인터페이스 구현: BGR 이미지 -> NailMask 리스트"""
        h, w = image_bgr.shape[:2]

        # ── 추론 실행 ──
        # predict()가 검출 전 과정을 처리하고 결과 객체 리스트를 반환한다.
        # 이미지 1장을 넣었으니 results[0]만 쓰면 된다.
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        result = results[0]

        # 마스크가 하나도 없으면 빈 리스트 반환
        if result.masks is None or len(result.masks.data) == 0:
            print(
                f"[WARN] YOLO가 마스크를 하나도 찾지 못했습니다 (conf={self.conf}). "
                "--conf 값을 낮춰서 다시 시도해보세요 (예: --conf 0.1)."
            )
            return []

        # ── 결과 꺼내기 ──
        # result.masks.data는 PyTorch 텐서(GPU에 있을 수 있음)라서
        # .cpu().numpy()로 일반 numpy 배열로 변환해야 한다.
        # 모양: (검출개수 N, 마스크높이, 마스크폭), 값은 0~1 확률
        mask_data = result.masks.data.cpu().numpy()
        # 검출별 신뢰도 점수 (boxes가 없으면 전부 1.0으로 처리)
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None
            else np.ones(mask_data.shape[0])
        )

        # ── 마스크 후처리 ──
        candidates = []
        min_area = self.min_area_ratio * w * h
        for i in range(mask_data.shape[0]):
            # YOLO 마스크는 원본보다 작은 해상도로 나오므로 원본 크기로 확대.
            # INTER_NEAREST = 마스크(0/1 값)에 적합한 보간 (중간값 안 만듦)
            resized = cv2.resize(mask_data[i], (w, h), interpolation=cv2.INTER_NEAREST)
            # 확률(0~1)을 0.5 기준으로 잘라 0/255 이진 마스크로 변환
            binary = (resized > 0.5).astype(np.uint8) * 255
            area = int(np.count_nonzero(binary))    # 흰 픽셀 개수 = 면적
            if area < min_area:                     # 너무 작으면 노이즈로 버림
                continue
            candidates.append((binary, float(confs[i]), area))

        print(f"[INFO] YOLO 검출: {len(candidates)}개 (필터 전 {mask_data.shape[0]}개), conf={self.conf}")

        if len(candidates) < 5:
            print(
                f"[WARN] 손톱 5개 중 {len(candidates)}개만 검출됐습니다. "
                "--conf 값을 낮춰서 다시 시도하거나 조명/각도를 조정해보세요."
            )
        elif len(candidates) > 5:
            # 5개 초과면 신뢰도 높은 순으로 5개만 (한 손 = 손톱 최대 5개 가정)
            candidates.sort(key=lambda c: c[1], reverse=True)
            candidates = candidates[:5]
            print("[INFO] 5개 초과 검출되어 confidence 상위 5개만 사용합니다.")

        # NailMask 객체로 포장해서 반환
        nail_masks = []
        for binary, conf, _area in candidates:
            x, y, bw, bh = cv2.boundingRect(binary)    # 마스크를 감싸는 사각형
            nail_masks.append(NailMask(mask=binary, confidence=conf, bbox=(x, y, bw, bh)))
        return nail_masks
