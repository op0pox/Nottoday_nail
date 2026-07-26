# -*- coding: utf-8 -*-
"""
detection/dl_segmenter.py
===========================
[역할] 향후 경량 세그멘테이션 모델(ONNX/TensorRT, 예: YOLOv8n-seg 또는
U-Net 계열)로 손톱 마스크를 얻을 자리를 미리 잡아두는 스텁이다.

지금은 실제로 동작하지 않는다 (모델이 아직 없음 - training/ 로드맵 참고,
README의 "SAM 라벨링 -> 학습 -> ONNX -> TensorRT 배포" 절 참고).
NailSegmenter 인터페이스(detection/base.py)를 그대로 구현해뒀기 때문에,
모델이 준비되면 main.py에서 이 클래스로 교체하기만 하면 된다
(예: --segmenter dl 옵션 추가 등).

Jetson Nano에서는 TensorRT 엔진(.engine/.trt)으로 변환한 모델을 쓰는 것이
일반적이고, 노트북/개발 PC에서는 ONNX Runtime으로 우선 검증하는 흐름을
가정한다. 어느 쪽이든 이 클래스의 __init__에서 모델 경로를 받아 로드하고,
segment()에서 추론 결과를 이진 마스크(HxW uint8, 0/255)로 변환해서
반환하면 detection/base.py의 KeypointDetector가 그대로 재사용한다.
"""

from detection.base import NailSegmenter


class OnnxNailSegmenter(NailSegmenter):
    """
    TODO(추후 구현): ONNX Runtime 기반 세그멘테이션 모델 로더.

    구현 골격:
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=[...])
        입력 전처리(리사이즈/정규화) -> self.session.run(...) -> 출력 후처리
        (임계값 적용, 가장 큰 연결요소 선택 등)로 바이너리 마스크 생성

    지금은 model_path가 주어져도 실제 로드를 시도하지 않고, segment() 호출 시
    NotImplementedError를 낸다 - 모델이 준비되기 전까지 실수로 이 백엔드가
    선택되면 조용히 틀린 결과를 내는 대신 명확히 실패하게 하기 위함이다.
    """

    name = "dl_onnx"

    def __init__(self, model_path, config, providers=None):
        self.model_path = model_path
        self.config = config
        self.providers = providers
        self.session = None  # 실제 구현 시 ort.InferenceSession(...)

    def segment(self, image_bgr, roi=None):
        raise NotImplementedError(
            "OnnxNailSegmenter는 아직 구현되지 않았습니다.\n"
            "README.md의 로드맵(SAM 라벨링 -> YOLOv8n-seg/U-Net 학습 -> ONNX 변환 -> "
            "TensorRT 배포)을 따라 모델을 준비한 뒤, 이 클래스의 __init__/segment()를 채워 넣으세요.\n"
            "그동안은 detection/contour.py의 ThresholdNailSegmenter(고전 CV)를 사용하세요."
        )


class TensorRTNailSegmenter(NailSegmenter):
    """
    TODO(추후 구현): Jetson Nano 배포용 TensorRT 엔진 기반 세그멘테이션.

    구현 골격:
        import tensorrt as trt
        import pycuda.driver as cuda
        엔진(.engine) 로드 -> 실행 컨텍스트 생성 -> GPU 메모리 바인딩 ->
        추론 -> 마스크 후처리

    ONNX -> TensorRT 변환은 보통 `trtexec --onnx=model.onnx --saveEngine=model.engine`
    으로 하고, 그렇게 만든 .engine 파일 경로를 engine_path로 받는다.
    """

    name = "dl_tensorrt"

    def __init__(self, engine_path, config):
        self.engine_path = engine_path
        self.config = config
        self.engine = None  # 실제 구현 시 TensorRT 런타임으로 로드

    def segment(self, image_bgr, roi=None):
        raise NotImplementedError(
            "TensorRTNailSegmenter는 아직 구현되지 않았습니다. "
            "README의 배포 로드맵을 참고해서 TensorRT 엔진을 준비한 뒤 구현하세요."
        )
