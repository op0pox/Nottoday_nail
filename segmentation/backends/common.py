"""
backends/common.py
===================
세그멘테이션 백엔드들이 공통으로 따르는 인터페이스(규격)를 정의한다.
이 파일은 직접 실행하지 않는다. 다른 백엔드 파일에서
    from .common import NailMask, SegmentationBackend
형태로 불러와서 사용한다.

[왜 이런 파일이 필요한가?]
    손톱을 찾는 방법은 여러 가지다 (YOLO 딥러닝, 피부색 규칙 등).
    방법이 달라도 "입력과 출력의 모양"을 똑같이 맞춰두면,
    사용하는 쪽(seg_gui.py)은 어떤 방법이 쓰였는지 몰라도 된다.
    -> 나중에 더 좋은 모델로 갈아끼워도 GUI 코드는 수정할 필요 없음.
    (객체지향에서 말하는 "다형성"의 실전 예시)

인터페이스 규칙:
    입력  = BGR 이미지 (np.ndarray, cv2.imread로 읽은 것과 동일한 형식)
    출력  = NailMask 객체 리스트 (손톱 1개당 1개)
"""


class NailMask:
    """
    손톱 하나에 대한 세그멘테이션 결과를 담는 작은 상자(데이터 클래스).

    핵심은 mask 속성이다:
      원본 이미지와 같은 크기의 흑백 배열(H x W)이고,
      손톱인 픽셀만 255, 나머지는 전부 0이다.
      "세그멘테이션한다" = 이 배열을 만든다는 뜻.
    """

    def __init__(self, mask, finger=None, confidence=1.0, bbox=None):
        self.mask = mask            # HxW uint8 numpy 배열, 0 또는 255 (255=손톱 영역)
        self.finger = finger        # "thumb"/"index"/... 아직 모르면 None
        self.confidence = confidence  # 백엔드가 준 신뢰도 0~1 (규칙 기반 백엔드는 1.0 고정)
        self.bbox = bbox            # 손톱을 감싸는 사각형 (x, y, w, h) 또는 None


class SegmentationBackend:
    """
    모든 세그멘테이션 백엔드가 상속해야 하는 기본 클래스 (추상 클래스 역할).

    이 클래스를 상속한 뒤 segment()를 구현하면 새 백엔드가 완성된다.
    현재 구현체: dl_yolo.py(YoloNailBackend), classical.py(ClassicalNailBackend)
    """

    name = "base"    # 백엔드 이름 (로그/디버그 표시용, 하위 클래스에서 덮어씀)

    def segment(self, image_bgr):
        """
        image_bgr(BGR 이미지)을 입력받아 NailMask 리스트를 반환한다.
        하위 클래스에서 반드시 구현해야 한다.

        NotImplementedError: "이 함수는 자식 클래스가 구현해야 해요"라는
        표시. 실수로 부모 클래스를 그대로 쓰면 즉시 오류가 나게 한다.
        """
        raise NotImplementedError
