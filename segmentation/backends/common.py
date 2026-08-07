"""
segmentation/common.py
=======================
세그멘테이션 백엔드들이 공통으로 따르는 인터페이스를 정의한다.
이 파일은 직접 실행하지 않는다. 다른 세그멘테이션 백엔드 파일에서
    from .common import NailMask, SegmentationBackend
형태로 불러와서 사용한다.

인터페이스 규칙:
    입력  = BGR 이미지 (np.ndarray, cv2.imread로 읽은 것과 동일한 형식)
    출력  = NailMask 객체 리스트 (손톱 1개당 1개)
"""


class NailMask:
    """손톱 하나에 대한 세그멘테이션 결과."""

    def __init__(self, mask, finger=None, confidence=1.0, bbox=None):
        self.mask = mask            # HxW uint8 numpy 배열, 0 또는 255 (255=손톱 영역)
        self.finger = finger        # "thumb"/"index"/... 아직 모르면 None (label_fingers()가 채움)
        self.confidence = confidence  # 백엔드가 준 신뢰도 (수동/고전 백엔드는 보통 1.0)
        self.bbox = bbox            # (x, y, w, h) 또는 None


class SegmentationBackend:
    """모든 세그멘테이션 백엔드가 상속해야 하는 기본 클래스."""

    name = "base"

    def segment(self, image_bgr):
        """
        image_bgr(BGR 이미지)을 입력받아 NailMask 리스트를 반환한다.
        하위 클래스에서 반드시 구현해야 한다.
        """
        raise NotImplementedError
