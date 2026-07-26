# -*- coding: utf-8 -*-
"""
detection/base.py
===================
손톱 마스크 획득(NailSegmenter)과 마스크->키포인트 산출을 묶는 상위
파이프라인(KeypointDetector)의 인터페이스를 정의한다.

구조:
    NailSegmenter   : "이미지 -> 마스크" 만 담당. 교체 가능한 지점.
                      지금은 detection/contour.py의 ThresholdNailSegmenter(고전 CV)를
                      쓰고, 나중에 detection/dl_segmenter.py의 AI 세그멘터로 바꿀 수 있다.
    KeypointDetector : NailSegmenter로 마스크를 얻은 뒤, 마스크->키포인트 변환은
                      "마스크가 어디서 왔는지"와 무관하게 항상 같은 공용 로직
                      (detection/contour.py의 extract_front_keypoints/
                      extract_side_keypoints)을 사용한다.

Python 3.6 호환을 위해 dataclass 대신 plain class + dict를 사용한다.
"""


class NailSegmenter(object):
    """
    손톱(또는 측면 뷰의 손가락 실루엣) 마스크를 얻는 인터페이스.
    하위 클래스는 segment()만 구현하면 된다.
    """

    name = "base"

    def segment(self, image_bgr, roi=None):
        """
        image_bgr: BGR 이미지 (numpy array)
        roi: (x, y, w, h) 또는 None (None이면 이미지 전체를 대상으로 함)

        반환: mask (HxW uint8, 0 또는 255). 실패 시 None을 반환할 수 있다
        (호출부에서 "검출 실패" 오류/수동 보정 흐름으로 이어짐).
        """
        raise NotImplementedError


class KeypointDetectionResult(object):
    """
    detect()의 반환값을 담는 간단한 값 객체. 아래 필드들을 갖는다.
        keypoints_px : {"1": (x,y), ...} 픽셀 좌표
        mask         : 사용된 바이너리 마스크
        view         : "front" 또는 "side"
        auto_detected: True면 전부 자동 검출, False가 섞여있으면 일부/전부 수동 보정됨
    """

    def __init__(self, keypoints_px, mask, view, auto_detected=True):
        self.keypoints_px = keypoints_px
        self.mask = mask
        self.view = view
        self.auto_detected = auto_detected

    def to_dict(self):
        return {
            "keypoints_px": {k: list(v) for k, v in self.keypoints_px.items()},
            "view": self.view,
            "auto_detected": self.auto_detected,
        }


class KeypointDetector(object):
    """
    NailSegmenter(마스크 획득) + 공용 마스크->키포인트 로직을 묶는 파이프라인.
    main.py/collect.py 등에서는 이 클래스만 사용하면 되고, 세그멘터 종류
    (고전 CV vs 추후 AI 모델)는 생성자에 주입하는 segmenter로만 결정된다.
    """

    def __init__(self, segmenter, config):
        self.segmenter = segmenter
        self.config = config

    def detect(self, image_bgr, view, roi=None):
        """
        view: "front" 또는 "side"
        반환: KeypointDetectionResult (검출 실패 시 mask=None, keypoints_px={})
        """
        # 여기서 import하는 이유: contour.py가 detection 패키지의 공용 마스크->키포인트
        # 로직을 갖고 있고, base.py는 세그멘터 구현에 의존하지 않아야 하기 때문에
        # (순환 임포트 방지) 함수 내부에서 지연 임포트한다.
        from detection.contour import extract_front_keypoints, extract_side_keypoints

        mask = self.segmenter.segment(image_bgr, roi=roi)
        if mask is None:
            return KeypointDetectionResult(keypoints_px={}, mask=None, view=view, auto_detected=False)

        if view == "front":
            keypoints_px = extract_front_keypoints(mask, self.config)
        elif view == "side":
            keypoints_px = extract_side_keypoints(mask, self.config)
        else:
            raise ValueError("view는 'front' 또는 'side'여야 합니다: %s" % view)

        return KeypointDetectionResult(keypoints_px=keypoints_px, mask=mask, view=view, auto_detected=True)
