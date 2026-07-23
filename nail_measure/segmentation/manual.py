"""
segmentation/manual.py
========================
[역할]
    기존 manual_measure.py의 "손톱 뿌리/끝 2점 클릭" 방식을
    SegmentationBackend와 같은 공통 인터페이스로 감싼 백엔드다.
    클릭한 두 점을 잇는 얇은 선을 마스크로 만들어서,
    measure_from_mask.py의 PCA 기반 길이 계산 로직을 다른 백엔드와
    똑같이 재사용할 수 있게 한다. 이렇게 하면 manual/yolo/classical
    백엔드 결과가 전부 같은 CSV 포맷으로 저장되어 곧바로 비교할 수 있다.

    이 파일은 직접 실행하지 않는다. measure_auto.py --backend manual 로
    사용한다. 마우스 클릭이 필요하므로 화면(GUI, SSH -X 또는 모니터
    직결)이 있는 환경에서 실행해야 한다.
"""

import cv2
import numpy as np

from .common import NailMask, SegmentationBackend
from utils import collect_points, FINGERS, FINGER_LABELS_KO

LINE_THICKNESS = 5  # 클릭한 두 점을 잇는 합성 마스크 선 두께(px)


class ManualNailBackend(SegmentationBackend):
    name = "manual"

    def __init__(self, line_thickness=LINE_THICKNESS):
        self.line_thickness = line_thickness

    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        nail_masks = []

        print("[INFO] 각 손가락마다 '손톱 뿌리' -> '손톱 끝' 순서로 클릭하세요.")
        for finger_key in FINGERS:
            label_ko = FINGER_LABELS_KO[finger_key]
            labels = [
                f"[{label_ko}] 손톱 뿌리를 클릭하세요",
                f"[{label_ko}] 손톱 끝을 클릭하세요",
            ]
            points = collect_points(
                image_bgr, num_points=2, point_labels=labels, window_name=f"Measure - {label_ko}"
            )
            if points is None:
                print(f"[INFO] '{label_ko}' 측정을 건너뛰었습니다(취소).")
                continue

            root, tip = points
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(mask, root, tip, 255, self.line_thickness)
            x, y, bw, bh = cv2.boundingRect(mask)
            nail_masks.append(
                NailMask(mask=mask, finger=finger_key, confidence=1.0, bbox=(x, y, bw, bh))
            )

        cv2.destroyAllWindows()
        return nail_masks
