"""
segmentation/classical.py
===========================
[역할]
    딥러닝(YOLO) 백엔드 로드/추론이 실패했을 때 쓰는 간단한 폴백
    베이스라인이다. 피부색으로 손 영역을 먼저 찾고, 그 안에서 채도가
    낮고 밝은(손톱 특유의 광택/색) 영역을 후보로 추출한다.

    YOLO 백엔드보다 훨씬 부정확하다. 정밀 측정용이 아니라
    "그래도 뭔가 자동 결과가 나오게 하는" 최후 수단으로만 쓴다.
    조명, 손 각도, 매니큐어 유무에 따라 결과가 크게 흔들릴 수 있다.

    이 파일은 직접 실행하지 않는다. measure_auto.py --backend classical
    로 사용하거나, --backend yolo 로드가 실패했을 때 measure_auto.py가
    자동으로 이 백엔드로 폴백한다.
"""

import cv2
import numpy as np

from .common import NailMask, SegmentationBackend

MIN_AREA_RATIO = 0.0003   # 이미지 전체 면적 대비 이보다 작은 후보는 노이즈로 버림
MAX_AREA_RATIO = 0.02     # 이보다 크면 손톱이 아니라 다른 밝은 영역(반사 등)일 가능성이 높음
MIN_ASPECT_RATIO = 1.3    # 손톱은 둥근 손가락 살보다 길쭉하다고 가정


class ClassicalNailBackend(SegmentationBackend):
    name = "classical"

    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        img_area = h * w

        hand_mask = self._skin_mask(image_bgr)
        if hand_mask is None:
            print("[WARN] classical 백엔드가 피부색 영역(손)을 찾지 못했습니다.")
            return []

        nail_candidate = self._nail_candidate_mask(image_bgr, hand_mask)
        contours, _ = cv2.findContours(nail_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA_RATIO * img_area or area > MAX_AREA_RATIO * img_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect < MIN_ASPECT_RATIO:
                continue
            candidates.append((contour, area, (x, y, bw, bh)))

        print(f"[INFO] classical 백엔드 후보: {len(candidates)}개")
        if len(candidates) < 5:
            print(
                f"[WARN] 손톱 5개 중 {len(candidates)}개만 후보로 검출됐습니다. "
                "이 백엔드는 정확도가 낮으니 가능하면 --backend yolo를 사용하세요."
            )
        elif len(candidates) > 5:
            candidates.sort(key=lambda c: c[1], reverse=True)
            candidates = candidates[:5]

        nail_masks = []
        for contour, _area, bbox in candidates:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            nail_masks.append(NailMask(mask=mask, confidence=1.0, bbox=bbox))
        return nail_masks

    @staticmethod
    def _skin_mask(image_bgr):
        """YCrCb 색공간 기반 피부색 임계값으로 손 영역(가장 큰 살색 덩어리)을 찾는다."""
        h, w = image_bgr.shape[:2]
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        hand_contour = max(contours, key=cv2.contourArea)
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(hand_mask, [hand_contour], -1, 255, thickness=cv2.FILLED)
        return hand_mask

    @staticmethod
    def _nail_candidate_mask(image_bgr, hand_mask):
        """
        손 영역 안에서 채도가 낮고(彩度 낮음=손톱 특유의 옅은 색) 밝은 영역을
        손톱 후보로 뽑는다. 임계값은 손 영역 채도 분포의 하위 35 퍼센타일로
        자동 조정해서 조명 변화에 어느 정도 대응한다.
        """
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        hand_sat_values = sat[hand_mask > 0]
        if hand_sat_values.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        sat_threshold = float(np.percentile(hand_sat_values, 35))

        candidate = np.zeros((h, w), dtype=np.uint8)
        candidate[(hand_mask > 0) & (sat < sat_threshold) & (val > 80)] = 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return candidate
