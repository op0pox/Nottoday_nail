"""
backends/classical.py
======================
[역할]
    딥러닝(YOLO) 백엔드 로드/추론이 실패했을 때 쓰는 간단한 폴백
    베이스라인이다. 피부색으로 손 영역을 먼저 찾고, 그 안에서 채도가
    낮고 밝은(손톱 특유의 광택/색) 영역을 후보로 추출한다.

    YOLO 백엔드보다 훨씬 부정확하다. 정밀 측정용이 아니라
    "그래도 뭔가 자동 결과가 나오게 하는" 최후 수단으로만 쓴다.
    조명, 손 각도, 매니큐어 유무에 따라 결과가 크게 흔들릴 수 있다.

    이 파일은 직접 실행하지 않는다. seg_gui.py에서 YOLO 백엔드 로드가
    실패했을 때 자동으로 이 백엔드로 폴백한다.

[전체 알고리즘 3단계]
    1. _skin_mask()           : 피부색 픽셀을 찾아 "손 영역" 마스크 생성
    2. _nail_candidate_mask() : 손 영역 안에서 밝고 채도 낮은 픽셀 = 손톱 후보
    3. segment()              : 후보 덩어리들을 크기/모양으로 걸러 최종 선택
"""

import cv2
import numpy as np

from .common import NailMask, SegmentationBackend

# ── 후보 필터링 기준 (이미지 크기에 대한 비율이라 해상도와 무관하게 동작) ──
MIN_AREA_RATIO = 0.0003   # 이미지 전체 면적 대비 이보다 작은 후보는 노이즈로 버림
MAX_AREA_RATIO = 0.02     # 이보다 크면 손톱이 아니라 다른 밝은 영역(반사 등)일 가능성이 높음
MIN_ASPECT_RATIO = 1.3    # 가로세로 비율. 손톱은 둥근 손가락 살보다 길쭉하다고 가정


class ClassicalNailBackend(SegmentationBackend):
    name = "classical"     # 부모 클래스의 name을 덮어씀

    def segment(self, image_bgr):
        """common.py 인터페이스 구현: BGR 이미지 -> NailMask 리스트"""
        h, w = image_bgr.shape[:2]
        img_area = h * w                     # 전체 픽셀 수 (면적 비율 계산용)

        # ── 1단계: 손 영역 찾기 ──
        hand_mask = self._skin_mask(image_bgr)
        if hand_mask is None:
            print("[WARN] classical 백엔드가 피부색 영역(손)을 찾지 못했습니다.")
            return []

        # ── 2단계: 손 영역 안에서 손톱 후보 픽셀 추출 ──
        nail_candidate = self._nail_candidate_mask(image_bgr, hand_mask)

        # findContours = 흰 픽셀 덩어리들의 테두리를 각각 뽑아준다
        # RETR_EXTERNAL = 바깥 테두리만, CHAIN_APPROX_SIMPLE = 좌표 압축 저장
        contours, _ = cv2.findContours(nail_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ── 3단계: 덩어리들을 크기/모양 기준으로 필터링 ──
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)              # 덩어리 면적(픽셀 수)
            if area < MIN_AREA_RATIO * img_area or area > MAX_AREA_RATIO * img_area:
                continue                                 # 너무 작거나 크면 제외
            x, y, bw, bh = cv2.boundingRect(contour)     # 덩어리를 감싸는 사각형
            # 긴 변 / 짧은 변 = 길쭉한 정도. max(1, ...)는 0으로 나누기 방지
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect < MIN_ASPECT_RATIO:
                continue                                 # 동그란 덩어리는 제외
            candidates.append((contour, area, (x, y, bw, bh)))

        print(f"[INFO] classical 백엔드 후보: {len(candidates)}개")
        if len(candidates) < 5:
            print(
                f"[WARN] 손톱 5개 중 {len(candidates)}개만 후보로 검출됐습니다. "
                "이 백엔드는 정확도가 낮으니 가능하면 --backend yolo를 사용하세요."
            )
        elif len(candidates) > 5:
            # 5개보다 많으면 면적이 큰 순서로 상위 5개만 남긴다
            candidates.sort(key=lambda c: c[1], reverse=True)
            candidates = candidates[:5]

        # 선택된 후보들을 NailMask 객체로 변환
        nail_masks = []
        for contour, _area, bbox in candidates:
            # 빈 검은 배열을 만들고, 그 위에 이 덩어리만 흰색(255)으로 채운다
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            nail_masks.append(NailMask(mask=mask, confidence=1.0, bbox=bbox))
        return nail_masks

    @staticmethod
    def _skin_mask(image_bgr):
        """
        YCrCb 색공간 기반 피부색 임계값으로 손 영역(가장 큰 살색 덩어리)을 찾는다.

        왜 YCrCb인가? BGR에서는 조명이 바뀌면 피부색 값이 크게 흔들리지만,
        YCrCb는 밝기(Y)와 색(Cr, Cb)이 분리되어 있어서 "밝기는 무시하고
        색만으로" 피부를 골라내기 좋다. (Cr 133~173, Cb 77~127은
        피부 검출에서 널리 쓰이는 경험적 범위)
        """
        h, w = image_bgr.shape[:2]
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)   # 색공간 변환
        # inRange = 범위 안의 픽셀만 255, 나머지 0인 마스크 생성
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        # 모폴로지 연산으로 마스크 다듬기:
        #   CLOSE(닫힘) = 마스크 안의 작은 구멍 메우기
        #   OPEN(열림)  = 마스크 밖의 작은 점 노이즈 제거
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # 피부색 덩어리가 여러 개면(얼굴, 배경 등) 가장 큰 것 = 손이라고 가정
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
        손 영역 안에서 채도가 낮고(=손톱 특유의 옅은 색) 밝은 영역을
        손톱 후보로 뽑는다.

        HSV 색공간 사용: H(색상), S(채도=색의 진함), V(명도=밝기).
        손톱은 살보다 "덜 붉고(채도 낮음) 더 밝다(명도 높음)"는 성질 이용.

        임계값은 고정하지 않고 손 영역 채도 분포의 하위 35 퍼센타일로
        자동 조정해서 조명 변화에 어느 정도 대응한다.
        """
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]     # 채도 채널만 추출 (H x W 2차원 배열)
        val = hsv[:, :, 2]     # 명도 채널만 추출

        # 손 영역에 해당하는 픽셀들의 채도 값만 모은다
        # (배열[불리언마스크] = 조건에 맞는 값들만 1차원으로 뽑는 numpy 문법)
        hand_sat_values = sat[hand_mask > 0]
        if hand_sat_values.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        # 하위 35% 지점의 채도값 = "손에서 상대적으로 채도가 낮은" 기준선
        sat_threshold = float(np.percentile(hand_sat_values, 35))

        # 세 조건을 모두 만족하는 픽셀만 후보로: 손 영역 & 저채도 & 충분히 밝음
        candidate = np.zeros((h, w), dtype=np.uint8)
        candidate[(hand_mask > 0) & (sat < sat_threshold) & (val > 80)] = 255

        # 작은 노이즈 제거(OPEN) + 끊어진 조각 잇기(CLOSE)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return candidate
