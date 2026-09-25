# -*- coding: utf-8 -*-
"""
세그 폴리곤으로 원본을 잘라 분류 모델 입력을 만든다.

    원본 + 폴리곤 -> 바깥은 검은 배경 -> 손톱만 크롭 -> 그레이스케일

정렬은 하지 않는다. utils/01 의 검은 배경 합성·크롭과 utils/02 의 그레이만 가져왔다.
"""
import cv2
import numpy as np


def prepare_nail_gray(image_bgr, polygon):
    """BGR 원본과 세그 폴리곤으로 그레이스케일 손톱 이미지를 반환한다. 실패하면 None."""
    if image_bgr is None or polygon is None:
        return None
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None

    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    fill = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [fill], 255)
    if not np.any(mask):
        return None

    black = np.zeros_like(image_bgr)
    cut = np.where(mask[:, :, None] > 0, image_bgr, black)
    x, y, box_w, box_h = cv2.boundingRect(mask)
    if box_w < 1 or box_h < 1:
        return None
    crop = cut[y:y + box_h, x:x + box_w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
