# -*- coding: utf-8 -*-
"""
analysis/body_length.py
=========================
정면 세로/가로 비율 R = H_front / W_front로 바디 기장(Long/Medium/Short)을 분류한다.

    R >= long_threshold(기본 1.5)                -> Long
    medium_threshold(기본 1.0) <= R < long_threshold -> Medium
    R < medium_threshold                            -> Short

임계값은 config["body_length"]에서 읽는다.
"""


def classify_body_length(h_front_mm, w_front_mm, config):
    """
    반환: {"ratio": R, "label": "Long"/"Medium"/"Short"}
    """
    if w_front_mm <= 0:
        raise ValueError("W_front는 0보다 커야 합니다 (입력: %s)" % w_front_mm)

    ratio = h_front_mm / w_front_mm
    thresholds = config["body_length"]
    long_threshold = thresholds["long_threshold"]
    medium_threshold = thresholds["medium_threshold"]

    if ratio >= long_threshold:
        label = "Long"
    elif ratio >= medium_threshold:
        label = "Medium"
    else:
        label = "Short"

    return {"ratio": ratio, "label": label}
