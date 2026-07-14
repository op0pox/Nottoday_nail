# -*- coding: utf-8 -*-
"""
analysis/curve.py
==================
손톱 유형(P/S/B/C)과 정면 가로너비(W_front), 측면 너비(W_side)로부터
"곡면 길이"(팁이 감싸야 하는 곡선의 실제 길이, mm)를 계산한다.

계산 절차 (PDF 원문 판단):
    flat = W_front * flat_ratio          # 중심 플랫(직선) 구간
    a = (W_front - flat) / 2             # 한쪽 경사 구간의 수평 성분
    b = W_side                           # 수직 성분
    c = sqrt(a^2 + b^2)                  # 현(chord)
    h = c * h_ratio                      # 현의 수직 높이(sagitta)
    r = c^2 / (8h) + h/2                 # 반지름
    theta = 2 * arcsin(c / (2r))         # 중심각 (radian)
    L = r * theta                        # 한쪽 호의 길이
    곡면길이 = 2L + flat

유형별 상수(flat_ratio, h_ratio)는 config.yaml의 curve_constants에서 읽어온다.
PDF 원문 표기(P형=2L+1/2 a, S/C형=2L+1/3 a, B형=2L+2/3 a)와 위 flat_ratio
기반 표기가 같은 값을 낸다는 것은 tests/test_curve.py에서 검증한다.
"""

import math


def compute_curve_length(w_front_mm, w_side_mm, flat_ratio, h_ratio):
    """
    핵심 공식 그 자체. 유형(P/S/B/C) 이름이 아니라 flat_ratio/h_ratio
    상수를 직접 받는다 (유형->상수 매핑은 이 함수를 호출하는 쪽의 책임).

    반환값: dict. 최종 곡면길이는 반환값["curve_length_mm"]에 있고,
    나머지(a, b, c, h, r, theta_rad, theta_deg, l, flat)는 디버깅/검증용
    중간값이다.
    """
    if w_front_mm <= 0 or w_side_mm <= 0:
        raise ValueError(
            "W_front, W_side는 0보다 커야 합니다 (입력: w_front_mm=%s, w_side_mm=%s)"
            % (w_front_mm, w_side_mm)
        )

    flat = w_front_mm * flat_ratio
    a = (w_front_mm - flat) / 2.0
    b = w_side_mm
    c = math.sqrt(a * a + b * b)
    h = c * h_ratio

    if h <= 0:
        raise ValueError("h(sagitta)가 0 이하입니다. h_ratio 설정을 확인하세요.")

    r = (c * c) / (8.0 * h) + h / 2.0

    ratio = c / (2.0 * r)
    # 부동소수점 오차로 ratio가 1을 아주 살짝 넘는 경우를 방지
    ratio = max(-1.0, min(1.0, ratio))
    theta_rad = 2.0 * math.asin(ratio)

    l = r * theta_rad
    curve_length_mm = 2.0 * l + flat

    return {
        "flat": flat,
        "a": a,
        "b": b,
        "c": c,
        "h": h,
        "r": r,
        "theta_rad": theta_rad,
        "theta_deg": math.degrees(theta_rad),
        "l": l,
        "curve_length_mm": curve_length_mm,
    }


def compute_curve_length_for_type(w_front_mm, w_side_mm, nail_type, curve_constants):
    """
    nail_type("P"/"S"/"B"/"C")과 config의 curve_constants dict(
    {"P": {"flat_ratio":.., "h_ratio":..}, ...})를 받아서
    compute_curve_length()를 호출하는 편의 함수.
    """
    if nail_type not in curve_constants:
        raise ValueError(
            "알 수 없는 손톱 유형: %s (curve_constants에 없음, 가능: %s)"
            % (nail_type, ", ".join(sorted(curve_constants.keys())))
        )
    consts = curve_constants[nail_type]
    return compute_curve_length(
        w_front_mm, w_side_mm, consts["flat_ratio"], consts["h_ratio"]
    )
