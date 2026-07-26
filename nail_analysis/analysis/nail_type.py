# -*- coding: utf-8 -*-
"""
analysis/nail_type.py
=======================
손톱 유형(P=평행 / S=부채 / B=버드 / C=원통)을 정면+측면 키포인트로 판정한다.

좌표계 규약(중요): 이 모듈이 받는 키포인트는 모두 mm 단위이고,
이미지 좌표계와 동일하게 y는 아래로 갈수록 커진다고 가정한다
(카메라 프레임 기준 "위(높음)" = y가 더 작은 쪽). calibration.py가
호모그래피로 변환한 mm 좌표도 이 규약을 따른다.

판정 절차:
    1. 정면 사다리꼴 분석 (6.1) -> P/C 후보 또는 S/B 후보 힌트(front_hint)
    2. 측면 다이아몬드 분석 (6.2, 판단 트리) -> 최종 유형(final_type)
    3. front_hint와 측면 트리 결과 계열이 다르면 측면 트리를 우선하고
       mismatch=True, confidence를 낮춰서 표시한다.

원본 PDF 판단 트리 3단계(P/B 분기)의 "점5/점10 높이" 표현이 문장 두 개에서
서로 모순되게 읽혀서, 이 구현은 "점5(손톱끝)가 점10(살끝)보다 명확히 높으면
(더 위로 튀어나오면) P, 아니면 B"로 해석했다. config의
nail_type.side_pb_invert를 true로 바꾸면 방향을 뒤집을 수 있다
(실측 검증 후 반대로 나오면 이 값을 바꿀 것).
"""

import math


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _is_higher(y_a, y_b, tolerance):
    """a가 b보다 명확히 높은지(작은 y인지) 여부. tolerance 이내 차이는 '같음'으로 본다."""
    return (y_b - y_a) > tolerance


def analyze_front_slope(front_kp_mm, config):
    """
    정면 사다리꼴 분석 (6.1).
    윗변(점4-점6) vs 아랫변(점2-점8) 너비 차이로 기울기 지표 s(0~5)를 계산하고,
    s 값에 따라 P/C 후보인지 S/B 후보인지 힌트를 반환한다.
    """
    nt_cfg = config["nail_type"]

    top_width = _dist(front_kp_mm["4"], front_kp_mm["6"])
    bottom_width = _dist(front_kp_mm["2"], front_kp_mm["8"])

    if bottom_width <= 1e-6:
        diff_ratio = 0.0
    else:
        diff_ratio = (top_width - bottom_width) / bottom_width

    scale = nt_cfg.get("front_slope_scale", 5.0)
    s = max(0.0, min(5.0, diff_ratio * scale))

    if s < nt_cfg["front_slope_pc_max"]:
        hint = "PC"
    elif s >= nt_cfg["front_slope_sb_min"]:
        hint = "SB"
    else:
        # front_slope_pc_max와 front_slope_sb_min 사이의 애매한 구간
        hint = "PC" if s < (nt_cfg["front_slope_pc_max"] + nt_cfg["front_slope_sb_min"]) / 2 else "SB"

    return {
        "top_width_mm": top_width,
        "bottom_width_mm": bottom_width,
        "diff_ratio": diff_ratio,
        "slope_s": s,
        "hint": hint,  # "PC" 또는 "SB"
    }


def analyze_side_tree(side_kp_mm, w_side_mm, config):
    """
    측면 판단 트리 (6.2). 최종 유형을 이 함수 하나로 결정한다 (정면 결과는
    참고/불일치 표시용일 뿐 최종 판정에는 영향을 주지 않는다 - "측면 트리 우선").

    반환: {"final_type": "P"/"S"/"B"/"C", "branch": "SC"/"PB", "notes": [...], "low_confidence": bool}
    """
    nt_cfg = config["nail_type"]
    notes = []
    low_confidence = False

    y3 = side_kp_mm["3"][1]
    y4 = side_kp_mm["4"][1]
    height_diff = y3 - y4  # 양수면 4가 3보다 위(작은 y)

    height_tol = nt_cfg["side_height_tolerance_mm"]

    if abs(height_diff) <= height_tol:
        branch = "SC"
    elif height_diff > 0:
        # 점4가 점3보다 명확히 위 (높음) -> P/B 분기
        branch = "PB"
    else:
        # PDF 트리에 명시되지 않은 경우(점3가 점4보다 위). PB 분기로 취급하되
        # 신뢰도를 낮춘다.
        branch = "PB"
        low_confidence = True
        notes.append("점3이 점4보다 높음 - PDF 트리에 명시되지 않은 케이스라 PB로 임시 분류")

    if branch == "SC":
        # 2단계: 최대너비(W_side)와 깊이(depth_ratio)로 S/C 판정
        depth_ratio = _depth_ratio(side_kp_mm)
        is_deep = depth_ratio <= nt_cfg["side_c_depth_ratio_max"]
        is_wide = w_side_mm >= nt_cfg["side_c_width_threshold_mm"]

        if is_deep and is_wide:
            final_type = "C"
        else:
            final_type = "S"
        notes.append(
            "SC 분기: depth_ratio=%.3f(<=%.2f? %s), W_side=%.2fmm(>=%.2f? %s) -> %s"
            % (
                depth_ratio,
                nt_cfg["side_c_depth_ratio_max"],
                is_deep,
                w_side_mm,
                nt_cfg["side_c_width_threshold_mm"],
                is_wide,
                final_type,
            )
        )
    else:
        # 3단계: 점5(손톱끝) vs 점10(살끝) 높이로 P/B 판정
        y5 = side_kp_mm["5"][1]
        y10 = side_kp_mm["10"][1]
        tip_tol = nt_cfg["side_tip_tolerance_mm"]

        nail_tip_higher = _is_higher(y5, y10, tip_tol)
        if nt_cfg.get("side_pb_invert", False):
            nail_tip_higher = not nail_tip_higher

        final_type = "P" if nail_tip_higher else "B"
        notes.append(
            "PB 분기: 점5 y=%.2f, 점10 y=%.2f, tol=%.2f, nail_tip_higher=%s -> %s"
            % (y5, y10, tip_tol, nail_tip_higher, final_type)
        )

    return {
        "final_type": final_type,
        "branch": branch,
        "notes": notes,
        "low_confidence": low_confidence,
    }


def _depth_ratio(side_kp_mm):
    """
    점3(또는 9)이 큐티클(점1)~손톱끝(점5) 축 상에서 어디쯤 위치하는지를
    0(점1, 큐티클)~1(점5, 손톱끝) 사이 비율로 반환한다. 값이 작을수록
    큐티클 쪽으로 "깊다(안쪽)"고 본다.
    """
    p1 = side_kp_mm["1"]
    p5 = side_kp_mm["5"]
    p3 = side_kp_mm["3"]

    axis = (p5[0] - p1[0], p5[1] - p1[1])
    axis_len_sq = axis[0] ** 2 + axis[1] ** 2
    if axis_len_sq <= 1e-9:
        return 0.5

    vec = (p3[0] - p1[0], p3[1] - p1[1])
    t = (vec[0] * axis[0] + vec[1] * axis[1]) / axis_len_sq
    return max(0.0, min(1.0, t))


def classify_nail_type(front_kp_mm, side_kp_mm, w_side_mm, config):
    """
    정면+측면 키포인트로 최종 손톱 유형을 판정한다.

    front_kp_mm: {"1":(x,y), ..., "9":(x,y)} (mm)
    side_kp_mm:  {"1":(x,y), "2":(x,y), "3":(x,y), "4":(x,y), "5":(x,y), "9":(x,y), "10":(x,y)} (mm)
    w_side_mm: 측면 최대너비 (mm)

    반환: {
        "final_type": "P"/"S"/"B"/"C",
        "front": analyze_front_slope() 결과,
        "side": analyze_side_tree() 결과,
        "mismatch": bool,   # 정면 힌트 계열과 최종유형 계열이 다른지
        "confidence": float (0~1),
    }
    """
    front_result = analyze_front_slope(front_kp_mm, config)
    side_result = analyze_side_tree(side_kp_mm, w_side_mm, config)

    final_type = side_result["final_type"]
    final_family = "PC" if final_type in ("P", "C") else "SB"
    mismatch = front_result["hint"] != final_family

    confidence = 1.0
    if mismatch:
        confidence -= 0.3
    if side_result["low_confidence"]:
        confidence -= 0.2
    confidence = max(0.0, min(1.0, confidence))

    return {
        "final_type": final_type,
        "front": front_result,
        "side": side_result,
        "mismatch": mismatch,
        "confidence": confidence,
    }
