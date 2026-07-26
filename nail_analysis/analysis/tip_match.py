# -*- coding: utf-8 -*-
"""
analysis/tip_match.py
=======================
계산된 곡면길이(mm)와 유형(P/S/B/C)으로, config의 tip_size_table에서
가장 가까운 사이즈 번호를 찾는다.

tip_size_table의 실제 값(전개도 최대너비 mm)은 아직 확정 전이라 config.yaml에
placeholder로 들어있다. 이 테이블만 바꾸면 매칭 결과가 실제 제품 사이즈에 맞게 바뀐다.
"""


def match_tip_size(curve_length_mm, nail_type, config):
    """
    반환: {
        "size": "6" (문자열 번호),
        "size_label": "S6",
        "table_value_mm": 15.5,
        "diff_mm": -0.4,   # curve_length_mm - table_value_mm (음수면 계산값이 더 작음)
    }
    """
    table = config["tip_size_table"].get(nail_type)
    if not table:
        raise ValueError("tip_size_table에 '%s' 유형이 없습니다." % nail_type)

    best_size = None
    best_value = None
    best_diff_abs = None

    for size, value_mm in table.items():
        diff_abs = abs(curve_length_mm - value_mm)
        if best_diff_abs is None or diff_abs < best_diff_abs:
            best_diff_abs = diff_abs
            best_size = size
            best_value = value_mm

    return {
        "size": best_size,
        "size_label": "%s%s" % (nail_type, best_size),
        "table_value_mm": best_value,
        "diff_mm": curve_length_mm - best_value,
    }
