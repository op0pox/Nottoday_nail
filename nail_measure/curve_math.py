"""
curve_math.py
=============
[1. 역할]
    손톱을 원의 일부(호)로 근사해서, 정면 가로너비와 측면 가로너비 두
    측정값만으로 손톱 곡면을 펼쳤을 때의 너비(곡면 길이)를 계산하는
    순수 계산 함수 모음이다. OpenCV/마우스 클릭 등 UI 코드와 완전히
    분리되어 있어서 숫자만 넣으면 되고, 이 파일만 단독으로 실행해서
    자체 테스트를 돌려볼 수도 있다.

    계산 순서 (측정값 2개 -> 최종 곡면 길이):
        a = 정면 가로너비 * 유형별 a_ratio        (중앙 구간 길이)
        b = 측면 가로너비
        c = sqrt(a^2 + b^2)                        (빗변/현)
        h = c * 유형별 h_ratio                     (현에 세우는 수직 높이)
        r = c^2 / (8h) + h/2                        (근사 원의 반지름)
        theta = 2 * arcsin(c / (2r))                 (중심각, 라디안)
        L = r * theta                                (호 길이)
        curve_length = 2L + 유형별 k * a             (최종 곡면 길이)

    유형별 a_ratio/h_ratio/k 값은 curve_config.py의 NAIL_TYPE_PARAMS에
    분리되어 있다 (실험 중인 가설값이라 코드에 직접 박아두지 않았다).

[2. 실행 명령어]
    python3 curve_math.py     # 자체 테스트만 실행하고 결과를 출력한다

    다른 스크립트(measure_curve.py 등)에서는 아래처럼 불러와서 쓴다.
        from curve_math import compute_curve_length

[3. 어디에 입력해야 하는가]
    -> Jetson Nano든 노트북이든 터미널에서 바로 실행 가능하다
       (마우스 클릭/카메라 등이 전혀 필요 없는 순수 계산 코드).

[4. 정상적으로 실행되면]
    python3 curve_math.py 실행 시 아래처럼 출력된다.

        [OK] curve_math.py 자체 테스트 전부 통과

[5. 오류가 발생하면 확인할 것]
    - "알 수 없는 손톱 유형입니다": nail_type이 curve_config.py의
      NAIL_TYPE_PARAMS에 있는 키(P/S/B/C)와 정확히 같은 문자열인지 확인.
    - "수직 높이(h)가 0 이하입니다" / "반지름(r)이 0 이하입니다":
      정면/측면 너비 측정값이 0이거나 음수로 잘못 들어간 경우가 많다.
      measure_curve.py에서 클릭한 두 점이 실제로 손톱 너비 양 끝인지 확인.
    - 자체 테스트가 실패한다: curve_config.py의 NAIL_TYPE_PARAMS 값을
      최근에 바꿨다면, 아래 self-test의 회귀 검증 수식과 값이 여전히
      맞는지 확인 (값을 바꾼 것 자체는 테스트 실패 원인이 아니어야 한다 -
      테스트는 "공식이 올바르게 조합됐는지"를 보는 것이지 특정 비율
      값을 고정해서 검증하는 게 아니다).
"""

import math

from curve_config import NAIL_TYPE_PARAMS


def _get_type_params(nail_type, params=None):
    """nail_type에 해당하는 {a_ratio, h_ratio, k} dict를 반환한다."""
    table = params if params is not None else NAIL_TYPE_PARAMS
    if nail_type not in table:
        raise ValueError(f"알 수 없는 손톱 유형입니다: {nail_type} (가능한 값: {', '.join(table.keys())})")
    return table[nail_type]


def compute_a(front_width_mm, nail_type, params=None):
    """정면 가로너비에서 유형별 비율만큼의 "중앙 구간" 길이 a를 계산한다."""
    type_params = _get_type_params(nail_type, params)
    return front_width_mm * type_params["a_ratio"]


def compute_chord(a_mm, b_mm):
    """a, b를 직각변으로 하는 빗변(현) c = sqrt(a^2 + b^2)를 계산한다."""
    return math.sqrt(a_mm ** 2 + b_mm ** 2)


def compute_height(chord_mm, nail_type, params=None):
    """현(c)에 유형별 비율을 곱해 수직 높이 h를 계산한다."""
    type_params = _get_type_params(nail_type, params)
    return chord_mm * type_params["h_ratio"]


def compute_radius(chord_mm, height_mm):
    """현(c)과 높이(h)로부터 근사 원의 반지름 r = c^2/(8h) + h/2 를 계산한다."""
    if height_mm <= 0:
        raise ValueError(f"수직 높이(h)가 0 이하입니다 (h={height_mm}). 정면/측면 너비 측정값을 확인하세요.")
    return (chord_mm ** 2) / (8 * height_mm) + height_mm / 2


def compute_theta(chord_mm, radius_mm):
    """현(c)과 반지름(r)으로부터 중심각 theta = 2*arcsin(c/(2r))를 라디안으로 계산한다."""
    if radius_mm <= 0:
        raise ValueError(f"반지름(r)이 0 이하입니다 (r={radius_mm}).")
    ratio = chord_mm / (2 * radius_mm)
    # 부동소수점 오차로 ratio가 [-1, 1]을 살짝 벗어나 asin 정의역 오류가 나는 것을 방지
    ratio = max(-1.0, min(1.0, ratio))
    return 2 * math.asin(ratio)


def compute_arc_length(radius_mm, theta_rad):
    """반지름(r)과 중심각(theta)으로부터 호 길이 L = r*theta를 계산한다."""
    return radius_mm * theta_rad


def compute_curve_length(front_width_mm, side_width_mm, nail_type, params=None):
    """
    정면 가로너비(front_width_mm)와 측면 가로너비(side_width_mm), 손톱 유형만으로
    최종 곡면 길이까지 전 과정을 계산해서 모든 중간값을 dict로 반환한다.
    """
    type_params = _get_type_params(nail_type, params)

    a = front_width_mm * type_params["a_ratio"]
    b = side_width_mm
    c = compute_chord(a, b)
    h = compute_height(c, nail_type, params)
    r = compute_radius(c, h)
    theta = compute_theta(c, r)
    L = compute_arc_length(r, theta)
    curve_length = 2 * L + type_params["k"] * a

    return {
        "nail_type": nail_type,
        "front_width_mm": front_width_mm,
        "side_width_mm": side_width_mm,
        "a_mm": a,
        "b_mm": b,
        "c_mm": c,
        "h_mm": h,
        "r_mm": r,
        "theta_rad": theta,
        "L_mm": L,
        "curve_length_mm": curve_length,
    }


def find_closest_tip_size(curve_length_mm, nail_type, tips_rows):
    """
    tips_rows(예: [{"type": "P", "size_no": "P6", "max_width_mm": 12.0}, ...])
    중에서 nail_type이 같은 행들만 골라, curve_length_mm과 max_width_mm 차이가
    가장 작은 행을 찾아 반환한다. 일치하는 유형이 없으면 None.

    반환값: {"size_no": ..., "max_width_mm": ..., "diff_mm": ...} 또는 None
        diff_mm = max_width_mm - curve_length_mm (양수면 팁이 더 크다는 뜻)
    """
    candidates = [row for row in tips_rows if row.get("type") == nail_type]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: abs(float(row["max_width_mm"]) - curve_length_mm))
    return {
        "size_no": best["size_no"],
        "max_width_mm": float(best["max_width_mm"]),
        "diff_mm": float(best["max_width_mm"]) - curve_length_mm,
    }


# ---------------------------------------------------------------------------
# 자체 테스트 (pytest 등 별도 의존성 없이 python3 curve_math.py로 바로 실행)
# ---------------------------------------------------------------------------
def _self_test():
    # 1) 피타고라스 확인: a=3, b=4 -> c=5
    c = compute_chord(3, 4)
    assert abs(c - 5.0) < 1e-9, f"[FAIL] compute_chord(3, 4) = {c} (기대값 5.0)"

    # 2) theta/L 수치 검증: 반지름 r, 중심각 theta로 만든 현(c)을 거꾸로 넣었을 때
    #    theta가 원래 값으로 복원되는지 확인 (호/현/반지름 관계식 검증)
    r_known = 10.0
    theta_known = 1.0
    c_known = 2 * r_known * math.sin(theta_known / 2)
    theta_calc = compute_theta(c_known, r_known)
    assert abs(theta_calc - theta_known) < 1e-9, f"[FAIL] theta 복원 실패: {theta_calc} != {theta_known}"
    L_calc = compute_arc_length(r_known, theta_calc)
    assert abs(L_calc - r_known * theta_known) < 1e-9, f"[FAIL] arc_length 계산 실패: {L_calc}"

    # 3) compute_curve_length가 공식을 그대로 조합한 값과 일치하는지(회귀 검증), 유형 전부
    for nail_type, type_params in NAIL_TYPE_PARAMS.items():
        result = compute_curve_length(front_width_mm=15.0, side_width_mm=8.0, nail_type=nail_type)

        a = 15.0 * type_params["a_ratio"]
        b = 8.0
        c = math.sqrt(a ** 2 + b ** 2)
        h = c * type_params["h_ratio"]
        r = c ** 2 / (8 * h) + h / 2
        theta = 2 * math.asin(min(1.0, c / (2 * r)))
        L = r * theta
        expected = 2 * L + type_params["k"] * a

        assert abs(result["curve_length_mm"] - expected) < 1e-6, (
            f"[FAIL] {nail_type} 유형 곡면길이 불일치: {result['curve_length_mm']} != {expected}"
        )
        assert result["r_mm"] > 0 and result["theta_rad"] > 0, f"[FAIL] {nail_type} 유형 r/theta 값 이상"

    # 4) 알 수 없는 유형은 에러가 나야 한다
    try:
        compute_curve_length(front_width_mm=15.0, side_width_mm=8.0, nail_type="X")
        raise AssertionError("[FAIL] 알 수 없는 유형('X')에서 오류가 나지 않음")
    except ValueError:
        pass

    # 5) 팁 사이즈 매칭
    tips_rows = [
        {"type": "P", "size_no": "P4", "max_width_mm": 10.0},
        {"type": "P", "size_no": "P6", "max_width_mm": 12.0},
        {"type": "P", "size_no": "P8", "max_width_mm": 14.0},
        {"type": "S", "size_no": "S6", "max_width_mm": 11.0},
    ]
    match = find_closest_tip_size(11.3, "P", tips_rows)
    assert match is not None and match["size_no"] == "P6", f"[FAIL] 팁 매칭 실패: {match}"
    assert find_closest_tip_size(11.3, "C", tips_rows) is None, "[FAIL] 없는 유형인데 매칭됨"

    print("[OK] curve_math.py 자체 테스트 전부 통과")


if __name__ == "__main__":
    _self_test()
