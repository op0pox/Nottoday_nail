"""
curve_config.py
================
[1. 역할]
    measure_curve.py / curve_math.py가 사용하는 손톱 유형(P/S/B/C)별
    곡면 계산 상수를 모아둔 설정 파일이다. 이 값들은 자료마다 표기가
    엇갈리는 부분(특히 B형)이 있는 "실험 중인 가설값"이라서, 코드에
    하드코딩하지 않고 이 파일에서만 관리한다. 재실험 시 이 dict 값만
    바꾸면 된다 (코드 수정 불필요).

[2. 실행 명령어]
    이 파일은 직접 실행하지 않는다. curve_math.py / measure_curve.py에서
        from curve_config import NAIL_TYPE_PARAMS
    형태로 불러와서 사용한다.

[3. 어디에 입력해야 하는가]
    -> 유형별 비율 값을 바꾸고 싶을 때만 이 파일을 직접 수정한다.

[4. 정상적으로 실행되면]
    (해당 없음 - 설정값만 모아둔 파일)

[5. 오류가 발생하면 확인할 것]
    - 새 유형을 추가했는데 인식이 안 된다: NAIL_TYPE_PARAMS의 키(P/S/B/C)와
      measure_curve.py --type 인자의 choices가 같은 문자열인지 확인.
    - 곡면 길이가 이상하게 나온다: 여기 비율 값이 최신 가설값인지 확인
      (특히 B형은 자료 간 표기가 엇갈려 불확실하다고 표시돼 있음).
"""

# 유형별 곡면 계산 파라미터 (실험 중인 가설값)
#   a_ratio : 정면 가로너비 중 "중앙 구간" a의 비율  (a = front_width_mm * a_ratio)
#   h_ratio : 현(c)에 대한 수직 높이 h의 비율        (h = c * h_ratio)
#   k       : 최종 공식의 a 계수                    (curve_length = 2L + k * a)
NAIL_TYPE_PARAMS = {
    "P": {"a_ratio": 1 / 5, "h_ratio": 1 / 4, "k": 1 / 2},  # 평행형
    "S": {"a_ratio": 1 / 7, "h_ratio": 1 / 7, "k": 1 / 3},  # 부채형
    "B": {"a_ratio": 1 / 4, "h_ratio": 1 / 6, "k": 2 / 3},  # 버드형 (자료상 1/7 표기도 있어 불확실 - 실험 후 조정 대상)
    "C": {"a_ratio": 1 / 7, "h_ratio": 1 / 5, "k": 1 / 3},  # 원통형
}

NAIL_TYPES = list(NAIL_TYPE_PARAMS.keys())

# --- Phase 2 (유형 자동 분류) 임계값 -----------------------------------------
# 정면 사다리꼴 좌우 변 기울기로 1차 분류, 측면 마름모꼴 상대 높이로 P/B/C를 구분한다.
# 이 값들도 실험 중인 가설이므로 하드코딩하지 않고 여기서만 관리한다.
CLASSIFICATION_THRESHOLDS = {
    # 정면 판정: 좌우 변 기울기(delta_x/delta_y 등, curve_classify.py 참고)
    "front_slope_parallel_max": 1.0,  # 0~이 값: 거의 평행 -> P/B/C 후보
    "front_slope_fan_max": 3.0,  # front_slope_parallel_max~이 값: 사다리꼴 -> S형 확정
}
