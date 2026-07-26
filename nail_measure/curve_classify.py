"""
curve_classify.py
==================
[1. 역할]
    Phase 2(선택 기능): 손톱 유형(P/S/B/C)을 사람이 완전히 눈으로 판단해서
    `measure_curve.py --type`에 직접 넣는 대신, **키포인트는 사람이 클릭하고
    분류 판정만 자동으로** 내려주는 모듈이다. 완전 자동 디텍션(YOLO 등)은
    아직 다루지 않는다.

    분류는 2단계다.
      1) 정면 판정: 정면 사진에서 손톱 좌/우 변의 위/아래 4점(사다리꼴
         꼭짓점: top_left, bottom_left, top_right, bottom_right)을 클릭받아
         좌우 변의 평균 기울기(가로 이동량/세로 이동량)를 계산한다.
           - 기울기가 0~1 구간(거의 평행) -> P/B/C 후보, 2)로 넘어감
           - 기울기가 1~3 구간(사다리꼴) -> S형으로 확정
      2) 측면 판정 (1)에서 P/B/C 후보로 나온 경우만): 측면 사진에서
         최대너비점(3번), 곡률점(4번), 살 끝점(5번), 손톱 끝점(10번)
         4점을 클릭받아 mm 좌표(ChArUco 호모그래피 변환)로 상대 높이를 비교한다.
           - 3번과 4번 높이가 같음 -> C형
           - 4번이 3번보다 높고, 손톱 끝(10)이 살 끝(5)보다 낮음 -> P형
           - 4번이 3번보다 높고, 5번이 10번보다 높거나 같음 -> B형
           - 그 외 패턴 -> 판정 불가(None), 근거와 함께 보고

    이 값들(기울기 임계값, 높이 동일 판정 허용오차)은 실험 중인 가설이라
    curve_config.py의 CLASSIFICATION_THRESHOLDS에 분리돼 있다.

[2. 실행 명령어]
    이 파일은 직접 실행하지 않는다. measure_curve.py가 `--type auto`일 때
        from curve_classify import run_interactive_classification
    형태로 불러와서 사용한다. 순수 판정 로직만 테스트하려면 아래처럼
    자체 테스트를 돌릴 수 있다.
        python3 curve_classify.py

[3. 어디에 입력해야 하는가]
    -> `measure_curve.py --type auto`를 실행하는 터미널에서 그대로 이어서
       입력한다 (정면/측면 사진 위에 마우스로 4점씩 추가 클릭).

[4. 정상적으로 실행되면]
    measure_curve.py --type auto 실행 시, 곡면 너비 클릭(정면 2점 + 측면
    2점) 뒤에 분류용 클릭(정면 4점 + 측면 4점)이 이어지고, 아래처럼
    판정 결과와 근거가 출력된다.

        [INFO] --type auto: 유형 자동 분류를 먼저 진행합니다 (정면 4점 + 측면 4점 클릭).
        [CLASSIFY] 판정 결과: P형
                   근거: front_slope=0.42 (0~1: 평행 후보), diff_34_mm=1.85 (4번이 3번보다 높음), 10번<5번(낮음)
        [SAVED] data/results/type_classification.csv

    python3 curve_classify.py 를 직접 실행하면 자체 테스트만 돈다.
        [OK] curve_classify.py 자체 테스트 전부 통과

[5. 오류가 발생하면 확인할 것]
    - 판정 결과가 계속 None(판정 불가)으로 나온다: curve_config.py의
      CLASSIFICATION_THRESHOLDS 값이 실제 촬영 환경과 맞는지, 4점을
      정확한 순서(정면: 왼쪽위->왼쪽아래->오른쪽위->오른쪽아래, 측면:
      3번->4번->5번->10번)로 클릭했는지 확인.
    - 판정이 눈으로 보기에 이상하다: 측면 판정은 측면 사진의 ChArUco
      호모그래피로 mm 변환한 좌표를 기준으로 하므로, `--side-calib`가
      올바른 보정 JSON인지 확인 (엉뚱한 사진의 보정 JSON을 넣으면
      상대 높이 비교가 깨진다).
    - 클릭을 잘못했다: 'r' 키로 마지막 점 되돌리기, 전체 취소는 'q'
      (취소하면 measure_curve.py가 "--type을 수동으로 지정해서 다시
      시도하라"는 안내와 함께 중단된다).
"""

from utils import collect_points, transform_points_to_mm, append_csv_rows, FINGER_LABELS_KO
from curve_config import CLASSIFICATION_THRESHOLDS

CLASSIFICATION_CSV_HEADER = [
    "timestamp",
    "front_image",
    "side_image",
    "finger",
    "nail_type",
    "front_slope",
    "side_diff_34_mm",
    "reason",
]


def compute_front_slope(top_left, bottom_left, top_right, bottom_right):
    """정면 사다리꼴 좌/우 변의 평균 기울기(|가로 이동량| / |세로 이동량|)를 계산한다."""

    def _slope(top, bottom):
        dx = abs(bottom[0] - top[0])
        dy = abs(bottom[1] - top[1])
        return dx / dy if dy else float("inf")

    left_slope = _slope(top_left, bottom_left)
    right_slope = _slope(top_right, bottom_right)
    return (left_slope + right_slope) / 2


def classify_front(slope, thresholds=None):
    """
    기울기 값으로 1차 분류한다.
    반환값: "S"(확정) / "candidate"(P/B/C 후보, 측면 판정 필요) / None(범위 밖, 판정 불가)
    """
    t = thresholds or CLASSIFICATION_THRESHOLDS
    if slope <= t["front_slope_parallel_max"]:
        return "candidate"
    if slope <= t["front_slope_fan_max"]:
        return "S"
    return None


def classify_side(p3, p4, p5, p10, thresholds=None):
    """
    측면 4점(mm 좌표)의 상대 높이로 P/B/C를 구분한다. 좌표는 y값이 작을수록
    "높다"(이미지/보드 좌표계 그대로, 원본 사진에서 위쪽일수록 y가 작다고 가정).

    반환값: (nail_type 또는 None, basis dict)
    """
    t = thresholds or CLASSIFICATION_THRESHOLDS
    tie_tolerance_mm = t.get("side_height_tie_mm", 0.3)

    diff_34 = p3[1] - p4[1]  # 양수면 4번이 3번보다 위(높음)
    basis = {"diff_34_mm": diff_34}

    if abs(diff_34) <= tie_tolerance_mm:
        basis["reason"] = f"3번과 4번 높이가 비슷함(|diff|<={tie_tolerance_mm}mm) -> C형"
        return "C", basis

    if diff_34 > 0:
        # 가설 원문의 P/B 조건("10이 5보다 낮음" vs "5가 10보다 높거나 같음")은 표현이
        # 사실상 같은 방향을 가리켜서 문자 그대로 구현하면 B가 동률 경계에서만
        # 도달 가능해진다. 여기서는 P/C처럼 서로 배타적인 이분류가 되도록,
        # "10이 5보다 낮으면 P, 그렇지 않으면(=10이 5보다 높거나 같으면) B"로 해석한다.
        # 이 해석 자체도 검증 전 가설이니 실측 데이터로 재확인이 필요하다.
        if p10[1] > p5[1]:  # 10번 y가 5번보다 큼 = 10번이 더 아래(낮음)
            basis["reason"] = "4번이 3번보다 높고, 손톱 끝(10)이 살 끝(5)보다 낮음 -> P형"
            return "P", basis
        basis["reason"] = "4번이 3번보다 높고, 손톱 끝(10)이 살 끝(5)보다 높거나 같음 -> B형"
        return "B", basis

    basis["reason"] = "4번이 3번보다 낮음 - 가설에 없는 패턴, 판정 불가"
    return None, basis


def classify_type(front_points, side_points_mm, thresholds=None):
    """
    front_points: {"top_left","bottom_left","top_right","bottom_right"} 픽셀 좌표
    side_points_mm: {"p3","p4","p5","p10"} mm 좌표 (호모그래피 변환된 값)

    반환값: {"nail_type": "P"/"S"/"B"/"C"/None, "basis": {...근거값들...}}
    """
    slope = compute_front_slope(
        front_points["top_left"], front_points["bottom_left"], front_points["top_right"], front_points["bottom_right"]
    )
    front_result = classify_front(slope, thresholds)
    basis = {"front_slope": slope}

    if front_result == "S":
        basis["reason"] = "정면 기울기 1~3 구간(사다리꼴) -> S형 확정"
        return {"nail_type": "S", "basis": basis}

    if front_result is None:
        basis["reason"] = "정면 기울기가 예상 범위(0~3)를 벗어남 - 판정 불가"
        return {"nail_type": None, "basis": basis}

    side_type, side_basis = classify_side(
        side_points_mm["p3"], side_points_mm["p4"], side_points_mm["p5"], side_points_mm["p10"], thresholds
    )
    basis.update(side_basis)
    return {"nail_type": side_type, "basis": basis}


def run_interactive_classification(front_image, side_image, side_homography, finger_key):
    """
    정면 이미지에서 사다리꼴 4점, 측면 이미지에서 마름모꼴 4점을 클릭받아
    classify_type()까지 실행한다. 중간에 취소('q')하면 None을 반환한다.
    """
    label_ko = FINGER_LABELS_KO[finger_key]

    front_labels = [
        f"[{label_ko}] 정면: 왼쪽 변 위쪽 점을 클릭하세요",
        f"[{label_ko}] 정면: 왼쪽 변 아래쪽 점을 클릭하세요",
        f"[{label_ko}] 정면: 오른쪽 변 위쪽 점을 클릭하세요",
        f"[{label_ko}] 정면: 오른쪽 변 아래쪽 점을 클릭하세요",
    ]
    front_pts = collect_points(front_image, num_points=4, point_labels=front_labels, window_name=f"Classify Front - {label_ko}")
    if front_pts is None:
        return None
    top_left, bottom_left, top_right, bottom_right = front_pts

    side_labels = [
        f"[{label_ko}] 측면: 최대너비점(3번)을 클릭하세요",
        f"[{label_ko}] 측면: 곡률점(4번)을 클릭하세요",
        f"[{label_ko}] 측면: 살 끝점(5번)을 클릭하세요",
        f"[{label_ko}] 측면: 손톱 끝점(10번)을 클릭하세요",
    ]
    side_pts_px = collect_points(side_image, num_points=4, point_labels=side_labels, window_name=f"Classify Side - {label_ko}")
    if side_pts_px is None:
        return None

    side_pts_mm = transform_points_to_mm(side_homography, side_pts_px)
    p3_mm, p4_mm, p5_mm, p10_mm = [tuple(float(v) for v in pt) for pt in side_pts_mm]

    result = classify_type(
        {"top_left": top_left, "bottom_left": bottom_left, "top_right": top_right, "bottom_right": bottom_right},
        {"p3": p3_mm, "p4": p4_mm, "p5": p5_mm, "p10": p10_mm},
    )
    result["front_points_px"] = {
        "top_left": top_left, "bottom_left": bottom_left, "top_right": top_right, "bottom_right": bottom_right,
    }
    result["side_points_px"] = {"p3": side_pts_px[0], "p4": side_pts_px[1], "p5": side_pts_px[2], "p10": side_pts_px[3]}
    return result


def append_classification_row(csv_path, row):
    """type_classification.csv에 판정 결과 1행을 누적 저장한다."""
    append_csv_rows(csv_path, [row], CLASSIFICATION_CSV_HEADER)


# ---------------------------------------------------------------------------
# 자체 테스트 (pytest 없이 python3 curve_classify.py로 바로 실행)
# ---------------------------------------------------------------------------
def _self_test():
    # 1) 거의 평행(기울기 낮음) -> candidate
    slope_parallel = compute_front_slope((100, 50), (102, 150), (300, 50), (298, 150))
    assert classify_front(slope_parallel) == "candidate", f"[FAIL] 평행 케이스가 candidate가 아님: {slope_parallel}"

    # 2) 사다리꼴(기울기 1~3) -> S 확정
    slope_fan = compute_front_slope((100, 50), (250, 150), (300, 50), (150, 150))
    assert classify_front(slope_fan) == "S", f"[FAIL] 부채꼴 케이스가 S가 아님: {slope_fan}"

    # 3) 측면 판정 - C형 (3번/4번 높이 비슷)
    side_type, _ = classify_side(p3=(0, 10.0), p4=(0, 10.1), p5=(0, 5.0), p10=(0, 15.0))
    assert side_type == "C", f"[FAIL] C형 판정 실패: {side_type}"

    # 4) 측면 판정 - P형 (4번이 3번보다 높고, 10번이 5번보다 낮음)
    side_type, _ = classify_side(p3=(0, 10.0), p4=(0, 8.0), p5=(0, 5.0), p10=(0, 12.0))
    assert side_type == "P", f"[FAIL] P형 판정 실패: {side_type}"

    # 5) 측면 판정 - B형 (4번이 3번보다 높고, 손톱 끝(10)이 살 끝(5)보다 낮지 않음)
    side_type, _ = classify_side(p3=(0, 10.0), p4=(0, 8.0), p5=(0, 9.0), p10=(0, 6.0))
    assert side_type == "B", f"[FAIL] B형 판정 실패: {side_type}"

    # 6) classify_type 종합 - 평행 + P패턴 -> P
    result = classify_type(
        front_points={"top_left": (100, 50), "bottom_left": (102, 150), "top_right": (300, 50), "bottom_right": (298, 150)},
        side_points_mm={"p3": (0, 10.0), "p4": (0, 8.0), "p5": (0, 5.0), "p10": (0, 12.0)},
    )
    assert result["nail_type"] == "P", f"[FAIL] classify_type 종합 실패: {result}"

    print("[OK] curve_classify.py 자체 테스트 전부 통과")


if __name__ == "__main__":
    _self_test()
