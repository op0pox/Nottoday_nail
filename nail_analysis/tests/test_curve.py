# -*- coding: utf-8 -*-
"""
tests/test_curve.py
====================
[역할] analysis/curve.py의 곡면 길이 공식을 검증한다.

실행 명령어:
    cd nail_analysis
    python -m unittest tests.test_curve -v

포함 내용:
    1. 실측 검증 케이스 (C형, W_front=8.3mm, W_side=4.4mm -> 곡면길이 ~= 13.59mm)
       실제 자로 잰 값(~13mm)과도 같은 범위인지 참고로 같이 출력한다.
    2. flat_ratio 표기와 PDF 원문 표기(P=2L+1/2 a, S/C=2L+1/3 a, B=2L+2/3 a)가
       동일한 값을 내는지 여러 입력값에 대해 확인한다.
    3. 잘못된 입력(0 이하)에 대한 예외 처리 확인.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.curve import compute_curve_length, compute_curve_length_for_type  # noqa: E402


# config.yaml의 curve_constants와 반드시 같은 값을 유지해야 한다.
CURVE_CONSTANTS = {
    "P": {"flat_ratio": 1.0 / 5, "h_ratio": 1.0 / 4},
    "S": {"flat_ratio": 1.0 / 7, "h_ratio": 1.0 / 7},
    "B": {"flat_ratio": 1.0 / 4, "h_ratio": 1.0 / 6},
    "C": {"flat_ratio": 1.0 / 7, "h_ratio": 1.0 / 5},
}

# PDF 원문 표기: flat = k * a  (P=1/2, S/C=1/3, B=2/3)
PDF_FLAT_K = {"P": 0.5, "S": 1.0 / 3, "B": 2.0 / 3, "C": 1.0 / 3}


class TestCurveLengthRealWorldCase(unittest.TestCase):
    """PDF/실측 검증 케이스: C형, W_front=8.3mm, W_side=4.4mm."""

    def test_c_type_matches_reference_value(self):
        result = compute_curve_length_for_type(8.3, 4.4, "C", CURVE_CONSTANTS)

        # 중간값들도 PDF 원문의 근사치와 크게 어긋나지 않는지 대략 확인 (느슨한 허용오차)
        self.assertAlmostEqual(result["a"], 3.557, delta=0.01)
        self.assertAlmostEqual(result["b"], 4.4, delta=0.001)

        # 최종 곡면길이는 PDF 기준값 13.59mm에 ±0.15mm로 일치해야 한다.
        self.assertAlmostEqual(result["curve_length_mm"], 13.59, delta=0.15)

        # 참고용 출력: 실제 자로 잰 값(~13mm)과 같은 자릿수 범위인지 (엄격한 assert는 아님 -
        # 곡선을 평평한 자로 재면 항상 실제보다 짧게 측정되므로 정확히 같을 수 없다)
        print(
            "\n[참고] 공식 계산값 %.3fmm vs 실제 자 측정값 13mm (곡선을 평평한 자로 재서 "
            "짧게 측정되는 것이 정상)" % result["curve_length_mm"]
        )


class TestFlatRatioEquivalence(unittest.TestCase):
    """
    flat = W_front * flat_ratio 표기와, PDF 원문의 flat = k * a 표기가
    수학적으로 동일한 결과를 내는지 확인한다.

    a = (W_front - flat) / 2 이므로, flat = k*a 를 만족하는 flat_ratio는
    대수적으로 flat_ratio = k / (k + 2) 가 된다. 이 항등식과, 실제로 두
    표기로 각각 flat을 계산했을 때 값이 같은지를 여러 W_front 값에 대해 확인한다.
    """

    def test_flat_ratio_derived_from_k_matches_config(self):
        for nail_type, k in PDF_FLAT_K.items():
            derived_flat_ratio = k / (k + 2.0)
            configured_flat_ratio = CURVE_CONSTANTS[nail_type]["flat_ratio"]
            self.assertAlmostEqual(
                derived_flat_ratio,
                configured_flat_ratio,
                places=6,
                msg="%s: k=%.4f로부터 유도한 flat_ratio가 config 값과 다릅니다" % (nail_type, k),
            )

    def test_flat_value_matches_across_representations(self):
        for nail_type, k in PDF_FLAT_K.items():
            flat_ratio = CURVE_CONSTANTS[nail_type]["flat_ratio"]
            for w_front in (5.0, 8.3, 12.0, 20.0):
                flat_from_ratio = w_front * flat_ratio
                a_from_ratio = (w_front - flat_from_ratio) / 2.0
                flat_from_k = k * a_from_ratio
                self.assertAlmostEqual(
                    flat_from_ratio,
                    flat_from_k,
                    places=6,
                    msg="%s (W_front=%.1f): flat_ratio 표기와 k*a 표기가 불일치" % (nail_type, w_front),
                )


class TestCurveLengthAllTypes(unittest.TestCase):
    """4개 유형 모두 예외 없이 계산되고, 곡면길이가 W_front보다는 크다는 상식적 검증."""

    def test_all_types_produce_positive_reasonable_length(self):
        for nail_type in ("P", "S", "B", "C"):
            result = compute_curve_length_for_type(9.0, 5.0, nail_type, CURVE_CONSTANTS)
            self.assertGreater(result["curve_length_mm"], 0)
            # 곡면길이는 평평한 너비(W_front)보다 항상 같거나 길어야 한다 (호가 직선보다 길거나 같음)
            self.assertGreaterEqual(result["curve_length_mm"], 9.0 - 1e-6)
            self.assertTrue(math.isfinite(result["curve_length_mm"]))


class TestCurveLengthInvalidInput(unittest.TestCase):
    def test_zero_or_negative_width_raises(self):
        with self.assertRaises(ValueError):
            compute_curve_length(0, 4.4, 1.0 / 7, 1.0 / 5)
        with self.assertRaises(ValueError):
            compute_curve_length(8.3, -1, 1.0 / 7, 1.0 / 5)

    def test_unknown_nail_type_raises(self):
        with self.assertRaises(ValueError):
            compute_curve_length_for_type(8.3, 4.4, "X", CURVE_CONSTANTS)


if __name__ == "__main__":
    unittest.main()
