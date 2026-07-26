# -*- coding: utf-8 -*-
"""
visualize.py
=============
[역할] 키포인트/측정선/판정 결과를 원본 이미지 위에 오버레이로 그리는
함수 모음. 한글 표시는 OpenCV 기본 폰트로 깨지기 때문에 결과 텍스트는
영문으로 표시한다 (예: Type=S, Body=Medium, Curve=13.59mm, Tip=S6).

[실행 위치] 직접 실행하는 파일이 아니다. main.py/collect.py가 불러와서 쓴다.
"""

import cv2

FRONT_LABEL_COLOR = (0, 255, 0)
SIDE_LABEL_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
MEASURE_LINE_COLOR = (0, 255, 255)


def draw_keypoints(image, keypoints_px, color=POINT_COLOR, radius=6):
    """키포인트를 번호와 함께 원본 이미지 위에 그린 사본을 반환한다."""
    annotated = image.copy()
    for kid, (x, y) in keypoints_px.items():
        pt = (int(round(x)), int(round(y)))
        cv2.circle(annotated, pt, radius, color, -1)
        cv2.putText(
            annotated,
            str(kid),
            (pt[0] + radius + 2, pt[1] - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            FRONT_LABEL_COLOR,
            2,
        )
    return annotated


def draw_front_measurements(image, front_kp_px):
    """정면 W_front(점3-점7), H_front(점1-점5) 측정선을 그린다."""
    annotated = draw_keypoints(image, front_kp_px)
    p3 = tuple(int(round(v)) for v in front_kp_px["3"])
    p7 = tuple(int(round(v)) for v in front_kp_px["7"])
    p1 = tuple(int(round(v)) for v in front_kp_px["1"])
    p5 = tuple(int(round(v)) for v in front_kp_px["5"])
    cv2.line(annotated, p3, p7, MEASURE_LINE_COLOR, 2)
    cv2.line(annotated, p1, p5, MEASURE_LINE_COLOR, 2)
    return annotated


def draw_side_measurements(image, side_kp_px):
    """측면 W_side(점3-점9) 측정선을 그린다."""
    annotated = draw_keypoints(image, side_kp_px)
    p3 = tuple(int(round(v)) for v in side_kp_px["3"])
    p9 = tuple(int(round(v)) for v in side_kp_px["9"])
    cv2.line(annotated, p3, p9, MEASURE_LINE_COLOR, 2)
    return annotated


def draw_result_text(image, lines, origin=(10, 30), line_height=32, color=(0, 255, 255)):
    """
    결과 요약 텍스트(영문)를 이미지 좌상단에 여러 줄로 그린다.
    lines: ["Type=S", "Body=Medium", "Curve=13.59mm", "Tip=S6"] 같은 문자열 리스트
    """
    annotated = image.copy()
    x, y = origin
    for i, line in enumerate(lines):
        pos = (x, y + i * line_height)
        # 가독성을 위해 검은 테두리 + 컬러 글자
        cv2.putText(annotated, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(annotated, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return annotated


def build_result_lines(result):
    """
    analysis 모듈들의 결과 dict를 받아서 draw_result_text에 넣을 영문 문자열 리스트를 만든다.
    result 예시 키: body_length(label), nail_type(final_type), curve_length_mm, tip_match(size_label), confidence
    """
    lines = []
    if "body_length" in result:
        lines.append("Body=%s" % result["body_length"]["label"])
    if "nail_type" in result:
        lines.append("Type=%s" % result["nail_type"]["final_type"])
        if result["nail_type"].get("mismatch"):
            lines.append("(front/side mismatch, conf=%.2f)" % result["nail_type"]["confidence"])
    if "curve_length_mm" in result:
        lines.append("Curve=%.2fmm" % result["curve_length_mm"])
    if "tip_match" in result:
        lines.append("Tip=%s (diff %.2fmm)" % (result["tip_match"]["size_label"], result["tip_match"]["diff_mm"]))
    return lines


def draw_finger_guide_overlay(image, guide_box, alpha=0.35, color=(0, 255, 255), label="Place finger here"):
    """
    라이브 프리뷰에 손가락 배치 가이드(반투명 사각형)를 그린다.
    guide_box: (x, y, w, h) 픽셀 좌표
    """
    x, y, w, h = guide_box
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=-1)
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.rectangle(blended, (x, y), (x + w, y + h), color, thickness=2)
    if label:
        cv2.putText(blended, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return blended


def draw_scale_status(image, status, origin=(10, 60)):
    """
    ScaleProvider의 상태("live"/"held"/"fallback"/"none")를 화면에 표시한다.
    "live" 이외는 경고 색(주황/빨강)으로 눈에 띄게 표시한다.
    """
    status_colors = {
        "live": (0, 200, 0),
        "held": (0, 165, 255),
        "fallback": (0, 140, 255),
        "none": (0, 0, 255),
    }
    status_text = {
        "live": "SCALE: live",
        "held": "SCALE: held (board not visible)",
        "fallback": "SCALE: fallback (saved calibration)",
        "none": "SCALE: UNAVAILABLE",
    }
    color = status_colors.get(status, (0, 0, 255))
    text = status_text.get(status, "SCALE: unknown")
    annotated = image.copy()
    cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return annotated


def stack_front_side(front_image, side_image, gap=10):
    """정면/측면 이미지를 같은 높이로 맞춰서 좌우로 이어붙인다 (요약 화면용)."""
    import numpy as np

    h = min(front_image.shape[0], side_image.shape[0])

    def _resize_to_height(img, target_h):
        scale = target_h / float(img.shape[0])
        new_w = int(img.shape[1] * scale)
        return cv2.resize(img, (new_w, target_h))

    f = _resize_to_height(front_image, h)
    s = _resize_to_height(side_image, h)
    separator = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.hstack([f, separator, s])
