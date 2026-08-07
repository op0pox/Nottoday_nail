# -*- coding: utf-8 -*-
"""
debug_viz.py
==============
세그멘테이션 결과(마스크 윤곽선)와 측정 지점(점/선/mm값)을 원본 사진 위에
그려서 저장한다. 성공/실패 상관없이 항상 만들어서, 폰 화면에서 바로
"어디를 손가락/손톱으로 인식했는지"를 눈으로 확인할 수 있게 하기 위함.
"""

import cv2


def draw_finger_debug(image_bgr, mask, width_pts=None, length_pts=None, width_mm=None, length_mm=None, error=None):
    """
    손가락 1개 사진(정면/측면 공용)의 디버그 오버레이.
    width_pts(가로, 빨간 선)는 정면/측면 둘 다 그리고, length_pts(세로, 파란 선)는
    정면에서만 넘겨준다.
    """
    overlay = image_bgr.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 4)

    if width_pts is not None:
        p1, p2 = width_pts
        p1i, p2i = tuple(map(int, p1)), tuple(map(int, p2))
        cv2.circle(overlay, p1i, 10, (0, 0, 255), -1)
        cv2.circle(overlay, p2i, 10, (0, 0, 255), -1)
        cv2.line(overlay, p1i, p2i, (0, 0, 255), 4)
        if width_mm is not None:
            cv2.putText(
                overlay, "W %.2fmm" % width_mm, (p2i[0] + 20, p2i[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
            )

    if length_pts is not None:
        lp1, lp2 = length_pts
        lp1i, lp2i = tuple(map(int, lp1)), tuple(map(int, lp2))
        cv2.circle(overlay, lp1i, 10, (255, 0, 0), -1)
        cv2.circle(overlay, lp2i, 10, (255, 0, 0), -1)
        cv2.line(overlay, lp1i, lp2i, (255, 0, 0), 4)
        if length_mm is not None:
            cv2.putText(
                overlay, "L %.2fmm" % length_mm, (lp2i[0] + 20, lp2i[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3,
            )

    if error:
        _draw_error_banner(overlay, error)
    return overlay


def _draw_error_banner(image, message):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(image, message[:60], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
