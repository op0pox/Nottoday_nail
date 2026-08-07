# -*- coding: utf-8 -*-
"""
curve_calc.py
===============
curvature/curvature.py의 calc() 순수 계산 로직만 옮긴 것.
원본은 tkinter/matplotlib(GUI)을 모듈 최상단에서 import해서 서버 프로세스에
그대로 끌어오면 무겁고 깨지기 쉬우므로, 계산 로직만 의존성 없이 복제해서 쓴다.
공식/비율(P형 k_cal=0.86 보정 포함)은 curvature.py와 반드시 동일하게 유지할 것.
"""

import math

TYPES = {
    "P": dict(name="P형 (평행)", k_a=2 / 5, k_h=1 / 4, k_f=1 / 2, k_cal=0.86),
    "B": dict(name="B형 (버드)", k_a=3 / 8, k_h=1 / 6, k_f=2 / 3, k_cal=1.0),
    "S": dict(name="S형 (부채)", k_a=3 / 7, k_h=1 / 7, k_f=1 / 3, k_cal=1.0),
    "C": dict(name="C형 (평행)", k_a=3 / 7, k_h=1 / 5, k_f=1 / 3, k_cal=1.0),
}


def calc(category, front_width_mm, side_width_mm):
    p = TYPES[category]
    a = p["k_a"] * front_width_mm
    b = side_width_mm
    c = math.hypot(a, b)
    h = p["k_h"] * c
    r = c ** 2 / (8 * h) + h / 2
    theta = 2 * math.atan2(c / 2, r - h)
    L = r * theta
    flat = p["k_f"] * a
    total = (2 * L + flat) * p["k_cal"]
    return dict(
        category=category, name=p["name"],
        front_width_mm=front_width_mm, side_width_mm=side_width_mm,
        a=a, b=b, c=c, h=h, r=r, theta_deg=math.degrees(theta), L=L, flat=flat,
        total_mm=total,
    )
