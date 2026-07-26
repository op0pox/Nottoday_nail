# -*- coding: utf-8 -*-
"""
generate_fake_dashboard_data.py
=================================
[1. 역할]
    웹 대시보드(webapp/, main.py dashboard)를 로컬 PC(카메라도, Jetson에서 찍은
    실측 데이터도 없는 환경)에서 검증하기 위한 가짜 샘플 데이터 생성기.
    data/collect/session_*_fake/, data/results/measure_*.json,
    data/results/analyze_*.json(+overlay.jpg)를 만든다.

    실측 파이프라인(main.py의 calibrate/measure/analyze/collect)은 전혀 건드리지
    않는다 - 개발 중에만 쓰고, 실제 Jetson 배포/실측에는 쓰지 않는다.

[2. 실행 명령어 예시]
    python3 generate_fake_dashboard_data.py
    python3 generate_fake_dashboard_data.py --sessions 3 --fingers-per-session 10
    python3 generate_fake_dashboard_data.py --seed 42

[3. 어디에서 실행하는가]
    nail_analysis/ 안에서 실행한다 (config.yaml의 data 경로가 상대경로이므로).

[4. 정상적으로 실행되면]
    data/collect/, data/results/ 아래에 가짜 세션/사진/결과 JSON이 생성되고,
    콘솔에 생성된 경로가 출력된다. 이후 'python3 main.py dashboard --debug'로
    브라우저에서 확인할 수 있다.

[5. 오류가 발생하면 확인할 것]
    - "설정 파일을 찾을 수 없습니다": nail_analysis/ 디렉터리에서 실행했는지 확인.
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta

import cv2
import numpy as np

from analysis.body_length import classify_body_length
from analysis.tip_match import match_tip_size
from config_loader import load_config

FRONT_KEYPOINT_IDS = [str(i) for i in range(1, 10)]
SIDE_KEYPOINT_IDS = ["1", "2", "3", "4", "5", "9", "10"]
NAIL_TYPES = ["P", "S", "B", "C"]


def _make_placeholder_image(text, width=480, height=360):
    color = (random.randint(150, 220), random.randint(150, 220), random.randint(150, 220))
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.putText(img, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "FAKE DATA (dev only)",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return img


def _fake_keypoints(ids):
    return {k: [round(random.uniform(0, 500), 1), round(random.uniform(0, 500), 1)] for k in ids}


def _fake_result(config):
    """_result_to_json_dict()(main.py)와 같은 모양의 dict를 무작위 값으로 만든다."""
    nail_type = random.choice(NAIL_TYPES)
    w_front_mm = round(random.uniform(6.0, 12.0), 2)
    h_front_mm = round(random.uniform(8.0, 16.0), 2)
    w_side_mm = round(random.uniform(3.0, 6.0), 2)
    curve_length_mm = round(random.uniform(10.0, 20.0), 2)
    confidence = round(random.uniform(0.6, 1.0), 2)

    body_length = classify_body_length(h_front_mm, w_front_mm, config)
    tip_match = match_tip_size(curve_length_mm, nail_type, config)

    return {
        "front_keypoints_px": _fake_keypoints(FRONT_KEYPOINT_IDS),
        "side_keypoints_px": _fake_keypoints(SIDE_KEYPOINT_IDS),
        "front_keypoints_mm": _fake_keypoints(FRONT_KEYPOINT_IDS),
        "side_keypoints_mm": _fake_keypoints(SIDE_KEYPOINT_IDS),
        "w_front_mm": w_front_mm,
        "h_front_mm": h_front_mm,
        "w_side_mm": w_side_mm,
        "body_length": body_length,
        "nail_type": {
            "final_type": nail_type,
            "front": {},
            "side": {},
            "mismatch": confidence < 0.85,
            "confidence": confidence,
        },
        "curve": {},
        "curve_length_mm": curve_length_mm,
        "tip_match": tip_match,
        "auto_detected": True,
    }


def generate_collect_session(config, session_offset, num_fingers):
    """collect.CollectSession.save_finger()와 같은 파일 구조를 만든다 (makedirs만 직접 함)."""
    collect_dir = config["paths"]["collect_dir"]
    timestamp = (datetime.now() - timedelta(minutes=session_offset)).strftime("%Y%m%d_%H%M%S")
    session_name = "session_%s_fake" % timestamp
    session_dir = os.path.join(collect_dir, session_name)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    finger_ids = list(config["fingers"]["order"])[:num_fingers]
    for finger_id in finger_ids:
        cv2.imwrite(
            os.path.join(session_dir, "%s_front.jpg" % finger_id),
            _make_placeholder_image("%s front" % finger_id),
        )
        cv2.imwrite(
            os.path.join(session_dir, "%s_side.jpg" % finger_id),
            _make_placeholder_image("%s side" % finger_id),
        )

        result = _fake_result(config)
        meta = {
            "finger_id": finger_id,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "front_keypoints_px": result["front_keypoints_px"],
            "side_keypoints_px": result["side_keypoints_px"],
            "files": {
                "front_image": "%s_front.jpg" % finger_id,
                "side_image": "%s_side.jpg" % finger_id,
            },
        }
        meta.update(result)
        with open(os.path.join(session_dir, "%s_meta.json" % finger_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[생성] %s (%d개 손가락)" % (session_dir, len(finger_ids)))
    return session_name


def generate_measure_summary(config, num_fingers):
    """main.py의 _save_measure_summary()와 같은 모양의 JSON을 만든다."""
    results_dir = config["paths"]["results_dir"]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finger_ids = list(config["fingers"]["order"])[:num_fingers]
    summary = {
        "timestamp": timestamp,
        "fingers": {fid: _fake_result(config) for fid in finger_ids},
        "config_snapshot": {},
    }
    path = os.path.join(results_dir, "measure_%s.json" % timestamp)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[생성] %s" % path)


def generate_analyze_result(config, finger_id):
    """main.py의 cmd_analyze()가 저장하는 <finger_id>.json + <finger_id>_overlay.jpg를 만든다."""
    results_dir = config["paths"]["results_dir"]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    payload = {"finger_id": finger_id, "result": _fake_result(config), "config_snapshot": {}}
    json_path = os.path.join(results_dir, "%s.json" % finger_id)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    overlay_path = os.path.join(results_dir, "%s_overlay.jpg" % finger_id)
    cv2.imwrite(overlay_path, _make_placeholder_image("%s overlay" % finger_id, width=640, height=480))
    print("[생성] %s (+ overlay.jpg)" % json_path)


def main():
    parser = argparse.ArgumentParser(description="대시보드 로컬 검증용 가짜 데이터 생성 (실측 데이터 아님)")
    parser.add_argument("--sessions", type=int, default=2, help="생성할 collect 세션 수 (기본 2)")
    parser.add_argument("--fingers-per-session", type=int, default=5, help="세션당 손가락 수 (기본 5)")
    parser.add_argument("--seed", type=int, default=None, help="재현 가능한 값이 필요하면 지정")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    config = load_config()

    for i in range(args.sessions):
        generate_collect_session(config, i, args.fingers_per_session)

    generate_measure_summary(config, args.fingers_per_session)

    for finger_id in list(config["fingers"]["order"])[:2]:
        generate_analyze_result(config, "analyze_%s" % finger_id)

    print("\n[완료] 가짜 데이터 생성 끝. 'python3 main.py dashboard --debug'로 http://127.0.0.1:5000 에서 확인하세요.")


if __name__ == "__main__":
    main()
