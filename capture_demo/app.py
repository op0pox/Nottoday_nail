# -*- coding: utf-8 -*-
"""
app.py
========
[역할] 폰 브라우저로 사진을 업로드하면 서버(랩탑)가 자동으로 손톱 가로/세로/측면
너비를 측정하고 곡면 길이를 계산해서 보여주는 데모용 웹앱.

흐름:
  1. "/" 에서 새 세션 시작 (오른손/왼손 선택)
  2. "/s/<sid>/finger" 에서 손가락 하나씩(엄지→소지) 정면사진 1장 + 측면사진 1장 +
     유형(P/B/S/C)을 한 번에 업로드 -> 자동으로 가로/세로/측면 너비(mm) 계산 -> 곡면 길이 계산
     (그룹사진 한 장으로 5개 손톱을 한 번에 구분하는 방식도 시도했지만, 손가락
     하이라이트가 손톱보다 더 "손톱스럽게" 잡혀 라벨이 계속 엉켜서 포기했다.
     사진 1장당 손가락 1개면 "이게 어느 손가락이냐"는 모호함 자체가 없다.)
  3. "/s/<sid>/result" 에서 5개 손가락 전체 결과 표

[실행] python app.py  (기본 포트 5000, 0.0.0.0으로 바인딩해서 같은 와이파이의
폰에서 http://<랩탑IP>:5000 으로 접속)

[주의] 세션 데이터는 메모리에만 저장한다 (서버 재시작하면 사라짐). 데모 하루용.
업로드된 원본 사진은 data/uploads/<세션ID>/ 에 저장해서 문제 생기면 나중에
디버그용으로 다시 볼 수 있게 한다.
"""

import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for

import board_calib
import debug_viz
import segment
from curve_calc import calc as curve_calc

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "data", "uploads")

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_LABELS_KO = segment.FINGER_LABELS_KO
NAIL_TYPES = ["P", "B", "S", "C"]

app = Flask(__name__)
SESSIONS = {}


def _new_session(hand):
    sid = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    SESSIONS[sid] = {
        "hand": hand,
        "fingers": {},   # {finger: {front_width_mm, front_length_mm, side_width_mm, type, curve}}
        "error": None,
        "debug_images": {},  # {finger: {"front": 파일명, "side": 파일명}}
    }
    os.makedirs(os.path.join(UPLOAD_DIR, sid), exist_ok=True)
    return sid


def _get_session(sid):
    return SESSIONS.get(sid)


MAX_DIM_PX = 1600  # 폰 원본(3000~4000px대) 그대로 처리하면 마커 검출이 수십 초씩 걸려서 축소한다


def _read_image(file_storage):
    if file_storage is None or file_storage.filename == "":
        return None
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > MAX_DIM_PX:
            scale = MAX_DIM_PX / float(longest)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _save_upload(sid, name, img):
    path = os.path.join(UPLOAD_DIR, sid, name)
    cv2.imwrite(path, img)
    return path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    hand = request.form.get("hand", "right")
    sid = _new_session(hand)
    return redirect(url_for("finger_step", sid=sid))


@app.route("/s/<sid>/photo/<filename>")
def session_photo(sid, filename):
    if _get_session(sid) is None or "/" in filename or "\\" in filename:
        abort(404)
    folder = os.path.join(UPLOAD_DIR, sid)
    if not os.path.isfile(os.path.join(folder, filename)):
        abort(404)
    return send_from_directory(folder, filename)


def _process_front(sid, finger, file):
    img = _read_image(file)
    if img is None:
        return None, "정면사진을 선택해주세요."
    _save_upload(sid, "front_%s.jpg" % finger, img)

    result, error = None, None
    try:
        detection = board_calib.detect_board(img)
        result = segment.measure_front_finger(img, detection.homography)
    except (board_calib.BoardDetectionError, segment.SegmentationError) as exc:
        error = "[%s 정면] %s" % (FINGER_LABELS_KO.get(finger, finger), exc)

    if result is not None:
        # 실제로 측정에 쓰인 마스크(YOLO 성공 시 손톱, 폴백 시 위쪽 절반 근사)를 그대로 보여준다
        mask = result["mask"]
        w_pts = (result["point_a_px"], result["point_b_px"])
        l_pts = (result["length_point_a_px"], result["length_point_b_px"])
        width_mm, length_mm = result["front_width_mm"], result["front_length_mm"]
    else:
        # 실패해도 뭘 잘못 잡았는지 보여주기 위해 원시 마스크를 따로 계산한다
        mask = segment.yolo_nail_mask(img)
        if mask is None:
            finger_mask = segment.single_finger_mask(img)
            mask = segment.tip_region_mask(finger_mask, segment.FRONT_TIP_RATIO) if finger_mask is not None else None
        w_pts = segment.mask_width_points(mask) if mask is not None else None
        l_pts = segment.mask_height_points(mask) if mask is not None else None
        width_mm = length_mm = None

    debug_name = "front_%s_debug.jpg" % finger
    debug_img = debug_viz.draw_finger_debug(img, mask, w_pts, l_pts, width_mm, length_mm, error=error)
    _save_upload(sid, debug_name, debug_img)

    return result, error, debug_name


def _process_side(sid, finger, file):
    img = _read_image(file)
    if img is None:
        return None, "측면사진을 선택해주세요.", None
    _save_upload(sid, "side_%s.jpg" % finger, img)

    mask = segment.best_nail_mask(img)
    w_pts = segment.mask_width_points(mask) if mask is not None else None

    result, error = None, None
    try:
        detection = board_calib.detect_board(img)
        result = segment.measure_side_width(img, detection.homography)
    except (board_calib.BoardDetectionError, segment.SegmentationError) as exc:
        error = "[%s 측면] %s" % (FINGER_LABELS_KO.get(finger, finger), exc)

    debug_name = "side_%s_debug.jpg" % finger
    width_mm = result["side_width_mm"] if result else None
    debug_img = debug_viz.draw_finger_debug(img, mask, w_pts, None, width_mm, None, error=error)
    _save_upload(sid, debug_name, debug_img)

    return result, error, debug_name


@app.route("/s/<sid>/finger", methods=["GET", "POST"])
def finger_step(sid):
    session = _get_session(sid)
    if session is None:
        return redirect(url_for("index"))

    remaining = [f for f in FINGER_ORDER if f not in session["fingers"]]

    if request.method == "POST":
        finger = request.form.get("finger")
        nail_type = request.form.get("type")
        if finger not in FINGER_ORDER or nail_type not in NAIL_TYPES:
            session["error"] = "손가락/유형 값이 올바르지 않습니다."
            return redirect(url_for("finger_step", sid=sid))

        front_result, front_error, front_debug = _process_front(sid, finger, request.files.get("front_photo"))
        side_result, side_error, side_debug = _process_side(sid, finger, request.files.get("side_photo"))

        session["debug_images"][finger] = {"front": front_debug, "side": side_debug}

        error = front_error or side_error
        if error:
            session["error"] = error
            return redirect(url_for("finger_step", sid=sid))

        curve = curve_calc(nail_type, front_result["front_width_mm"], side_result["side_width_mm"])
        session["fingers"][finger] = {
            "front_width_mm": front_result["front_width_mm"],
            "front_length_mm": front_result["front_length_mm"],
            "side_width_mm": side_result["side_width_mm"],
            "type": nail_type,
            "curve": curve,
        }
        session["error"] = None
        return redirect(url_for("finger_step", sid=sid))

    if not remaining:
        return redirect(url_for("result_step", sid=sid))

    next_finger = remaining[0]
    return render_template(
        "finger.html",
        sid=sid,
        finger=next_finger,
        finger_label=FINGER_LABELS_KO.get(next_finger, next_finger),
        remaining_count=len(remaining),
        total_count=len(FINGER_ORDER),
        types=NAIL_TYPES,
        error=session["error"],
    )


@app.route("/s/<sid>/result")
def result_step(sid):
    session = _get_session(sid)
    if session is None:
        return redirect(url_for("index"))
    if len(session["fingers"]) < len(FINGER_ORDER):
        return redirect(url_for("finger_step", sid=sid))

    rows = []
    for finger in FINGER_ORDER:
        f = session["fingers"][finger]
        rows.append(
            {
                "finger": finger,
                "label": FINGER_LABELS_KO.get(finger, finger),
                "front_width_mm": f["front_width_mm"],
                "front_length_mm": f["front_length_mm"],
                "side_width_mm": f["side_width_mm"],
                "type": f["type"],
                "curve_length_mm": f["curve"]["total_mm"],
                "front_debug_image": session["debug_images"].get(finger, {}).get("front"),
                "side_debug_image": session["debug_images"].get(finger, {}).get("side"),
            }
        )

    return render_template("result.html", sid=sid, hand=session["hand"], rows=rows)


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
