# -*- coding: utf-8 -*-
"""
webapp/app.py
==============
[1. 역할]
    data/collect/, data/results/에 저장된 사진·측정 결과를 브라우저로 보는
    읽기 전용 대시보드(Phase A). 쓰기/삭제 라우트는 없다. 카메라를 열지
    않으므로 measure --save-data와 동시에 실행해도 충돌하지 않는다.

[2. 실행 명령어 예시]
    python3 main.py dashboard
    python3 main.py dashboard --host 0.0.0.0 --port 5000
    python3 main.py dashboard --debug   (로컬 개발용, LAN에 노출 시 사용 금지)

[3. 어디에서 실행하는가]
    Jetson Nano에서 실행하는 것이 기본이다(다른 서브커맨드가 저장한 data/를
    바로 읽어야 하므로). 같은 LAN의 노트북에서 브라우저로 http://<jetson-ip>:포트
    접속해서 본다.

[4. 정상적으로 실행되면]
    "/"에서 Collect Sessions / Measure Runs / Analyze Results 3개 목록이 보이고,
    각 항목을 클릭하면 사진(있으면)과 유형/바디기장/곡률/팁사이즈 결과가 보인다.

[5. 오류가 발생하면 확인할 것]
    - "Flask가 설치되어 있지 않습니다": requirements.txt의 Flask 관련 패키지 설치 확인.
    - 목록이 비어있음: data/collect, data/results에 실제로 저장된 파일이 있는지 확인
      (measure --save-data 또는 collect, analyze를 먼저 실행해야 함).
"""

from flask import Flask, abort, render_template, send_from_directory

from . import data_sources as ds


def create_app(config):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            sessions=ds.list_collect_sessions(config),
            measure_runs=ds.list_measure_summaries(config),
            analyze_results=ds.list_analyze_results(config),
        )

    @app.route("/collect/<session_name>")
    def collect_session(session_name):
        finger_ids = ds.list_session_fingers(config, session_name)
        if finger_ids is None:
            abort(404)

        fingers = []
        for finger_id in finger_ids:
            meta = ds.load_finger_meta(config, session_name, finger_id) or {}
            files = meta.get("files", {})
            fingers.append(
                {
                    "finger_id": finger_id,
                    "meta": meta,
                    "front_image": files.get("front_image"),
                    "side_image": files.get("side_image"),
                }
            )

        return render_template("collect_session.html", session_name=session_name, fingers=fingers)

    @app.route("/collect/<session_name>/img/<filename>")
    def collect_image(session_name, filename):
        session_dir, safe_filename = ds.resolve_session_image(config, session_name, filename)
        if session_dir is None:
            abort(404)
        return send_from_directory(session_dir, safe_filename)

    @app.route("/measure/<timestamp>")
    def measure_detail(timestamp):
        summary = ds.load_measure_summary(config, timestamp)
        if summary is None:
            abort(404)
        fingers_order = config["fingers"]["order"]
        ordered_fingers = [
            (fid, summary["fingers"][fid]) for fid in fingers_order if fid in summary["fingers"]
        ]
        return render_template(
            "measure_summary.html", timestamp=timestamp, fingers=ordered_fingers
        )

    @app.route("/analyze/<finger_id>")
    def analyze_detail(finger_id):
        data = ds.load_analyze_result(config, finger_id)
        if data is None:
            abort(404)
        return render_template("analyze_result.html", finger_id=finger_id, data=data)

    @app.route("/analyze/<finger_id>/overlay.jpg")
    def analyze_image(finger_id):
        results_dir, safe_filename = ds.resolve_analyze_overlay(config, finger_id)
        if results_dir is None:
            abort(404)
        return send_from_directory(results_dir, safe_filename)

    return app
