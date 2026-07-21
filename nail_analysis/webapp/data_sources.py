# -*- coding: utf-8 -*-
"""
webapp/data_sources.py
========================
[역할]
    data/collect/, data/results/에 이미 저장된 파일을 읽기만 하는 헬퍼 모음.
    쓰기(os.makedirs 등 부작용 포함)는 절대 하지 않는다 - collect.CollectSession은
    생성자에서 os.makedirs를 호출하므로 여기서는 재사용하지 않는다.

    모든 함수는 요청에서 온 이름(session_name, finger_id, filename)을 그대로
    경로에 꽂지 않고, 반드시 해당 디렉터리의 실제 os.listdir() 결과와 먼저
    대조한다("요청을 믿지 말고 listdir을 믿는다"). 없는 이름이면 None을 반환하고,
    라우트 쪽에서 404로 처리한다.

[실행 위치] 직접 실행하는 파일이 아니다. webapp/app.py가 불러와서 쓴다.
"""

import json
import os


def _safe_child_name(base_dir, name):
    """
    name이 os.listdir(base_dir) 안에 실제로 존재하는 항목이면 그 이름을 그대로,
    아니면 None을 반환한다. '..'/절대경로/URL 인코딩된 구분자는 listdir 결과에
    애초에 나타날 수 없으므로 이 대조만으로 경로 조작을 막을 수 있다.
    """
    if not os.path.isdir(base_dir):
        return None
    if name in os.listdir(base_dir):
        return name
    return None


def _mtime_desc(base_dir, names):
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(base_dir, n)), reverse=True)


# ---------------------------------------------------------------------------
# data/collect/session_*/ (measure --save-data 또는 collect 서브커맨드가 생성)
# ---------------------------------------------------------------------------
def list_collect_sessions(config):
    """
    [{"name": "session_20260101_120000", "finger_count": 3}, ...] 를 최신순으로 반환.
    """
    collect_dir = config["paths"]["collect_dir"]
    if not os.path.isdir(collect_dir):
        return []

    names = [
        n
        for n in os.listdir(collect_dir)
        if n.startswith("session_") and os.path.isdir(os.path.join(collect_dir, n))
    ]
    names = _mtime_desc(collect_dir, names)

    sessions = []
    for name in names:
        session_dir = os.path.join(collect_dir, name)
        finger_count = len([f for f in os.listdir(session_dir) if f.endswith("_meta.json")])
        sessions.append({"name": name, "finger_count": finger_count})
    return sessions


def list_session_fingers(config, session_name):
    """
    session_name이 실제 존재하는 세션이면 finger_id 목록(정렬됨)을, 아니면 None을 반환한다.
    """
    collect_dir = config["paths"]["collect_dir"]
    safe_name = _safe_child_name(collect_dir, session_name)
    if safe_name is None:
        return None

    session_dir = os.path.join(collect_dir, safe_name)
    fingers = [f[: -len("_meta.json")] for f in os.listdir(session_dir) if f.endswith("_meta.json")]
    return sorted(fingers)


def load_finger_meta(config, session_name, finger_id):
    """
    <finger_id>_meta.json을 읽어 dict로 반환한다. session_name/finger_id가 실제
    존재하지 않으면 None을 반환한다.
    """
    collect_dir = config["paths"]["collect_dir"]
    safe_session = _safe_child_name(collect_dir, session_name)
    if safe_session is None:
        return None

    session_dir = os.path.join(collect_dir, safe_session)
    meta_filename = "%s_meta.json" % finger_id
    safe_meta = _safe_child_name(session_dir, meta_filename)
    if safe_meta is None:
        return None

    with open(os.path.join(session_dir, safe_meta), "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_session_image(config, session_name, filename):
    """
    (session_dir, safe_filename) 튜플을 반환한다. 유효하지 않으면 (None, None).
    라우트에서 flask.send_from_directory(session_dir, safe_filename)로 서빙한다.

    session_dir는 반드시 절대경로로 반환한다 - Flask의 send_from_directory는
    상대경로를 프로세스의 cwd가 아니라 앱의 root_path(webapp/) 기준으로 풀기
    때문에, config.yaml의 상대경로(data/collect 등)를 그대로 넘기면 엉뚱한
    위치를 찾다가 404가 난다.
    """
    collect_dir = config["paths"]["collect_dir"]
    safe_session = _safe_child_name(collect_dir, session_name)
    if safe_session is None:
        return None, None

    session_dir = os.path.abspath(os.path.join(collect_dir, safe_session))
    safe_filename = _safe_child_name(session_dir, filename)
    if safe_filename is None:
        return None, None

    return session_dir, safe_filename


# ---------------------------------------------------------------------------
# data/results/measure_<timestamp>.json (measure, --save-data 없이 실행)
# ---------------------------------------------------------------------------
def list_measure_summaries(config):
    """[{"timestamp": "20260101_120000", "finger_count": N}, ...] 최신순."""
    results_dir = config["paths"]["results_dir"]
    if not os.path.isdir(results_dir):
        return []

    names = [n for n in os.listdir(results_dir) if n.startswith("measure_") and n.endswith(".json")]
    names = _mtime_desc(results_dir, names)

    summaries = []
    for name in names:
        timestamp = name[len("measure_") : -len(".json")]
        try:
            with open(os.path.join(results_dir, name), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError):
            continue
        summaries.append({"timestamp": timestamp, "finger_count": len(data.get("fingers", {}))})
    return summaries


def load_measure_summary(config, timestamp):
    """measure_<timestamp>.json을 읽어 반환한다. 없으면 None."""
    results_dir = config["paths"]["results_dir"]
    filename = "measure_%s.json" % timestamp
    safe_filename = _safe_child_name(results_dir, filename)
    if safe_filename is None:
        return None

    with open(os.path.join(results_dir, safe_filename), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# data/results/<finger_id>.json (+ <finger_id>_overlay.jpg) - analyze 서브커맨드
# ---------------------------------------------------------------------------
_EXCLUDED_RESULT_PREFIXES = ("measure_",)
_EXCLUDED_RESULT_NAMES = ("calib_front.json", "calib_side.json")


def list_analyze_results(config):
    """[{"finger_id": ..., "has_overlay": bool}, ...] 최신순."""
    results_dir = config["paths"]["results_dir"]
    if not os.path.isdir(results_dir):
        return []

    candidates = []
    for name in os.listdir(results_dir):
        if not name.endswith(".json"):
            continue
        if name in _EXCLUDED_RESULT_NAMES:
            continue
        if name.startswith(_EXCLUDED_RESULT_PREFIXES):
            continue
        try:
            with open(os.path.join(results_dir, name), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, OSError):
            continue
        # analyze가 저장한 결과인지 스키마로 판별 (finger_id + result 키)
        if "finger_id" not in data or "result" not in data:
            continue
        candidates.append(name)

    candidates = _mtime_desc(results_dir, candidates)

    results = []
    for name in candidates:
        finger_id = name[: -len(".json")]
        overlay_name = "%s_overlay.jpg" % finger_id
        has_overlay = os.path.exists(os.path.join(results_dir, overlay_name))
        results.append({"finger_id": finger_id, "has_overlay": has_overlay})
    return results


def load_analyze_result(config, finger_id):
    """<finger_id>.json을 읽어 반환한다 (없으면 None). has_overlay 여부도 같이 담아준다."""
    results_dir = config["paths"]["results_dir"]
    filename = "%s.json" % finger_id
    safe_filename = _safe_child_name(results_dir, filename)
    if safe_filename is None:
        return None

    with open(os.path.join(results_dir, safe_filename), "r", encoding="utf-8") as f:
        data = json.load(f)

    overlay_name = "%s_overlay.jpg" % finger_id
    data["has_overlay"] = os.path.exists(os.path.join(results_dir, overlay_name))
    return data


def resolve_analyze_overlay(config, finger_id):
    """
    (results_dir, safe_overlay_filename) 튜플. 유효하지 않으면 (None, None).
    resolve_session_image과 같은 이유로 절대경로로 반환한다.
    """
    results_dir = os.path.abspath(config["paths"]["results_dir"])
    overlay_name = "%s_overlay.jpg" % finger_id
    safe_filename = _safe_child_name(results_dir, overlay_name)
    if safe_filename is None:
        return None, None
    return results_dir, safe_filename
