import json
import re
from pathlib import Path

from classification.nail_classification import SHAPE_PATH, contour_to_xy

SHAPE_CODE_RE = re.compile(r"_([A-Za-z]+)\([^)]+\)$")  # 키 끝에서 글자 코드 (SC)
STEM_RE = re.compile(r"^(\d+)_[LR]_f(\d+)$")  # 사람 ID·손가락 번호
FINGERS_DIR = Path(SHAPE_PATH).parent / "fingers"  # 라벨 JSON 폴더


# ===== 라벨·정답 불러오기 =====

with open(SHAPE_PATH, "r", encoding="utf-8") as f:
    _catalog_raw = json.load(f)
if not _catalog_raw:
    raise RuntimeError(f"No shape templates found in {SHAPE_PATH}")

# 분류 버튼이 비교 대상으로 쓰는 카탈로그: (키, 원본 좌표)
CATALOG_TEMPLATES = [(name, contour_to_xy(points)) for name, points in _catalog_raw.items()]


# 키에서 P/S
def extract_shape_code(template_name):
    match = SHAPE_CODE_RE.search(template_name)
    if match:
        return match.group(1)[0].upper()
    return template_name


# LabelMe에서 가장 큰 폴리곤
def points_from_labelme(data):
    shapes = data.get("shapes") or []
    polygons = [s for s in shapes if s.get("points") and len(s["points"]) >= 3]
    if not polygons:
        raise ValueError("JSON에 다각형 라벨이 없습니다")
    shape = max(polygons, key=lambda s: len(s["points"]))
    return contour_to_xy(shape["points"])


# fingers 윤곽과 정답 글자 로드
def load_finger_items():
    with open(SHAPE_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)
    letters = {SHAPE_CODE_RE.sub("", name): extract_shape_code(name) for name in templates}

    items = []
    skipped = []
    for path in sorted(FINGERS_DIR.glob("*.json")):
        stem = path.stem
        letter = letters.get(stem)
        if not letter:
            skipped.append(stem)
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items.append((stem, points_from_labelme(raw), letter))
    return items, skipped


# ===== METRIC 분기 =====

# 카탈로그에서 가장 가까운 템플릿
def nearest_template(query_points, templates, n_samples, region, metric):
    return metric.nearest_template(query_points, templates, n_samples, region)


# 한 장 빼고 같은 손가락끼리만 검증
def leave_one_out_fingers(n_samples, region, metric):
    items, skipped = load_finger_items()
    return metric.leave_one_out_fingers(items, skipped, n_samples, region)


# 맞춰가는 단계별 PNG와 최종 점수
def visualize_steps(query_pts, template_pts, n_samples, region, metric):
    return metric.visualize_steps(query_pts, template_pts, n_samples, region)
