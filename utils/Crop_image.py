import json
import os
from copy import deepcopy

from PIL import Image, ImageOps

PADDING_RATIO = 0.20
IMAGE_EXTS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS = [
    (
        os.path.join(SCRIPT_DIR, "TestDataset", "FullData_checkerboard"),
        os.path.join(SCRIPT_DIR, "TestDataset", "CropedData_checkerboard"),
    ),
    (
        os.path.join(SCRIPT_DIR, "TestDataset", "FullData_online"),
        os.path.join(SCRIPT_DIR, "TestDataset", "CropedData_online"),
    ),
]
SRC_DIR = JOBS[0][0]
DST_DIR = JOBS[0][1]

# 이미지 파일 찾기
def find_image_filename(src_dir, base_name):
    for ext in IMAGE_EXTS:
        filename = base_name + ext
        if os.path.exists(os.path.join(src_dir, filename)):
            return filename
    return None

# json라벨링 좌표 수집
def collect_polygon_points(shapes):
    points = []
    for shape in shapes:
        if shape.get("shape_type", "polygon") not in ("polygon", "linestrip"):
            continue
        for point in shape.get("points", []):
            if len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    return points

# json라벨링 + 원본이미지 크기를 받아 얼마나 자를지 정함
def padded_crop_box(points, image_width, image_height, pad_ratio=PADDING_RATIO):
    # 리스트 컴프리핸션 => points(라벨링 좌표)에서 각각 x와 y만을 뺀다음 4개의 변수에 각각 x의 최대최소 y의 최대최소 저장
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # 패딩계산 => 기존 타이트한 박스에서 설정된 패딩값만큼을 곱해 러프하게 잡음
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio

    #만약 int안의 값이 음수면 이미지 밖으로 나가버린단뜻이므로 0으로 잡음(예외처리)
    left = max(0, int(x1 - pad_x)) # 이게 왼쪽인데 최솟값에서 패딩값(남길박스의 20%)을 빼면 더 왼쪽가서 자른다는뜻이므로 러프하게 잘려짐 나머지 변수도 동일하고 방향만 잘 잡으면됨
    top = max(0, int(y1 - pad_y))
    right = min(image_width, int(round(x2 + pad_x)))
    bottom = min(image_height, int(round(y2 + pad_y)))

    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)

    return left, top, right, bottom

# 크롭 후 json 라벨링 좌표조정
def shift_shapes(shapes, left, top):
    shifted = deepcopy(shapes)
    for shape in shifted:
        shape["points"] = [
            [point[0] - left, point[1] - top] if len(point) >= 2 else point
            for point in shape.get("points", [])
        ]
    return shifted

# 크롭 후 이미지 저장
def save_image(image, dest_path):
    ext = os.path.splitext(dest_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(dest_path, quality=95, subsampling=0)
    else:
        image.save(dest_path)

# 한개 크롭
def crop_one(json_name):
    json_path = os.path.join(SRC_DIR, json_name)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_name = os.path.splitext(json_name)[0]
    image_filename = find_image_filename(SRC_DIR, base_name)
    if image_filename is None:
        return "skip", f"이미지 없음: {json_name}"

    image_path = os.path.join(SRC_DIR, image_filename)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened)
        image = image.copy()

    points = collect_polygon_points(data.get("shapes", []))
    if not points:
        return "skip", f"폴리곤 없음: {json_name}"

    left, top, right, bottom = padded_crop_box(points, image.width, image.height)
    cropped = image.crop((left, top, right, bottom))

    cropped_data = deepcopy(data)
    cropped_data["shapes"] = shift_shapes(data.get("shapes", []), left, top)
    cropped_data["imagePath"] = image_filename
    cropped_data["imageWidth"] = cropped.width
    cropped_data["imageHeight"] = cropped.height
    cropped_data["imageData"] = None

    os.makedirs(DST_DIR, exist_ok=True)
    save_image(cropped, os.path.join(DST_DIR, image_filename))
    with open(os.path.join(DST_DIR, json_name), "w", encoding="utf-8") as f:
        json.dump(cropped_data, f, ensure_ascii=False, indent=2)

    return "ok", (
        f"{json_name}: {image.width}x{image.height} -> "
        f"{cropped.width}x{cropped.height} (box {left},{top},{right},{bottom})"
    )


def run_job():
    if not os.path.isdir(SRC_DIR):
        raise FileNotFoundError(f"원본 폴더가 없습니다: {SRC_DIR}")

    os.makedirs(DST_DIR, exist_ok=True)

    json_names = sorted(
        name for name in os.listdir(SRC_DIR) if name.lower().endswith(".json")
    )
    ok_count = 0
    skip_count = 0

    print(f"원본: {SRC_DIR}")
    print(f"저장: {DST_DIR}")
    print(f"JSON {len(json_names)}개, 패딩 {int(PADDING_RATIO * 100)}%")
    print("원본은 읽기만 합니다.")

    for json_name in json_names:
        status, message = crop_one(json_name)
        if status == "ok":
            ok_count += 1
        else:
            skip_count += 1
            print(f"[SKIP] {message}")

        if ok_count <= 5 or ok_count % 50 == 0:
            print(f"[OK {ok_count}] {message}")

    print(f"완료: {ok_count}개 crop, {skip_count}개 skip")


def main():
    global SRC_DIR, DST_DIR
    for src_dir, dst_dir in JOBS:
        SRC_DIR = src_dir
        DST_DIR = dst_dir
        run_job()
        print()


if __name__ == "__main__":
    main()
