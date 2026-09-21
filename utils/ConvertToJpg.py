import os

from PIL import Image, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FOLDER_DIR = r"C:\Users\USER\Downloads\수민씨데이터_체커보드"

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".heic", ".heif",
}
JPEG_QUALITY = 95
HEIF_BRANDS = {b"heic", b"heif", b"mif1", b"msf1", b"heix", b"hevc"}


def register_heif():
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        return True
    except ImportError:
        return False


def make_target_dir(origin_dir):
    origin_dir = os.path.abspath(origin_dir)
    parent = os.path.dirname(origin_dir)
    return os.path.join(parent, os.path.basename(origin_dir) + "_to_jpg")


def sniff_is_image(path):
    try:
        with open(path, "rb") as f:
            head = f.read(32)
    except OSError:
        return False
    if len(head) < 12:
        return False
    if head.startswith(b"\xff\xd8\xff"):
        return True
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if head.startswith(b"BM"):
        return True
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return True
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return True
    if head[4:8] == b"ftyp" and head[8:12] in HEIF_BRANDS:
        return True
    return False


def is_image_file(path, name):
    ext = os.path.splitext(name)[1].lower()
    if ext in IMAGE_EXTS:
        return True
    if ext == "":
        return sniff_is_image(path)
    return False


def unique_jpg_path(dst_dir, stem):
    path = os.path.join(dst_dir, stem + ".jpg")
    if not os.path.exists(path):
        return path
    index = 2
    while True:
        path = os.path.join(dst_dir, f"{stem}_{index}.jpg")
        if not os.path.exists(path):
            return path
        index += 1


def image_to_rgb(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


def convert_one(src_path, dst_path):
    with Image.open(src_path) as opened:
        rgb = image_to_rgb(opened)
        rgb.save(dst_path, "JPEG", quality=JPEG_QUALITY)


def convert_tree(origin_dir, target_dir):
    count = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(origin_dir):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, origin_dir)
        dst_dir = target_dir if rel_dir == "." else os.path.join(target_dir, rel_dir)
        for name in sorted(filenames):
            src_path = os.path.join(dirpath, name)
            if not is_image_file(src_path, name):
                continue
            os.makedirs(dst_dir, exist_ok=True)
            stem = os.path.splitext(name)[0] or name
            dst_path = unique_jpg_path(dst_dir, stem)
            try:
                convert_one(src_path, dst_path)
            except Exception as exc:
                skipped += 1
                print(f"건너뜀 {os.path.relpath(src_path, origin_dir)}: {exc}")
                continue
            count += 1
            print(f"{os.path.relpath(src_path, origin_dir)} -> {os.path.relpath(dst_path, target_dir)}")
    return count, skipped


def main():
    if not ORIGIN_FOLDER_DIR:
        raise ValueError("ORIGIN_FOLDER_DIR를 지정하세요.")
    if not os.path.isdir(ORIGIN_FOLDER_DIR):
        raise FileNotFoundError(f"폴더가 없습니다: {ORIGIN_FOLDER_DIR}")

    heif_ok = register_heif()
    origin_dir = os.path.abspath(ORIGIN_FOLDER_DIR)
    target_dir = make_target_dir(origin_dir)
    if os.path.commonpath([origin_dir, target_dir]) == origin_dir:
        raise RuntimeError(f"출력 폴더가 원본 안에 있습니다: {target_dir}")

    os.makedirs(target_dir, exist_ok=True)
    total, skipped = convert_tree(origin_dir, target_dir)
    if total == 0:
        hint = ""
        if not heif_ok:
            hint = " HEIC는 pillow-heif 설치가 필요합니다: pip install pillow-heif"
        raise RuntimeError(f"이미지가 없습니다: {origin_dir}.{hint}")
    print(f"\n완료: {total}장 -> {target_dir}")
    if skipped:
        print(f"건너뜀: {skipped}장")
        if not heif_ok:
            print("HEIC를 쓰려면: pip install pillow-heif")


if __name__ == "__main__":
    main()
