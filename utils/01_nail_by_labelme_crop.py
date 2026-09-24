# -*- coding: utf-8 -*-
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
폴더 내 모든 이미지(하위 폴더 포함)를 LabelMe JSON으로 처리한다.

이미지마다 같은 폴더의 같은 이름 .json을 읽어서
  <이름>_mask.jpg : 원본 위에 마스크 경계를 선으로 그린 이미지
  <이름>_nail.jpg : 마스크 영역만 잘라낸 이미지
를 만들고, 입력 폴더의 하위 구조 그대로 출력 폴더에 저장한다.

사용법:
  python mask_nail_batch.py <입력_폴더>
      -> 결과: <입력_폴더>_result/ (입력 폴더 옆에 생성)
  python mask_nail_batch.py <입력_폴더> <출력_폴더>

예) data/A/x.jpg, data/A/x.json, data/B/C/y.jpg, data/B/C/y.json
    -> data_result/A/x_mask.jpg, x_nail.jpg
       data_result/B/C/y_mask.jpg, y_nail.jpg

필요 패키지: pip install pillow
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LINE_COLOR = "lime"   # 외곽선 색
LINE_WIDTH = 6        # 외곽선 두께(px)
BG_COLOR = "black"    # _nail.jpg 의 영역 밖 배경색
JPG_QUALITY = 95


def load_image(img_path: Path, json_size: tuple) -> Image.Image:
    """JSON의 (width, height)와 크기가 맞는 이미지를 반환 (EXIF 회전 보정 포함)."""
    raw = Image.open(img_path)
    for cand in (ImageOps.exif_transpose(raw), raw):
        if cand.size == json_size:
            return cand.convert("RGB")
    raise ValueError(f"이미지 크기 {raw.size}가 JSON 크기 {json_size}와 맞지 않음")


def process_one(img_path: Path, json_path: Path, out_dir: Path) -> tuple:
    """이미지 한 장 처리. 저장한 nail 이미지 크기를 반환."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img = load_image(img_path, (data["imageWidth"], data["imageHeight"]))

    polygons = [
        [tuple(p) for p in s["points"]]
        for s in data.get("shapes", [])
        if s.get("shape_type", "polygon") == "polygon" and len(s["points"]) >= 3
    ]
    if not polygons:
        raise ValueError("JSON에 polygon이 없음")

    # 마스크 생성 (polygon 내부 = 255)
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for pts in polygons:
        md.polygon(pts, fill=255)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    # 1) 원본 위에 선으로 표시 -> <이름>_mask.jpg
    outlined = img.copy()
    od = ImageDraw.Draw(outlined)
    for pts in polygons:
        od.line(pts + [pts[0]], fill=LINE_COLOR, width=LINE_WIDTH, joint="curve")
    outlined.save(out_dir / f"{stem}_mask.jpg", quality=JPG_QUALITY)

    # 2) 마스크 영역만 추출 -> <이름>_nail.jpg
    nail = Image.composite(img, Image.new("RGB", img.size, BG_COLOR), mask)
    nail = nail.crop(mask.getbbox())
    nail.save(out_dir / f"{stem}_nail.jpg", quality=JPG_QUALITY)
    return nail.size


def main():

    in_root = Path(r"C:\Users\user\Desktop\mirocore\nail_data\06").resolve()
    if not in_root.is_dir():
        sys.exit(f"폴더가 아닙니다: {in_root}")
    out_root = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) == 3
        else in_root.with_name(in_root.name + "_result")
    )

    # 하위 폴더까지 모든 이미지 검색 (출력 폴더가 입력 안에 있으면 제외)
    images = sorted(
        p for p in in_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
        and out_root not in p.parents
    )
    print(f"입력: {in_root}\n출력: {out_root}\n이미지 {len(images)}장 발견\n")

    ok, skipped, failed = 0, [], []
    for img_path in images:
        rel = img_path.relative_to(in_root)
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            skipped.append(rel)
            print(f"[건너뜀] JSON 없음: {rel}")
            continue
        try:
            size = process_one(img_path, json_path, out_root / rel.parent)
            ok += 1
            print(f"[완료] {rel}  -> nail {size[0]}x{size[1]}")
        except Exception as e:  # 한 장이 실패해도 계속 진행
            failed.append((rel, str(e)))
            print(f"[실패] {rel}: {e}")

    print(f"\n성공 {ok} / JSON 없음 {len(skipped)} / 실패 {len(failed)}")

if __name__ == "__main__":
    main()