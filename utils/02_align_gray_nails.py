# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
02_align_gray_nails.py
01_name_by_labelme_crop.py 실행 후 라벨에 따라 폴더로 나눠 저장, 폴더의 이미지(*_nail.jpg)를 그레이스케일로 바꾸고 방향을 맞춰 저장

처리 순서 (이미지마다):
  1) 그레이스케일 변환
  2) 측면(_side) 이미지 자동 방향 보정 (--side-mode)
       rot90  : 가로로 누운 경우(가로>세로)만 90° 회전  [기본값, 보간 없음]
       deskew : 기울기까지 보정 (폭이 가장 좁아지는 각도로 회전)
       none   : 자동 보정 안 함
  3) 수동 보정 목록(--fix CSV)에 있으면 추가 회전
  4) 검은 배경 여백을 다시 잘라냄
  * 정면(_front) 이미지는 자동 회전하지 않음 (손톱이 정사각형에 가까워 방향 판단이 불확실)
    -> 가로로 넓은 정면은 [확인 필요]로 표시만 하므로, 눈으로 보고 --fix 로 고친다.
  * 180° 방향(손톱 끝이 위/아래)은 자동으로 알 수 없으므로 --fix 로 고친다.

수동 보정 CSV 예 (fix.csv, 시계방향 각도 90/180/270):
    file,rotate_cw
    P/036_L_01_front_nail.jpg,90
    036_R_01_side_nail.jpg,180        <- 파일명만 적으면 모든 하위폴더에서 일치하는 파일에 적용

사용법:
    python align_gray_nails.py --src "C:\\...\\06_result" --dst "C:\\...\\06_aligned"
    python align_gray_nails.py --src ... --dst ... --fix fix.csv --side-mode deskew

출력: 입력과 같은 하위 폴더 구조 + 같은 파일명(학습 코드의 _front_nail/_side_nail 규칙 유지)
      dst/orientation_log.csv 에 파일별 처리 내역 기록
"""
import argparse
import csv
from pathlib import Path

from PIL import Image

# ---- 기본 경로 (필요 시 수정) ----
DEFAULT_SRC = r"C:\Users\user\Desktop\mirocore\nail_data\07_PS"
DEFAULT_DST = r"C:\Users\user\Desktop\mirocore\nail_data\07_PS_aligned"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BG_THRESHOLD = 15  # 이 값 이하 픽셀은 배경(검정)으로 간주 (JPEG 경계 노이즈 대응)

# Pillow 버전 호환 (9.1+ 는 Image.Transpose, 구버전은 Image.ROTATE_*)
_T = getattr(Image, "Transpose", Image)
ROT_CW = {90: _T.ROTATE_270, 180: _T.ROTATE_180, 270: _T.ROTATE_90}


def foreground(img: Image.Image) -> Image.Image:
    """손톱 영역(배경보다 밝은 픽셀) 이진 마스크."""
    return img.point(lambda p: 255 if p > BG_THRESHOLD else 0)


def crop_to_content(img: Image.Image) -> Image.Image:
    bbox = foreground(img).getbbox()
    return img.crop(bbox) if bbox else img


def best_vertical_angle(img: Image.Image) -> int:
    """회전 후 폭이 가장 좁아지는 각도(반시계, -89~90도)를 찾는다 = 긴 축을 세로로."""
    mask = foreground(img)
    best, best_w = 0, None
    for a in sorted(range(-89, 91), key=abs):  # 동률이면 회전량이 작은 쪽 우선
        bbox = mask.rotate(a, resample=Image.NEAREST, expand=True).getbbox()
        if bbox is None:
            continue
        w = bbox[2] - bbox[0]
        if best_w is None or w < best_w:
            best, best_w = a, w
    return best


def load_fixes(path):
    """수동 보정 CSV -> {상대경로 또는 파일명: 시계방향 각도}"""
    fixes = {}
    if not path:
        return fixes
    with open(path, encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            key = row["file"].strip().replace("\\", "/")
            ang = int(row["rotate_cw"]) % 360
            if ang not in (0, 90, 180, 270):
                raise ValueError(f"rotate_cw 는 0/90/180/270 만 가능: {row}")
            fixes[key] = ang
    return fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="mask_nail_batch.py 결과 폴더")
    ap.add_argument("--dst", default=DEFAULT_DST, help="저장 폴더 (src와 달라야 함)")
    ap.add_argument("--nail-suffix", default="_nail", help="처리할 파일명 끝 (예: _nail)")
    ap.add_argument("--front-tag", default="_front")
    ap.add_argument("--side-tag", default="_side")
    ap.add_argument("--side-mode", choices=["rot90", "deskew", "none"], default="rot90")
    ap.add_argument("--side-rot-dir", choices=["cw", "ccw"], default="cw",
                    help="rot90 모드에서 누운 측면을 돌리는 방향")
    ap.add_argument("--front-warn-ratio", type=float, default=1.2,
                    help="정면의 가로/세로가 이 값보다 크면 [확인 필요] 표시")
    ap.add_argument("--fix", default=None, help="수동 보정 CSV (file,rotate_cw)")
    ap.add_argument("--format", choices=["keep", "png"], default="keep")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {src}")
    if src == dst or src in dst.parents:
        raise ValueError("출력 폴더는 입력 폴더와 같거나 그 안에 있으면 안 됩니다.")
    fixes = load_fixes(args.fix)
    used_fixes = set()

    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS
                   and p.stem.endswith(args.nail_suffix))  # *_mask.jpg 등은 제외
    print(f"대상 이미지 {len(files)}장: {src}\n")

    log, warns = [], 0
    for f in files:
        rel = f.relative_to(src)
        rel_key = rel.as_posix()
        view = ("side" if args.side_tag in f.stem else
                "front" if args.front_tag in f.stem else "unknown")
        try:
            with Image.open(f) as im:
                img = im.convert("L")
            w0, h0 = img.size
            auto = "none"

            if view == "side" and args.side_mode == "rot90" and w0 > h0:
                img = img.transpose(ROT_CW[90] if args.side_rot_dir == "cw" else ROT_CW[270])
                auto = f"rot90_{args.side_rot_dir}"
            elif view == "side" and args.side_mode == "deskew":
                a = best_vertical_angle(img)
                if a != 0:
                    img = img.rotate(a, resample=Image.BICUBIC, expand=True, fillcolor=0)
                    auto = f"deskew_{a:+d}deg(ccw)"

            manual = fixes.get(rel_key, fixes.get(f.name, 0))
            if rel_key in fixes or f.name in fixes:
                used_fixes.add(rel_key if rel_key in fixes else f.name)
            if manual:
                img = img.transpose(ROT_CW[manual])

            img = crop_to_content(img)
            w1, h1 = img.size

            flag = ""
            if view == "front" and not manual and w1 / h1 > args.front_warn_ratio:
                flag = "확인 필요: 정면이 가로로 넓음"
            if view == "side" and w1 > h1:
                flag = "확인 필요: 측면이 여전히 가로로 넓음"
            if view == "unknown":
                flag = "확인 필요: front/side 태그 없음"
            if flag:
                warns += 1
                print(f"[{flag}] {rel_key}  {w1}x{h1}")

            out = dst / rel
            if args.format == "png":
                out = out.with_suffix(".png")
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix.lower() in {".jpg", ".jpeg"}:
                img.save(out, quality=args.quality)
            else:
                img.save(out)
            log.append(dict(file=rel_key, view=view, size_in=f"{w0}x{h0}", auto=auto,
                            manual_cw=manual, size_out=f"{w1}x{h1}", flag=flag))
        except Exception as e:  # 한 장 실패해도 계속
            print(f"[실패] {rel_key}: {e}")
            log.append(dict(file=rel_key, view=view, size_in="", auto="", manual_cw="",
                            size_out="", flag=f"실패: {e}"))

    for k in sorted(set(fixes) - used_fixes):
        print(f"[경고] 수동 보정 목록의 파일을 찾지 못함: {k}")

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "orientation_log.csv", "w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.DictWriter(fp, fieldnames=["file", "view", "size_in", "auto", "manual_cw",
                                            "size_out", "flag"])
        wr.writeheader()
        wr.writerows(log)
    n_auto = sum(1 for r in log if r["auto"] not in ("none", ""))
    print(f"\n완료: {len(log)}장 저장, 자동 회전 {n_auto}장, 확인 필요 {warns}장")
    print(f"저장 위치: {dst}\n처리 기록: {dst / 'orientation_log.csv'}")


if __name__ == "__main__":
    main()