#!/usr/bin/env python3
"""
seg_gui.py — 손톱 세그멘테이션 GUI
====================================
[역할]
    위에서(Top) / 측면(Side) 사진을 한 장씩 불러오면, 이미지에서
    인식된 손톱을 전부 자동 세그멘테이션해서 오른쪽에 미리보기로
    보여주고, 실측값이 표기된 세그멘테이션 이미지를 results/ 폴더에
    저장한다. (손 전체 사진이면 손톱 5개가 전부, 손가락 1개 사진이면
    1개가 표시된다. 여러 개일 때는 사진 왼쪽부터 1, 2, ... 번호를 붙인다)

    사진 안에 ChArUco 체커보드가 함께 찍혀 있으면 보드를 자동 인식해서
    픽셀 <-> mm 호모그래피를 계산하고, 손톱의 길이/폭을 mm 단위로
    실측해 이미지 위에 표시하고 measurements.json으로도 저장한다.
    (보드가 없으면 픽셀 단위로만 표시)

[사용법]
    cd segmentation
    KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python seg_gui.py

    1. [위에서(Top) 이미지 열기] / [측면(Side) 이미지 열기]로 사진 선택
       (한 장만 선택해도 동작)
    2. [세그멘테이션 실행] 클릭
    3. 오른쪽 미리보기 확인 → results/<날짜시간>/ 폴더에 자동 저장됨

[백엔드]
    기본은 YOLOv8-seg (backends/dl_yolo.py, 최초 실행 시 모델 자동 다운로드).
    ultralytics가 없거나 로드에 실패하면 classical(피부색 기반) 백엔드로
    자동 폴백한다 (정확도 낮음, 참고용).

[오류가 발생하면 확인할 것]
    - "ultralytics 패키지가 설치되어 있지 않습니다": pip3 install -r requirements.txt
    - macOS에서 OpenMP 충돌로 죽는다면: KMP_DUPLICATE_LIB_OK=TRUE 붙여서 실행
    - 체커보드가 인식되지 않는다: 보드 전체가 사진에 나오는지, 반사/초점 확인.
      보드 스펙이 다르면 charuco_calibration.py 상단의 SQUARES_X/Y, SQUARE_MM 수정.
"""

import base64
import os
import threading
import traceback
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

from utils import ensure_dir, save_json
from backends.measure_from_mask import (
    measure_nail_from_mask,
    find_vertical_axis_endpoints,
    find_horizontal_width_endpoints,
)
import charuco_calibration

# 색상 팔레트 — curvature/curvature.py와 동일한 디자인 언어
BG        = '#f5f5f7'
SURFACE   = '#ffffff'
BORDER    = '#e8e8ec'
BORDER2   = '#d4d4da'
TEXT      = '#1d1d1f'
MUTED     = '#86868b'
ACCENT    = '#2f6fed'
ACCENT_DK = '#1d4fc4'
ACCENT_BG = '#eef3fe'
RED       = '#d64545'
GREEN     = '#3d9960'

FONT = 'Helvetica'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DEFAULT_CONF = 0.25
PREVIEW_MAX_W = 470
PREVIEW_MAX_H = 560

KIND_TITLES = {"top": "위에서 (Top)", "side": "측면 (Side)"}


# ── 세그멘테이션 파이프라인 (GUI와 분리된 순수 로직) ─────────────────────────
def load_backend(conf):
    """YOLO 백엔드를 먼저 시도하고, 실패하면 classical로 폴백한다."""
    try:
        from backends.dl_yolo import YoloNailBackend
        return YoloNailBackend(conf=conf), None
    except RuntimeError as e:
        from backends.classical import ClassicalNailBackend
        return ClassicalNailBackend(), str(e)


def try_charuco(image):
    """
    사진 속 ChArUco 보드를 자동 인식해서 (homography, rms_mm)를 반환한다.
    보드가 없거나 인식에 실패하면 (None, None)을 반환한다.
    """
    args = SimpleNamespace(
        squares_x=charuco_calibration.SQUARES_X,
        squares_y=charuco_calibration.SQUARES_Y,
        square_mm=charuco_calibration.SQUARE_MM,
        marker_mm=charuco_calibration.MARKER_MM,
        dict=charuco_calibration.DICT_NAME,
    )
    try:
        result = charuco_calibration.calibrate(image, args)
        return np.asarray(result["homography"]), result["rms_reprojection_mm"]
    except (RuntimeError, ValueError) as e:
        print(f"[INFO] 체커보드 인식 실패: {e}")
        return None, None


def sort_masks_by_x(nail_masks):
    """검출된 마스크들을 사진 왼쪽부터 순서대로(중심 x좌표 기준) 정렬한다."""
    def center_x(nm):
        ys, xs = np.nonzero(nm.mask)
        return float(xs.mean()) if len(xs) else 0.0
    return sorted(nail_masks, key=center_x)


def measure_pixels(mask):
    """보드가 없을 때: 세로 길이/중간 폭을 픽셀 단위로만 계산한다."""
    endpoints = find_vertical_axis_endpoints(mask)
    if endpoints is None:
        return None
    p1, p2 = endpoints
    y_mid = (float(p1[1]) + float(p2[1])) / 2.0
    width_endpoints = find_horizontal_width_endpoints(mask, y_mid)
    result = {
        "point_a": (float(p1[0]), float(p1[1])),
        "point_b": (float(p2[0]), float(p2[1])),
        "pixel_distance": float(np.linalg.norm(p2 - p1)),
        "length_mm": None,
        "width_point_a": None,
        "width_point_b": None,
        "width_pixel_distance": None,
        "width_mm": None,
    }
    if width_endpoints is not None:
        w1, w2 = width_endpoints
        result["width_point_a"] = (float(w1[0]), float(w1[1]))
        result["width_point_b"] = (float(w2[0]), float(w2[1]))
        result["width_pixel_distance"] = float(np.linalg.norm(w2 - w1))
    return result


def format_measure(m):
    """측정 결과를 'L 12.34 W 8.21 mm' 또는 'L 340 W 120 px' 형태로 요약한다."""
    if m is None:
        return "측정 실패"
    if m["length_mm"] is not None:
        w = f"  W {m['width_mm']:.2f}" if m["width_mm"] else ""
        return f"L {m['length_mm']:.2f}{w} mm"
    w = f"  W {m['width_pixel_distance']:.0f}" if m["width_pixel_distance"] else ""
    return f"L {m['pixel_distance']:.0f}{w} px"


def build_overlay(image, nail_masks, measures):
    """
    기존 measure_auto.py 디버그 이미지와 같은 스타일로, 검출된 손톱을
    전부 그린다: 초록 윤곽선 + 빨간 세로 길이선 + 파란 폭선 + 노란 mm 텍스트.
    손톱이 2개 이상이면 왼쪽부터 "1:", "2:" 번호를 붙인다.
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    scale = max(1.0, max(h, w) / 1920.0)
    font_scale = 0.8 * scale
    thickness = max(2, int(round(2 * scale)))

    if not nail_masks:
        cv2.putText(overlay, "no nail detected", (10, int(50 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                    thickness, cv2.LINE_AA)
        return overlay

    for i, (nail_mask, m) in enumerate(zip(nail_masks, measures)):
        # 초록 윤곽선 (색 채우기 없음)
        contours, _ = cv2.findContours(nail_mask.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness)

        if m is None:
            continue

        # 빨간 세로 길이선
        pa = tuple(int(round(v)) for v in m["point_a"])
        pb = tuple(int(round(v)) for v in m["point_b"])
        cv2.line(overlay, pa, pb, (0, 0, 255), thickness)
        # 파란 폭선
        if m["width_point_a"] is not None:
            wa = tuple(int(round(v)) for v in m["width_point_a"])
            wb = tuple(int(round(v)) for v in m["width_point_b"])
            cv2.line(overlay, wa, wb, (255, 0, 0), thickness)

        # 노란 텍스트 — 손톱 위쪽에 표기 (예: "1:10.4mm w5.0mm")
        if m["length_mm"] is not None:
            text = f"{m['length_mm']:.1f}mm"
            if m["width_mm"]:
                text += f" w{m['width_mm']:.1f}mm"
        else:
            text = f"{m['pixel_distance']:.0f}px"
            if m["width_pixel_distance"]:
                text += f" w{m['width_pixel_distance']:.0f}px"
        if len(nail_masks) > 1:
            text = f"{i + 1}:{text}"

        ys, xs = np.nonzero(nail_mask.mask)
        tx = max(0, int(xs.mean()) - int(60 * scale))
        ty = max(int(30 * scale), int(ys.min()) - int(12 * scale))
        cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return overlay


def process_image(image, backend, use_charuco):
    """
    이미지 한 장 처리: 세그멘테이션(검출된 손톱 전부) → (선택) 체커보드 실측
    → 오버레이. 손 전체 사진이면 손톱 5개가 전부 표시된다.
    반환: dict(overlay, measures, rms_mm, homography_found, detected, count)
    """
    nail_masks = sort_masks_by_x(backend.segment(image))

    homography, rms_mm = (None, None)
    if use_charuco and nail_masks:
        homography, rms_mm = try_charuco(image)

    measures = []
    for nm in nail_masks:
        if homography is not None:
            # finger를 지정하면 기존 measure_auto.py와 동일한 세로(수직) 측정 방식을 쓴다
            measures.append(measure_nail_from_mask(nm.mask, homography, finger="nail"))
        else:
            measures.append(measure_pixels(nm.mask))

    return {
        "overlay": build_overlay(image, nail_masks, measures),
        "measures": measures,
        "rms_mm": rms_mm,
        "homography_found": homography is not None,
        "detected": bool(nail_masks),
        "count": len(nail_masks),
    }


# ── 커스텀 위젯 ───────────────────────────────────────────────────────────────
class FlatButton(tk.Label):
    """macOS에서도 색이 먹는 플랫 버튼 (hover 효과 포함)"""

    def __init__(self, master, text, command, bg, fg, hover_bg,
                 font=(FONT, 11, 'bold'), padx=12, pady=6, **kw):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font,
                         padx=padx, pady=pady, cursor='hand2', **kw)
        self._bg, self._hover = bg, hover_bg
        self._command = command
        self.enabled = True
        self.bind('<Enter>', lambda e: self.enabled and self.config(bg=self._hover))
        self.bind('<Leave>', lambda e: self.config(bg=self._bg))
        self.bind('<Button-1>', lambda e: self.enabled and self._command())

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.config(fg='white' if enabled else '#b6c6ee')


class ImagePicker(tk.Frame):
    """이미지 파일을 한 장 고르는 [버튼 + 선택된 파일명] 위젯"""

    FILETYPES = [("이미지", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("모든 파일", "*.*")]

    def __init__(self, master, label, on_pick):
        super().__init__(master, bg=BG)
        self.path = None
        self.on_pick = on_pick

        FlatButton(self, f"{label} 열기", self._pick,
                   bg=SURFACE, fg=TEXT, hover_bg=ACCENT_BG,
                   font=(FONT, 11, 'bold'), padx=12, pady=8,
                   highlightthickness=1, highlightbackground=BORDER2
                   ).pack(fill=tk.X)
        self.name_lbl = tk.Label(self, text="선택된 파일 없음", bg=BG, fg=MUTED,
                                 font=(FONT, 9), anchor='w')
        self.name_lbl.pack(fill=tk.X, pady=(3, 0))

    def _pick(self):
        path = filedialog.askopenfilename(filetypes=self.FILETYPES)
        if not path:
            return
        self.path = path
        self.name_lbl.config(text=os.path.basename(path), fg=TEXT)
        self.on_pick()


# ── GUI ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("손톱 세그멘테이션 — Top / Side")
        root.configure(bg=BG)
        self.backend = None
        self.backend_conf = None
        self._running = False
        self._photos = {}   # PhotoImage 참조 유지 (GC 방지)
        self._build_ui()

    # ── 왼쪽 패널 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        left = tk.Frame(self.root, bg=BG, padx=22, pady=20, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="손톱 세그멘테이션", font=(FONT, 18, 'bold'),
                 bg=BG, fg=TEXT, anchor='w').pack(fill=tk.X)
        tk.Label(left, text="Top/Side 각 1장 → 인식된 손톱 전부 자동 실측",
                 font=(FONT, 11), bg=BG, fg=MUTED, anchor='w'
                 ).pack(fill=tk.X, pady=(2, 14))

        tk.Label(left, text="입력 이미지", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 6))
        self.top_picker = ImagePicker(left, "위에서(Top) 이미지", self._clear_status)
        self.top_picker.pack(fill=tk.X, pady=(0, 8))
        self.side_picker = ImagePicker(left, "측면(Side) 이미지", self._clear_status)
        self.side_picker.pack(fill=tk.X, pady=(0, 12))

        tk.Label(left, text="옵션", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 6))

        opt = tk.Frame(left, bg=BG)
        opt.pack(fill=tk.X)
        tk.Label(opt, text="검출 신뢰도(conf)", bg=BG, fg=TEXT,
                 font=(FONT, 10)).pack(side=tk.LEFT)
        self.conf_var = tk.StringVar(value=f"{DEFAULT_CONF:g}")
        tk.Entry(opt, textvariable=self.conf_var, width=6, justify='center',
                 font=(FONT, 11), bg=SURFACE, fg=TEXT, relief='flat',
                 highlightthickness=1, highlightbackground=BORDER2,
                 highlightcolor=ACCENT).pack(side=tk.RIGHT, ipady=3)

        self.charuco_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="체커보드(ChArUco) 인식해서 mm 실측",
                       variable=self.charuco_var, bg=BG, fg=TEXT,
                       activebackground=BG, font=(FONT, 10), anchor='w'
                       ).pack(fill=tk.X, pady=(6, 12))

        self.run_btn = FlatButton(left, "세그멘테이션 실행", self._run,
                                  bg=ACCENT, fg='white', hover_bg=ACCENT_DK,
                                  font=(FONT, 12, 'bold'), padx=12, pady=10)
        self.run_btn.pack(fill=tk.X)

        self.status = tk.Label(left, text="이미지를 선택하고 실행을 누르세요",
                               bg=BG, fg=MUTED, font=(FONT, 10),
                               anchor='w', wraplength=270, justify='left')
        self.status.pack(fill=tk.X, pady=(8, 10))

        tk.Label(left, text="실측 결과", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 6))
        card = tk.Frame(left, bg=SURFACE, padx=12, pady=10,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill=tk.BOTH, expand=True)
        self.result_lbl = tk.Label(card, text="—", bg=SURFACE, fg=TEXT,
                                   font=('Courier', 11), justify='left', anchor='nw')
        self.result_lbl.pack(fill=tk.BOTH, expand=True)

        # ── 오른쪽 미리보기 패널: 위/측면 2칸 ───────────────────────────────
        right = tk.Frame(self.root, bg=BG, padx=10, pady=14)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.previews = {}
        for kind in ("top", "side"):
            col = tk.Frame(right, bg=BG)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
            tk.Label(col, text=KIND_TITLES[kind], bg=BG, fg=TEXT,
                     font=(FONT, 12, 'bold')).pack(pady=(0, 6))
            frame = tk.Frame(col, bg=SURFACE, highlightthickness=1,
                             highlightbackground=BORDER)
            frame.pack(fill=tk.BOTH, expand=True)
            lbl = tk.Label(frame, text="미리보기 없음", bg=SURFACE, fg=MUTED,
                           font=(FONT, 11))
            lbl.pack(fill=tk.BOTH, expand=True)
            sub = tk.Label(col, text="", bg=BG, fg=MUTED, font=(FONT, 9))
            sub.pack(pady=(4, 0))
            self.previews[kind] = (lbl, sub)

    # ── 동작 ─────────────────────────────────────────────────────────────────
    def _clear_status(self):
        if not self._running:
            self.status.config(text="준비 완료 — [세그멘테이션 실행]을 누르세요", fg=MUTED)

    def _set_status(self, text, color=MUTED):
        self.root.after(0, lambda: self.status.config(text=text, fg=color))

    def _run(self):
        if self._running:
            return
        inputs = {}
        for kind, picker in (("top", self.top_picker), ("side", self.side_picker)):
            if picker.path:
                inputs[kind] = picker.path
        if not inputs:
            self.status.config(text="⚠ 이미지를 최소 한 장 선택하세요", fg=RED)
            return
        try:
            conf = float(self.conf_var.get())
            assert 0.0 < conf < 1.0
        except (ValueError, AssertionError):
            self.status.config(text="⚠ conf는 0과 1 사이 숫자여야 합니다 (예: 0.25)", fg=RED)
            return

        self._running = True
        self.run_btn.set_enabled(False)
        self.status.config(text="처리 중...", fg=MUTED)
        threading.Thread(
            target=self._worker,
            args=(inputs, conf, self.charuco_var.get()),
            daemon=True,
        ).start()

    def _worker(self, inputs, conf, use_charuco):
        try:
            if self.backend is None or self.backend_conf != conf:
                self._set_status("세그멘테이션 백엔드 로딩 중...")
                self.backend, backend_warn = load_backend(conf)
                self.backend_conf = conf
                if backend_warn:
                    print(f"[WARN] YOLO 로드 실패, classical 폴백:\n{backend_warn}")

            session_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
            results, measurements_json, saved = {}, {}, []
            for kind, path in inputs.items():
                self._set_status(f"{'위에서' if kind == 'top' else '측면'} 이미지 처리 중...")
                image = cv2.imread(path)
                if image is None:
                    raise RuntimeError(f"이미지를 열 수 없습니다: {path}")
                res = process_image(image, self.backend, use_charuco)
                results[kind] = res

                out_path = os.path.join(session_dir, f"{kind}.png")
                ensure_dir(out_path)
                cv2.imwrite(out_path, res["overlay"])
                saved.append(out_path)

                nails = []
                for idx, m in enumerate(res["measures"]):
                    if m is None:
                        continue
                    nails.append({
                        "index": idx + 1,
                        "length_mm": round(m["length_mm"], 2) if m["length_mm"] else None,
                        "width_mm": round(m["width_mm"], 2) if m["width_mm"] else None,
                        "length_px": round(m["pixel_distance"], 1),
                    })
                measurements_json[kind] = {
                    "image_path": path,
                    "detected_count": res["count"],
                    "board_rms_mm": round(res["rms_mm"], 4) if res["rms_mm"] else None,
                    "nails": nails,
                }

            if measurements_json:
                json_path = os.path.join(session_dir, "measurements.json")
                save_json(json_path, measurements_json)

            self.root.after(0, lambda: self._show_results(results, session_dir, saved))
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            self._set_status(f"⚠ 오류: {msg[:300]}", RED)
            self.root.after(0, self._done)

    def _show_results(self, results, session_dir, saved):
        for kind, (lbl, sub) in self.previews.items():
            res = results.get(kind)
            if res is None:
                continue
            photo = self._to_photo(res["overlay"])
            self._photos[kind] = photo
            lbl.config(image=photo, text="")
            if not res["detected"]:
                sub.config(text="검출 실패", fg=RED)
            elif res["homography_found"]:
                sub.config(text=f"손톱 {res['count']}개 · 보드 RMS {res['rms_mm']:.2f}mm",
                           fg=TEXT)
            else:
                sub.config(text=f"손톱 {res['count']}개 · 보드 미검출", fg=MUTED)

        self.result_lbl.config(text=self._format_measurements(results))
        rel = os.path.relpath(session_dir, os.path.dirname(RESULTS_DIR))
        self.status.config(text=f"✓ 완료 — {rel}/ 에 {len(saved)}개 이미지 저장됨", fg=GREEN)
        self._done()

    @staticmethod
    def _format_measurements(results):
        lines = []
        for kind, title in (("top", "[위에서]"), ("side", "[측면]")):
            res = results.get(kind)
            if res is None:
                continue
            lines.append(title)
            if not res["detected"]:
                lines.append(" 검출 실패")
            for idx, m in enumerate(res["measures"]):
                lines.append(f" {idx + 1}: {format_measure(m)}")
            lines.append("")
        return "\n".join(lines).strip() or "—"

    def _done(self):
        self._running = False
        self.run_btn.set_enabled(True)

    @staticmethod
    def _to_photo(image_bgr):
        """OpenCV BGR 이미지를 미리보기 크기로 줄여 tk.PhotoImage로 변환한다."""
        h, w = image_bgr.shape[:2]
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        resized = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", resized)
        if not ok:
            raise RuntimeError("미리보기 이미지 인코딩에 실패했습니다")
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1320x680")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
