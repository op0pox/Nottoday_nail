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

[전체 처리 흐름 — 이 순서대로 코드를 읽으면 이해하기 쉽다]
    1. 사용자가 [이미지 열기] 버튼으로 사진 선택      -> ImagePicker 클래스
    2. [세그멘테이션 실행] 클릭                        -> App._run()
    3. 백그라운드 스레드에서 이미지 처리               -> App._worker()
       (스레드를 쓰는 이유: YOLO 추론이 몇 초 걸리는데
        메인 스레드에서 돌리면 GUI 창이 그동안 멈춰버린다)
    4. 이미지 한 장 처리                               -> process_image()
       4-1. YOLO로 손톱 마스크 검출                    -> backends/dl_yolo.py
       4-2. 체커보드 인식해서 픽셀<->mm 변환행렬 계산   -> try_charuco()
       4-3. 마스크마다 길이/폭 측정                    -> backends/measure_from_mask.py
       4-4. 원본 위에 윤곽선/측정선/mm 텍스트 그리기    -> build_overlay()
    5. 결과 이미지를 results/에 저장                   -> App._worker() 뒷부분
    6. 미리보기 갱신 (메인 스레드로 돌아와서)          -> App._show_results()

[사용법]
    cd segmentation
    KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python seg_gui.py

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

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
import base64        # 이미지를 문자열로 인코딩 (tkinter PhotoImage에 넣기 위해)
import os            # 경로 조작, 폴더 생성
import threading     # 무거운 작업(YOLO 추론)을 GUI와 분리해서 돌리기 위한 스레드
import traceback     # 오류 발생 시 상세한 위치를 콘솔에 출력
from datetime import datetime          # 결과 폴더 이름에 쓸 날짜/시간
from types import SimpleNamespace      # 간단한 "속성 묶음" 객체 (charuco 설정 전달용)

# ── 외부 라이브러리 ──────────────────────────────────────────────────────────
import cv2                             # OpenCV: 이미지 읽기/그리기/변환 전부 담당
import numpy as np                     # 배열 연산 (마스크는 전부 numpy 배열이다)
import tkinter as tk                   # 파이썬 기본 GUI 라이브러리
from tkinter import filedialog         # "파일 열기" 대화상자

# ── 프로젝트 내부 모듈 ───────────────────────────────────────────────────────
from utils import ensure_dir, save_json                    # 폴더 생성, JSON 저장
from backends.measure_from_mask import (
    measure_nail_from_mask,        # 마스크 + 호모그래피 -> mm 길이/폭 계산
    find_vertical_axis_endpoints,  # 마스크의 세로 방향 양 끝점(뿌리/끝) 찾기
    find_horizontal_width_endpoints,  # 특정 높이에서 가로 폭의 양 끝점 찾기
)
import charuco_calibration             # 체커보드 인식 + 호모그래피 계산

# ── 색상 팔레트 (GUI 디자인용) ───────────────────────────────────────────────
# curvature/curvature.py와 동일한 디자인 언어를 쓴다.
# tkinter는 '#RRGGBB' 형식의 16진수 색상 문자열을 받는다.
BG        = '#f5f5f7'   # 창 전체 배경 (연한 회색)
SURFACE   = '#ffffff'   # 카드/입력창 배경 (흰색)
BORDER    = '#e8e8ec'   # 옅은 테두리
BORDER2   = '#d4d4da'   # 조금 진한 테두리
TEXT      = '#1d1d1f'   # 본문 글자색 (거의 검정)
MUTED     = '#86868b'   # 보조 글자색 (회색)
ACCENT    = '#2f6fed'   # 포인트 색 (파랑) — 실행 버튼 등
ACCENT_DK = '#1d4fc4'   # 포인트 색의 어두운 버전 (hover 시)
ACCENT_BG = '#eef3fe'   # 포인트 색의 아주 연한 배경
RED       = '#d64545'   # 오류 표시용
GREEN     = '#3d9960'   # 성공 표시용

FONT = 'Helvetica'      # GUI 전체에서 쓰는 글꼴 이름

# 결과 저장 폴더: 이 파일(seg_gui.py)이 있는 폴더 밑의 results/
# __file__ = 지금 실행 중인 이 파이썬 파일의 경로
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DEFAULT_CONF = 0.25     # YOLO 검출 신뢰도 기본값 (낮출수록 더 많이 검출됨)
PREVIEW_MAX_W = 470     # 미리보기 이미지 최대 가로 픽셀
PREVIEW_MAX_H = 560     # 미리보기 이미지 최대 세로 픽셀

# 화면에 표시할 제목 (딕셔너리: 내부 이름 "top"/"side" -> 표시용 문자열)
KIND_TITLES = {"top": "위에서 (Top)", "side": "측면 (Side)"}


# ═════════════════════════════════════════════════════════════════════════════
# 1부: 세그멘테이션 파이프라인 (GUI와 무관한 순수 이미지 처리 로직)
#      GUI 없이도 이 함수들만 불러서 테스트할 수 있게 분리해뒀다.
# ═════════════════════════════════════════════════════════════════════════════
def load_backend(conf):
    """
    세그멘테이션 백엔드를 준비한다.
    1순위: YOLO (딥러닝, 정확함) — ultralytics 패키지가 필요
    2순위: classical (피부색 규칙 기반, 부정확) — YOLO 로드 실패 시 자동 폴백

    반환값: (백엔드 객체, 경고 메시지 또는 None)
    """
    try:
        # import를 함수 안에서 하는 이유: ultralytics는 무거워서(PyTorch 포함)
        # 실제로 필요할 때만 불러오고, 없어도 프로그램이 죽지 않게 하기 위함
        from backends.dl_yolo import YoloNailBackend
        return YoloNailBackend(conf=conf), None
    except RuntimeError as e:
        # YOLO 로드 실패(패키지 없음/다운로드 실패) -> classical로 대체
        from backends.classical import ClassicalNailBackend
        return ClassicalNailBackend(), str(e)


def try_charuco(image):
    """
    사진 속 ChArUco 보드(체커보드+마커 보정판)를 자동 인식해서
    "이미지 픽셀 좌표 -> 보드 mm 좌표" 변환 행렬(호모그래피)을 계산한다.

    호모그래피란? 3x3 행렬 하나로 "이 픽셀이 보드 위에서 몇 mm 지점인지"를
    알려주는 변환이다. 이게 있으면 사진 속 어떤 두 점 사이의 실제 mm 거리든
    계산할 수 있다. (카메라가 약간 기울어져 있어도 보정된다)

    반환값: (호모그래피 3x3 배열, 재투영 오차 mm) — 인식 실패 시 (None, None)
    """
    # calibrate()는 원래 명령줄 스크립트용이라 argparse 결과 객체를 받는다.
    # SimpleNamespace로 같은 모양의 객체를 만들어 전달한다 (보드 스펙 설정).
    args = SimpleNamespace(
        squares_x=charuco_calibration.SQUARES_X,   # 보드 가로 칸 수 (18)
        squares_y=charuco_calibration.SQUARES_Y,   # 보드 세로 칸 수 (26)
        square_mm=charuco_calibration.SQUARE_MM,   # 한 칸 실제 크기 (10mm)
        marker_mm=charuco_calibration.MARKER_MM,   # 마커 실제 크기 (7mm)
        dict=charuco_calibration.DICT_NAME,        # 마커 종류 (DICT_4X4_250)
    )
    try:
        result = charuco_calibration.calibrate(image, args)
        # rms_reprojection_mm = 계산된 변환의 오차 (작을수록 정확, 0.5mm 이하 권장)
        return np.asarray(result["homography"]), result["rms_reprojection_mm"]
    except (RuntimeError, ValueError) as e:
        # 보드가 사진에 없거나 너무 흐릿하면 여기로 온다 -> mm 측정 불가
        print(f"[INFO] 체커보드 인식 실패: {e}")
        return None, None


def sort_masks_by_x(nail_masks):
    """
    검출된 마스크들을 사진 왼쪽부터 순서대로(중심 x좌표 기준) 정렬한다.
    -> 결과 화면에 "1:, 2:, ..." 번호를 왼쪽부터 일관되게 붙이기 위함.
    """
    def center_x(nm):
        # np.nonzero(마스크) = 마스크에서 값이 0이 아닌(=손톱인) 픽셀들의 좌표
        # ys, xs 두 배열로 나오고, xs.mean()이 손톱의 가로 중심이 된다.
        ys, xs = np.nonzero(nm.mask)
        return float(xs.mean()) if len(xs) else 0.0
    return sorted(nail_masks, key=center_x)


def measure_pixels(mask):
    """
    체커보드가 없을 때의 대체 측정: mm 변환 없이 픽셀 단위로만
    세로 길이와 중간 높이의 가로 폭을 계산한다.

    반환값 딕셔너리의 키는 measure_nail_from_mask()와 동일하게 맞춰서
    (mm 값들만 None으로) 이후 코드가 둘을 구분 없이 다루게 한다.
    """
    # 마스크의 세로 방향 양 끝점 (손톱 뿌리 <-> 끝)
    endpoints = find_vertical_axis_endpoints(mask)
    if endpoints is None:            # 마스크가 비어있으면 측정 불가
        return None
    p1, p2 = endpoints

    # 길이선의 중간 높이(y)에서 가로 폭을 잰다
    y_mid = (float(p1[1]) + float(p2[1])) / 2.0
    width_endpoints = find_horizontal_width_endpoints(mask, y_mid)

    result = {
        "point_a": (float(p1[0]), float(p1[1])),        # 길이선 시작점 (픽셀)
        "point_b": (float(p2[0]), float(p2[1])),        # 길이선 끝점 (픽셀)
        "pixel_distance": float(np.linalg.norm(p2 - p1)),  # 길이 (픽셀), norm=거리
        "length_mm": None,                               # 보드가 없으니 mm는 없음
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
    """
    측정 결과 딕셔너리를 사람이 읽을 한 줄 문자열로 요약한다.
    mm가 있으면 "L 12.34  W 8.21 mm", 없으면 "L 340  W 120 px" 형태.
    """
    if m is None:
        return "측정 실패"
    if m["length_mm"] is not None:
        # f-string의 :.2f = 소수점 둘째 자리까지 표시
        w = f"  W {m['width_mm']:.2f}" if m["width_mm"] else ""
        return f"L {m['length_mm']:.2f}{w} mm"
    w = f"  W {m['width_pixel_distance']:.0f}" if m["width_pixel_distance"] else ""
    return f"L {m['pixel_distance']:.0f}{w} px"


def build_overlay(image, nail_masks, measures):
    """
    원본 사진 위에 검출/측정 결과를 그려서 "결과 이미지"를 만든다.
    스타일은 기존 measure_auto.py 디버그 이미지와 동일:
      초록 윤곽선 + 빨간 세로 길이선 + 파란 폭선 + 노란 mm 텍스트
    손톱이 2개 이상이면 왼쪽부터 "1:", "2:" 번호를 붙인다.

    ※ OpenCV의 색은 (B, G, R) 순서다! (0,255,0)=초록, (0,0,255)=빨강
    """
    overlay = image.copy()     # 원본을 건드리지 않도록 복사본에 그린다
    h, w = image.shape[:2]     # shape = (세로, 가로, 채널) — 앞의 둘만 사용

    # 사진 해상도가 제각각이므로, 선 굵기/글자 크기를 해상도에 비례시킨다
    scale = max(1.0, max(h, w) / 1920.0)
    font_scale = 0.8 * scale
    thickness = max(2, int(round(2 * scale)))

    # 검출이 하나도 없으면 안내 문구만 쓰고 반환
    if not nail_masks:
        cv2.putText(overlay, "no nail detected", (10, int(50 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                    thickness, cv2.LINE_AA)
        return overlay

    # zip(A, B) = 두 리스트를 짝지어 순회 (마스크와 그 측정 결과를 함께)
    for i, (nail_mask, m) in enumerate(zip(nail_masks, measures)):
        # ── 초록 윤곽선 ──
        # findContours = 마스크(흰 영역)의 테두리 좌표들을 뽑아준다
        contours, _ = cv2.findContours(nail_mask.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness)

        if m is None:      # 측정 실패한 마스크는 윤곽선만 그리고 넘어감
            continue

        # ── 빨간 세로 길이선 (뿌리 <-> 끝) ──
        # 좌표는 float이므로 int로 반올림해야 cv2.line에 넣을 수 있다
        pa = tuple(int(round(v)) for v in m["point_a"])
        pb = tuple(int(round(v)) for v in m["point_b"])
        cv2.line(overlay, pa, pb, (0, 0, 255), thickness)

        # ── 파란 폭선 (중간 높이 가로) ──
        if m["width_point_a"] is not None:
            wa = tuple(int(round(v)) for v in m["width_point_a"])
            wb = tuple(int(round(v)) for v in m["width_point_b"])
            cv2.line(overlay, wa, wb, (255, 0, 0), thickness)

        # ── 노란 텍스트 (예: "1:10.4mm w5.0mm") ──
        if m["length_mm"] is not None:
            text = f"{m['length_mm']:.1f}mm"
            if m["width_mm"]:
                text += f" w{m['width_mm']:.1f}mm"
        else:                                    # 보드 미검출 -> 픽셀로 표기
            text = f"{m['pixel_distance']:.0f}px"
            if m["width_pixel_distance"]:
                text += f" w{m['width_pixel_distance']:.0f}px"
        if len(nail_masks) > 1:                  # 여러 개일 때만 번호 붙임
            text = f"{i + 1}:{text}"

        # 텍스트 위치: 손톱 중심 x 근처, 손톱 맨 위(y 최솟값)보다 약간 위
        ys, xs = np.nonzero(nail_mask.mask)
        tx = max(0, int(xs.mean()) - int(60 * scale))          # 왼쪽으로 살짝 이동
        ty = max(int(30 * scale), int(ys.min()) - int(12 * scale))  # 화면 밖 방지
        cv2.putText(overlay, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return overlay


def process_image(image, backend, use_charuco):
    """
    이미지 한 장의 전체 처리 과정 (이 프로그램의 핵심 함수):
      세그멘테이션(검출된 손톱 전부) → (선택) 체커보드 실측 → 오버레이 생성

    반환값 딕셔너리:
      overlay          : 결과가 그려진 이미지 (numpy 배열)
      measures         : 손톱별 측정 결과 리스트
      rms_mm           : 체커보드 보정 오차 (없으면 None)
      homography_found : 체커보드 인식 성공 여부
      detected         : 손톱이 1개라도 검출됐는지
      count            : 검출된 손톱 개수
    """
    # 1) 세그멘테이션: 백엔드가 NailMask 객체 리스트를 반환 (왼쪽부터 정렬)
    nail_masks = sort_masks_by_x(backend.segment(image))

    # 2) 체커보드 인식 (옵션이 켜져 있고, 손톱이 검출된 경우에만 시도)
    homography, rms_mm = (None, None)
    if use_charuco and nail_masks:
        homography, rms_mm = try_charuco(image)

    # 3) 손톱마다 길이/폭 측정
    measures = []
    for nm in nail_masks:
        if homography is not None:
            # finger를 지정하면 기존 measure_auto.py와 동일한 세로(수직) 측정 방식을 쓴다
            measures.append(measure_nail_from_mask(nm.mask, homography, finger="nail"))
        else:
            measures.append(measure_pixels(nm.mask))   # 보드 없음 -> 픽셀만

    # 4) 결과 이미지 생성 + 요약 정보 반환
    return {
        "overlay": build_overlay(image, nail_masks, measures),
        "measures": measures,
        "rms_mm": rms_mm,
        "homography_found": homography is not None,
        "detected": bool(nail_masks),
        "count": len(nail_masks),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2부: 커스텀 위젯 (tkinter 기본 위젯을 조합해 만든 부품들)
# ═════════════════════════════════════════════════════════════════════════════
class FlatButton(tk.Label):
    """
    macOS에서도 배경색이 제대로 먹는 플랫 버튼.

    왜 tk.Button을 안 쓰나? macOS의 tk.Button은 시스템 스타일이 강제되어
    bg 색이 무시된다. 그래서 Label에 클릭 이벤트를 직접 연결해(bind)
    버튼처럼 동작하게 만들었다.
    """

    def __init__(self, master, text, command, bg, fg, hover_bg,
                 font=(FONT, 11, 'bold'), padx=12, pady=6, **kw):
        # super().__init__ = 부모 클래스(tk.Label)의 생성자 호출
        super().__init__(master, text=text, bg=bg, fg=fg, font=font,
                         padx=padx, pady=pady, cursor='hand2', **kw)
        self._bg, self._hover = bg, hover_bg   # 평소 색 / 마우스 올렸을 때 색
        self._command = command                # 클릭 시 실행할 함수
        self.enabled = True                    # False면 클릭 무시 (실행 중 잠금)

        # bind(이벤트, 함수) = 해당 이벤트가 발생하면 함수 실행
        # '<Enter>' = 마우스가 위젯 위로 들어옴, '<Leave>' = 나감,
        # '<Button-1>' = 마우스 왼쪽 버튼 클릭
        self.bind('<Enter>', lambda e: self.enabled and self.config(bg=self._hover))
        self.bind('<Leave>', lambda e: self.config(bg=self._bg))
        self.bind('<Button-1>', lambda e: self.enabled and self._command())

    def set_enabled(self, enabled):
        """실행 중에는 버튼을 잠가서 중복 클릭을 막는다."""
        self.enabled = enabled
        self.config(fg='white' if enabled else '#b6c6ee')   # 잠기면 흐린 색


class ImagePicker(tk.Frame):
    """
    이미지 파일을 한 장 고르는 위젯: [열기 버튼] + 아래에 선택된 파일명 표시.
    tk.Frame을 상속해서 "버튼+라벨" 묶음을 하나의 부품처럼 쓸 수 있게 했다.
    """

    # 파일 열기 대화상자에서 보여줄 확장자 필터
    FILETYPES = [("이미지", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("모든 파일", "*.*")]

    def __init__(self, master, label, on_pick):
        super().__init__(master, bg=BG)
        self.path = None          # 아직 선택된 파일 없음 (App이 이 값을 읽어감)
        self.on_pick = on_pick    # 파일을 고른 직후 호출할 콜백 함수

        FlatButton(self, f"{label} 열기", self._pick,
                   bg=SURFACE, fg=TEXT, hover_bg=ACCENT_BG,
                   font=(FONT, 11, 'bold'), padx=12, pady=8,
                   highlightthickness=1, highlightbackground=BORDER2
                   ).pack(fill=tk.X)    # pack(fill=X) = 가로로 꽉 채워 배치
        self.name_lbl = tk.Label(self, text="선택된 파일 없음", bg=BG, fg=MUTED,
                                 font=(FONT, 9), anchor='w')
        self.name_lbl.pack(fill=tk.X, pady=(3, 0))

    def _pick(self):
        """버튼 클릭 -> 파일 대화상자 열기 -> 선택된 경로 저장 + 파일명 표시"""
        path = filedialog.askopenfilename(filetypes=self.FILETYPES)
        if not path:              # 사용자가 취소를 누르면 빈 문자열이 온다
            return
        self.path = path
        # basename = 경로에서 파일 이름만 추출 (/a/b/c.jpg -> c.jpg)
        self.name_lbl.config(text=os.path.basename(path), fg=TEXT)
        self.on_pick()            # App에 "파일 골랐어요" 알림


# ═════════════════════════════════════════════════════════════════════════════
# 3부: 메인 GUI (화면 구성 + 버튼 동작 + 스레드 처리)
# ═════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root: tk.Tk):
        self.root = root                       # 최상위 창
        root.title("손톱 세그멘테이션 — Top / Side")   # 창 제목
        root.configure(bg=BG)
        self.backend = None        # 세그멘테이션 백엔드 (첫 실행 때 로드해서 재사용)
        self.backend_conf = None   # 백엔드를 로드했을 때의 conf 값 (바뀌면 재로드)
        self._running = False      # 지금 처리 중인지 (중복 실행 방지 플래그)
        self._photos = {}          # PhotoImage 참조 보관
                                   # (tkinter는 참조가 사라지면 이미지가 화면에서
                                   #  지워지는 함정이 있어서 여기에 붙잡아둔다)
        self._build_ui()           # 화면 부품들 생성

    # ── 화면 구성: 왼쪽 패널(입력/옵션/결과) + 오른쪽 패널(미리보기 2칸) ──────
    def _build_ui(self):
        # 왼쪽 세로 패널. width 고정 + pack_propagate(False)로 크기 고정
        left = tk.Frame(self.root, bg=BG, padx=22, pady=20, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # 제목/부제목
        tk.Label(left, text="손톱 세그멘테이션", font=(FONT, 18, 'bold'),
                 bg=BG, fg=TEXT, anchor='w').pack(fill=tk.X)
        tk.Label(left, text="Top/Side 각 1장 → 인식된 손톱 전부 자동 실측",
                 font=(FONT, 11), bg=BG, fg=MUTED, anchor='w'
                 ).pack(fill=tk.X, pady=(2, 14))

        # ── 입력 이미지 선택 (위에서 만든 ImagePicker 위젯 2개) ──
        tk.Label(left, text="입력 이미지", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 6))
        self.top_picker = ImagePicker(left, "위에서(Top) 이미지", self._clear_status)
        self.top_picker.pack(fill=tk.X, pady=(0, 8))
        self.side_picker = ImagePicker(left, "측면(Side) 이미지", self._clear_status)
        self.side_picker.pack(fill=tk.X, pady=(0, 12))

        # ── 옵션 ──
        tk.Label(left, text="옵션", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 6))

        # conf 입력칸: StringVar = tkinter 입력값을 담는 특수 변수
        # (Entry와 연결해두면 .get()으로 현재 입력 내용을 읽을 수 있다)
        opt = tk.Frame(left, bg=BG)
        opt.pack(fill=tk.X)
        tk.Label(opt, text="검출 신뢰도(conf)", bg=BG, fg=TEXT,
                 font=(FONT, 10)).pack(side=tk.LEFT)
        self.conf_var = tk.StringVar(value=f"{DEFAULT_CONF:g}")
        tk.Entry(opt, textvariable=self.conf_var, width=6, justify='center',
                 font=(FONT, 11), bg=SURFACE, fg=TEXT, relief='flat',
                 highlightthickness=1, highlightbackground=BORDER2,
                 highlightcolor=ACCENT).pack(side=tk.RIGHT, ipady=3)

        # 체커보드 실측 on/off 체크박스 (BooleanVar에 체크 상태가 담긴다)
        self.charuco_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="체커보드(ChArUco) 인식해서 mm 실측",
                       variable=self.charuco_var, bg=BG, fg=TEXT,
                       activebackground=BG, font=(FONT, 10), anchor='w'
                       ).pack(fill=tk.X, pady=(6, 12))

        # ── 실행 버튼 ──
        self.run_btn = FlatButton(left, "세그멘테이션 실행", self._run,
                                  bg=ACCENT, fg='white', hover_bg=ACCENT_DK,
                                  font=(FONT, 12, 'bold'), padx=12, pady=10)
        self.run_btn.pack(fill=tk.X)

        # ── 상태 표시줄 (진행 상황/오류/완료 메시지) ──
        self.status = tk.Label(left, text="이미지를 선택하고 실행을 누르세요",
                               bg=BG, fg=MUTED, font=(FONT, 10),
                               anchor='w', wraplength=270, justify='left')
        self.status.pack(fill=tk.X, pady=(8, 10))

        # ── 실측 결과 카드 (손톱별 L/W 목록이 여기 출력됨) ──
        tk.Label(left, text="실측 결과", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X, pady=(0, 6))
        card = tk.Frame(left, bg=SURFACE, padx=12, pady=10,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill=tk.BOTH, expand=True)
        # Courier = 고정폭 글꼴 -> 숫자 자릿수가 맞아 표처럼 보인다
        self.result_lbl = tk.Label(card, text="—", bg=SURFACE, fg=TEXT,
                                   font=('Courier', 11), justify='left', anchor='nw')
        self.result_lbl.pack(fill=tk.BOTH, expand=True)

        # ── 오른쪽 미리보기 패널: 위/측면 2칸 ──
        right = tk.Frame(self.root, bg=BG, padx=10, pady=14)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.previews = {}    # {"top": (이미지 라벨, 캡션 라벨), "side": (...)}
        for kind in ("top", "side"):
            col = tk.Frame(right, bg=BG)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
            tk.Label(col, text=KIND_TITLES[kind], bg=BG, fg=TEXT,
                     font=(FONT, 12, 'bold')).pack(pady=(0, 6))
            frame = tk.Frame(col, bg=SURFACE, highlightthickness=1,
                             highlightbackground=BORDER)
            frame.pack(fill=tk.BOTH, expand=True)
            # 이 Label에 나중에 이미지(PhotoImage)를 넣는다
            lbl = tk.Label(frame, text="미리보기 없음", bg=SURFACE, fg=MUTED,
                           font=(FONT, 11))
            lbl.pack(fill=tk.BOTH, expand=True)
            # 이미지 아래 요약 캡션 ("손톱 5개 · 보드 RMS 0.24mm" 등)
            sub = tk.Label(col, text="", bg=BG, fg=MUTED, font=(FONT, 9))
            sub.pack(pady=(4, 0))
            self.previews[kind] = (lbl, sub)

    # ── 상태 표시 도우미 ─────────────────────────────────────────────────────
    def _clear_status(self):
        """이미지를 새로 고르면 상태줄을 초기화 (처리 중이 아닐 때만)"""
        if not self._running:
            self.status.config(text="준비 완료 — [세그멘테이션 실행]을 누르세요", fg=MUTED)

    def _set_status(self, text, color=MUTED):
        """
        백그라운드 스레드에서 상태줄을 안전하게 바꾸는 함수.

        ※ 중요: tkinter 위젯은 메인 스레드에서만 건드려야 한다!
        root.after(0, 함수) = "메인 스레드야, 시간 나면 이 함수 실행해줘"
        라는 예약이다. 스레드에서 직접 config()를 부르면 프로그램이 죽을 수 있다.
        """
        self.root.after(0, lambda: self.status.config(text=text, fg=color))

    # ── 실행 버튼 클릭 시 ────────────────────────────────────────────────────
    def _run(self):
        if self._running:         # 이미 처리 중이면 무시
            return

        # 선택된 이미지 경로 수집 {"top": 경로, "side": 경로} (없는 쪽은 제외)
        inputs = {}
        for kind, picker in (("top", self.top_picker), ("side", self.side_picker)):
            if picker.path:
                inputs[kind] = picker.path
        if not inputs:
            self.status.config(text="⚠ 이미지를 최소 한 장 선택하세요", fg=RED)
            return

        # conf 입력값 검증 (숫자인지, 0~1 사이인지)
        try:
            conf = float(self.conf_var.get())
            assert 0.0 < conf < 1.0
        except (ValueError, AssertionError):
            self.status.config(text="⚠ conf는 0과 1 사이 숫자여야 합니다 (예: 0.25)", fg=RED)
            return

        # 처리 시작: 버튼 잠그고 백그라운드 스레드 시작
        self._running = True
        self.run_btn.set_enabled(False)
        self.status.config(text="처리 중...", fg=MUTED)
        # daemon=True: 창을 닫으면 스레드도 같이 종료되게 함
        threading.Thread(
            target=self._worker,
            args=(inputs, conf, self.charuco_var.get()),
            daemon=True,
        ).start()

    # ── 백그라운드 스레드: 실제 이미지 처리 + 저장 ───────────────────────────
    def _worker(self, inputs, conf, use_charuco):
        try:
            # 백엔드는 무거우니 한 번만 로드하고 재사용한다.
            # 단, conf가 바뀌었으면 새로 만들어야 함 (YOLO에 conf가 박혀있어서)
            if self.backend is None or self.backend_conf != conf:
                self._set_status("세그멘테이션 백엔드 로딩 중...")
                self.backend, backend_warn = load_backend(conf)
                self.backend_conf = conf
                if backend_warn:
                    print(f"[WARN] YOLO 로드 실패, classical 폴백:\n{backend_warn}")

            # 결과 저장 폴더: results/20260807_233950/ 처럼 실행 시각으로 생성
            session_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))

            results = {}            # 화면 표시용 결과 {"top": {...}, "side": {...}}
            measurements_json = {}  # JSON 저장용 수치 데이터
            saved = []              # 저장된 파일 경로 목록

            for kind, path in inputs.items():     # "top", "side" 순회
                self._set_status(f"{'위에서' if kind == 'top' else '측면'} 이미지 처리 중...")

                image = cv2.imread(path)          # 파일 -> numpy 배열 (BGR)
                if image is None:                 # 경로가 틀리거나 손상된 파일
                    raise RuntimeError(f"이미지를 열 수 없습니다: {path}")

                # ★ 핵심 처리 (1부의 파이프라인 함수 호출)
                res = process_image(image, self.backend, use_charuco)
                results[kind] = res

                # 결과 이미지 저장 (top.png / side.png)
                out_path = os.path.join(session_dir, f"{kind}.png")
                ensure_dir(out_path)              # 폴더가 없으면 만들어줌
                cv2.imwrite(out_path, res["overlay"])
                saved.append(out_path)

                # 손톱별 수치를 JSON용 딕셔너리로 정리
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

            # 수치 데이터를 measurements.json으로 저장
            if measurements_json:
                json_path = os.path.join(session_dir, "measurements.json")
                save_json(json_path, measurements_json)

            # 화면 갱신은 메인 스레드에 예약 (_set_status 주석 참고)
            self.root.after(0, lambda: self._show_results(results, session_dir, saved))

        except Exception as e:
            # 어떤 오류든 잡아서 콘솔에 상세 출력 + 상태줄에 요약 표시
            traceback.print_exc()
            msg = str(e)
            self._set_status(f"⚠ 오류: {msg[:300]}", RED)
            self.root.after(0, self._done)

    # ── 메인 스레드: 처리 완료 후 화면 갱신 ─────────────────────────────────
    def _show_results(self, results, session_dir, saved):
        for kind, (lbl, sub) in self.previews.items():
            res = results.get(kind)
            if res is None:              # 해당 방향 이미지를 안 넣었으면 건너뜀
                continue

            # 결과 이미지를 미리보기 크기로 변환해서 라벨에 표시
            photo = self._to_photo(res["overlay"])
            self._photos[kind] = photo   # 참조 보관 (안 하면 이미지가 사라진다!)
            lbl.config(image=photo, text="")

            # 캡션: 검출 개수 + 보드 인식 여부 요약
            if not res["detected"]:
                sub.config(text="검출 실패", fg=RED)
            elif res["homography_found"]:
                sub.config(text=f"손톱 {res['count']}개 · 보드 RMS {res['rms_mm']:.2f}mm",
                           fg=TEXT)
            else:
                sub.config(text=f"손톱 {res['count']}개 · 보드 미검출", fg=MUTED)

        # 왼쪽 실측 결과 카드 갱신
        self.result_lbl.config(text=self._format_measurements(results))

        # 상태줄에 저장 위치 표시 (relpath = 짧은 상대경로로 변환)
        rel = os.path.relpath(session_dir, os.path.dirname(RESULTS_DIR))
        self.status.config(text=f"✓ 완료 — {rel}/ 에 {len(saved)}개 이미지 저장됨", fg=GREEN)
        self._done()

    @staticmethod
    def _format_measurements(results):
        """실측 결과 카드에 넣을 여러 줄 문자열을 만든다."""
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
            lines.append("")                     # 섹션 사이 빈 줄
        return "\n".join(lines).strip() or "—"

    def _done(self):
        """처리 종료: 잠갔던 버튼을 다시 활성화"""
        self._running = False
        self.run_btn.set_enabled(True)

    @staticmethod
    def _to_photo(image_bgr):
        """
        OpenCV 이미지(numpy 배열)를 tkinter가 표시할 수 있는 PhotoImage로 변환.

        변환 경로: numpy 배열 -> PNG 바이트로 인코딩 -> base64 문자열 -> PhotoImage
        (Pillow 같은 추가 라이브러리 없이 변환하는 방법이다.
         tk.PhotoImage는 base64로 인코딩된 PNG 데이터를 바로 받을 수 있다)
        """
        h, w = image_bgr.shape[:2]
        # 미리보기 크기에 맞게 축소 배율 계산 (1.0 초과 금지 = 확대는 안 함)
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        resized = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)   # INTER_AREA = 축소에 적합
        ok, buf = cv2.imencode(".png", resized)              # 배열 -> PNG 바이트
        if not ok:
            raise RuntimeError("미리보기 이미지 인코딩에 실패했습니다")
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))


# ═════════════════════════════════════════════════════════════════════════════
# 실행 시작점
# ═════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()                 # 최상위 창 생성
    root.geometry("1320x680")      # 창 크기 (가로x세로 픽셀)
    App(root)                      # 화면 구성 (위의 App 클래스)
    root.mainloop()                # 이벤트 루프 시작 — 창을 닫을 때까지 여기서 대기

# 이 파일을 직접 실행했을 때만 main()을 호출
# (다른 파일에서 import할 때는 GUI가 뜨지 않게 하기 위한 파이썬 관용구)
if __name__ == "__main__":
    main()
