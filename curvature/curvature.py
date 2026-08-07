#!/usr/bin/env python3
"""
네일 팁 곡면 계산기 — P/B/S/C 유형별 곡면 공식

입력 (실측값, 2개만 필요):
  front_width : 정면 가로너비 (mm)
  side_width  : 측면 (가로)너비 (mm)

공통 계산 흐름 (유형별 비율 k_a, k_h, k_f 만 다름):
  a = k_a × front_width          측정 삼각형의 밑변
  b = side_width                 측정 삼각형의 높이
  c = √(a² + b²)                 측정 삼각형의 빗변 (원호 계산용 현)
  h = k_h × c                    현 c 에 대한 새그(sag)
  r = c²/(8h) + h/2              곡률 반지름
  θ = 2·atan2(c/2, r − h)        중심각 (arcsin의 수치적으로 안정된 형태)
  L = r·θ                        편측 원호 길이

  곡면 길이 = 2L + k_f × a       최종 결과값

유형별 비율 (P/B/S/C 계산법 PDF 기준):
  P형(평행) : k_a = 2/5, k_h = 1/4, k_f = 1/2
  B형(버드) : k_a = 3/8, k_h = 1/6, k_f = 2/3
  S형(부채) : k_a = 3/7, k_h = 1/7, k_f = 1/3
  C형(평행) : k_a = 3/7, k_h = 1/5, k_f = 1/3
"""
import math
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 한글 폰트
for _fn in ['Apple SD Gothic Neo', 'AppleGothic', 'NanumGothic', 'Malgun Gothic']:
    if any(f.name == _fn for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 — 뉴트럴 그레이 + 블루 포인트
BG        = '#f5f5f7'   # 창 배경
SURFACE   = '#ffffff'   # 카드 / 입력창
BORDER    = '#e8e8ec'
BORDER2   = '#d4d4da'
TEXT      = '#1d1d1f'   # 본문
MUTED     = '#86868b'   # 보조 텍스트
GRAY      = '#9ca3af'   # 치수선
ACCENT    = '#2f6fed'   # 포인트 (블루)
ACCENT_DK = '#1d4fc4'
ACCENT_BG = '#eef3fe'
RED       = '#d64545'   # 오류
GREEN     = '#3d9960'   # 성공

FONT = 'Helvetica'


# ── 유형별 계산 비율 ─────────────────────────────────────────────────────────
# k_cal: 실측 보정 계수. 원래 공식(k_cal=1.0)이 실측값보다 구조적으로 크게 나와서
# 추가한 값. P형은 실측 5건(검지/2번째/소지/5번째/6번째, 엄지 제외) 대비 계산값을
# 최소제곱으로 피팅해 0.86으로 보정. B/S/C는 아직 실측 데이터가 없어 1.0(보정 없음).
TYPES = {
    "P": dict(name="P형 (평행)", k_a=2 / 5, k_a_s="2/5",
              k_h=1 / 4, k_h_s="1/4", k_f=1 / 2, k_f_s="1/2", k_cal=0.86),
    "B": dict(name="B형 (버드)", k_a=3 / 8, k_a_s="3/8",
              k_h=1 / 6, k_h_s="1/6", k_f=2 / 3, k_f_s="2/3", k_cal=1.0),
    "S": dict(name="S형 (부채)", k_a=3 / 7, k_a_s="3/7",
              k_h=1 / 7, k_h_s="1/7", k_f=1 / 3, k_f_s="1/3", k_cal=1.0),
    "C": dict(name="C형 (평행)", k_a=3 / 7, k_a_s="3/7",
              k_h=1 / 5, k_h_s="1/5", k_f=1 / 3, k_f_s="1/3", k_cal=1.0),
}
CATEGORY_ORDER = ["P", "B", "S", "C"]


# ── 계산 ──────────────────────────────────────────────────────────────────────
def calc(category: str, front_width: float, side_width: float) -> dict:
    """유형(P/B/S/C)과 정면너비·측면너비로 곡면 길이를 계산한다."""
    p = TYPES[category]
    a = p["k_a"] * front_width
    b = side_width
    c = math.hypot(a, b)
    h = p["k_h"] * c
    r = c ** 2 / (8 * h) + h / 2
    # atan2를 쓰면 h가 c/2를 넘는 깊은 곡면에서도 θ가 올바르게 180°를 넘어간다.
    theta = 2 * math.atan2(c / 2, r - h)
    L = r * theta
    flat = p["k_f"] * a
    total = (2 * L + flat) * p["k_cal"]  # 보정: 실측 대비 과대추정 경향을 k_cal로 보정
    return dict(
        category=category, name=p["name"],
        front_width=front_width, side_width=side_width,
        a=a, b=b, c=c, h=h, r=r,
        theta_deg=math.degrees(theta), L=L, flat=flat, total=total,
        k_a_s=p["k_a_s"], k_h_s=p["k_h_s"], k_f_s=p["k_f_s"],
    )


def arc_profile(c: float, h: float, n: int = 150):
    """현(c)·새그(h)로 정의되는 단일 원호의 (x, z) 좌표. (0,0)→(c/2,h)→(c,0)"""
    R = c ** 2 / (8 * h) + h / 2
    Cx, Cz = c / 2, h - R

    a_start = math.atan2(0 - Cz, 0 - Cx)
    a_end = math.atan2(0 - Cz, c - Cx)
    if a_start < a_end:
        a_start += 2 * math.pi

    t = np.linspace(a_start, a_end, n)
    x = Cx + R * np.cos(t)
    z = Cz + R * np.sin(t)
    return x, z, R, Cx, Cz


def _bezier(p0, p1, p2, n=60):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return pts[:, 0], pts[:, 1]


# ── 커스텀 위젯 ───────────────────────────────────────────────────────────────
class FlatButton(tk.Label):
    """macOS에서도 색이 먹는 플랫 버튼 (hover 효과 포함)"""

    def __init__(self, master, text, command, bg, fg, hover_bg,
                 font=(FONT, 11, 'bold'), padx=12, pady=6, **kw):
        super().__init__(master, text=text, bg=bg, fg=fg, font=font,
                         padx=padx, pady=pady, cursor='hand2', **kw)
        self._bg, self._hover = bg, hover_bg
        self._command = command
        self.bind('<Enter>', lambda e: self.config(bg=self._hover))
        self.bind('<Leave>', lambda e: self.config(bg=self._bg))
        self.bind('<Button-1>', lambda e: self._command())


class NumberField(tk.Frame):
    """[-] 입력창 [+] 스테퍼가 붙은 숫자 입력 필드"""

    def __init__(self, master, label, default, step, on_change):
        super().__init__(master, bg=BG)
        self.step = step
        self.on_change = on_change

        head = tk.Frame(self, bg=BG)
        head.pack(fill=tk.X)
        tk.Label(head, text=label, bg=BG, fg=TEXT,
                 font=(FONT, 11)).pack(side=tk.LEFT)
        tk.Label(head, text='mm', bg=BG, fg=MUTED,
                 font=(FONT, 10)).pack(side=tk.RIGHT)

        box = tk.Frame(self, bg=SURFACE, highlightthickness=1,
                       highlightbackground=BORDER2, highlightcolor=ACCENT)
        box.pack(fill=tk.X, pady=(5, 0))

        FlatButton(box, '−', lambda: self._bump(-1),
                   bg=SURFACE, fg=MUTED, hover_bg=ACCENT_BG,
                   font=(FONT, 14), padx=13, pady=4).pack(side=tk.LEFT)

        self.var = tk.StringVar(value=default)
        self.entry = tk.Entry(box, textvariable=self.var,
                              font=(FONT, 16, 'bold'), justify='center',
                              bg=SURFACE, fg=TEXT, insertbackground=ACCENT,
                              relief='flat', bd=0,
                              highlightthickness=0)
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, ipady=7)
        self.entry.bind('<KeyRelease>', lambda e: self.on_change())
        self.entry.bind('<Up>',   lambda e: self._bump(+1))
        self.entry.bind('<Down>', lambda e: self._bump(-1))

        FlatButton(box, '+', lambda: self._bump(+1),
                   bg=SURFACE, fg=MUTED, hover_bg=ACCENT_BG,
                   font=(FONT, 14), padx=13, pady=4).pack(side=tk.RIGHT)

        self._box = box

    def _bump(self, sign):
        try:
            v = float(self.var.get())
        except ValueError:
            return
        v = max(round(v + sign * self.step, 2), self.step)
        self.var.set(f'{v:g}')
        self.on_change()

    def get(self) -> float:
        return float(self.var.get())

    def mark(self, ok: bool):
        self._box.config(highlightbackground=BORDER2 if ok else RED)


class CategorySelector(tk.Frame):
    """P / B / S / C 유형을 고르는 세그먼트 버튼"""

    def __init__(self, master, default, on_change):
        super().__init__(master, bg=BG)
        self.on_change = on_change
        self.value = default
        self.buttons = {}

        row = tk.Frame(self, bg=BG)
        row.pack(fill=tk.X)
        for i, key in enumerate(CATEGORY_ORDER):
            active = key == default
            b = FlatButton(
                row, key, lambda k=key: self._select(k),
                bg=ACCENT if active else SURFACE,
                fg='white' if active else TEXT,
                hover_bg=ACCENT_DK if active else ACCENT_BG,
                font=(FONT, 13, 'bold'), padx=8, pady=9)
            b.pack(side=tk.LEFT, expand=True, fill=tk.X,
                   padx=(0 if i == 0 else 4, 0))
            self.buttons[key] = b

        self.desc = tk.Label(self, text=TYPES[default]["name"], bg=BG,
                             fg=MUTED, font=(FONT, 10), anchor='w')
        self.desc.pack(fill=tk.X, pady=(5, 0))

    def _select(self, key):
        if key == self.value:
            return
        self.value = key
        for k, b in self.buttons.items():
            active = k == key
            b._bg = ACCENT if active else SURFACE
            b._hover = ACCENT_DK if active else ACCENT_BG
            b.config(bg=b._bg, fg='white' if active else TEXT)
        self.desc.config(text=TYPES[key]["name"])
        self.on_change()

    def get(self) -> str:
        return self.value


# ── GUI ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("네일 팁 곡면 계산기 — P/B/S/C")
        root.configure(bg=BG)
        self._after_id = None
        self._build_ui()
        self._update()

    # ── 왼쪽 패널 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        left = tk.Frame(self.root, bg=BG, padx=22, pady=20, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # 타이틀
        tk.Label(left, text="네일 곡면 계산기", font=(FONT, 18, 'bold'),
                 bg=BG, fg=TEXT, anchor='w').pack(fill=tk.X)
        tk.Label(left, text="유형별(P·B·S·C) 곡면 길이 계산", font=(FONT, 11),
                 bg=BG, fg=MUTED, anchor='w').pack(fill=tk.X, pady=(2, 14))

        # 유형 선택
        tk.Label(left, text="곡면 유형", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X,
                                                          pady=(0, 6))
        self.category = CategorySelector(left, "P", self._schedule)
        self.category.pack(fill=tk.X, pady=(0, 14))

        # 입력
        tk.Label(left, text="측정 입력", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X,
                                                          pady=(0, 6))
        self.fw_field = NumberField(left, "정면 너비", "14",
                                    step=0.5, on_change=self._schedule)
        self.fw_field.pack(fill=tk.X, pady=(0, 12))
        self.sw_field = NumberField(left, "측면 너비", "6",
                                    step=0.5, on_change=self._schedule)
        self.sw_field.pack(fill=tk.X, pady=(0, 6))

        # 상태 표시 (팝업 대신 인라인)
        self.status = tk.Label(left, text="", bg=BG, fg=MUTED,
                               font=(FONT, 10), anchor='w')
        self.status.pack(fill=tk.X, pady=(0, 10))

        # 계산 결과
        tk.Label(left, text="계산 결과", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X,
                                                          pady=(4, 6))

        # 메인 결과 카드 — 곡면 길이 (블루 틴트)
        card = tk.Frame(left, bg=ACCENT_BG, padx=16, pady=14,
                        highlightthickness=1, highlightbackground='#c8dafc')
        card.pack(fill=tk.X)
        top = tk.Frame(card, bg=ACCENT_BG)
        top.pack(fill=tk.X)
        self.formula_lbl = tk.Label(top, text="곡면 길이  = 2L + ½a",
                                    bg=ACCENT_BG, fg=ACCENT_DK,
                                    font=(FONT, 11, 'bold'))
        self.formula_lbl.pack(side=tk.LEFT)
        FlatButton(top, '복사', self._copy, bg=ACCENT, fg='white',
                   hover_bg=ACCENT_DK, font=(FONT, 9, 'bold'),
                   padx=10, pady=2).pack(side=tk.RIGHT)
        row = tk.Frame(card, bg=ACCENT_BG)
        row.pack(fill=tk.X, pady=(6, 0))
        self.total_label = tk.Label(row, text="—", bg=ACCENT_BG, fg=ACCENT_DK,
                                    font=(FONT, 30, 'bold'))
        self.total_label.pack(side=tk.LEFT)
        tk.Label(row, text=" mm", bg=ACCENT_BG, fg=MUTED,
                 font=(FONT, 13)).pack(side=tk.LEFT, anchor='s', pady=(0, 5))

        # 보조 결과 미니 카드 2×4 (a / b / c / h / r / θ / L / flat)
        grid = tk.Frame(left, bg=BG)
        grid.pack(fill=tk.X, pady=(10, 0))
        self._mini = {}
        mini_items = [
            ("a",     "a  (밑변)",     '#2f6fed'),
            ("b",     "b  (측면너비)",  '#2f6fed'),
            ("c",     "c  (빗변)",     '#5b8af0'),
            ("h",     "h  (새그)",     '#5b8af0'),
            ("r",     "r  (곡률반지름)", '#8fb0f5'),
            ("theta", "θ  (중심각)",   '#8fb0f5'),
            ("L",     "L  (편측 호)",  '#9ca3af'),
            ("flat",  "평탄부",        '#9ca3af'),
        ]
        for i, (key, lbl, dot) in enumerate(mini_items):
            cell = tk.Frame(grid, bg=SURFACE, padx=10, pady=7,
                            highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=i // 2, column=i % 2, sticky='nsew',
                      padx=(0 if i % 2 == 0 else 6, 0),
                      pady=(0 if i < 2 else 5, 0))
            grid.columnconfigure(i % 2, weight=1)
            head_ = tk.Frame(cell, bg=SURFACE)
            head_.pack(fill=tk.X)
            tk.Label(head_, text='●', bg=SURFACE, fg=dot,
                     font=(FONT, 6)).pack(side=tk.LEFT)
            tk.Label(head_, text=' ' + lbl, bg=SURFACE, fg=MUTED,
                     font=(FONT, 9)).pack(side=tk.LEFT)
            v = tk.Label(cell, text="—", bg=SURFACE, fg=TEXT,
                         font=(FONT, 12, 'bold'))
            v.pack(anchor='w', pady=(3, 0))
            self._mini[key] = v

        # 공식 카드 (하단 고정)
        fcard = tk.Frame(left, bg=SURFACE, padx=12, pady=10,
                         highlightthickness=1, highlightbackground=BORDER)
        fcard.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        self.fcard_lbl = tk.Label(fcard, text="", bg=SURFACE, fg='#6b7280',
                                  font=('Courier', 10), justify='left',
                                  anchor='w')
        self.fcard_lbl.pack(fill=tk.X)
        tk.Label(left, text="공식", bg=BG, fg=MUTED, font=(FONT, 9, 'bold'),
                 anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

        # 오른쪽 matplotlib 패널
        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(13, 5.2), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                         padx=4, pady=4)

    # ── 동작 ─────────────────────────────────────────────────────────────────
    def _schedule(self):
        """입력 후 250ms 디바운스로 실시간 재계산"""
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(250, self._update)

    def _copy(self):
        txt = self.total_label.cget('text')
        if txt != "—":
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self.status.config(text=f"✓ 복사됨: {txt} mm", fg=GREEN)

    def _update(self):
        self._after_id = None
        cat = self.category.get()
        try:
            fw = self.fw_field.get()
            sw = self.sw_field.get()
            assert fw > 0 and sw > 0
        except Exception:
            self.status.config(text="⚠ 정면너비, 측면너비 모두 양수를 입력하세요",
                               fg=RED)
            self.fw_field.mark(False)
            self.sw_field.mark(False)
            return
        self.fw_field.mark(True)
        self.sw_field.mark(True)
        try:
            r = calc(cat, fw, sw)
        except (ValueError, ZeroDivisionError) as e:
            self.status.config(text=f"⚠ {e}", fg=RED)
            self.sw_field.mark(False)
            return

        self.status.config(text="", fg=MUTED)
        self.formula_lbl.config(
            text=f"곡면 길이  = 2L + {r['k_f_s']}a")
        self.total_label.config(text=f"{r['total']:.2f}")
        self._mini["a"].config(text=f"{r['a']:.2f} mm")
        self._mini["b"].config(text=f"{r['b']:.2f} mm")
        self._mini["c"].config(text=f"{r['c']:.2f} mm")
        self._mini["h"].config(text=f"{r['h']:.2f} mm")
        self._mini["r"].config(text=f"{r['r']:.2f} mm")
        self._mini["theta"].config(text=f"{r['theta_deg']:.1f}°")
        self._mini["L"].config(text=f"{r['L']:.2f} mm")
        self._mini["flat"].config(text=f"{r['flat']:.2f} mm")
        self.fcard_lbl.config(
            text=(f"a = {r['k_a_s']}·정면너비\n"
                  f"b = 측면너비\n"
                  f"c = √(a²+b²)\n"
                  f"h = {r['k_h_s']}·c\n"
                  f"r = c²/(8h)+h/2\n"
                  f"θ = 2·asin(c/2r)\n"
                  f"L = r·θ\n"
                  f"곡면 길이 = 2L + {r['k_f_s']}a"))
        self._draw(r)

    # ── 그래프 ───────────────────────────────────────────────────────────────
    def _draw(self, r: dict):
        self.fig.clear()
        gs = gridspec.GridSpec(1, 3, figure=self.fig,
                               left=0.045, right=0.98,
                               top=0.86, bottom=0.16, wspace=0.32)
        ax1 = self.fig.add_subplot(gs[0])  # 측정 삼각형
        ax2 = self.fig.add_subplot(gs[1])  # 원호 계산
        ax3 = self.fig.add_subplot(gs[2])  # 최종 전개도

        a, b, c, h = r['a'], r['b'], r['c'], r['h']
        R = r['r']
        fw = r['front_width']
        flat = r['flat']

        # ── ① 측정 삼각형 (a·b·c) ───────────────────────────────────────────
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(SURFACE)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
            ax.set_xticks([])
            ax.set_yticks([])

        tx, ty = [0, a, a, 0], [0, 0, b, 0]
        ax1.fill(tx, ty, color=ACCENT, alpha=0.10)
        ax1.plot(tx, ty, color=ACCENT_DK, lw=2, solid_capstyle='round')
        ax1.text(a / 2, -b * 0.10, f'a = {a:.2f}', color=TEXT, fontsize=9,
                 ha='center', va='top')
        ax1.text(a * 1.05, b / 2, f'b = {b:.2f}', color=TEXT, fontsize=9,
                 ha='left', va='center')
        ax1.text(a * 0.32, b * 0.62, f'c = {c:.2f}', color=ACCENT_DK,
                 fontsize=9, ha='center', va='center', rotation=0)
        pad = max(a, b) * 0.25
        ax1.set_xlim(-pad, a + pad)
        ax1.set_ylim(-b * 0.3, b + pad)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title('① 측정 삼각형  c²=a²+b²', color=TEXT, fontsize=11, pad=8)

        # ── ② 원호 계산 (r·θ·L) ─────────────────────────────────────────────
        x_arr, z_arr, R_vis, Cx, Cz = arc_profile(c, h)
        circ_t = np.linspace(0, 2 * math.pi, 200)
        ax2.plot(Cx + R_vis * np.cos(circ_t), Cz + R_vis * np.sin(circ_t),
                 color=BORDER2, lw=1, ls='--')
        ax2.plot(x_arr, z_arr, color=ACCENT, lw=2.6, solid_capstyle='round')
        ax2.plot([0, c], [0, 0], color=GRAY, lw=1.1, ls=':')
        ax2.plot(Cx, Cz, marker='+', color=MUTED, ms=8)
        ax2.annotate('', xy=(c / 2, h), xytext=(c / 2, 0),
                     arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.1))
        ax2.text(c / 2 + c * 0.04, h / 2, f'h={h:.2f}', color=TEXT,
                 fontsize=8.5, va='center')
        ax2.text(c / 2, h + max(z_arr) * 0.08, f'L = {r["L"]:.2f}',
                 color=ACCENT_DK, fontsize=9.5, ha='center', fontweight='bold')
        ax2.text(c / 2, -max(b, c) * 0.14, f'c = {c:.2f}   θ = {r["theta_deg"]:.1f}°',
                 color=MUTED, fontsize=8.5, ha='center')
        pad2 = c * 0.25
        ax2.set_xlim(-pad2, c + pad2)
        ax2.set_ylim(-c * 0.25, max(z_arr) + pad2)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title('② 원호 계산  r·θ·L', color=TEXT, fontsize=11, pad=8)

        # ── ③ 최종 전개도 (2L + flat) ────────────────────────────────────────
        left_x = (fw - flat) / 2
        right_x = (fw + flat) / 2
        H = b  # 전개도의 시각적 높이 = 실측 측면너비 (원호 계산용 새그 h가 아님)

        ax3.plot([0, fw], [0, 0], color=BORDER2, lw=1.2)
        ax3.plot([left_x, right_x], [H, H], color=ACCENT_DK, lw=3,
                 solid_capstyle='round')

        # 제어점을 평탄부 쪽(높이 H)에 둬서 바닥 모서리에서 가파르게 올라와
        # 평탄부에 닿을 때는 수평으로 눕는 자연스러운 곡면 형태로 만든다.
        lx, lz = _bezier((0, 0), (left_x * 0.3, H), (left_x, H))
        rx, rz = _bezier((right_x, H), (fw - (fw - right_x) * 0.3, H), (fw, 0))
        ax3.plot(lx, lz, color=ACCENT, lw=2.4, solid_capstyle='round')
        ax3.plot(rx, rz, color=ACCENT, lw=2.4, solid_capstyle='round')
        ax3.fill(list(lx) + [right_x, left_x] + list(rx),
                 list(lz) + [H, H] + list(rz), color=ACCENT, alpha=0.08)

        ax3.text(left_x * 0.4, H * 0.55, 'L', color=ACCENT_DK, fontsize=11,
                 fontweight='bold', ha='center')
        ax3.text(fw - (fw - right_x) * 0.4, H * 0.55, 'L', color=ACCENT_DK,
                 fontsize=11, fontweight='bold', ha='center')
        ax3.text(fw / 2, H * 1.08, f'{r["k_f_s"]}·a = {flat:.2f}',
                 color=TEXT, fontsize=9, ha='center')

        pad3 = fw * 0.06
        ax3.set_xlim(-pad3, fw + pad3)
        ax3.set_ylim(-H * 0.15, H * 1.30)
        # 실측 mm 비율 그대로 반영 (측면너비가 커지면 실제로 더 도드라져 보인다)
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title(f'③ {r["name"]} 최종 전개도', color=TEXT, fontsize=11, pad=8)
        # 캡션은 축 상대좌표에 고정 — H가 아무리 작아도 겹치지 않는다
        ax3.text(0.5, -0.14,
                 f'곡면 길이 = 2L + {r["k_f_s"]}a = {r["total"]:.2f} mm',
                 transform=ax3.transAxes, color=ACCENT_DK, fontsize=10.5,
                 ha='center', va='top', fontweight='bold', clip_on=False)

        self.fig.suptitle(f'{r["name"]}  ·  정면너비 {r["front_width"]:g}mm · '
                          f'측면너비 {r["side_width"]:g}mm',
                          color=MUTED, fontsize=10, y=0.98)

        self.canvas.draw()


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1380x680")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
