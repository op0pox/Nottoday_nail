#!/usr/bin/env python3
"""
네일 팁 곡면 계산기 — N형 (단일 원호)

입력:
  w : 위에서 본 손톱 너비 (mm)
  h : 측면에서 측정한 최고 높이 = 새그 (mm)

계산:
  chord = w  (전체 너비)
  R     = chord² / (8h) + h/2
  θ     = 2 × arcsin(chord/2 / R)
  L     = R × θ   ← 곡면 길이
"""
import math
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa

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


# ── 계산 ──────────────────────────────────────────────────────────────────────
def calc(w: float, h: float) -> dict:
    chord = w  # 전체 너비를 현(chord)으로 사용
    R = chord ** 2 / (8 * h) + h / 2
    sin_v = (chord / 2) / R
    if sin_v > 1.0:
        raise ValueError(f"h({h:.2f}mm) 가 너무 큽니다.")
    theta = 2 * math.asin(sin_v)
    L = R * theta  # 단일 호 길이 = 곡면 길이
    return dict(chord=chord, h=h, R=R,
                theta_deg=math.degrees(theta), L=L)


# ── 단면 프로파일 (단일 원호) ──────────────────────────────────────────────────
def cross_section(w: float, h: float, n: int = 200):
    """
    N형 단면: 너비 w 전체가 하나의 원호
      (0, 0) → (w/2, h) → (w, 0)
    """
    R_vis = w ** 2 / (8 * h) + h / 2
    Cx, Cz = w / 2, h - R_vis

    a_start = math.atan2(0 - Cz, 0 - Cx)    # 왼쪽 끝 (0,0) 각도
    a_end   = math.atan2(0 - Cz, w - Cx)    # 오른쪽 끝 (w,0) 각도

    t = np.linspace(a_start, a_end, n)
    x = Cx + R_vis * np.cos(t)
    z = Cz + R_vis * np.sin(t)
    return x, z, R_vis


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


# ── GUI ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("네일 팁 곡면 계산기 — N형")
        root.configure(bg=BG)
        self._after_id = None
        self._build_ui()
        self._update()

    # ── 왼쪽 패널 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        left = tk.Frame(self.root, bg=BG, padx=22, pady=20, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # 타이틀
        tk.Label(left, text="네일 곡면 계산기", font=(FONT, 18, 'bold'),
                 bg=BG, fg=TEXT, anchor='w').pack(fill=tk.X)
        tk.Label(left, text="N형 · 단일 원호 모델", font=(FONT, 11),
                 bg=BG, fg=MUTED, anchor='w').pack(fill=tk.X, pady=(2, 18))

        # 입력
        tk.Label(left, text="측정 입력", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X,
                                                          pady=(0, 6))
        self.w_field = NumberField(left, "손톱 너비  w", "14",
                                   step=0.5, on_change=self._schedule)
        self.w_field.pack(fill=tk.X, pady=(0, 12))
        self.h_field = NumberField(left, "손톱 높이  h", "2",
                                   step=0.1, on_change=self._schedule)
        self.h_field.pack(fill=tk.X, pady=(0, 6))

        # 상태 표시 (팝업 대신 인라인)
        self.status = tk.Label(left, text="", bg=BG, fg=MUTED,
                               font=(FONT, 10), anchor='w')
        self.status.pack(fill=tk.X, pady=(0, 10))

        # 계산 결과
        tk.Label(left, text="계산 결과", bg=BG, fg=MUTED,
                 font=(FONT, 9, 'bold'), anchor='w').pack(fill=tk.X,
                                                          pady=(4, 6))

        # 메인 결과 카드 — 곡면 길이 L (블루 틴트)
        card = tk.Frame(left, bg=ACCENT_BG, padx=16, pady=14,
                        highlightthickness=1, highlightbackground='#c8dafc')
        card.pack(fill=tk.X)
        top = tk.Frame(card, bg=ACCENT_BG)
        top.pack(fill=tk.X)
        tk.Label(top, text="곡면 길이  L", bg=ACCENT_BG, fg=ACCENT_DK,
                 font=(FONT, 11, 'bold')).pack(side=tk.LEFT)
        FlatButton(top, '복사', self._copy, bg=ACCENT, fg='white',
                   hover_bg=ACCENT_DK, font=(FONT, 9, 'bold'),
                   padx=10, pady=2).pack(side=tk.RIGHT)
        row = tk.Frame(card, bg=ACCENT_BG)
        row.pack(fill=tk.X, pady=(6, 0))
        self.L_label = tk.Label(row, text="—", bg=ACCENT_BG, fg=ACCENT_DK,
                                font=(FONT, 30, 'bold'))
        self.L_label.pack(side=tk.LEFT)
        tk.Label(row, text=" mm", bg=ACCENT_BG, fg=MUTED,
                 font=(FONT, 13)).pack(side=tk.LEFT, anchor='s', pady=(0, 5))

        # 보조 결과 미니 카드 2×2 (R / θ / C커브 / chord)
        grid = tk.Frame(left, bg=BG)
        grid.pack(fill=tk.X, pady=(10, 0))
        self._mini = {}
        for i, (key, lbl, dot) in enumerate([
            ("R",      "곡률 반지름 R", '#2f6fed'),
            ("theta",  "호의 각도 θ",   '#5b8af0'),
            ("ccurve", "C커브",         '#8fb0f5'),
            ("chord",  "현 chord",      '#9ca3af'),
        ]):
            cell = tk.Frame(grid, bg=SURFACE, padx=10, pady=8,
                            highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=i // 2, column=i % 2, sticky='nsew',
                      padx=(0 if i % 2 == 0 else 6, 0),
                      pady=(0 if i < 2 else 6, 0))
            grid.columnconfigure(i % 2, weight=1)
            head_ = tk.Frame(cell, bg=SURFACE)
            head_.pack(fill=tk.X)
            tk.Label(head_, text='●', bg=SURFACE, fg=dot,
                     font=(FONT, 6)).pack(side=tk.LEFT)
            tk.Label(head_, text=' ' + lbl, bg=SURFACE, fg=MUTED,
                     font=(FONT, 9)).pack(side=tk.LEFT)
            v = tk.Label(cell, text="—", bg=SURFACE, fg=TEXT,
                         font=(FONT, 13, 'bold'))
            v.pack(anchor='w', pady=(3, 0))
            self._mini[key] = v

        # 공식 카드 (하단 고정)
        fcard = tk.Frame(left, bg=SURFACE, padx=12, pady=10,
                         highlightthickness=1, highlightbackground=BORDER)
        fcard.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        tk.Label(fcard,
                 text=("R = w²/(8h) + h/2\n"
                       "θ = 2·arcsin(w/2R)\n"
                       "L = R·θ"),
                 bg=SURFACE, fg='#6b7280', font=('Courier', 10),
                 justify='left', anchor='w').pack(fill=tk.X)
        tk.Label(left, text="공식", bg=BG, fg=MUTED, font=(FONT, 9, 'bold'),
                 anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

        # 오른쪽 matplotlib 패널
        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = plt.figure(figsize=(12, 7), facecolor=BG)
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
        txt = self.L_label.cget('text')
        if txt != "—":
            self.root.clipboard_clear()
            self.root.clipboard_append(txt)
            self.status.config(text=f"✓ 복사됨: {txt} mm", fg=GREEN)

    def _update(self):
        self._after_id = None
        try:
            w = self.w_field.get()
            h = self.h_field.get()
            assert w > 0 and h > 0
        except Exception:
            self.status.config(text="⚠ w, h 모두 양수를 입력하세요", fg=RED)
            self.w_field.mark(False)
            self.h_field.mark(False)
            return
        self.w_field.mark(True)
        self.h_field.mark(True)
        try:
            r = calc(w, h)
        except ValueError as e:
            self.status.config(text=f"⚠ {e}", fg=RED)
            self.h_field.mark(False)
            return

        self.status.config(text="", fg=MUTED)
        self.L_label.config(text=f"{r['L']:.2f}")
        self._mini["R"].config(text=f"{r['R']:.2f} mm")
        self._mini["theta"].config(text=f"{r['theta_deg']:.1f}°")
        self._mini["ccurve"].config(text=f"{r['theta_deg'] / 360 * 100:.0f} %")
        self._mini["chord"].config(text=f"{r['chord']:.2f} mm")
        self._draw(w, h, r)

    # ── 그래프 ───────────────────────────────────────────────────────────────
    def _draw(self, w: float, h: float, r: dict):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 1, figure=self.fig,
                               left=0.06, right=0.94,
                               top=0.93, bottom=0.09,
                               hspace=0.30, height_ratios=[1.5, 1])
        ax3 = self.fig.add_subplot(gs[0], projection='3d')
        ax2 = self.fig.add_subplot(gs[1])

        x_arr, z_arr, R_vis = cross_section(w, h)
        nail_len = w * 1.2

        # ── 3D 곡면 ──────────────────────────────────────────────────────────
        nx, ny = len(x_arr), 50
        ys = np.linspace(0, nail_len, ny)
        X3 = np.tile(x_arr, (ny, 1))
        Y3 = np.tile(ys, (nx, 1)).T
        Z3 = np.tile(z_arr, (ny, 1))

        # 높이에 따라 밝아지는 블루 그라데이션
        zn = (z_arr - z_arr.min()) / max(z_arr.max() - z_arr.min(), 1e-9)
        lo = np.array([0.28, 0.44, 0.82])
        hi = np.array([0.78, 0.86, 0.98])
        col = lo + (hi - lo) * zn[:, None]
        C3 = np.empty((ny, nx, 4))
        C3[..., :3] = np.tile(col, (ny, 1, 1))
        C3[..., 3] = 0.96

        # 길이 방향으로 미묘한 명암 변화 (입체감)
        shade = np.linspace(0.90, 1.06, ny)[:, None, None]
        C3[..., :3] = np.clip(C3[..., :3] * shade, 0, 1)

        ax3.plot_surface(X3, Y3, Z3, facecolors=C3,
                         shade=False, linewidth=0, antialiased=True)
        # 표면 결 라인 (하이라이트)
        for idx in range(12, nx - 12, 44):
            ax3.plot(np.full(ny, x_arr[idx]), ys, np.full(ny, z_arr[idx]),
                     color='white', lw=0.7, alpha=0.35)
        # 앞/뒤 실루엣
        for y0 in (0, nail_len):
            ax3.plot(x_arr, np.full(nx, y0), z_arr,
                     color=ACCENT_DK, lw=1.3, alpha=0.8)
        # 바닥 테두리
        ax3.plot([0, w, w, 0, 0], [0, 0, nail_len, nail_len, 0],
                 [0, 0, 0, 0, 0], color=BORDER2, lw=0.9)

        ax3.set_box_aspect([w, nail_len, max(h * 1.5, w * 0.08)])
        ax3.view_init(elev=25, azim=-60)
        ax3.set_facecolor(BG)
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(BORDER)
        ax3.grid(False)
        ax3.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax3.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax3.zaxis.set_major_locator(MaxNLocator(3))
        ax3.tick_params(colors=MUTED, labelsize=7)
        ax3.set_xlabel('너비 (mm)', color=MUTED, labelpad=4, fontsize=8)
        ax3.set_ylabel('길이 (mm)', color=MUTED, labelpad=4, fontsize=8)
        ax3.set_zlabel('높이 (mm)', color=MUTED, labelpad=2, fontsize=8)
        ax3.set_title('3D 미리보기', color=TEXT, pad=8, fontsize=12)

        # ── 2D 단면도 ─────────────────────────────────────────────────────────
        ax2.set_facecolor(SURFACE)
        ax2.grid(True, color='#eef0f4', lw=0.8)
        ax2.set_axisbelow(True)

        ax2.fill(list(x_arr) + [w, 0], list(z_arr) + [0, 0],
                 color=ACCENT, alpha=0.08)
        ax2.plot(x_arr, z_arr, color=ACCENT, lw=2.6,
                 solid_capstyle='round')
        ax2.plot([0, w], [0, 0], color=BORDER2, lw=1.2)

        # h 치수선
        cx = w / 2
        ax2.annotate('', xy=(cx, h), xytext=(cx, 0),
                     arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.3))
        ax2.text(cx + w * 0.03, h / 2, f'h = {h:g}',
                 color=TEXT, fontsize=9.5, va='center')

        # w 치수선
        yb = -h * 0.30
        ax2.annotate('', xy=(w, yb), xytext=(0, yb),
                     arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.3))
        ax2.text(w / 2, yb - h * 0.16, f'w = {w:g}',
                 color=TEXT, fontsize=9.5, ha='center', va='top')

        # 펼친 길이 L 비교 바 (호를 평평하게 폈을 때)
        yL = -h * 0.95
        x0, x1 = w / 2 - r['L'] / 2, w / 2 + r['L'] / 2
        ax2.plot([x0, x1], [yL, yL], color=ACCENT, lw=3.2,
                 solid_capstyle='round')
        for xe in (x0, x1):
            ax2.plot([xe, xe], [yL - h * 0.07, yL + h * 0.07],
                     color=ACCENT, lw=1.5)
        ax2.text(w / 2, yL - h * 0.16,
                 f'펼친 길이  L = {r["L"]:.2f} mm',
                 color=ACCENT_DK, fontsize=10, ha='center', va='top',
                 fontweight='bold')

        pad = w * 0.08
        ax2.set_xlim(min(0, x0) - pad, max(w, x1) + pad)
        ax2.set_ylim(-h * 1.55, h * 1.55)
        ax2.set_aspect('equal', adjustable='box')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(3))
        ax2.set_title('단면도 · 펼친 길이 비교 (실제 비율)', color=TEXT,
                      fontsize=12, pad=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for sp in ax2.spines.values():
            sp.set_color(BORDER)

        self.canvas.draw()


# ── 실행 ──────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1240x720")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
