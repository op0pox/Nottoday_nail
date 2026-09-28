# -*- coding: utf-8 -*-
"""
JetsonCheck.py
Jetson 촬영 사진, 3D 스캔, 손톱 분류 수집 현황을 순번(인덱스)별로 관리한다.
다른 툴(CheckData.py 등)에 의존하지 않는 단독 실행 파일이다.

폴더:
  utils/jetson_capture/<순번 또는 이름>/ 에 한 사람의 Jetson 사진이 들어간다.
  예: jetson_capture/001/  또는  jetson_capture/양재모/  (둘 다 있으면 둘 다 본다)
  [추가] 로 사람을 등록하면 순번 폴더가 만들어진다. (이름 폴더가 이미 있으면 안 만든다)

파일명 (앞부분은 순번이나 이름 아무거나):
  순번_손_손가락_side.jpg / 순번_손_손가락_front.jpg
  손: L 왼손, R 오른손   손가락: 01 엄지, 02 검지, 03 중지, 04 약지, 05 소지
  예: 001_L_01_side.jpg, 001_L_01_front.jpg
  관리자 페이지(Fast Api front)에서 손·손가락을 골라 [저장] 하면 이 이름으로 저장된다.
  확장자는 jpg, jpeg, png 모두 된다. 대소문자는 가리지 않는다.

체크 규칙:
  사진데이터   : 정면+측면 한 쌍이 10손가락 모두 있으면 자동으로 O.
                 버튼(또는 숫자키 1)으로 직접 O/X 할 수도 있다.
                 자동 체크는 X→O 로만 바꾸고, 직접 찍은 O 를 X 로 되돌리지 않는다.
  3d스캔데이터 : 따로 찍으므로 버튼(또는 숫자키 2)으로 직접 O/X.
  분류         : 손가락마다 드롭다운에서 정해진 형태만 고른다. 고르면 바로 저장.
                 사진 파일(순번_손_손가락)과 CSV 의 분류_손_손가락 칸이 같은 손가락이다.

저장:
  utils/data/dataset_jetson.csv
"""
import csv
import os
import tkinter as tk
from ctypes import byref, c_ulong, windll
from tkinter import font as tkfont
from tkinter import messagebox, ttk

# ---------- 화면 색·글꼴 (CheckData.py 와 같은 모양) ----------
BG = "#f5f5f5"
CARD = "#ffffff"
LINE = "#d4d4d4"
TEXT = "#111111"
MUTED = "#6b6b6b"
ACCENT = "#1a1a1a"
ACCENT_HOVER = "#333333"
NEUTRAL = "#ececec"
O_BG, O_FG = "#d9efe0", "#1f6b3a"
X_BG, X_FG = "#f8d7d4", "#b23a32"
FONT_FAMILY = "맑은 고딕"
FONT_UI = (FONT_FAMILY, 14)
FONT_TITLE = (FONT_FAMILY, 22, "bold")
FONT_SMALL = (FONT_FAMILY, 11)
FONT_STATUS = (FONT_FAMILY, 12)
HIGHLIGHT = {
    "highlightthickness": 1,
    "highlightbackground": LINE,
    "highlightcolor": ACCENT,
}
IME_NAMED_FONTS = (
    ("TkDefaultFont", 14),
    ("TkTextFont", 14),
    ("TkFixedFont", 14),
    ("TkMenuFont", 14),
    ("TkHeadingFont", 14),
    ("TkCaptionFont", 14),
    ("TkSmallCaptionFont", 11),
    ("TkIconFont", 14),
    ("TkTooltipFont", 11),
)
KOREAN_HKL = "00000412"
IME_CMODE_NATIVE = 0x0001
WM_INPUTLANGCHANGEREQUEST = 0x50
VK_HANGUL = 0x15
KEYEVENTF_KEYUP = 0x0002
IACE_DEFAULT = 0x0010


def label(parent, text, font=FONT_UI, fg=TEXT, bg=BG, **kwargs):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kwargs)


def button(parent, text, command, **kwargs):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=FONT_UI,
        bg=ACCENT,
        fg="white",
        activebackground=ACCENT_HOVER,
        activeforeground="white",
        relief="flat",
        cursor="hand2",
        **kwargs,
    )


def set_badge(badge, ox_label, value):
    if value == "O":
        bg, fg, text = O_BG, O_FG, "O"
    elif value == "X":
        bg, fg, text = X_BG, X_FG, "X"
    else:
        bg, fg, text = NEUTRAL, MUTED, "-"
    badge.config(bg=bg)
    ox_label.config(text=text, fg=fg, bg=bg)


def _enable_dpi_awareness():
    try:
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _apply_korean_input(root):
    try:
        root.tk.call("tk", "useinputmethods", 1)
    except tk.TclError:
        pass
    root.option_add("*Font", "{맑은 고딕} 14")
    for name, size in IME_NAMED_FONTS:
        try:
            tkfont.nametofont(name).configure(family=FONT_FAMILY, size=size)
        except tk.TclError:
            pass


def _widget_hwnd(widget):
    hwnd = int(widget.winfo_id())
    parent = windll.user32.GetParent(hwnd)
    return parent or hwnd


def _enable_hangul_ime(widget):
    try:
        hwnd = _widget_hwnd(widget)
        toplevel = _widget_hwnd(widget.winfo_toplevel())
        hkl = windll.user32.LoadKeyboardLayoutW(KOREAN_HKL, 1)
        windll.user32.ActivateKeyboardLayout(hkl, 0)
        switched = False
        for target in (hwnd, toplevel):
            windll.user32.PostMessageW(target, WM_INPUTLANGCHANGEREQUEST, 1, hkl)
            try:
                windll.imm32.ImmAssociateContextEx(target, 0, IACE_DEFAULT)
            except Exception:
                pass
            switched = _set_native_hangul(target, send_key=not switched) or switched
    except Exception:
        pass


def _set_native_hangul(hwnd, send_key=True):
    himc = windll.imm32.ImmGetContext(hwnd)
    if not himc:
        return False
    conv = c_ulong(0)
    sent = c_ulong(0)
    windll.imm32.ImmGetConversionStatus(himc, byref(conv), byref(sent))
    already_hangul = bool(conv.value & IME_CMODE_NATIVE)
    if not already_hangul:
        conv.value |= IME_CMODE_NATIVE
        windll.imm32.ImmSetConversionStatus(himc, conv.value, sent.value)
        if send_key:
            windll.user32.keybd_event(VK_HANGUL, 0, 0, 0)
            windll.user32.keybd_event(VK_HANGUL, 0, KEYEVENTF_KEYUP, 0)
    windll.imm32.ImmReleaseContext(hwnd, himc)
    return (not already_hangul) and send_key

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(SCRIPT_DIR, "jetson_capture")
JETSON_CSV = os.path.join(SCRIPT_DIR, "data", "dataset_jetson.csv")

ID_FIELD = "순번"
NAME_FIELD = "이름"
PHOTO_FIELD = "사진데이터"
SCAN_FIELD = "3d스캔데이터"

HANDS = (("L", "왼손"), ("R", "오른손"))
FINGERS = (("01", "엄지"), ("02", "검지"), ("03", "중지"), ("04", "약지"), ("05", "소지"))
FINGER_KEYS = [f"{hand}_{finger}" for hand, _ in HANDS for finger, _ in FINGERS]
VIEWS = ("front", "side")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# 드롭다운에서 고를 수 있는 형태. 형태가 늘거나 바뀌면 여기만 고친다.
SHAPES = ("P", "S", "B", "C")
EMPTY = "-"

FIELDS = [ID_FIELD, NAME_FIELD, PHOTO_FIELD, SCAN_FIELD] + [f"분류_{key}" for key in FINGER_KEYS]


def shape_field(key):
    return f"분류_{key}"


def read_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in FIELDS:
            row.setdefault(field, "")
    return rows


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def person_id(row):
    return (row.get(ID_FIELD) or "").strip()


def person_name(row):
    return (row.get(NAME_FIELD) or "").strip()


def display_name(row):
    pid = person_id(row)
    name = person_name(row) or "이름없음"
    return f"({pid}){name}" if pid else f"(-){name}"


def next_person_id(rows):
    ids = [int(person_id(row)) for row in rows if person_id(row).isdigit()]
    return "%03d" % (max(ids) + 1 if ids else 1)


def sort_by_id(rows):
    def key(row):
        pid = person_id(row)
        return (0, int(pid)) if pid.isdigit() else (1, person_name(row))
    return sorted(rows, key=key)


def find_matches(rows, query):
    query = query.strip()
    if not query:
        return sort_by_id(rows)
    id_hits = [row for row in rows if person_id(row) == query.zfill(3) or person_id(row) == query]
    exact = [row for row in rows if person_name(row) == query]
    partial = [row for row in rows if query in person_name(row)]
    return sort_by_id(id_hits or exact or partial)


def person_keys(row):
    """폴더명·파일명 앞부분으로 쓸 수 있는 값. 순번과 이름 둘 다 된다."""
    return [key for key in (person_id(row), person_name(row)) if key]


def person_dirs(row):
    """실제로 있는 이 사람의 폴더들 (순번 폴더, 이름 폴더)."""
    return [os.path.join(CAPTURE_DIR, key) for key in person_keys(row) if os.path.isdir(os.path.join(CAPTURE_DIR, key))]


def default_dir(row):
    """[폴더 열기]/[추가] 때 쓸 폴더. 있는 폴더가 우선, 없으면 순번 폴더."""
    dirs = person_dirs(row)
    if dirs:
        return dirs[0]
    return os.path.join(CAPTURE_DIR, person_id(row) or person_name(row))


def scan_photos(row):
    """순번/이름 폴더를 보고 손가락별 정면/측면이 있는지 돌려준다. {'L_01': {'front': True, 'side': False}, ...}"""
    found = {key: {view: False for view in VIEWS} for key in FINGER_KEYS}
    prefixes = {key.upper() for key in person_keys(row)}
    for folder in person_dirs(row):
        for filename in os.listdir(folder):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in IMAGE_EXTS:
                continue
            parts = stem.split("_")
            view = parts[-1].lower()
            if view not in VIEWS:
                continue
            # 앞부분_손_손가락_view. 앞부분(순번 또는 이름)에 _ 가 있어도 뒤에서부터 자른다.
            if len(parts) < 4 or "_".join(parts[:-3]).upper() not in prefixes:
                continue
            key = f"{parts[-3].upper()}_{parts[-2]}"
            if key in found:
                found[key][view] = True
    return found


def complete_pairs(found):
    return sum(1 for key in FINGER_KEYS if all(found[key].values()))


class JetsonCheckApp:
    def __init__(self, root):
        self.root = root
        self.rows = sort_by_id(read_rows(JETSON_CSV))
        self.current = None
        self.shown_rows = []
        self.shape_vars = {}
        self.shape_boxes = {}
        self._loading = False

        root.title("Jetson 데이터 수집 현황")
        root.configure(bg=BG)
        self._style_combobox()

        self._build_header()
        self._build_search()
        self._build_match_list()
        self._build_person_card()

        self.sync_all_photos()
        self.entry.focus_set()
        self.search()
        self._fit_window()
        self.root.after_idle(lambda: _enable_hangul_ime(self.entry))

    # ---------- 화면 ----------
    def _style_combobox(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Shape.TCombobox", padding=4, fieldbackground="white", background=CARD)
        style.map("Shape.TCombobox", fieldbackground=[("readonly", "white"), ("disabled", BG)])
        self.root.option_add("*TCombobox*Listbox.font", (FONT_FAMILY, 13))

    def _fit_window(self):
        self.root.update_idletasks()
        width = max(self.root.winfo_reqwidth() + 48, 860)
        height = max(self.root.winfo_reqheight() + 36, 820)
        self.root.minsize(width, height)
        self.root.geometry("%dx%d" % (width, height))

    def _build_header(self):
        label(self.root, "Jetson 데이터 수집 현황", font=FONT_TITLE).pack(pady=(22, 6))
        tk.Frame(self.root, bg=ACCENT, height=3).pack(fill="x", padx=220, pady=(0, 10))
        label(
            self.root,
            "사진은 jetson_capture/순번(또는 이름) 폴더에 10쌍(정면+측면)이 다 있으면 자동 O, 숫자키 1로 직접 체크도 됩니다. "
            "3D 스캔은 숫자키 2. 분류는 손가락마다 드롭다운에서 고르면 바로 저장됩니다.",
            font=FONT_SMALL,
            fg=MUTED,
            wraplength=780,
        ).pack(padx=28)

    def _build_search(self):
        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=28, pady=14)
        button(row, "새로고침", self.refresh, width=8).pack(side="right", ipady=4)
        button(row, "추가", self.add_person, width=8).pack(side="right", ipady=4, padx=(0, 8))
        button(row, "검색", self.search, width=8).pack(side="right", ipady=4, padx=(0, 8))
        label(row, "이름").pack(side="left")
        self.entry = tk.Entry(
            row,
            font=(FONT_FAMILY, 14),
            relief="flat",
            bg="white",
            fg=TEXT,
            insertbackground=TEXT,
            **HIGHLIGHT,
        )
        self.entry.pack(side="left", padx=10, ipady=8, fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self.search())
        self.entry.bind("<FocusIn>", self._on_entry_focus)
        self.entry.bind("<FocusOut>", self._on_entry_focus_out)
        self._bind_number_keys()

    def _build_match_list(self):
        self.match_list = tk.Listbox(
            self.root,
            font=FONT_UI,
            height=5,
            activestyle="none",
            relief="flat",
            bd=0,
            bg=CARD,
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground="white",
            **HIGHLIGHT,
        )
        self.match_list.pack(fill="both", expand=True, padx=28)
        self.match_list.bind("<<ListboxSelect>>", self.on_pick_match)

    def _build_person_card(self):
        self.person_label = label(self.root, "이름을 검색하세요", font=FONT_TITLE)
        self.person_label.pack(pady=(16, 10))

        card = tk.Frame(self.root, bg=CARD, **HIGHLIGHT)
        card.pack(fill="x", padx=28, pady=6)

        # 1. 사진데이터 (자동)
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=18, pady=(12, 4))
        label(row, "1. 사진데이터", bg=CARD, anchor="w", width=18).pack(side="left")
        self.photo_badge, self.photo_ox = self._badge(row)
        self.photo_count = label(row, "", bg=CARD, fg=MUTED, font=FONT_SMALL)
        self.photo_count.pack(side="left", padx=8)
        button(row, "1", self.toggle_photo, width=4).pack(side="right")
        button(row, "폴더 열기", self.open_folder, width=9).pack(side="right", padx=(0, 8))

        self.photo_missing = label(card, "", bg=CARD, fg=MUTED, font=FONT_SMALL, anchor="w", justify="left", wraplength=760)
        self.photo_missing.pack(fill="x", padx=18, pady=(0, 8))

        # 2. 3d스캔데이터 (수동)
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=18, pady=8)
        label(row, "2. 3D 스캔데이터", bg=CARD, anchor="w", width=18).pack(side="left")
        self.scan_badge, self.scan_ox = self._badge(row)
        button(row, "2", self.toggle_scan, width=4).pack(side="right")

        # 3. 분류 (드롭다운)
        tk.Frame(card, bg=HIGHLIGHT["highlightbackground"], height=1).pack(fill="x", padx=18, pady=(8, 10))
        label(card, "분류", bg=CARD, anchor="w").pack(fill="x", padx=18)
        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill="x", padx=18, pady=(6, 16))
        for col, (_, finger_name) in enumerate(FINGERS, start=1):
            label(grid, finger_name, bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=col, padx=6, pady=(0, 4))
        for r, (hand, hand_name) in enumerate(HANDS, start=1):
            label(grid, hand_name, bg=CARD, anchor="w", width=6).grid(row=r, column=0, sticky="w", pady=4)
            for col, (finger, _) in enumerate(FINGERS, start=1):
                key = f"{hand}_{finger}"
                var = tk.StringVar(value=EMPTY)
                box = ttk.Combobox(
                    grid,
                    textvariable=var,
                    values=(EMPTY,) + SHAPES,
                    state="disabled",
                    width=5,
                    justify="center",
                    font=(FONT_FAMILY, 13),
                    style="Shape.TCombobox",
                )
                box.grid(row=r, column=col, padx=6, pady=4)
                box.bind("<<ComboboxSelected>>", lambda _e, k=key: self.on_shape(k))
                self.shape_vars[key] = var
                self.shape_boxes[key] = box
        for col in range(1, len(FINGERS) + 1):
            grid.grid_columnconfigure(col, weight=1)

        self.status = label(self.root, "", font=FONT_STATUS, fg=ACCENT, wraplength=780)
        self.status.pack(fill="x", padx=28, pady=16)

    def _badge(self, parent):
        badge = tk.Frame(parent, bg=CARD, padx=12, pady=2)
        badge.pack(side="left", padx=8)
        ox_label = tk.Label(badge, text="-", font=(FONT_FAMILY, 20, "bold"), bg=CARD, fg=MUTED, width=2)
        ox_label.pack()
        return badge, ox_label

    # ---------- 입력 ----------
    def set_status(self, text, fg=MUTED):
        self.status.config(text=text, fg=fg)

    def query_text(self):
        return self.entry.get().strip()

    def set_query(self, text):
        self.entry.delete(0, tk.END)
        if text:
            self.entry.insert(0, text)

    def _bind_number_keys(self):
        for number in "12":
            self.root.bind(number, self._on_number_key)

    def _unbind_number_keys(self):
        for number in "12":
            self.root.unbind(number)

    def _on_number_key(self, event):
        if self.root.focus_get() == self.entry:
            return
        if isinstance(self.root.focus_get(), ttk.Combobox):
            return
        if event.char == "1":
            self.toggle_photo()
        elif event.char == "2":
            self.toggle_scan()

    def _on_entry_focus(self, _event=None):
        self._unbind_number_keys()
        self.root.after_idle(lambda: _enable_hangul_ime(self.entry))

    def _on_entry_focus_out(self, _event=None):
        self._bind_number_keys()

    # ---------- 동작 ----------
    def sync_all_photos(self):
        """모든 사람의 사진 폴더를 다시 보고 사진데이터 O/X 를 갱신한다."""
        changed = False
        for row in self.rows:
            if not person_keys(row):
                continue
            if row.get(PHOTO_FIELD) != "O" and complete_pairs(scan_photos(row)) == len(FINGER_KEYS):
                row[PHOTO_FIELD] = "O"
                changed = True
            elif row.get(PHOTO_FIELD) not in ("O", "X"):
                row[PHOTO_FIELD] = "X"
                changed = True
        if changed:
            write_rows(JETSON_CSV, self.rows)
        return changed

    def refresh(self):
        self.rows = sort_by_id(read_rows(JETSON_CSV))
        self.sync_all_photos()
        pid = person_id(self.current) if self.current else None
        self.search()
        if pid:
            for i, row in enumerate(self.shown_rows):
                if person_id(row) == pid:
                    self.match_list.selection_clear(0, tk.END)
                    self.match_list.selection_set(i)
                    self.set_person(row)
                    break
        self.set_status("폴더를 다시 확인했습니다.", O_FG)

    def search(self):
        query = self.query_text()
        matches = find_matches(self.rows, query)
        self.match_list.delete(0, tk.END)
        self.shown_rows = matches

        if not matches:
            self.set_person(None)
            if self.rows:
                self.set_status(f"'{query}' 이름이 없습니다. 추가로 등록할 수 있습니다.", X_FG)
            else:
                self.set_status("등록된 사람이 없습니다. 이름을 적고 [추가] 를 누르세요.")
            return

        for row in matches:
            self.match_list.insert(tk.END, display_name(row))
        self.match_list.selection_set(0)
        self.set_person(matches[0])

        if query:
            self.set_status(f"{len(matches)}명 검색됨")
            self.root.focus_set()
        else:
            self.set_status(f"전체 {len(matches)}명")
            self.entry.focus_set()

    def add_person(self):
        name = self.query_text()
        if not name:
            if not messagebox.askyesno("이름 없이 추가", "이름이 없습니다. 이름 없이 추가하시겠습니까?"):
                self.set_status("추가를 취소했습니다.")
                return
        else:
            if any(person_name(row) == name for row in self.rows):
                self.set_status(f"'{name}' 은(는) 이미 있습니다.", X_FG)
                return
            if not messagebox.askyesno("새 인원 추가", "'%s' 을(를) 추가하시겠습니까?" % name):
                self.set_status("추가를 취소했습니다.")
                return

        row = {field: "" for field in FIELDS}
        row[ID_FIELD] = next_person_id(self.rows)
        row[NAME_FIELD] = name
        row[PHOTO_FIELD] = "X"
        row[SCAN_FIELD] = "X"
        os.makedirs(default_dir(row), exist_ok=True)

        self.rows.append(row)
        self.rows = sort_by_id(self.rows)
        self.sync_all_photos()
        write_rows(JETSON_CSV, self.rows)

        self.set_query(person_id(row) if not name else name)
        self.search()
        self.set_person(row)
        folder = os.path.basename(default_dir(row))
        self.set_status("%s 추가됨. 사진은 jetson_capture/%s 폴더에 넣으세요. (이름 폴더도 됩니다)" % (display_name(row), folder), O_FG)

    def on_pick_match(self, _event=None):
        selected = self.match_list.curselection()
        if not selected:
            return
        self.set_person(self.shown_rows[selected[0]])

    def set_person(self, row):
        self.current = row
        self._loading = True
        if row is None:
            self.person_label.config(text="이름을 검색하세요")
            set_badge(self.photo_badge, self.photo_ox, "-")
            set_badge(self.scan_badge, self.scan_ox, "-")
            self.photo_count.config(text="")
            self.photo_missing.config(text="")
            for key in FINGER_KEYS:
                self.shape_vars[key].set(EMPTY)
                self.shape_boxes[key].config(state="disabled")
            self._loading = False
            return

        self.person_label.config(text=display_name(row))
        self.refresh_photo()
        set_badge(self.scan_badge, self.scan_ox, row.get(SCAN_FIELD) or "X")
        for key in FINGER_KEYS:
            value = (row.get(shape_field(key)) or "").strip()
            self.shape_vars[key].set(value if value in SHAPES else EMPTY)
            self.shape_boxes[key].config(state="readonly")
        self._loading = False

    def refresh_photo(self):
        found = scan_photos(self.current)
        pairs = complete_pairs(found)
        value = self.current.get(PHOTO_FIELD) or "X"
        if value != "O" and pairs == len(FINGER_KEYS):
            value = "O"
            self.current[PHOTO_FIELD] = value
            write_rows(JETSON_CSV, self.rows)
        set_badge(self.photo_badge, self.photo_ox, value)
        self.photo_count.config(text=f"파일 {pairs}/{len(FINGER_KEYS)}쌍")

        missing = []
        names = {f"{h}_{f}": f"{hn} {fn}" for h, hn in HANDS for f, fn in FINGERS}
        for key in FINGER_KEYS:
            lack = [("정면" if v == "front" else "측면") for v in VIEWS if not found[key][v]]
            if lack:
                missing.append(f"{names[key]}({'/'.join(lack)})")
        if not person_dirs(self.current):
            names = " 또는 ".join(f"jetson_capture/{key}" for key in person_keys(self.current))
            self.photo_missing.config(text=f"{names} 폴더가 없습니다. [폴더 열기] 를 누르면 순번 폴더가 만들어집니다.", fg=X_FG)
        elif missing:
            self.photo_missing.config(text="빠진 사진: " + ", ".join(missing), fg=MUTED)
        else:
            self.photo_missing.config(text="10쌍 모두 있습니다.", fg=O_FG)

    def open_folder(self):
        if self.current is None:
            self.set_status("먼저 이름을 검색하세요.", X_FG)
            return
        folder = default_dir(self.current)
        os.makedirs(folder, exist_ok=True)
        try:
            os.startfile(folder)
        except (AttributeError, OSError):
            self.set_status(folder)
        self.refresh_photo()

    def toggle_photo(self):
        self._toggle(PHOTO_FIELD, "사진데이터", self.photo_badge, self.photo_ox)

    def toggle_scan(self):
        self._toggle(SCAN_FIELD, "3D 스캔데이터", self.scan_badge, self.scan_ox)

    def _toggle(self, field, title, badge, ox_label):
        if self.current is None:
            self.set_status("먼저 이름을 검색하세요.", X_FG)
            return
        prev = self.current.get(field) or "X"
        if prev == "O":
            if not messagebox.askyesno("이미 수집한 데이터", f"이미 체크된 {title}입니다. X로 바꾸시겠습니까?"):
                self.set_status("O 유지")
                return
            next_value = "X"
        else:
            next_value = "O"
        self.current[field] = next_value
        write_rows(JETSON_CSV, self.rows)
        set_badge(badge, ox_label, next_value)
        note = ""
        if field == PHOTO_FIELD and next_value == "X" and complete_pairs(scan_photos(self.current)) == len(FINGER_KEYS):
            note = "  ※ 폴더에 10쌍이 다 있어서 새로고침하면 다시 O가 됩니다."
        self.set_status(f"{title}: {prev} → {next_value}  (저장됨){note}", O_FG)

    def on_shape(self, key):
        if self._loading or self.current is None:
            return
        value = self.shape_vars[key].get()
        if value not in SHAPES and value != EMPTY:
            self.shape_vars[key].set(EMPTY)
            return
        self.current[shape_field(key)] = "" if value == EMPTY else value
        write_rows(JETSON_CSV, self.rows)
        hand, finger = key.split("_")
        hand_name = dict(HANDS)[hand]
        finger_name = dict(FINGERS)[finger]
        self.set_status(f"{hand_name} {finger_name} 분류: {value}  (저장됨)", O_FG)
        self.root.focus_set()


def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    if not os.path.isfile(JETSON_CSV):
        write_rows(JETSON_CSV, [])
    _enable_dpi_awareness()
    root = tk.Tk()
    _apply_korean_input(root)
    JetsonCheckApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
