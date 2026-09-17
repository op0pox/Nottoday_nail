import csv
import os
import tkinter as tk
from ctypes import byref, c_ulong, windll
from tkinter import font as tkfont
from tkinter import messagebox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_CSV = os.path.join(SCRIPT_DIR, "data", "dataset_original.csv")

ID_FIELD = "순번"
NAME_FIELD = "이름"
DATA_FIELDS = [
    "체커보드 측면여부",
    "체커보드 정면 여부",
    "흰색배경 여부",
    "3d스캔데이터 여부",
]

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


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    fieldnames = [ID_FIELD, NAME_FIELD] + DATA_FIELDS
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def new_person_row(name, rows):
    return {
        ID_FIELD: next_person_id(rows),
        NAME_FIELD: name,
        **{field: "X" for field in DATA_FIELDS},
    }


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


class CheckDataApp:
    def __init__(self, root):
        self.root = root
        self.rows = sort_by_id(read_rows(ACTIVE_CSV))
        self.current = None
        self.shown_rows = []
        self.ox_badges = []
        self.ox_font = tkfont.Font(family=FONT_FAMILY, size=20, weight="bold")

        root.title("손톱 데이터 수집 현황")
        root.minsize(760, 720)
        root.geometry("780x740")
        root.resizable(True, True)
        root.configure(bg=BG)

        self._build_header()
        self._build_search()
        self._build_match_list()
        self._build_person_card()

        self.entry.focus_set()
        self.search()
        self._fit_window()
        self.root.after_idle(lambda: _enable_hangul_ime(self.entry))

    def _fit_window(self):
        self.root.update_idletasks()
        width = max(self.root.winfo_reqwidth() + 48, 780)
        height = max(self.root.winfo_reqheight() + 36, 740)
        self.root.minsize(width, height)
        self.root.geometry("%dx%d" % (width, height))

    def _build_header(self):
        label(self.root, "손톱 데이터 수집 현황", font=FONT_TITLE).pack(pady=(22, 6))
        tk.Frame(self.root, bg=ACCENT, height=3).pack(fill="x", padx=180, pady=(0, 10))
        label(
            self.root,
            "이름이 비어 있으면 전체 인원이 나옵니다. 이름 없이 추가도 가능합니다. 번호 키 1~4로도 바꿀 수 있습니다.",
            font=FONT_SMALL,
            fg=MUTED,
            wraplength=700,
        ).pack(padx=28)

    def _build_search(self):
        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=28, pady=14)
        button(row, "추가", self.add_person, width=8).pack(side="right", ipady=4)
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
            height=6,
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
        for i, field in enumerate(DATA_FIELDS, start=1):
            self._add_data_row(card, i, field)

        self.status = label(self.root, "", font=FONT_STATUS, fg=ACCENT, wraplength=700)
        self.status.pack(fill="x", padx=28, pady=16)

    def _add_data_row(self, parent, number, field):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=18, pady=10)
        label(row, f"{number}. {field}", bg=CARD, anchor="w", width=24).pack(side="left")

        badge = tk.Frame(row, bg=NEUTRAL, padx=12, pady=2)
        badge.pack(side="left", padx=8)
        ox_label = tk.Label(badge, text="-", font=self.ox_font, bg=NEUTRAL, fg=MUTED, width=2)
        ox_label.pack()
        self.ox_badges.append((badge, ox_label))

        button(row, str(number), lambda idx=number - 1: self.toggle(idx), width=4).pack(side="right")

    def set_status(self, text, fg=MUTED):
        self.status.config(text=text, fg=fg)

    def query_text(self):
        return self.entry.get().strip()

    def set_query(self, text):
        self.entry.delete(0, tk.END)
        if text:
            self.entry.insert(0, text)

    def _bind_number_keys(self):
        for number in "1234":
            self.root.bind(number, self._on_number_key)

    def _unbind_number_keys(self):
        for number in "1234":
            self.root.unbind(number)

    def _on_number_key(self, event):
        if self.root.focus_get() == self.entry:
            return
        if event.char in {"1", "2", "3", "4"}:
            self.toggle(int(event.char) - 1)

    def _on_entry_focus(self, _event=None):
        self._unbind_number_keys()
        self.root.after_idle(lambda: _enable_hangul_ime(self.entry))

    def _on_entry_focus_out(self, _event=None):
        self._bind_number_keys()

    def search(self):
        query = self.query_text()
        matches = find_matches(self.rows, query)
        self.match_list.delete(0, tk.END)
        self.shown_rows = matches

        if not matches:
            self.set_person(None)
            self.set_status(f"'{query}' 이름이 없습니다. 추가로 등록할 수 있습니다.", X_FG)
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
            if not messagebox.askyesno(
                "이름 없이 추가",
                "이름이 없습니다. 이름 없이 추가하시겠습니까?\n수집 항목은 전부 X로 저장됩니다.",
            ):
                self.set_status("추가를 취소했습니다.")
                return
        else:
            if any(person_name(row) == name for row in self.rows):
                self.set_status(f"'{name}' 은(는) 이미 있습니다.", X_FG)
                return
            if not messagebox.askyesno(
                "새 인원 추가",
                "'%s' 을(를) 추가하시겠습니까?\n수집 항목은 전부 X로 저장됩니다." % name,
            ):
                self.set_status("추가를 취소했습니다.")
                return

        row = new_person_row(name, self.rows)
        self.rows.append(row)
        self.rows = sort_by_id(self.rows)
        write_rows(ACTIVE_CSV, self.rows)

        self.set_query(person_id(row) if not name else name)
        self.search()
        self.set_person(row)
        self.set_status("%s 추가됨 (기본값 전부 X)" % display_name(row), O_FG)

    def on_pick_match(self, _event=None):
        selected = self.match_list.curselection()
        if not selected:
            return
        self.set_person(self.shown_rows[selected[0]])

    def set_person(self, row):
        self.current = row
        if row is None:
            self.person_label.config(text="이름을 검색하세요")
            for badge, ox_label in self.ox_badges:
                set_badge(badge, ox_label, "-")
            return
        self.person_label.config(text=display_name(row))
        self.refresh_ox()

    def refresh_ox(self):
        for (badge, ox_label), field in zip(self.ox_badges, DATA_FIELDS):
            set_badge(badge, ox_label, self.current[field])

    def toggle(self, index):
        if self.current is None:
            self.set_status("먼저 이름을 검색하세요.", X_FG)
            return

        field = DATA_FIELDS[index]
        prev = self.current[field]
        next_value = self._next_value(prev)
        if next_value is None:
            return

        self.current[field] = next_value
        write_rows(ACTIVE_CSV, self.rows)
        self.refresh_ox()
        self.set_status(f"{field}: {prev} → {next_value}  (저장됨)", O_FG)

    def _next_value(self, current):
        if current == "X":
            return "O"
        if current == "O":
            keep = not messagebox.askyesno(
                "이미 수집한 데이터",
                "이미 수집한 데이터입니다 X로 바꾸시겠습니까?\n\n예: X로 변경\n아니오: O 유지",
            )
            if keep:
                self.set_status("O 유지")
                return None
            return "X"
        self.set_status(f"값이 O/X가 아닙니다: {current}", X_FG)
        return None


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


def main():
    if not os.path.isfile(ACTIVE_CSV):
        raise FileNotFoundError(f"CSV가 없습니다: {ACTIVE_CSV}")
    _enable_dpi_awareness()
    root = tk.Tk()
    _apply_korean_input(root)
    CheckDataApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
