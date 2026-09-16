import csv
import os
import tkinter as tk
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

FONT_UI = ("Malgun Gothic", 12)
FONT_TITLE = ("Malgun Gothic", 17, "bold")
FONT_SMALL = ("Malgun Gothic", 9)
FONT_STATUS = ("Malgun Gothic", 10)

HIGHLIGHT = {
    "highlightthickness": 1,
    "highlightbackground": LINE,
    "highlightcolor": ACCENT,
}


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


def display_name(row):
    pid = person_id(row)
    return f"({pid}){row[NAME_FIELD]}" if pid else f"(-){row[NAME_FIELD]}"


def sort_by_id(rows):
    def key(row):
        pid = person_id(row)
        return (0, int(pid)) if pid.isdigit() else (1, row[NAME_FIELD])
    return sorted(rows, key=key)


def find_matches(rows, query):
    query = query.strip()
    if not query:
        return sort_by_id(rows)
    id_hits = [row for row in rows if person_id(row) == query.zfill(3) or person_id(row) == query]
    exact = [row for row in rows if row[NAME_FIELD] == query]
    partial = [row for row in rows if query in row[NAME_FIELD]]
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
        self.ox_font = tkfont.Font(family="Malgun Gothic", size=16, weight="bold")

        root.title("손톱 데이터 수집 현황")
        root.geometry("520x540")
        root.resizable(False, False)
        root.configure(bg=BG)

        self._build_header()
        self._build_search()
        self._build_match_list()
        self._build_person_card()

        root.bind("<Key>", self.on_key)
        self.entry.focus_set()
        self.search()

    def _build_header(self):
        label(self.root, "손톱 데이터 수집 현황", font=FONT_TITLE).pack(pady=(18, 4))
        tk.Frame(self.root, bg=ACCENT, height=3).pack(fill="x", padx=140, pady=(0, 8))
        label(
            self.root,
            "이름이 비어 있으면 전체 인원이 나옵니다.  번호 키 1~4로도 바꿀 수 있습니다.",
            font=FONT_SMALL,
            fg=MUTED,
        ).pack()

    def _build_search(self):
        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=24, pady=12)
        label(row, "이름").pack(side="left")
        self.name_var = tk.StringVar()
        self.entry = tk.Entry(
            row,
            textvariable=self.name_var,
            font=FONT_UI,
            width=18,
            relief="flat",
            bg="white",
            fg=TEXT,
            insertbackground=TEXT,
            **HIGHLIGHT,
        )
        self.entry.pack(side="left", padx=8, ipady=6)
        self.entry.bind("<Return>", lambda _e: self.search())
        button(row, "검색", self.search, width=8).pack(side="left", ipady=3)

    def _build_match_list(self):
        self.match_list = tk.Listbox(
            self.root,
            font=FONT_UI,
            height=4,
            activestyle="none",
            relief="flat",
            bd=0,
            bg=CARD,
            fg=TEXT,
            selectbackground=ACCENT,
            selectforeground="white",
            **HIGHLIGHT,
        )
        self.match_list.pack(fill="x", padx=24)
        self.match_list.bind("<<ListboxSelect>>", self.on_pick_match)

    def _build_person_card(self):
        self.person_label = label(self.root, "이름을 검색하세요", font=FONT_TITLE)
        self.person_label.pack(pady=(14, 8))

        card = tk.Frame(self.root, bg=CARD, **HIGHLIGHT)
        card.pack(fill="x", padx=24, pady=4)
        for i, field in enumerate(DATA_FIELDS, start=1):
            self._add_data_row(card, i, field)

        self.status = label(self.root, "", font=FONT_STATUS, fg=ACCENT)
        self.status.pack(pady=12)

    def _add_data_row(self, parent, number, field):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=14, pady=8)
        label(row, f"{number}. {field}", bg=CARD, anchor="w", width=22).pack(side="left")

        badge = tk.Frame(row, bg=NEUTRAL, padx=12, pady=2)
        badge.pack(side="left", padx=8)
        ox_label = tk.Label(badge, text="-", font=self.ox_font, bg=NEUTRAL, fg=MUTED, width=2)
        ox_label.pack()
        self.ox_badges.append((badge, ox_label))

        button(row, str(number), lambda idx=number - 1: self.toggle(idx), width=4).pack(side="right")

    def set_status(self, text, fg=MUTED):
        self.status.config(text=text, fg=fg)

    def on_key(self, event):
        if self.root.focus_get() is self.entry:
            return
        if event.char in {"1", "2", "3", "4"}:
            self.toggle(int(event.char) - 1)

    def search(self):
        query = self.name_var.get().strip()
        matches = find_matches(self.rows, query)
        self.match_list.delete(0, tk.END)
        self.shown_rows = matches

        if not matches:
            self.set_person(None)
            self.set_status(f"'{query}' 이름이 없습니다.", X_FG)
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


def main():
    if not os.path.isfile(ACTIVE_CSV):
        raise FileNotFoundError(f"CSV가 없습니다: {ACTIVE_CSV}")
    root = tk.Tk()
    CheckDataApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
