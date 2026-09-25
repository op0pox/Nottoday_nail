# -*- coding: utf-8 -*-
"""
폴더의 이미지와 같은 이름의 LabelMe json을 보면서
회전하거나 파일명을 바꾼다. 저장은 그 폴더에 바로 덮어쓴다.

    python ReviewImage.py
"""
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BAD_NAME = set('\\/:*?"<>|')

BG = "#f5f5f5"
TEXT = "#111111"
MUTED = "#6b6b6b"
ACCENT = "#1a1a1a"
FONT = ("맑은 고딕", 12)
FONT_TITLE = ("맑은 고딕", 18, "bold")
FONT_SMALL = ("맑은 고딕", 11)


def rotate_xy(x, y, width, height, turn):
    if turn == "cw":
        return height - 1 - y, x
    if turn == "ccw":
        return y, width - 1 - x
    return width - 1 - x, height - 1 - y


def rotate_image(image, turn):
    if turn == "cw":
        return image.transpose(Image.Transpose.ROTATE_270)
    if turn == "ccw":
        return image.transpose(Image.Transpose.ROTATE_90)
    return image.transpose(Image.Transpose.ROTATE_180)


def rotate_json(data, width, height, turn):
    for shape in data.get("shapes") or []:
        points = shape.get("points") or []
        shape["points"] = [
            [float(nx), float(ny)]
            for x, y in points
            for nx, ny in [rotate_xy(float(x), float(y), width, height, turn)]
        ]
    if turn == "180":
        data["imageWidth"] = width
        data["imageHeight"] = height
    else:
        data["imageWidth"] = height
        data["imageHeight"] = width
    data["imageData"] = None


def save_image(image, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(path, quality=95)
    else:
        image.save(path)


def list_pairs(folder):
    images = {}
    jsons = {}
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        low = ext.lower()
        if low in IMAGE_EXTS:
            images.setdefault(stem, path)
        elif low == ".json":
            jsons[stem] = path
    stems = sorted(set(images) | set(jsons), key=str.lower)
    return stems, images, jsons


def valid_stem(stem):
    stem = stem.strip()
    if not stem or stem in {".", ".."}:
        return "이름을 입력하세요."
    if any(ch in BAD_NAME for ch in stem) or stem.endswith(" ") or stem.endswith("."):
        return "파일명에 쓸 수 없는 문자가 있습니다."
    return ""


class ReviewApp:
    def __init__(self, root):
        self.root = root
        self.folder = ""
        self.stems = []
        self.images = {}
        self.jsons = {}
        self.index = 0
        self.photo = None

        root.title("이미지 / json 확인")
        root.geometry("1100x760")
        root.minsize(900, 640)
        root.configure(bg=BG)
        self._build()
        root.bind("<Left>", self._on_left)
        root.bind("<Right>", self._on_right)

    def _build(self):
        tk.Label(self.root, text="이미지 / json 확인", font=FONT_TITLE, bg=BG, fg=TEXT).pack(pady=(16, 8))

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=16)
        self.path_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.path_var, font=FONT)
        entry.pack(side="left", fill="x", expand=True, ipady=4)
        self._button(bar, "찾아보기", self.browse).pack(side="left", padx=(8, 0))
        self._button(bar, "열기", self.open_folder).pack(side="left", padx=(8, 0))

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        self.listbox = tk.Listbox(left, font=FONT, width=32, activestyle="none")
        self.listbox.pack(side="left", fill="y", expand=True)
        scroll = tk.Scrollbar(left, command=self.listbox.yview)
        scroll.pack(side="left", fill="y")
        self.listbox.config(yscrollcommand=scroll.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.canvas = tk.Canvas(right, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.info = tk.Label(right, text="폴더를 여세요.", font=FONT_SMALL, bg=BG, fg=MUTED, anchor="w")
        self.info.pack(fill="x", pady=(8, 0))

        tools = tk.Frame(self.root, bg=BG)
        tools.pack(fill="x", padx=16, pady=(0, 16))
        self._button(tools, "왼쪽 90°", lambda: self.rotate("ccw")).pack(side="left")
        self._button(tools, "오른쪽 90°", lambda: self.rotate("cw")).pack(side="left", padx=8)
        self._button(tools, "180°", lambda: self.rotate("180")).pack(side="left")
        tk.Label(tools, text="이름", font=FONT, bg=BG).pack(side="left", padx=(20, 6))
        self.name_var = tk.StringVar()
        tk.Entry(tools, textvariable=self.name_var, font=FONT, width=28).pack(side="left", ipady=4)
        self._button(tools, "이름 변경", self.rename).pack(side="left", padx=8)
        self.status = tk.Label(tools, text="", font=FONT_SMALL, bg=BG, fg=MUTED)
        self.status.pack(side="left", padx=8)

    def _button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command, font=FONT,
            bg=ACCENT, fg="white", activebackground="#333333", activeforeground="white",
            relief="flat", cursor="hand2", padx=10, pady=4,
        )

    def browse(self):
        path = filedialog.askdirectory(title="이미지와 json이 있는 폴더")
        if path:
            self.path_var.set(path)
            self.open_folder()

    def open_folder(self):
        folder = self.path_var.get().strip().strip('"')
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("폴더 없음", "폴더 경로를 확인하세요.")
            return
        stems, images, jsons = list_pairs(folder)
        if not stems:
            messagebox.showinfo("파일 없음", "이 폴더에 이미지와 json이 없습니다.")
            return
        self.folder = folder
        self.stems = stems
        self.images = images
        self.jsons = jsons
        self.listbox.delete(0, "end")
        for stem in stems:
            mark = []
            if stem not in images:
                mark.append("이미지 없음")
            if stem not in jsons:
                mark.append("json 없음")
            label = stem if not mark else "%s  (%s)" % (stem, ", ".join(mark))
            self.listbox.insert("end", label)
        self.index = 0
        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.show()
        self.status.config(text="%d개" % len(stems))

    def _on_select(self, _event):
        selected = self.listbox.curselection()
        if not selected or selected[0] == self.index:
            return
        self.index = selected[0]
        self.show()

    def _on_left(self, event):
        if isinstance(event.widget, tk.Entry):
            return
        self.move(-1)

    def _on_right(self, event):
        if isinstance(event.widget, tk.Entry):
            return
        self.move(1)

    def move(self, step):
        if not self.stems:
            return
        self.index = (self.index + step) % len(self.stems)
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.index)
        self.listbox.see(self.index)
        self.show()

    def current(self):
        if not self.stems:
            return None
        stem = self.stems[self.index]
        return stem, self.images.get(stem), self.jsons.get(stem)

    def show(self):
        item = self.current()
        if item is None:
            return
        stem, image_path, json_path = item
        self.name_var.set(stem)
        notes = [stem]
        if image_path and os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
            notes.append("%dx%d" % image.size)
            self._draw(image)
        else:
            self.canvas.delete("all")
            self.canvas.create_text(20, 20, anchor="nw", fill="white", text="이미지 없음", font=FONT)
        if json_path and os.path.isfile(json_path):
            try:
                data = json.load(open(json_path, encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                notes.append("json 읽기 실패")
            else:
                labels = [shape.get("label") or "?" for shape in data.get("shapes") or []]
                notes.append("json %d개" % len(labels))
                if labels:
                    shown = ", ".join(labels[:8])
                    if len(labels) > 8:
                        shown += " ..."
                    notes.append(shown)
        else:
            notes.append("json 없음")
        self.info.config(text="  |  ".join(notes))

    def _draw(self, image):
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        scale = min(cw / image.width, ch / image.height, 1.0)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        view = image.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(view)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo)

    def rotate(self, turn):
        item = self.current()
        if item is None:
            return
        stem, image_path, json_path = item
        if not image_path or not os.path.isfile(image_path):
            messagebox.showerror("이미지 없음", "회전할 이미지가 없습니다.")
            return
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        turned = rotate_image(image, turn)
        save_image(turned, image_path)
        if json_path and os.path.isfile(json_path):
            data = json.load(open(json_path, encoding="utf-8"))
            rotate_json(data, width, height, turn)
            data["imagePath"] = os.path.basename(image_path)
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
        names = {"cw": "오른쪽 90°", "ccw": "왼쪽 90°", "180": "180°"}
        self.status.config(text="%s 저장" % names[turn])
        self.show()

    def rename(self):
        item = self.current()
        if item is None:
            return
        stem, image_path, json_path = item
        new_stem = self.name_var.get().strip()
        error = valid_stem(new_stem)
        if error:
            messagebox.showerror("이름", error)
            return
        if new_stem == stem:
            return
        occupied = set(self.stems) - {stem}
        if new_stem in occupied:
            messagebox.showerror("이름", "같은 이름이 이미 있습니다.")
            return
        new_image = new_json = None
        if image_path:
            new_image = os.path.join(self.folder, new_stem + os.path.splitext(image_path)[1])
            if os.path.exists(new_image):
                messagebox.showerror("이름", "같은 이름의 이미지가 있습니다.")
                return
        if json_path:
            new_json = os.path.join(self.folder, new_stem + ".json")
            if os.path.exists(new_json):
                messagebox.showerror("이름", "같은 이름의 json이 있습니다.")
                return
        if image_path:
            os.rename(image_path, new_image)
            self.images.pop(stem, None)
            self.images[new_stem] = new_image
        if json_path:
            data = json.load(open(json_path, encoding="utf-8"))
            if new_image:
                data["imagePath"] = os.path.basename(new_image)
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            os.rename(json_path, new_json)
            self.jsons.pop(stem, None)
            self.jsons[new_stem] = new_json
        self.stems[self.index] = new_stem
        self.listbox.delete(self.index)
        self.listbox.insert(self.index, new_stem)
        self.listbox.selection_set(self.index)
        self.status.config(text="이름 변경: %s" % new_stem)
        self.show()


def main():
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
