# JetPack 4.6 / Python 3.6. 정면·측면 CSI 한 프레임을 JPEG로 주고,
# 실행하면 HDMI 화면에 사용자 화면 창(Tkinter)을 바로 띄운다.
import base64
import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:  # python3-tk 가 없으면 화면 없이 서버만 돈다.
    tk = None

HOST = "0.0.0.0"
PORT = 8080
WIDTH, HEIGHT = 1280, 720
FPS = 30
SENSOR_MODE = 4
# sensor 0은 옆모습(측면), sensor 1은 위에서 본 손톱(정면)이다.
SIDE_ID = 0
FRONT_ID = 1
JPEG_QUALITY = 90
# LED가 프레임에 들어오면 자동 노출이 그 밝기에 맞춰 손톱이 어둡게 나온다.
EXPOSURE_COMPENSATION = 1.0
# left top right bottom weight. 측면은 오른쪽 LED를 측광에서 뺀다.
AE_REGION = {
    SIDE_ID: "0 0 1400 1080 1",
    FRONT_ID: "400 480 1520 1080 1",
}

read_lock = threading.Lock()
frame_lock = threading.Lock()
display_lock = threading.Lock()
latest_display = {"views": []}
# HDMI 사용자 화면 창에 띄울 촬영 사진.
# /shot 은 재시도로 여러 번 불릴 수 있어서 바로 보여주지 않고 pending_shot 에 둔다.
# 측정 결과(POST /display)가 오면 그때의 pending_shot 을 latest_shot 으로 올린다.
latest_shot = None
pending_shot = None
# 화면 갱신용 번호. 결과가 올 때마다 1 늘어난다.
shot_id = 0
shooting_since = None
# 이 시간 안에 결과가 안 오면(실패, 캘리브레이션 촬영) '촬영중...' 을 풀고 이전 화면으로 돌아간다.
SHOOTING_TIMEOUT = 30.0

stop_event = threading.Event()
cap0 = None
cap1 = None
latest_front = None
latest_side = None
pump_thread = None


def gstreamer_pipeline(sensor_id):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d exposurecompensation=%.1f aeregion=\"%s\" ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
        "nvvidconv ! video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
        % (sensor_id, SENSOR_MODE, EXPOSURE_COMPENSATION, AE_REGION[sensor_id], WIDTH, HEIGHT, FPS, WIDTH, HEIGHT)
    )


def open_cameras():
    global cap0, cap1
    cap0 = cv2.VideoCapture(gstreamer_pipeline(FRONT_ID), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(gstreamer_pipeline(SIDE_ID), cv2.CAP_GSTREAMER)


def cameras_open():
    return cap0 is not None and cap1 is not None and cap0.isOpened() and cap1.isOpened()


def pump_loop():
    global latest_front, latest_side
    while not stop_event.is_set():
        with read_lock:
            ok0, front = cap0.read()
            ok1, side = cap1.read()
        if not ok0 or not ok1 or front is None or side is None:
            continue
        with frame_lock:
            latest_front = front.copy()
            latest_side = side.copy()


def wait_first_frame(timeout=8):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with frame_lock:
            if latest_front is not None and latest_side is not None:
                return True
        time.sleep(0.01)
    return False


def jpeg_b64(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def grab_pair():
    with frame_lock:
        if latest_front is None or latest_side is None:
            return None
        front = latest_front
        side = latest_side
    front_b64 = jpeg_b64(front)
    side_b64 = jpeg_b64(side)
    if not front_b64 or not side_b64:
        return None
    return {"front": front_b64, "side": side_b64}


def focus_score(frame):
    h, w = frame.shape[:2]
    center = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def preview_jpeg():
    with frame_lock:
        if latest_front is None or latest_side is None:
            return None
        front, side = latest_front, latest_side
    tiles = []
    for name, frame in (("side", side), ("front", front)):
        score = focus_score(frame)
        small = cv2.resize(frame, (640, 360))
        cv2.putText(small, "%s focus %.0f" % (name, score), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        tiles.append(small)
    ok, buf = cv2.imencode(".jpg", cv2.hconcat(tiles), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        global pending_shot, shooting_since
        path = self.path.split("?", 1)[0]
        if path == "/health":
            with frame_lock:
                has_frame = latest_front is not None and latest_side is not None
            self._json(200, {
                "ok": cameras_open() and has_frame,
                "front_open": cap0 is not None and cap0.isOpened(),
                "side_open": cap1 is not None and cap1.isOpened(),
            })
            return
        if path == "/preview":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()
            try:
                while not stop_event.is_set():
                    jpg = preview_jpeg()
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if path == "/display":
            with display_lock:
                payload = latest_display
            self._json(200, payload)
            return
        if path == "/shot":
            pair = grab_pair()
            if pair is None:
                self._json(503, {"detail": "카메라 프레임을 읽지 못했습니다."})
                return
            with display_lock:
                pending_shot = pair
                if shooting_since is None:
                    shooting_since = time.time()
            self._json(200, pair)
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self):
        global latest_display, latest_shot, pending_shot, shot_id, shooting_since
        path = self.path.split("?", 1)[0]
        if path != "/display":
            self._json(404, {"detail": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError):
            self._json(400, {"detail": "잘못된 화면 데이터입니다."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("views"), list):
            self._json(400, {"detail": "views 가 필요합니다."})
            return
        with display_lock:
            latest_display = payload
            # 결과와 같은 촬영 사진을 함께 올린다. 파일 업로드 측정이면 Jetson 사진은 비운다.
            latest_shot = pending_shot
            pending_shot = None
            shooting_since = None
            shot_id += 1
        self._json(200, {"ok": True})

    def log_message(self, fmt, *args):
        return


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# HDMI 사용자 화면 (Tkinter 창)
# 서버와 같은 프로세스라 HTTP 로 다시 묻지 않고 메모리의 상태를 바로 읽는다.
# ---------------------------------------------------------------------------
BG = "#ffffff"
PANEL = "#f4f5f7"
BORDER = "#e2e4e9"
TEXT = "#1f2330"
MUTED = "#9aa0ab"
ACCENT = "#2f5bd3"
# 윤곽선 색 (BGR). 관리자 화면과 같은 #ff0055.
CONTOUR_BGR = (85, 0, 255)
POLL_MS = 150


def screen_state():
    """화면에 필요한 상태를 한 번에 복사해 온다."""
    global pending_shot, shooting_since
    with display_lock:
        if shooting_since is not None and time.time() - shooting_since > SHOOTING_TIMEOUT:
            shooting_since = None
            pending_shot = None
        return shot_id, shooting_since is not None, latest_shot, latest_display


def decode_b64_image(data):
    """jpeg base64 또는 data URL 을 BGR 이미지로."""
    if not data:
        return None
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    except (ValueError, TypeError):
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def fit_image(img, box_w, box_h, bg):
    """비율을 지키며 box 안에 맞추고 남는 곳은 bg 색으로 채운다."""
    h, w = img.shape[:2]
    scale = min(float(box_w) / w, float(box_h) / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((box_h, box_w, 3), bg, dtype=np.uint8)
    x, y = (box_w - nw) // 2, (box_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas


def draw_contours(img, contour_groups):
    """원본 좌표 윤곽선을 반투명 채우기 + 선으로 그린다."""
    polys = []
    for contours in contour_groups:
        for cnt in contours or []:
            if cnt and len(cnt) >= 3:
                polys.append(np.round(np.array(cnt, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2))
    if not polys:
        return img
    out = img.copy()
    overlay = img.copy()
    cv2.fillPoly(overlay, polys, CONTOUR_BGR)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    thickness = max(2, img.shape[1] // 400)
    cv2.polylines(out, polys, True, CONTOUR_BGR, thickness, cv2.LINE_AA)
    return out


def to_photo(img):
    """BGR 이미지를 Tk PhotoImage 로. Tk 8.6 기본 PNG 지원만 쓴다(PIL 불필요)."""
    ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        return None
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))


def hex_bgr(color):
    color = color.lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return (b, g, r)


def view_key(label):
    return "front" if "정면" in str(label or "") else "side"


class ScreenApp(object):
    def __init__(self, root):
        self.root = root
        self.version = None
        self.mode = ""
        self.photos = []

        root.title("손톱 측정")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry("%dx%d+0+0" % (sw, sh))
        root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.bind("<F11>", lambda _e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
        try:
            root.config(cursor="none")  # 키오스크 화면이라 마우스 커서를 숨긴다.
        except tk.TclError:
            pass

        family = self._font_family()
        self.f_title = (family, max(14, sh // 36), "bold")
        self.f_empty = (family, max(12, sh // 45))
        self.f_tag = (family, max(10, sh // 55))
        self.f_mm = (family, max(12, sh // 40))
        self.f_shape = (family, max(20, sh // 22), "bold")
        self.f_none = (family, max(12, sh // 45))

        pad_x, gap = int(sw * 0.025), int(sw * 0.02)
        self.shot_w = (sw - pad_x * 2 - gap) // 2
        self.shot_h = int(sh * 0.44)
        self.prev_size = int(sh * 0.2)

        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=pad_x, pady=(int(sh * 0.03), 0))
        self.shot_labels = {}
        for col, (key, title) in enumerate((("side", "측면"), ("front", "정면"))):
            box = tk.Frame(top, bg=BG)
            box.grid(row=0, column=col, padx=(0 if col == 0 else gap, 0))
            tk.Label(box, text=title, font=self.f_title, bg=BG, fg=TEXT).pack(pady=(0, int(sh * 0.012)))
            holder = tk.Frame(box, width=self.shot_w, height=self.shot_h, bg=PANEL,
                              highlightthickness=1, highlightbackground=BORDER)
            holder.pack()
            holder.pack_propagate(False)
            label = tk.Label(holder, bg=PANEL, fg=MUTED, font=self.f_empty, bd=0)
            label.pack(fill="both", expand=True)
            self.shot_labels[key] = label

        self.bottom = tk.Frame(root, bg=BG)
        self.bottom.pack(fill="both", expand=True, padx=pad_x, pady=(int(sh * 0.02), int(sh * 0.03)))

        self.show_waiting()
        self.root.after(50, self.poll)

    def _font_family(self):
        try:
            families = set(tkfont.families(self.root))
        except tk.TclError:
            families = set()
        for name in ("Noto Sans CJK KR", "NanumGothic", "UnDotum", "Malgun Gothic"):
            if name in families:
                return name
        return "TkDefaultFont"

    # ---------- 상태별 화면 ----------
    def set_shot_text(self, key, text):
        self.shot_labels[key].config(image="", text=text)

    def clear_bottom(self):
        for child in self.bottom.winfo_children():
            child.destroy()

    def show_waiting(self):
        for key in ("side", "front"):
            self.set_shot_text(key, "촬영 대기")
        self.show_bottom_text("측정 대기")

    def show_bottom_text(self, text):
        self.clear_bottom()
        if text:
            tk.Label(self.bottom, text=text, font=self.f_empty, bg=BG, fg=MUTED).place(relx=0.5, rely=0.5, anchor="center")

    def show_shooting(self):
        for key in ("side", "front"):
            self.set_shot_text(key, "촬영중...")
        self.show_bottom_text("")

    def poll(self):
        try:
            version, shooting, shot, display = screen_state()
            if shooting:
                # 새 측정 시작: 재시도 촬영은 보여주지 않고 결과가 올 때까지 '촬영중...' 만 띄운다.
                if self.mode != "shooting":
                    self.mode = "shooting"
                    self.show_shooting()
            elif self.mode != "shown" or version != self.version:
                self.mode = "shown"
                self.version = version
                self.render(shot, display)
        except Exception as exc:  # 화면 오류로 서버까지 죽지 않게 한다.
            print("화면 갱신 실패: %s" % exc)
        self.root.after(POLL_MS, self.poll)

    # ---------- 결과 그리기 ----------
    def render(self, shot, display):
        grouped = {"side": [], "front": []}
        for view in (display or {}).get("views") or []:
            grouped[view_key(view.get("label"))].extend(view.get("items") or [])

        photos = []
        for key in ("side", "front"):
            img = decode_b64_image(shot.get(key)) if shot else None
            if img is None:
                self.set_shot_text(key, "촬영 대기")
                continue
            img = draw_contours(img, [item.get("contours") for item in grouped[key]])
            photo = to_photo(fit_image(img, self.shot_w - 2, self.shot_h - 2, hex_bgr(PANEL)))
            if photo is None:
                self.set_shot_text(key, "사진을 불러오지 못했습니다")
                continue
            self.shot_labels[key].config(image=photo, text="")
            photos.append(photo)

        count = max(len(grouped["side"]), len(grouped["front"]))
        if not count:
            self.show_bottom_text("측정 대기")
        else:
            self.clear_bottom()
            row = tk.Frame(self.bottom, bg=BG)
            row.place(relx=0.5, rely=0.5, anchor="center")
            for i in range(count):
                side = grouped["side"][i] if i < len(grouped["side"]) else None
                front = grouped["front"][i] if i < len(grouped["front"]) else None
                photos.extend(self.make_pair(row, side, front).photos)
        # PhotoImage 는 참조가 없으면 사라지므로 들고 있는다.
        self.photos = photos

    def make_pair(self, parent, side, front):
        card = tk.Frame(parent, bg=BG, highlightthickness=1, highlightbackground=BORDER, padx=30, pady=16)
        card.pack(side="left", padx=10)
        card.photos = []
        self._view(card, "측면", side).grid(row=0, column=0, padx=20)
        value = (side or {}).get("shape") or (front or {}).get("shape")
        if value:
            badge = tk.Label(card, text="%s형" % value, font=self.f_shape, bg=ACCENT, fg="#ffffff", padx=24, pady=4)
        else:
            badge = tk.Label(card, text="형태 -", font=self.f_none, bg=BORDER, fg="#6b7280", padx=16, pady=4)
        badge.grid(row=0, column=1, padx=30)
        self._view(card, "정면", front).grid(row=0, column=2, padx=20)
        return card

    def _view(self, card, tag, item):
        box = tk.Frame(card, bg=BG)
        tk.Label(box, text=tag, font=self.f_tag, bg=BG, fg=MUTED).pack()
        img = decode_b64_image((item or {}).get("preview"))
        size = self.prev_size
        if img is None:
            img = np.zeros((size, size, 3), dtype=np.uint8)
        photo = to_photo(fit_image(img, size, size, (0, 0, 0)))
        tk.Label(box, image=photo, bd=0, bg=BG).pack(pady=(6, 8))
        card.photos.append(photo)
        if item:
            length = item.get("length_mm")
            width = item.get("width_mm")
            text = "길이 %smm\n폭 %s" % ("-" if length is None else length, "%smm" % width if width else "측정 불가")
        else:
            text = "-"
        tk.Label(box, text=text, font=self.f_mm, bg=BG, fg=TEXT, justify="center").pack()
        return box


def start_gui(server):
    """창을 띄우고 창이 닫힐 때까지 돌린다. 창을 못 띄우면 False."""
    if tk is None:
        print("tkinter 가 없어 화면 없이 서버만 켭니다. (sudo apt-get install python3-tk)")
        return False
    if not os.environ.get("DISPLAY"):
        # SSH 로 켰을 때도 보드에 연결된 HDMI 화면에 띄운다.
        os.environ["DISPLAY"] = ":0"
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print("화면을 열지 못해 서버만 켭니다: %s" % exc)
        return False
    ScreenApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.bind("<Control-q>", lambda _e: root.destroy())
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    server.shutdown()
    return True


def main():
    global pump_thread
    started = time.time()
    print("카메라 여는 중")
    open_cameras()
    if not cameras_open():
        print("카메라 열기 실패. /shot 은 503 입니다.")
    else:
        pump_thread = threading.Thread(target=pump_loop)
        pump_thread.daemon = True
        pump_thread.start()
        if wait_first_frame():
            print("준비완료 %.1f초" % (time.time() - started))
        else:
            print("첫 프레임 대기 실패. /shot 은 503 입니다.")
    server = ThreadingServer((HOST, PORT), Handler)
    print("capture server http://%s:%d" % (HOST, PORT))
    try:
        use_gui = "--no-gui" not in sys.argv
        if not (use_gui and start_gui(server)):
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if pump_thread is not None:
            pump_thread.join(timeout=1)
        if cap0 is not None:
            cap0.release()
        if cap1 is not None:
            cap1.release()
        server.server_close()


if __name__ == "__main__":
    main()
