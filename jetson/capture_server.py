# JetPack 4.6 / Python 3.6. 정면·측면 CSI 한 프레임을 JPEG로 준다.
import base64
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

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
# 마지막 /shot 사진. HDMI 화면(/screen)에 원본 촬영 사진을 바로 띄울 때 쓴다.
latest_shot = None
shot_id = 0

USER_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>손톱</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: #0b0c0f; color: #f2f3f5; font-family: "Noto Sans CJK KR", "Malgun Gothic", sans-serif; }
  body { display: flex; flex-direction: column; padding: 2.5vh 3vw; gap: 2.5vh; }
  .label { font-size: 2.2vh; color: #9aa0ab; margin: 0 0 1vh; font-weight: 500; }
  #shots { display: flex; gap: 2vw; justify-content: center; height: 38vh; }
  #shots:empty { display: none; }
  .shot { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; align-items: center; }
  .shot img { flex: 1; min-height: 0; max-width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #000; border: 1px solid #262a33; border-radius: 12px; }
  #root { flex: 1; min-height: 0; display: flex; flex-direction: column; justify-content: center; gap: 2vh; }
  .view { display: flex; align-items: center; gap: 1.5vw; }
  .view h2 { width: 5vw; margin: 0; font-size: 2.6vh; font-weight: 500; color: #9aa0ab; text-align: center; }
  .cards { flex: 1; display: flex; gap: 1vw; justify-content: center; }
  .card { flex: 0 1 17vw; display: flex; align-items: center; gap: 1vw; padding: 1.2vh 1vw; background: #16181d; border: 1px solid #262a33; border-radius: 14px; }
  .card img { width: 13vh; height: 13vh; flex: none; object-fit: contain; background: #000; border-radius: 8px; }
  .info { flex: 1; min-width: 0; text-align: left; }
  .mm { font-size: 2vh; line-height: 1.5; white-space: nowrap; }
  .shape { display: inline-block; font-size: 2.8vh; font-weight: 700; margin-top: 0.8vh; padding: 0.2vh 1vw; border-radius: 999px; background: #f5c542; color: #0b0c0f; }
  .wait { font-size: 5vh; color: #9aa0ab; text-align: center; margin: 0; }
</style>
</head>
<body>
<section id="shots"></section>
<main id="root"><p class="wait">측정 대기</p></main>
<script>
var last = "";
var lastShot = 0;
function renderShots(info) {
  var id = info && info.id;
  if (!id || id === lastShot) return;
  lastShot = id;
  var box = document.getElementById("shots");
  box.innerHTML = "";
  [["side", "측면 촬영"], ["front", "정면 촬영"]].forEach(function (pair) {
    var el = document.createElement("div");
    el.className = "shot";
    var title = document.createElement("p");
    title.className = "label";
    title.textContent = pair[1];
    el.appendChild(title);
    var img = document.createElement("img");
    img.src = "/last_shot/" + pair[0] + ".jpg?id=" + id;
    img.alt = pair[1];
    el.appendChild(img);
    box.appendChild(el);
  });
}
function render(data) {
  var raw = JSON.stringify(data);
  if (raw === last) return;
  last = raw;
  var root = document.getElementById("root");
  var views = (data && data.views) || [];
  var total = 0;
  views.forEach(function (view) { total += (view.items || []).length; });
  if (!total) {
    root.innerHTML = '<p class="wait">측정 대기</p>';
    return;
  }
  root.innerHTML = "";
  views.forEach(function (view) {
    var items = view.items || [];
    if (!items.length) return;
    var row = document.createElement("section");
    row.className = "view";
    var title = document.createElement("h2");
    title.textContent = view.label || "";
    row.appendChild(title);
    var list = document.createElement("div");
    list.className = "cards";
    items.forEach(function (item) {
      var el = document.createElement("div");
      el.className = "card";
      if (item.preview) {
        var img = document.createElement("img");
        img.src = item.preview;
        img.alt = view.label || "전처리";
        el.appendChild(img);
      }
      var info = document.createElement("div");
      info.className = "info";
      var mm = document.createElement("div");
      mm.className = "mm";
      mm.appendChild(document.createTextNode("길이 " + (item.length_mm == null ? "-" : item.length_mm) + "mm"));
      mm.appendChild(document.createElement("br"));
      mm.appendChild(document.createTextNode("폭 " + (item.width_mm ? item.width_mm + "mm" : "측정 불가")));
      info.appendChild(mm);
      if (item.shape) {
        var shape = document.createElement("div");
        shape.className = "shape";
        shape.textContent = item.shape + "형";
        info.appendChild(shape);
      }
      el.appendChild(info);
      list.appendChild(el);
    });
    row.appendChild(list);
    root.appendChild(row);
  });
}
function tick() {
  fetch("/display").then(function (r) { return r.json(); }).then(render).catch(function () {});
  fetch("/shot_info").then(function (r) { return r.json(); }).then(renderShots).catch(function () {});
}
tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""
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
        global latest_shot, shot_id
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
        if path == "/screen":
            body = USER_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/shot":
            pair = grab_pair()
            if pair is None:
                self._json(503, {"detail": "카메라 프레임을 읽지 못했습니다."})
                return
            with display_lock:
                latest_shot = pair
                shot_id += 1
            self._json(200, pair)
            return
        if path == "/shot_info":
            with display_lock:
                current_id = shot_id
            self._json(200, {"id": current_id})
            return
        if path in ("/last_shot/front.jpg", "/last_shot/side.jpg"):
            with display_lock:
                shot = latest_shot
            key = "front" if path.endswith("front.jpg") else "side"
            if shot is None or not shot.get(key):
                self._json(404, {"detail": "촬영 사진이 없습니다."})
                return
            body = base64.b64decode(shot[key])
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self):
        global latest_display
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
        self._json(200, {"ok": True})

    def log_message(self, fmt, *args):
        return


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


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
