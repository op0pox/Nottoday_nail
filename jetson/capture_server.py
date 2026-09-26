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
# IMX219 mode 4는 1280x720. 모드를 비우면 탐색이 느리고, 60fps로 두 대를 열면 두 번째 open이 멈춘다.
# tnr/ee를 켜 두면 재시작 직후 버퍼가 찰 때까지 첫 프레임이 밀린다.
WIDTH, HEIGHT = 1280, 720
FPS = 30
SENSOR_MODE = 4
FRONT_ID = 0
SIDE_ID = 1
JPEG_QUALITY = 90

read_lock = threading.Lock()
frame_lock = threading.Lock()
stop_event = threading.Event()
cap0 = None
cap1 = None
latest_front = None
latest_side = None
pump_thread = None


def gstreamer_pipeline(sensor_id):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d tnr-mode=0 ee-mode=0 ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
        "nvvidconv ! video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id, SENSOR_MODE, WIDTH, HEIGHT, FPS, WIDTH, HEIGHT)
    )


def _open_one(sensor_id, out, index):
    out[index] = cv2.VideoCapture(gstreamer_pipeline(sensor_id), cv2.CAP_GSTREAMER)


def open_cameras():
    global cap0, cap1
    opened = [None, None]
    front = threading.Thread(target=_open_one, args=(FRONT_ID, opened, 0))
    side = threading.Thread(target=_open_one, args=(SIDE_ID, opened, 1))
    front.start()
    side.start()
    front.join()
    side.join()
    cap0, cap1 = opened


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


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
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
        if path == "/shot":
            pair = grab_pair()
            if pair is None:
                self._json(503, {"detail": "카메라 프레임을 읽지 못했습니다."})
                return
            self._json(200, pair)
            return
        self._json(404, {"detail": "not found"})

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
