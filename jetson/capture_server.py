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
WIDTH, HEIGHT = 1920, 1080
FPS = 30
SENSOR_MODE = 2
# sensor 0은 옆모습(측면), sensor 1은 위에서 본 손톱(정면)이다.
SIDE_ID = 0
FRONT_ID = 1
JPEG_QUALITY = 90
# LED가 프레임에 들어오면 자동 노출이 그 밝기에 맞춰 손톱이 어둡게 나온다.
EXPOSURE_COMPENSATION = 2.0
# left top right bottom weight. 측면은 오른쪽 LED를 측광에서 뺀다.
AE_REGION = {
    SIDE_ID: "0 0 1400 1080 1",
    FRONT_ID: "400 480 1520 1080 1",
}

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
