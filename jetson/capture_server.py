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
# IMX219 sensor-mode 4. 없는 해상도를 요구하면 모드를 찾느라 느려진다.
# tnr/ee 를 켜 두면 nvargus 재시작 직후 버퍼를 채울 때까지 프레임이 밀린다.
SENSOR_WIDTH, SENSOR_HEIGHT = 1280, 720
SENSOR_MODE = 4
SENSOR_FPS = 60
WIDTH, HEIGHT = SENSOR_WIDTH, SENSOR_HEIGHT
FRONT_ID = 0
SIDE_ID = 1
JPEG_QUALITY = 90
WARMUP_FRAMES = 8

read_lock = threading.Lock()
cap0 = None
cap1 = None


def gstreamer_pipeline(sensor_id=0):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d tnr-mode=0 ee-mode=0 ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
        "nvvidconv ! video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id, SENSOR_MODE, SENSOR_WIDTH, SENSOR_HEIGHT, SENSOR_FPS, WIDTH, HEIGHT)
    )


def open_cameras():
    global cap0, cap1
    cap0 = cv2.VideoCapture(gstreamer_pipeline(FRONT_ID), cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(gstreamer_pipeline(SIDE_ID), cv2.CAP_GSTREAMER)


def jpeg_b64(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _read_into(cap, box):
    box.append(cap.read())


def _encode_into(frame, box):
    box.append(jpeg_b64(frame))


def grab_pair():
    if cap0 is None or cap1 is None or not cap0.isOpened() or not cap1.isOpened():
        return None
    with read_lock:
        frames = [[], []]
        readers = [
            threading.Thread(target=_read_into, args=(cap0, frames[0])),
            threading.Thread(target=_read_into, args=(cap1, frames[1])),
        ]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        if len(frames[0]) != 1 or len(frames[1]) != 1:
            return None
        ok0, front = frames[0][0]
        ok1, side = frames[1][0]
        if not ok0 or not ok1 or front is None or side is None:
            return None
        encoded = [[], []]
        encoders = [
            threading.Thread(target=_encode_into, args=(front, encoded[0])),
            threading.Thread(target=_encode_into, args=(side, encoded[1])),
        ]
        for encoder in encoders:
            encoder.start()
        for encoder in encoders:
            encoder.join()
    if len(encoded[0]) != 1 or len(encoded[1]) != 1:
        return None
    front_b64, side_b64 = encoded[0][0], encoded[1][0]
    if not front_b64 or not side_b64:
        return None
    return {"front": front_b64, "side": side_b64}


def warmup():
    for _ in range(WARMUP_FRAMES):
        if grab_pair() is None:
            return


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
            front_open = cap0 is not None and cap0.isOpened()
            side_open = cap1 is not None and cap1.isOpened()
            self._json(200, {"ok": front_open and side_open, "front_open": front_open, "side_open": side_open})
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
    started = time.time()
    open_cameras()
    print("카메라 열기 %.1f초" % (time.time() - started))
    if cap0 is None or not cap0.isOpened() or cap1 is None or not cap1.isOpened():
        print("카메라 열기 실패. /shot 은 503 입니다.")
    else:
        started = time.time()
        warmup()
        print("워밍업 %.1f초" % (time.time() - started))
    server = ThreadingServer((HOST, PORT), Handler)
    print("capture server http://%s:%d" % (HOST, PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if cap0 is not None:
            cap0.release()
        if cap1 is not None:
            cap1.release()
        server.server_close()


if __name__ == "__main__":
    main()
