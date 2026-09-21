# JetPack 4.6 / Python 3.6. 정면·측면 CSI 한 프레임을 JPEG로 준다.
import base64
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

HOST = "0.0.0.0"
PORT = 8080
WIDTH, HEIGHT = 640, 360
FRONT_ID = 0
SIDE_ID = 1
JPEG_QUALITY = 90

read_lock = threading.Lock()
cap0 = None
cap1 = None


def gstreamer_pipeline(sensor_id=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id, WIDTH, HEIGHT, WIDTH, HEIGHT)
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


def grab_pair():
    if cap0 is None or cap1 is None or not cap0.isOpened() or not cap1.isOpened():
        return None
    with read_lock:
        ok0, front = cap0.read()
        ok1, side = cap1.read()
    if not ok0 or not ok1 or front is None or side is None:
        return None
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
    open_cameras()
    if cap0 is None or not cap0.isOpened() or cap1 is None or not cap1.isOpened():
        print("카메라 열기 실패. /shot 은 503 입니다.")
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
