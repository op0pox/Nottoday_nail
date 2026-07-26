# -*- coding: utf-8 -*-
"""
camera.py
==========
[역할] Jetson Nano B01 CSI 포트 2개(정면 sensor-id=0, 측면 sensor-id=1)에서
IMX219(라즈베리파이 카메라 V2 호환) 프레임을 GStreamer로 동시에 읽어온다.
카메라가 없는 PC에서도 개발/검증할 수 있도록, --front/--side 이미지 파일을
넣으면 그 이미지를 "찍은 것처럼" 반환하는 정지 이미지 입력 모드도 제공한다.

[실행 위치] 직접 실행하는 파일이 아니다. main.py가 불러와서 쓴다.
"""

import cv2


def gstreamer_pipeline(
    sensor_id, capture_width, capture_height, display_width, display_height, framerate, flip_method
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height)
    )


class DualCameraCapture(object):
    """정면/측면 CSI 카메라 2대를 동시에 여는 래퍼."""

    def __init__(self, config):
        cam_cfg = config["camera"]
        self.cap_front = self._open_one(cam_cfg["front"]["sensor_id"], cam_cfg)
        self.cap_side = self._open_one(cam_cfg["side"]["sensor_id"], cam_cfg)

    @staticmethod
    def _open_one(sensor_id, cam_cfg):
        pipeline = gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=cam_cfg["capture_width"],
            capture_height=cam_cfg["capture_height"],
            display_width=cam_cfg["capture_width"],
            display_height=cam_cfg["capture_height"],
            framerate=cam_cfg["framerate"],
            flip_method=cam_cfg["flip_method"],
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(
                "\n"
                "===================== 카메라 열기 실패 (sensor-id=%d) =====================\n"
                "확인할 것:\n"
                "  1) CSI 케이블이 제대로 꽂혔는지 (Jetson Nano B01은 CSI 포트가 2개:\n"
                "     CAM0=sensor-id 0(정면), CAM1=sensor-id 1(측면))\n"
                "  2) Jetson Nano 터미널에서 'ls /dev/video*'로 장치가 보이는지 확인\n"
                "  3) 다른 프로세스가 카메라를 점유하고 있지 않은지 확인\n"
                "     ('ps aux | grep python' 등)\n"
                "  4) 카메라가 아예 없는 PC라면 --front img1.jpg --side img2.jpg 로\n"
                "     정지 이미지 입력 모드를 대신 쓸 수 있습니다\n"
                "==========================================================================\n"
                % sensor_id
            )
        return cap

    def read(self):
        """반환: (front_frame, side_frame) BGR numpy array 쌍. 읽기 실패 시 (None, None)."""
        ok_f, frame_f = self.cap_front.read()
        ok_s, frame_s = self.cap_side.read()
        if not ok_f or not ok_s:
            return None, None
        return frame_f, frame_s

    def release(self):
        if self.cap_front is not None:
            self.cap_front.release()
        if self.cap_side is not None:
            self.cap_side.release()


class StillImageCapture(object):
    """카메라 없는 PC에서 --front img1.jpg --side img2.jpg 로 정지 이미지를 흉내낸다."""

    def __init__(self, front_path, side_path):
        self.frame_front = cv2.imread(front_path)
        self.frame_side = cv2.imread(side_path)
        if self.frame_front is None:
            raise RuntimeError("정면 이미지를 열 수 없습니다: %s" % front_path)
        if self.frame_side is None:
            raise RuntimeError("측면 이미지를 열 수 없습니다: %s" % side_path)

    def read(self):
        return self.frame_front.copy(), self.frame_side.copy()

    def release(self):
        pass


def open_dual_camera(config, front_image=None, side_image=None):
    """
    front_image/side_image가 둘 다 주어지면 정지 이미지 모드,
    둘 다 생략되면 실제 CSI 카메라(DualCameraCapture)를 연다.
    """
    if front_image and side_image:
        return StillImageCapture(front_image, side_image)
    if front_image or side_image:
        raise ValueError("--front와 --side는 함께 지정해야 합니다 (둘 다 이미지 경로이거나 둘 다 생략).")
    return DualCameraCapture(config)


def downscale_for_preview(frame, config):
    """
    Jetson Nano 성능을 고려해 라이브 프리뷰는 config의 preview_width/height
    이하로 다운스케일한다. 실제 분석은 항상 캡처된 원본 고해상도 프레임으로 한다.
    """
    cam_cfg = config["camera"]
    target_w, target_h = cam_cfg["preview_width"], cam_cfg["preview_height"]
    h, w = frame.shape[:2]
    if w <= target_w and h <= target_h:
        return frame
    scale = min(target_w / float(w), target_h / float(h))
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
