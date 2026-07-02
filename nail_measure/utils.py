"""
utils.py
========
이 파일은 프로젝트의 다른 스크립트들(camera_test.py, capture_image.py,
grid_calibration.py, manual_measure.py, compare_actual.py, height_experiment.py,
main.py)이 공통으로 사용하는 함수들을 모아둔 "도구 상자"이다.

이 파일은 직접 실행하는 파일이 아니다. 다른 파일에서
    from utils import ...
형태로 불러와서 사용한다.

여기 있는 함수들:
- ensure_dir()            : 폴더가 없으면 자동 생성
- timestamp_filename()    : 날짜/시간이 들어간 파일명 생성
- gstreamer_pipeline()    : Jetson Nano CSI 카메라용 GStreamer 파이프라인 문자열 생성
- open_camera()           : CSI 또는 USB 카메라를 안전하게 여는 함수 (실패 시 친절한 오류)
- check_gui_available()   : SSH 환경에서 화면(GUI) 창을 띄울 수 있는지 미리 확인
- collect_points()        : 이미지 위에서 사용자가 마우스로 점을 클릭하게 하고 좌표를 반환
- load_json() / save_json(): JSON 파일 읽기/쓰기
- FINGERS, FINGER_LABELS_KO : 5개 손가락 이름 상수
"""

import os
import sys
import json
from datetime import datetime

import cv2

# ---------------------------------------------------------------------------
# 손가락 관련 상수
# ---------------------------------------------------------------------------
# 내부적으로(코드, CSV 컬럼 등) 사용하는 영문 이름은 이 순서를 그대로 따른다.
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# 화면에 출력할 때 사용하는 한글 이름
FINGER_LABELS_KO = {
    "thumb": "엄지",
    "index": "검지",
    "middle": "중지",
    "ring": "약지",
    "pinky": "소지",
}


# ---------------------------------------------------------------------------
# 폴더 / 파일 유틸
# ---------------------------------------------------------------------------
def ensure_dir(path):
    """
    path가 파일 경로든 폴더 경로든 상관없이,
    필요한 상위 폴더가 없으면 자동으로 만들어준다.

    예: ensure_dir("data/captured/2026_07_02.jpg")
        -> "data/captured" 폴더가 없으면 생성한다.
    """
    folder = path
    # 확장자가 있는 "파일 경로"로 보이면 상위 폴더만 추출
    if os.path.splitext(path)[1]:
        folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def timestamp_filename(prefix="capture", ext="jpg"):
    """
    날짜와 시간이 포함된 파일명을 만들어 준다.
    예: capture_20260702_193045.jpg
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}.{ext}"


def load_json(path):
    """JSON 파일을 읽어서 dict로 반환한다. 파일이 없으면 None을 반환한다."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """dict를 JSON 파일로 저장한다. 상위 폴더가 없으면 자동으로 만든다."""
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 카메라 관련 유틸
# ---------------------------------------------------------------------------
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1280,
    display_height=720,
    framerate=21,
    flip_method=0,
):
    """
    Jetson Nano의 CSI 카메라(Raspberry Pi Camera Module 2/3, Arducam CSI 등)를
    OpenCV에서 열기 위한 GStreamer 파이프라인 문자열을 만든다.

    Jetson Nano는 CSI 카메라를 열 때 v4l2가 아니라 nvarguscamerasrc라는
    NVIDIA 전용 GStreamer 플러그인을 사용해야 한다. 이게 CSI 카메라를
    USB 웹캠처럼 cv2.VideoCapture(0) 로 바로 열 수 없는 이유이다.

    flip_method: 카메라가 거꾸로 장착되어 화면이 뒤집혀 보이면 값을 바꿔본다.
                 (0=회전없음, 2=180도 회전 이 가장 흔하게 씀)
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def open_camera(
    camera_type="csi",
    device=0,
    width=1280,
    height=720,
    sensor_id=0,
    flip_method=0,
):
    """
    카메라를 연다. camera_type에 따라 CSI 또는 USB 카메라를 연다.

    camera_type:
        "csi" -> Jetson Nano CSI 포트에 연결된 카메라
                 (Raspberry Pi Camera Module 2/3, Arducam CSI 등)
        "usb" -> USB로 연결된 웹캠 (Arducam USB 버전 포함)

    성공하면 cv2.VideoCapture 객체를 반환한다.
    실패하면 초보자가 이해할 수 있는 한글 오류 메시지와 함께
    RuntimeError를 발생시킨다.
    """
    if camera_type == "csi":
        pipeline = gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=width,
            capture_height=height,
            display_width=width,
            display_height=height,
            flip_method=flip_method,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif camera_type == "usb":
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        raise ValueError(f"알 수 없는 camera_type: {camera_type} (csi 또는 usb만 가능)")

    if not cap.isOpened():
        raise RuntimeError(
            "\n"
            "===================== 카메라 열기 실패 =====================\n"
            f" camera_type = {camera_type}\n"
            "다음을 순서대로 확인해보세요.\n"
            "  1. 카메라 케이블/USB가 Jetson Nano에 제대로 연결되어 있는가?\n"
            "     (CSI 케이블은 파란 면이 방열판 반대쪽(이더넷 포트 쪽)을 향해야 함)\n"
            "  2. Jetson Nano 터미널에서 카메라가 보이는지 확인:\n"
            "       ls /dev/video*\n"
            "  3. CSI 카메라는 'gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink' 로\n"
            "     먼저 단독 테스트가 되는지 확인 (Jetson Nano 터미널, 모니터 연결 시)\n"
            "  4. USB 카메라는 camera_type='usb', device 번호(0,1,2...)가 맞는지 확인\n"
            "  5. 다른 프로그램이 이미 카메라를 점유하고 있지 않은지 확인\n"
            "     (예: cheese, 다른 python 프로세스 등 -> 'ps aux | grep python')\n"
            "=============================================================\n"
        )
    return cap


def check_gui_available():
    """
    현재 환경에서 OpenCV 창(cv2.imshow)을 띄울 수 있는지 간단히 확인한다.

    SSH로 접속만 한 상태(X11 포워딩 없음)에서는 화면이 없기 때문에
    cv2.imshow()를 호출하면 아래와 비슷한 오류가 난다.
        cv2.error: ... The function is not implemented ...
        또는 Could not connect to display

    이 함수는 완벽한 검증은 아니지만, 흔한 실수를 미리 막기 위해
    DISPLAY 환경변수가 비어있는 리눅스 환경이면 경고를 출력한다.
    """
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            print(
                "\n"
                "[경고] 현재 화면(DISPLAY) 정보가 없습니다.\n"
                "일반 SSH 접속(예: ssh user@jetson-ip) 상태라면 카메라 창이 뜨지 않습니다.\n"
                "해결 방법 중 하나를 사용하세요.\n"
                "  1) X11 포워딩으로 SSH 접속: ssh -X user@jetson-ip  (또는 -Y)\n"
                "     -> 이 경우 노트북에 X 서버(Xming, VcXsrv, XQuartz 등)가 필요합니다.\n"
                "  2) Jetson Nano에 모니터/키보드를 직접 연결해서 그 화면에서 실행\n"
                "  3) 화면 없이 사진만 저장하는 방식으로 사용 (자동 촬영 모드)\n"
            )
            return False
    return True


# ---------------------------------------------------------------------------
# 마우스 클릭으로 점 좌표 수집하기
# ---------------------------------------------------------------------------
def collect_points(image, num_points, point_labels=None, window_name="Click Points"):
    """
    이미지를 화면에 띄우고, 사용자가 마우스로 점을 num_points개 클릭하면
    그 좌표 리스트를 반환한다.

    point_labels: 각 점에 대해 화면 상단에 보여줄 안내 문구 리스트.
                  예: ["모눈 시작점을 클릭하세요", "모눈 끝점을 클릭하세요"]
                  생략하면 "점 1/2 클릭" 식으로 자동 표시한다.

    조작 방법:
        - 왼쪽 마우스 클릭: 점 찍기
        - 'r' 키: 마지막으로 찍은 점 취소(되돌리기)
        - 'q' 키: 중간에 그만두기 (측정 취소, None 반환)

    반환값:
        점을 모두 찍으면 [(x1, y1), (x2, y2), ...] 형태의 리스트
        중간에 'q'로 취소하면 None
    """
    if not check_gui_available():
        raise RuntimeError(
            "화면(GUI)을 사용할 수 없는 환경이라 마우스 클릭 방식을 쓸 수 없습니다.\n"
            "SSH -X 옵션으로 재접속하거나, Jetson Nano에 모니터를 연결한 뒤 다시 시도하세요."
        )

    points = []
    display_img = image.copy()

    def _label_for(idx):
        if point_labels and idx < len(point_labels):
            return point_labels[idx]
        return f"점 {idx + 1}/{num_points} 클릭하세요"

    def _redraw():
        nonlocal display_img
        display_img = image.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(display_img, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                display_img,
                str(i + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        # 안내 문구 (다음에 찍어야 할 점)
        if len(points) < num_points:
            guide = _label_for(len(points))
        else:
            guide = "모든 점을 찍었습니다. 아무 키나 누르면 계속합니다."
        cv2.putText(
            display_img,
            guide,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_img,
            "[클릭: 점 찍기] [r: 되돌리기] [q: 취소]",
            (10, display_img.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            _redraw()
            cv2.imshow(window_name, display_img)

    _redraw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, _on_mouse)

    cancelled = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cancelled = True
            break
        if key == ord("r") and points:
            points.pop()
            _redraw()
            cv2.imshow(window_name, display_img)
            continue
        if len(points) >= num_points and key != 255:
            # 점을 다 찍은 뒤 아무 키나 누르면 완료
            break

    cv2.destroyWindow(window_name)

    if cancelled:
        return None
    return points
