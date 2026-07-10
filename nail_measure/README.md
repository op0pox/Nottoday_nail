# nail_measure

Jetson Nano + 카메라로 손톱 사진을 찍고, ChArUco 보정판(체스보드 +
ArUco 마커)을 자동 인식해서 이미지<->mm 호모그래피를 계산한 뒤,
엄지/검지/중지/약지/소지 5개 손톱의 길이(뿌리~끝)를 mm 단위로
측정하는 프로젝트다.

- **v1**: 모눈판 두 점 클릭 -> 스칼라 px_per_mm, 손톱도 사람이 직접
  클릭해서 측정 (`grid_calibration.py`, 지금도 그대로 남아있음, 필요하면 계속 쓸 수 있다)
- **v2(현재)**: ChArUco 보드 자동 검출 -> 호모그래피 기반 mm 변환
  (`charuco_calibration.py`), 손톱은 사전학습 YOLOv8-seg 모델로 자동
  검출(`measure_auto.py --backend yolo`)하거나, 기존처럼 수동 클릭
  (`--backend manual` / `manual_measure.py`)으로도 측정 가능. 두 방식
  모두 같은 CSV 포맷을 쓰기 때문에 서로 정확도를 비교할 수 있다
  (수동 클릭 결과를 자동 측정의 "정답"처럼 사용).
- 카메라 높이는 **29.5cm 고정**을 기본값으로 쓴다 (모든 결과에 메타데이터로 기록됨).

## 목차

1. [사용 장비](#사용-장비)
2. [프로젝트 구조](#프로젝트-구조)
3. [ChArUco 보드 준비](#charuco-보드-준비)
4. [Jetson Nano 카메라 연결 방법](#jetson-nano-카메라-연결-방법)
5. [필요한 패키지 설치](#필요한-패키지-설치)
6. [전체 실행 흐름](#전체-실행-흐름)
7. [단계별 실행 방법](#단계별-실행-방법)
8. [Nano 촬영 -> 노트북 측정 워크플로](#nano-촬영--노트북-측정-워크플로)
9. [결과 CSV 확인 방법](#결과-csv-확인-방법)
10. [자주 발생하는 오류와 해결 방법](#자주-발생하는-오류와-해결-방법)
11. [모델 정확도가 부족할 때](#모델-정확도가-부족할-때)

---

## 사용 장비

- Jetson Nano
- 카메라 후보 (아무거나 하나면 됨)
  - Arducam 16MP Auto Focus (CSI 또는 USB 버전)
  - Raspberry Pi Camera Module 3 (CSI)
  - Raspberry Pi Camera Module 2 (CSI)
- ChArUco 보정판 인쇄물 (18x26칸, 한 칸 10mm 기준. [ChArUco 보드 준비](#charuco-보드-준비) 참고)
- (선택, v1 방식용) 모눈판
- 일반 자 (실제 길이 검증용)
- 카메라 고정 거치대 (높이 29.5cm 고정용)
- 조명
- 노트북(리눅스): YOLO 자동 세그멘테이션은 여기서 실행한다

---

## 프로젝트 구조

```text
nail_measure/
├── main.py                  # 통합 메뉴 + --mode 스크립트 실행(capture/calibrate/measure-*/compare)
├── utils.py                 # 공용 함수 (카메라 열기, 클릭 UI, 호모그래피 변환, CSV 저장 등)
├── camera_test.py           # 카메라 연결 확인
├── capture_image.py         # 사진 촬영 (s=저장, q=종료)
├── charuco_calibration.py   # [v2] ChArUco 보드 자동 검출 -> 호모그래피 계산 (메인 보정 방식)
├── grid_calibration.py      # [v1] 모눈 두 점 클릭 -> px_per_mm 계산 (예전 방식, 계속 사용 가능)
├── manual_measure.py        # 5개 손가락 손톱 길이 수동 클릭 측정 (호모그래피 기반)
├── measure_auto.py          # [v2] 세그멘테이션 백엔드(yolo/classical/manual)로 자동 측정
├── segmentation/
│   ├── common.py             # NailMask / SegmentationBackend 공통 인터페이스
│   ├── dl_yolo.py             # (메인) 사전학습 YOLOv8-seg 모델 다운로드+추론
│   ├── classical.py           # 피부색 기반 폴백 베이스라인
│   ├── manual.py               # 기존 클릭 방식을 같은 인터페이스로 감싼 것
│   └── measure_from_mask.py    # 마스크 -> PCA 뿌리/끝 추정 -> mm 계산, 손가락 자동 라벨링
├── models/                  # YOLO 가중치 저장 위치 (최초 실행 시 자동 다운로드, git에는 안 올림)
├── training/
│   └── finetune_guide.md     # 모델 정확도가 부족할 때 파인튜닝하는 절차 (문서만)
├── compare_actual.py        # 실제 자 측정값과 비교(오차/오차율), 백엔드별로 기록
├── height_experiment.py     # 카메라 높이별 실험 기록 및 평균 오차 요약 (v1, 높이를 바꿀 때 사용)
├── requirements.txt         # 노트북(YOLO/ChArUco/비교)용 패키지 목록
├── requirements_nano.txt    # Jetson Nano(촬영/보정/수동측정)용 패키지 목록
├── README.md                # 이 문서
├── data/
│   ├── captured/             # 촬영된 원본 사진 저장 위치
│   ├── results/               # *_charuco.json, measurements.csv, comparison.csv 등 결과물
│   │   └── debug/              # 보정/자동측정 디버그 오버레이 이미지
│   └── test_images/          # 테스트용 샘플 이미지 (직접 넣어서 사용 가능)
└── docs/
    ├── experiment_log_template.md   # 실험 결과를 사람이 보기 좋게 기록하는 템플릿
    └── height_test_cheatsheet.md    # 높이별/카메라별 반복 테스트용 명령어 모음
```

각 파일의 역할은 파일 맨 위 docstring에도 "1. 역할 / 2. 실행 명령어 /
3. 어디에 입력하는지 / 4. 정상 화면 / 5. 오류 대처" 형식으로 자세히
적어두었다.

---

## ChArUco 보드 준비

기본 보드 스펙은 `charuco_calibration.py` 상단에 상수로 정의돼 있다.

```text
SQUARES_X = 18       # 가로 칸 수
SQUARES_Y = 26       # 세로 칸 수
SQUARE_MM = 10.0     # 체스보드 한 칸 크기 (인쇄 후 실측값으로 --square-mm로 덮어쓸 수 있음)
MARKER_MM = 7.0      # ArUco 마커 한 변 크기 (SQUARE_MM 바꾸면 0.7 비율 유지해서 같이 바꿀 것)
DICT     = DICT_4X4_250
```

- 인쇄 후에는 **반드시 실제 자로 한 칸 크기를 재보고**, 인쇄 오차가
  있으면 `--square-mm`(과 비율 유지한 `--marker-mm`)로 실측치를 넘겨준다.
  이 값이 틀리면 모든 손톱 길이가 그 비율만큼 통째로 틀어진다.
- 보드는 평평하게(휘거나 구겨지지 않게) 고정하고, 촬영 시 보드 전체가
  프레임에 들어오도록 한다.
- 조명이 보드 표면에 강하게 반사되어 하얗게 날아가면 마커 인식이
  실패하거나 중복 마커로 오검출될 수 있다. 보드에 고르게 빛이
  닿도록 조명을 조정한다.

---

## Jetson Nano 카메라 연결 방법

> 아래 명령어는 특별한 언급이 없으면 전부 **Jetson Nano 터미널**
> (SSH로 접속한 뒤의 화면)에서 실행한다. 노트북 자체 터미널이 아니다.

### 1) 카메라 모듈 연결

- **CSI 카메라** (Raspberry Pi Camera Module 2/3, Arducam CSI 버전)
  1. Jetson Nano 전원을 완전히 끈다 (분리 중 통전되면 카메라가 손상될 수 있음).
  2. Jetson Nano 보드의 CSI 커넥터(카메라 포트) 검은 플라스틱 걸쇠를
     위로 살짝 당겨서 연다.
  3. 리본 케이블을 끼운다. **파란색 면(또는 접점이 보이는 면)이
     이더넷 포트 쪽을 향하고, 은색 접점이 있는 면이 방열판(HDMI 포트) 반대 방향을
     향하도록** 넣는다. (보드 실크스크린에 방향이 인쇄되어 있는 경우가 많으니
     확인한다.)
  4. 케이블이 삐뚤어지지 않고 커넥터 끝까지 평행하게 들어갔는지 확인한 뒤
     걸쇠를 눌러 고정한다.
  5. 전원을 켠다.

- **USB 카메라** (Arducam USB 버전 등)
  - Jetson Nano의 USB 포트에 꽂기만 하면 된다. 별도 드라이버 설치가
    보통 필요 없다.

### 2) CSI 카메라 사용 시 주의사항

- 케이블 방향이 반대로 꽂히면 카메라가 아예 인식되지 않거나
  (`/dev/video0`가 안 보임), 화면이 깨져서 나올 수 있다.
- 케이블이 완전히 끝까지 삽입되지 않고 살짝 걸린 상태면 접촉 불량으로
  영상이 뚝뚝 끊기거나 초록/보라색 노이즈가 낀 화면이 나올 수 있다.
- 전원이 켜진 상태에서 케이블을 꽂거나 빼지 않는다.
- CSI 카메라는 `/dev/video0`로 보이더라도 **일반 v4l2 방식(cv2.VideoCapture(0))으로는
  안 열리는 경우가 많다.** 반드시 GStreamer의 `nvarguscamerasrc` 파이프라인을
  사용해야 한다. (이 프로젝트의 `utils.py`가 이미 처리해준다.)

### 3) Jetson Nano에서 카메라 인식 확인

Jetson Nano 터미널에서:

```bash
ls /dev/video*
```

- CSI 카메라 정상 연결 시 보통 `/dev/video0`가 보인다.
- USB 카메라를 추가로 연결하면 `/dev/video1` 등으로 추가 표시된다.
- 아무것도 안 보이면 케이블/연결 상태를 다시 확인한다.

CSI 카메라 센서 정보(모델명 등)를 좀 더 자세히 보고 싶다면:

```bash
v4l2-ctl --list-devices
```

### 4) GStreamer로 카메라 단독 테스트 (모니터가 Jetson Nano에 직접 연결된 경우)

이 방법은 **Jetson Nano에 모니터가 직접 연결되어 있을 때만** 화면이 뜬다.
순수 SSH 터미널(모니터 없음)에서는 창이 뜰 곳이 없으므로 이 명령들은
생략하고 아래 "5) OpenCV로 카메라 열기 테스트"로 넘어가도 된다.

CSI 카메라:

```bash
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
```

또는 구형 Jetson Nano 이미지에 포함된 전용 도구:

```bash
nvgstcapture-1.0
```

USB 카메라 (Jetson Nano 터미널에서):

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink
```

이 창들은 `Ctrl+C`로 종료한다.

### 5) OpenCV에서 카메라 열기 (이 프로젝트가 사용하는 방식)

이 프로젝트는 `utils.py`의 `open_camera()` 함수가 아래처럼
CSI 카메라는 GStreamer 파이프라인으로, USB 카메라는 일반
`cv2.VideoCapture(device_index)`로 카메라를 연다.

```python
# CSI 카메라 (Jetson Nano 전용 nvarguscamerasrc 파이프라인 사용)
cap = cv2.VideoCapture(gstreamer_pipeline_string, cv2.CAP_GSTREAMER)

# USB 카메라
cap = cv2.VideoCapture(0)
```

직접 함수를 호출할 필요 없이, `python3 camera_test.py`를 실행하면
이 과정을 자동으로 해준다. (아래 "단계별 실행 방법" 참고)

### 6) 카메라가 안 열릴 때 확인 목록

1. `ls /dev/video*` 에 장치가 보이는가?
2. CSI 케이블 방향이 맞는가? (파란/접점 면 방향, 완전히 삽입됐는지)
3. 다른 프로그램(예: 이전에 죽지 않고 남아있는 python 프로세스)이
   카메라를 이미 점유하고 있지 않은가?
   ```bash
   ps aux | grep python
   ```
4. USB 카메라라면 `--camera usb --device 0` (또는 1, 2...)로
   올바른 장치 번호를 지정했는가?
5. Jetson Nano를 재부팅해본다 (`sudo reboot`). CSI 카메라는 간혹
   재부팅 전까지 인식이 꼬여 있는 경우가 있다.

---

## 필요한 패키지 설치

Nano용과 노트북용 요구사항 파일이 분리되어 있다. **각 기기에서 서로 다른
파일을 설치**해야 한다.

### Jetson Nano (촬영 + ChArUco 보정 + 수동 측정)

```bash
cd ~/nail_measure
pip3 install -r requirements_nano.txt
```

**중요:** JetPack(Jetson Nano 공식 이미지)에는 GStreamer(CSI 카메라)와
aruco(ChArUco) 모듈을 지원하는 OpenCV가 시스템에 이미 설치되어 있는
경우가 많다. 이 경우 `pip3 install opencv-python`을 따로 실행하면
GStreamer 지원이 빠진 버전으로 덮어써져서 **CSI 카메라가 갑자기 안 열리게
될 수 있다.** 그래서 `requirements_nano.txt`에는 `opencv-python`을 기본적으로
주석 처리해 두었다. 아래 명령으로 시스템에 OpenCV/aruco가 이미 있는지 먼저 확인하자.

```bash
python3 -c "import cv2; print(cv2.__version__); print('aruco:', hasattr(cv2, 'aruco'))"
```

- 버전과 `aruco: True`가 출력되면 추가 설치가 필요 없다.
- `ModuleNotFoundError`가 나면 그때 `requirements_nano.txt`의
  `opencv-python` 줄 주석을 해제하고 다시 설치한다 (단, 이 경우 CSI
  카메라는 안 될 수 있고 USB 카메라만 동작할 가능성이 높다).
- `aruco: False`가 나오면 `pip3 install opencv-contrib-python`이 필요할 수
  있다 (역시 GStreamer 지원이 빠질 위험이 있으니, 먼저 시스템 OpenCV에
  aruco가 정말 없는지 확실히 확인한 뒤 설치할 것).

### 노트북(리눅스 랩탑) (YOLO 자동 세그멘테이션 + 오차 비교)

```bash
cd ~/nail_measure   # 또는 프로젝트를 clone/복사한 경로
pip3 install -r requirements.txt
```

`ultralytics`(PyTorch 포함)와 `opencv-contrib-python`(ChArUco 지원)이
설치된다. 노트북은 GStreamer/CSI 카메라를 안 쓰므로 OpenCV 버전 충돌
걱정 없이 편하게 설치해도 된다. 최초로 `measure_auto.py --backend yolo`를
실행하면 모델 가중치(`models/nails_seg_s_yolov8_v1.pt`, 약 23.9MB)를
자동으로 다운로드한다.

---

## 전체 실행 흐름

```text
[Jetson Nano]
1. SSH 접속
2. (카메라 연결 확인) python3 camera_test.py
3. (사진 촬영)        python3 capture_image.py
4. (ChArUco 보정)     python3 charuco_calibration.py --image <사진경로>
5. (손톱 측정, 선택)  python3 manual_measure.py --image <사진경로> --calib <보정json경로>

[노트북] (사진을 Nano -> 노트북으로 scp 이동한 뒤)
6. (자동 손톱 측정)   python3 measure_auto.py --image <사진경로> --calib <보정json경로>
7. (실측값 입력/비교) python3 compare_actual.py --image <사진경로>
8. (백엔드별 오차 비교) python3 main.py --mode compare
```

`python3 main.py`를 실행하면 위 과정을 번호 메뉴로 순서대로 안내받으며
실행할 수 있고, `python3 main.py --mode <단계>`로 메뉴 없이 바로 한
단계만 실행할 수도 있다. ([전체 실행 흐름 자동화 참고](#nano-촬영--노트북-측정-워크플로))

---

## 단계별 실행 방법

### 카메라 테스트

```bash
python3 camera_test.py                # CSI 카메라
python3 camera_test.py --camera usb   # USB 카메라
```

### 사진 촬영

```bash
python3 capture_image.py
```

- 화면이 뜨면 `s` = 저장, `q` = 종료.
- 저장 위치: `data/captured/capture_YYYYMMDD_HHMMSS.jpg`

손은 ChArUco 보드 위에 자연스럽게 펴서 올리고, 손톱 뿌리와 끝이
보드와 함께 선명하게 보이도록 촬영한다. 카메라 높이는 29.5cm 고정을
기본으로 한다.

### ChArUco 보정 (v2, 기본)

```bash
python3 charuco_calibration.py --image data/captured/capture_20260704_101500.jpg
```

- 마우스 클릭이 필요 없다 — 보드를 자동으로 검출해서 호모그래피를 계산한다.
- 인쇄한 보드의 한 칸 크기가 기본값(10mm)과 다르면 `--square-mm`로
  실측치를 넘긴다 (`--marker-mm`도 같은 비율(0.7)로 같이 조정).
- 결과는 `data/results/<이미지이름>_charuco.json`에 저장되고,
  검출 결과를 눈으로 확인할 수 있는 디버그 이미지가
  `data/results/debug/<이미지이름>_charuco_debug.jpg`에 저장된다.
- 재투영 RMS가 0.5mm를 넘으면 경고가 뜬다 — 보드 평탄도/각도/`--square-mm`값을 재확인한다.

(v1의 모눈판 방식이 필요하면 `grid_calibration.py --image ... --cells 10`을
그대로 쓸 수 있다. 단, 이 경우 결과 JSON에 `homography`가 없어서
`manual_measure.py`/`measure_auto.py`에는 쓸 수 없다 — v1 JSON은 v1
워크플로 전용이다.)

### 손톱 길이 측정 - 수동 클릭

```bash
python3 manual_measure.py \
  --image data/captured/capture_20260704_101500.jpg \
  --calib data/results/capture_20260704_101500_charuco.json
```

- 엄지 -> 검지 -> 중지 -> 약지 -> 소지 순서로, 각 손가락마다
  "손톱 뿌리" -> "손톱 끝" 두 점을 클릭한다.
- 손가락이 보드 평면보다 떠 있는 정도를 보정하려면 `--nail-height <mm>`을 준다.
- 결과는 `data/results/measurements.csv`에 `backend=manual`로 누적 저장된다.

### 손톱 길이 측정 - 자동 (YOLO, 노트북에서 실행)

```bash
python3 measure_auto.py \
  --image data/captured/capture_20260704_101500.jpg \
  --calib data/results/capture_20260704_101500_charuco.json
```

- 기본 백엔드는 `yolo`다. 검출 개수가 5개가 아니면 `--conf 0.15`처럼
  threshold를 낮춰서 다시 시도해본다.
- 손가락 자동 라벨링은 기본적으로 마스크의 x좌표 순서로 이뤄진다.
  왼손을 촬영했다면 `--hand left`를 지정한다. (자동 매칭이 애매하면
  터미널에서 마스크 번호별로 직접 손가락을 입력하는 폴백이 뜬다)
- 결과는 `data/results/measurements.csv`에 `backend=yolo`로 누적 저장되고,
  디버그 오버레이 이미지가 `data/results/debug/`에 저장된다.
- YOLO 백엔드가 실패하면(모델 다운로드 실패 등) `classical`(피부색 기반)
  백엔드로 자동 폴백한다.

### 실제 자 측정값과 비교

```bash
python3 compare_actual.py --image data/captured/capture_20260704_101500.jpg
```

- 화면 안내에 따라 5개 손가락의 실제 길이(mm)를 입력한다.
- 같은 사진을 manual/yolo 등 여러 백엔드로 측정해뒀다면, 백엔드별로
  각각 비교 결과가 남는다 (`--backends manual yolo`로 특정 백엔드만 비교 가능).
- 결과는 `data/results/comparison.csv`에 누적 저장된다.

### 백엔드별 평균 오차 비교

여러 사진에 대해 manual/yolo(/classical)로 각각 측정 + 비교를 반복해서
`comparison.csv`가 쌓인 뒤, 아래 명령으로 백엔드별 평균 오차를 한눈에 본다.

```bash
python3 main.py --mode compare
```

```text
[비교 결과]
manual    평균 오차: 0.32mm (n=15)
yolo      평균 오차: 0.41mm (n=15)
classical 평균 오차: 0.78mm (n=15)
```

### 카메라 높이별 실험 기록 및 요약 (v1, 높이를 바꿔볼 때)

카메라 높이를 29.5cm 고정이 아니라 바꿔가며 비교하고 싶을 때 쓴다.
(높이가 29.5cm로 고정이라면 이 단계는 건너뛰어도 된다)

```bash
# 사진을 찍을 때마다 촬영 조건 기록
python3 height_experiment.py log \
  --image data/captured/capture_20260704_101500.jpg \
  --camera-model "Arducam 16MP AF" --height 29.5 \
  --lighting "책상 스탠드" --focus auto --note "3차 시도"

# 여러 높이에서 반복한 뒤 평균 오차 요약
python3 height_experiment.py summary
```

### 통합 메뉴로 한 번에

```bash
python3 main.py --image data/captured/capture_20260704_101500.jpg --backend yolo
```

---

## Nano 촬영 -> 노트북 측정 워크플로

YOLO(`--backend yolo`) 자동 측정은 ultralytics/PyTorch가 필요해서
**노트북에서만** 실행한다. Jetson Nano에서 찍은 사진을 노트북으로
옮기는 흐름은 다음과 같다.

```bash
# --- Jetson Nano 터미널 ---
python3 capture_image.py                     # s로 저장, q로 종료
python3 charuco_calibration.py --image data/captured/capture_20260704_101500.jpg
# (여기까지는 GUI 없이도 되고, 수동 측정을 원하면 manual_measure.py도 Nano에서 가능)

# --- 노트북 터미널 (Jetson Nano가 아니라 노트북에서 실행!) ---
mkdir -p ~/nail_measure/data/captured ~/nail_measure/data/results
scp jetson사용자명@Jetson_IP:~/nail_measure/data/captured/capture_20260704_101500.jpg \
    ~/nail_measure/data/captured/
scp jetson사용자명@Jetson_IP:~/nail_measure/data/results/capture_20260704_101500_charuco.json \
    ~/nail_measure/data/results/

cd ~/nail_measure
python3 measure_auto.py \
  --image data/captured/capture_20260704_101500.jpg \
  --calib data/results/capture_20260704_101500_charuco.json
python3 compare_actual.py --image data/captured/capture_20260704_101500.jpg
```

측정 결과(`measurements.csv`, `comparison.csv`)를 다시 Nano와
합치고 싶다면 반대 방향으로 `scp`하거나, 둘 중 한쪽(보통 노트북)을
기준 저장소로 정해서 그쪽에서만 비교/요약을 진행하는 게 편하다.

---

## 결과 CSV 확인 방법

Nano/노트북 어느 터미널에서든 바로 내용을 볼 수 있다.

```bash
cat data/results/measurements.csv
cat data/results/comparison.csv
cat data/results/experiment_log.csv
```

디버그 이미지(보정/자동측정 결과를 눈으로 확인)는 `data/results/debug/`에 저장된다.

```bash
ls data/results/debug/
```

파일을 다른 기기로 옮겨서 엑셀/구글시트나 이미지 뷰어로 열어보고 싶다면
`scp`를 사용한다 ([Nano 촬영 -> 노트북 측정 워크플로](#nano-촬영--노트북-측정-워크플로) 참고).

```bash
# 노트북 터미널에서 실행 (Jetson Nano 터미널이 아님)
scp jetson사용자명@Jetson_IP:~/nail_measure/data/results/*.csv ./
scp jetson사용자명@Jetson_IP:~/nail_measure/data/results/debug/*.jpg ./
```

---

## 자주 발생하는 오류와 해결 방법

| 증상 | 원인/해결 방법 |
| --- | --- |
| `카메라 열기 실패` 메시지 | `ls /dev/video*`로 장치 확인, CSI 케이블 방향/삽입 상태 재확인, USB면 `--device` 번호 확인 |
| `cv2.error: ... not implemented ...` 또는 창이 안 뜸 | 순수 SSH 접속(모니터/X11 없음) 상태. `ssh -X 사용자명@IP`로 재접속하거나 Jetson Nano에 모니터 연결 (촬영/보정 단계 중 `charuco_calibration.py`는 GUI 없이도 동작함) |
| CSI 카메라가 `cv2.VideoCapture(0)`으로 안 열림 | Jetson Nano CSI 카메라는 `nvarguscamerasrc` GStreamer 파이프라인이 필요함 (이 프로젝트는 자동 처리됨). `camera_type="csi"`로 실행했는지 확인 |
| `pip3 install opencv-python` 이후 카메라가 갑자기 안 됨 | JetPack 기본 OpenCV가 pip 버전으로 덮어써져 GStreamer 지원이 빠진 경우. `pip3 uninstall opencv-python` 후 시스템 기본 OpenCV로 되돌리기 |
| `cv2.aruco 모듈이 없습니다` | OpenCV가 opencv-contrib 없이 빌드됨. `python3 -c "import cv2; print(hasattr(cv2,'aruco'))"`로 확인 후 [필요한 패키지 설치](#필요한-패키지-설치) 참고 |
| ChArUco "마커를 하나도 찾지 못했습니다" | 보드 전체가 프레임에 잘 들어왔는지, 초점이 맞는지 확인. 조명이 보드에 반사돼 하얗게 날아간 부분이 없는지 확인 |
| ChArUco 중복 마커 ID가 많이 나옴(`[INFO] ... 중복 ID N개 제거됨`) | 체크무늬 일부가 가짜 마커로 오검출된 것 (자동으로 필터링은 되지만 너무 많으면 정확도가 떨어짐). 조명을 보드에 고르게 비추고 반사를 줄여서 재촬영 |
| ChArUco 재투영 RMS가 0.5mm보다 훨씬 큼 | 보드가 평평하지 않거나(휘어짐/구겨짐) 카메라 각도가 너무 기울어짐. `--square-mm`가 실제 인쇄 크기와 맞는지도 확인 |
| `보정 파일에 homography 값이 없습니다` | v1(`grid_calibration.py`)이 만든 JSON을 `manual_measure.py`/`measure_auto.py`에 넣은 경우. `charuco_calibration.py`로 만든 JSON을 사용할 것 |
| `보정 파일을 찾을 수 없습니다` | `charuco_calibration.py`를 먼저 실행해서 JSON을 만들었는지, `--calib` 경로가 정확한지 확인 |
| `ultralytics 패키지가 설치되어 있지 않습니다` | 노트북에서 `pip3 install -r requirements.txt` 실행 (Nano에는 설치하지 않음) |
| `모델 자동 다운로드에 실패했습니다` | 네트워크/방화벽 확인, 또는 안내된 URL을 브라우저로 열어 `models/nails_seg_s_yolov8_v1.pt`에 수동 저장 |
| YOLO 검출 개수가 5개가 아님 | `--conf` 값을 낮춰서(예: 0.1) 재시도. 계속 안 맞으면 `--backend classical`이나 `--backend manual`로 대체하거나, [모델 정확도가 부족할 때](#모델-정확도가-부족할-때) 참고 |
| 손가락 라벨이 계속 틀림 | `--hand left`/`right` 값이 실제 촬영한 손과 맞는지 확인. 마스크가 정확히 5개가 아니면 자동으로 뜨는 터미널 수동 입력 프롬프트를 이용 |
| `측정 데이터가 없습니다` (compare_actual.py) | `manual_measure.py` 또는 `measure_auto.py`를 먼저 실행했는지, `--image` 경로 문자열이 measurements.csv에 저장된 것과 정확히 같은지 확인 |
| `height_experiment.py summary`에서 표본이 안 잡힘 | `height_experiment.py log`로 이미지별 높이를 먼저 기록했는지, CSV들의 `image_path` 값이 서로 동일한 문자열인지 확인 |
| 사진이 뿌옇게 나옴 | Auto Focus 카메라의 초점이 안 맞은 상태. 초점이 맞을 때까지 기다리거나 살짝 흔들어 재초점, 렌즈 보호 필름 제거 여부 확인 |
| 손톱 뿌리/끝 위치를 잘못 클릭함 | 클릭 도중 `r` 키로 마지막 점 되돌리기, 전체를 취소하려면 `q` |

---

## 모델 정확도가 부족할 때

공개 사전학습 YOLO 모델은 우리 촬영 환경(수직 촬영, ChArUco 배경,
29.5cm 고정 높이)으로 학습된 게 아니라서, `--conf`를 조정해도 정확도가
계속 부족할 수 있다. `python3 main.py --mode compare`로 `manual`
백엔드 대비 오차가 얼마나 큰지 확인하고, 필요하면 우리가 찍은 사진으로
모델을 직접 파인튜닝하는 절차를 `training/finetune_guide.md`에
정리해뒀다 (라벨링 도구, 전이학습 명령, 모델 교체 방법 등, 문서만
제공하며 코드 구현은 포함하지 않는다).
