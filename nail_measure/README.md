# nail_measure

Jetson Nano + 카메라로 손톱 사진을 찍고, 0.5mm 모눈판을 기준으로
픽셀-mm 비율을 계산한 뒤, 엄지/검지/중지/약지/소지 5개 손톱의
길이(뿌리~끝)를 mm 단위로 측정하는 프로젝트다.

초기 버전은 딥러닝 자동 검출이 아니라, **사용자가 이미지에서 직접
점을 클릭하는 방식**으로 정확한 측정 흐름을 먼저 검증하는 것을
목표로 한다.

## 목차

1. [사용 장비](#사용-장비)
2. [프로젝트 구조](#프로젝트-구조)
3. [Jetson Nano 카메라 연결 방법](#jetson-nano-카메라-연결-방법)
4. [필요한 패키지 설치](#필요한-패키지-설치)
5. [전체 실행 흐름](#전체-실행-흐름)
6. [단계별 실행 방법](#단계별-실행-방법)
7. [결과 CSV 확인 방법](#결과-csv-확인-방법)
8. [자주 발생하는 오류와 해결 방법](#자주-발생하는-오류와-해결-방법)

---

## 사용 장비

- Jetson Nano
- 카메라 후보 (아무거나 하나면 됨)
  - Arducam 16MP Auto Focus (CSI 또는 USB 버전)
  - Raspberry Pi Camera Module 3 (CSI)
  - Raspberry Pi Camera Module 2 (CSI)
- 0.5mm 모눈판
- 일반 자 (실제 길이 검증용)
- 카메라 고정 거치대 (높이를 바꿔가며 촬영하기 위함)
- 조명

---

## 프로젝트 구조

```text
nail_measure/
├── main.py                  # 모든 기능을 번호로 실행하는 통합 메뉴
├── utils.py                 # 공용 함수 모음 (카메라 열기, 클릭 UI, 파일 저장 등)
├── camera_test.py           # 1단계: 카메라 연결 확인
├── capture_image.py         # 2단계: 사진 촬영 (s=저장, q=종료)
├── grid_calibration.py      # 3단계: 0.5mm 모눈 기준 px_per_mm 계산
├── manual_measure.py        # 4단계: 5개 손가락 손톱 길이 수동 측정
├── compare_actual.py        # 5단계: 실제 자 측정값과 비교(오차/오차율)
├── height_experiment.py     # 6단계: 카메라 높이별 실험 기록 및 평균 오차 요약
├── requirements.txt         # 파이썬 패키지 목록
├── README.md                # 이 문서
├── data/
│   ├── captured/             # 촬영된 원본 사진 저장 위치
│   ├── results/               # calib.json, measurements.csv, comparison.csv 등 결과물
│   └── test_images/          # 테스트용 샘플 이미지 (직접 넣어서 사용 가능)
└── docs/
    └── experiment_log_template.md   # 실험 결과를 사람이 보기 좋게 기록하는 템플릿
```

각 파일의 역할은 파일 맨 위 docstring에도 "1. 역할 / 2. 실행 명령어 /
3. 어디에 입력하는지 / 4. 정상 화면 / 5. 오류 대처" 형식으로 자세히
적어두었다.

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

Jetson Nano 터미널에서 실행한다.

```bash
cd ~/nail_measure   # 프로젝트를 clone/복사한 경로로 이동
pip3 install -r requirements.txt
```

**중요:** JetPack(Jetson Nano 공식 이미지)에는 GStreamer(CSI 카메라)를
지원하는 OpenCV가 시스템에 이미 설치되어 있는 경우가 많다. 이 경우
`pip3 install opencv-python`을 따로 실행하면 GStreamer 지원이 빠진
버전으로 덮어써져서 **CSI 카메라가 갑자기 안 열리게 될 수 있다.**

그래서 `requirements.txt`에는 `opencv-python`을 기본적으로 주석 처리해
두었다. 아래 명령으로 시스템에 OpenCV가 이미 있는지 먼저 확인하자.

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

- 버전이 출력되면 OpenCV가 이미 있는 것이므로 추가 설치가 필요 없다.
- `ModuleNotFoundError`가 나면 그때 `requirements.txt`의
  `opencv-python` 줄 주석을 해제하고 다시 `pip3 install -r requirements.txt`를
  실행한다. (단, 이 경우 CSI 카메라는 안 될 수 있고 USB 카메라만 동작할
  가능성이 높다.)

---

## 전체 실행 흐름

```text
1. Jetson Nano에 SSH 접속
2. (카메라 연결 확인) python3 camera_test.py
3. (사진 촬영)        python3 capture_image.py
4. (모눈 보정)        python3 grid_calibration.py --image <사진경로>
5. (손톱 측정)        python3 manual_measure.py --image <사진경로> --calib <보정json경로>
6. (실측값 입력/비교) python3 compare_actual.py --image <사진경로>
7. (촬영조건 기록)    python3 height_experiment.py log --image <사진경로> --camera-model "..." --height 20
8. (높이별 오차 요약) python3 height_experiment.py summary
```

`python3 main.py`를 실행하면 위 과정을 번호 메뉴로 순서대로 안내받으며
실행할 수 있다.

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

손은 0.5mm 모눈판 위에 자연스럽게 펴서 올리고, 손톱 뿌리와 끝이
모눈판과 함께 선명하게 보이도록 촬영한다.

### 모눈(0.5mm) 보정

```bash
python3 grid_calibration.py --image data/captured/capture_20260702_193045.jpg --cells 10
```

- 모눈 10칸(=5.0mm) 정도의 시작점/끝점을 클릭한다. 칸 수를 다르게
  세었다면 `--cells` 값을 그에 맞게 바꾼다.
- 결과는 `data/results/<이미지이름>_calib.json`에 저장된다.

### 손톱 길이 측정

```bash
python3 manual_measure.py \
  --image data/captured/capture_20260702_193045.jpg \
  --calib data/results/capture_20260702_193045_calib.json
```

- 엄지 -> 검지 -> 중지 -> 약지 -> 소지 순서로, 각 손가락마다
  "손톱 뿌리" -> "손톱 끝" 두 점을 클릭한다.
- 결과는 `data/results/measurements.csv`에 누적 저장된다.

### 실제 자 측정값과 비교

```bash
python3 compare_actual.py --image data/captured/capture_20260702_193045.jpg
```

- 화면 안내에 따라 5개 손가락의 실제 길이(mm)를 입력한다.
- 결과는 `data/results/comparison.csv`에 누적 저장된다.

### 카메라 높이별 실험 기록 및 요약

```bash
# 사진을 찍을 때마다 촬영 조건 기록
python3 height_experiment.py log \
  --image data/captured/capture_20260702_193045.jpg \
  --camera-model "Arducam 16MP AF" --height 20 \
  --lighting "책상 스탠드" --focus auto --note "3차 시도"

# 여러 높이에서 반복한 뒤 평균 오차 요약
python3 height_experiment.py summary
```

### 통합 메뉴로 한 번에

```bash
python3 main.py --camera usb --camera-model "Arducam 16MP AF" --height 20
```

---

## 결과 CSV 확인 방법

Jetson Nano 터미널에서 바로 내용을 볼 수 있다.

```bash
cat data/results/measurements.csv
cat data/results/comparison.csv
cat data/results/experiment_log.csv
```

파일을 노트북으로 옮겨서 엑셀/구글시트로 열어보고 싶다면, 노트북
터미널에서 `scp`를 사용한다 (Jetson Nano가 아니라 **노트북 터미널**에서 실행).

```bash
scp jetson사용자명@Jetson_IP:~/nail_measure/data/results/*.csv ./
```

---

## 자주 발생하는 오류와 해결 방법

| 증상 | 원인/해결 방법 |
| --- | --- |
| `카메라 열기 실패` 메시지 | `ls /dev/video*`로 장치 확인, CSI 케이블 방향/삽입 상태 재확인, USB면 `--device` 번호 확인 |
| `cv2.error: ... not implemented ...` 또는 창이 안 뜸 | 순수 SSH 접속(모니터/X11 없음) 상태. `ssh -X 사용자명@IP`로 재접속하거나 Jetson Nano에 모니터 연결 |
| CSI 카메라가 `cv2.VideoCapture(0)`으로 안 열림 | Jetson Nano CSI 카메라는 `nvarguscamerasrc` GStreamer 파이프라인이 필요함 (이 프로젝트는 자동 처리됨). `camera_type="csi"`로 실행했는지 확인 |
| `pip3 install opencv-python` 이후 카메라가 갑자기 안 됨 | JetPack 기본 OpenCV가 pip 버전으로 덮어써져 GStreamer 지원이 빠진 경우. `pip3 uninstall opencv-python` 후 시스템 기본 OpenCV로 되돌리기 |
| `보정 파일을 찾을 수 없습니다` | `grid_calibration.py`를 먼저 실행해서 JSON을 만들었는지, `--calib` 경로가 정확한지 확인 |
| `측정 데이터가 없습니다` (compare_actual.py) | `manual_measure.py`를 먼저 실행했는지, `--image` 경로 문자열이 measurements.csv에 저장된 것과 정확히 같은지 확인 |
| `height_experiment.py summary`에서 표본이 안 잡힘 | `height_experiment.py log`로 이미지별 높이를 먼저 기록했는지, 세 CSV(experiment_log/measurements/comparison)의 `image_path` 값이 서로 동일한 문자열인지 확인 |
| 사진이 뿌옇게 나옴 | Auto Focus 카메라의 초점이 안 맞은 상태. 초점이 맞을 때까지 기다리거나 살짝 흔들어 재초점, 렌즈 보호 필름 제거 여부 확인 |
| 손톱 뿌리/끝 위치를 잘못 클릭함 | 클릭 도중 `r` 키로 마지막 점 되돌리기, 전체를 취소하려면 `q` |

---

## 다음 단계 (자동 검출 확장 예정)

현재는 사용자가 직접 클릭해서 측정하는 수동 방식이다. 이후에는
`manual_measure.py`의 "손톱 뿌리/끝 좌표 구하기" 부분만
딥러닝 기반 손톱 검출 모델로 교체하고, 픽셀->mm 변환 로직(px_per_mm)과
CSV 저장 형식은 그대로 재사용할 수 있도록 함수를 분리해서 작성했다.
