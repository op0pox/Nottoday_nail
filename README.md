# Nottoday_nail

네일 팁 제작 자동화를 위한 손톱 촬영·세그멘테이션·곡면 계산 프로젝트.

## 폴더 구성

```
Nottoday_nail/
├── capture.py        # Jetson CSI 카메라 정지 이미지 캡처 CLI
├── curvature/        # 곡면 길이 계산기 GUI (P/B/S/C 유형)
└── segmentation/     # 손톱 세그멘테이션 + 체커보드 실측 GUI
    ├── seg_gui.py            # 메인 GUI
    ├── check_setup.py        # 설치 상태 점검 + YOLO 모델 다운로드
    ├── charuco_calibration.py# ChArUco 보드 인식 → 픽셀↔mm 변환
    ├── backends/             # 세그멘테이션 백엔드 (YOLO / classical)
    ├── models/               # YOLO 가중치 (자동 다운로드, git 미포함)
    └── results/              # 실행 결과 저장 (git 미포함)
```

## 설치 (macOS 기준)

```bash
# 1. tkinter (GUI 라이브러리) — Homebrew Python 사용 시
brew install python-tk@3.14

# 2. 가상환경 생성 + 패키지 설치
cd segmentation
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 3. 설치 점검 (YOLO 모델 약 24MB 자동 다운로드)
./.venv/bin/python check_setup.py
```

리눅스는 `sudo apt install python3-tk` 후 같은 방식으로 진행한다.

## 사용법

### 손톱 세그멘테이션 + 실측 — [`segmentation/`](segmentation/README.md)

손가락을 **한 개씩** 촬영한 위에서(Top)/측면(Side) 사진을 한 장씩 넣으면
손톱을 자동 세그멘테이션하고, 사진 속 ChArUco 체커보드를 인식해
길이/폭을 mm로 실측한다. 결과 이미지는 `results/`에 자동 저장.

```bash
cd segmentation
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python seg_gui.py
```

### 곡면 길이 계산기 — [`curvature/`](curvature/README.md)

손톱의 정면 너비·측면 너비 실측값과 곡면 유형(P/B/S/C)을 입력하면
네일 팁 제작에 필요한 곡면 길이를 계산한다.
tkinter + matplotlib이 필요한데 `segmentation/.venv`에 이미 들어있으므로
같은 가상환경으로 실행하면 된다.

```bash
KMP_DUPLICATE_LIB_OK=TRUE segmentation/.venv/bin/python curvature/curvature.py
```

### 촬영 — `capture.py`

Jetson Nano CSI 카메라로 정지 이미지를 빠르게 캡처한다 (Enter=촬영, q=종료).

```bash
python3 capture.py
```

## 참고

- Python 3.9+ 권장
- macOS에서 OpenMP 충돌 오류(`libomp` 관련)가 나면 `KMP_DUPLICATE_LIB_OK=TRUE`를 앞에 붙여 실행
- 체커보드 스펙(칸 수/크기)은 `segmentation/charuco_calibration.py` 상단 상수와
  인쇄물이 일치해야 한다 (기본: 18x26칸, 한 칸 10mm, 마커 7mm, DICT_4X4_250)
