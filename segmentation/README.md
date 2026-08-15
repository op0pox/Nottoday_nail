# segmentation — 손톱 세그멘테이션 + 실측 GUI

위에서(Top) / 측면(Side) 사진을 한 장씩 넣으면:

1. 이미지에서 **인식된 손톱 전부**를 자동 세그멘테이션해서 **오른쪽 미리보기**에 표시
   (손 전체 사진이면 5개 전부, 손가락 1개 사진이면 1개 — 여러 개일 때는
   사진 왼쪽부터 1, 2, ... 번호를 붙여 각각 실측)
2. **실측값이 표기된 세그멘테이션 이미지**를 `results/<날짜시간>/`에 저장
   (`top.png`, `side.png` + 수치 데이터 `measurements.json`)
3. 손톱 길이/폭을 **mm로 실측**해 이미지 위에 표기. 보정 방식은 3가지:
   - **고정 거리 보정 (기본)** — 카메라가 장치에 고정된 경우, 렌즈~손톱
     거리(cm)만 입력하면 자동 보정 (기본값: 위 10.5cm / 측면 7.0cm).
     체커보드 촬영 불필요. `nano_capture/dual_capture.py`로 찍은
     원본(1920x1080)을 그대로 넣으면 된다.
   - **체커보드(ChArUco) 인식** — 사진 안에 보드가 함께 찍힌 경우
   - **보정 없음** — 픽셀 단위로만 표기

결과 이미지 스타일: 초록 윤곽선 + 빨간 세로 길이선 + 파란 폭선 + 노란 mm 텍스트
(기존 measure_auto.py 디버그 이미지와 동일)

## 설치

```bash
# tkinter — Homebrew Python(macOS): brew install python-tk@3.14
#           리눅스: sudo apt install python3-tk

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 설치 점검 + YOLO 모델(약 24MB) 자동 다운로드
./.venv/bin/python check_setup.py
```

## 실행

```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python seg_gui.py
```

1. **[위에서(Top) 이미지 열기]** / **[측면(Side) 이미지 열기]**로 사진을
   한 장씩 고른다 (한 장만 골라도 동작)
2. 필요하면 옵션 조정
   - **검출 신뢰도(conf)**: 손톱이 검출되지 않으면 낮춘다 (예: 0.1)
   - **mm 보정 방식**: 고정 거리 보정(기본) / 체커보드 인식 / 보정 없음
   - **위 거리(cm) / 측면 거리(cm)**: 고정 거리 모드에서 렌즈~손톱 거리.
     장치가 바뀌지 않는 한 그대로 두면 된다
3. **[세그멘테이션 실행]** 클릭 → 미리보기 확인, 저장 경로는 상태줄에 표시

## 파일 구성

| 파일 | 역할 |
|---|---|
| `seg_gui.py` | 메인 GUI (이것만 실행하면 됨) |
| `check_setup.py` | 설치 상태 점검 + YOLO 모델 다운로드 |
| `charuco_calibration.py` | ChArUco 체커보드 자동 인식 → 픽셀↔mm 호모그래피 (단독 CLI 실행도 가능) |
| `utils.py` | 공용 유틸 (mm 변환, JSON 저장 등) |
| `backends/common.py` | 백엔드 공통 인터페이스 (`NailMask`, `SegmentationBackend`) |
| `backends/dl_yolo.py` | YOLOv8-seg 자동 세그멘테이션 (메인, 사전학습 공개 모델) |
| `backends/classical.py` | 피부색 기반 폴백 백엔드 (YOLO 로드 실패 시 자동 사용) |
| `backends/measure_from_mask.py` | 마스크에서 세로 길이/중간 폭 추출 + mm 계산 |
| `models/` | YOLO 가중치 자동 다운로드 위치 (git 미포함) |
| `results/` | 실행 결과 저장 위치 (git 미포함) |

## 동작 원리

1. **세그멘테이션**: 사전학습 YOLOv8-seg 손톱 모델(mnemic/nails_seg_yolov8,
   CC BY 4.0)로 손톱 마스크를 얻는다. 검출된 손톱은 전부 사용하며,
   사진 왼쪽부터 순서대로 번호를 붙인다.
2. **픽셀↔mm 보정**:
   - 고정 거리 모드 — 핀홀 카메라 모델로 `px_per_mm = 초점거리(px) / 거리(mm)`를
     계산한다. 초점거리(px)는 사진 해상도로 센서 모드(IMX219 비닝 여부)를
     판별해 정한다. 지원 해상도: 3264x2464, 1920x1080, 1640x1232, 1280x720.
   - 체커보드 모드 — 사진 속 ChArUco 보드의 마커를 자동 검출해
     이미지 픽셀 ↔ 보드 mm 좌표 호모그래피를 계산한다 (보드 일부만 나와도 동작,
     재투영 RMS가 0.5mm를 넘으면 경고).
3. **실측**: 마스크의 세로(y) 범위를 길이로, 길이선 중간 높이의 가로 폭을
   폭으로 재서 호모그래피로 mm 변환한다.

## 체커보드(ChArUco) 스펙 — 체커보드 모드를 쓸 때만 해당

`charuco_calibration.py` 상단 상수와 **인쇄된 보드가 반드시 일치해야 한다**:

- 18 x 26칸, 한 칸 10mm, 마커 7mm, `DICT_4X4_250`

인쇄 후 실측값이 다르면 `SQUARES_X/Y`, `SQUARE_MM`, `MARKER_MM`을 수정할 것.
보드 스펙이 틀리면 mm 실측값이 통째로 배수만큼 틀어진다.

## 문제 해결

| 증상 | 해결 |
|---|---|
| 손톱 검출 실패 | conf를 0.1 이하로 낮춤, 조명/초점 확인 |
| 체커보드 미검출 | 보드가 사진에 충분히 나오는지, 반사/그림자 확인 |
| 고정 거리 모드 "지원 안 되는 해상도" | 촬영 원본을 자르거나 리사이즈하지 말고 그대로 넣기 |
| 고정 거리 모드 측정값이 몇 % 다름 | 거리(cm)를 렌즈 표면~손톱 표면 기준으로 다시 재서 입력 |
| macOS에서 실행 중 죽음 (OpenMP) | `KMP_DUPLICATE_LIB_OK=TRUE` 붙여서 실행 |
| `cv2.aruco` 없음 오류 | `pip3 uninstall opencv-python` 후 `opencv-contrib-python` 재설치 |
| tkinter 없음 오류 | macOS: `brew install python-tk@3.14`, 리눅스: `apt install python3-tk` |
