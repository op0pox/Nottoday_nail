# segmentation — 손톱 세그멘테이션 + 체커보드 실측 GUI

손가락을 **한 개씩** 촬영한 위에서(Top) / 측면(Side) 사진을 한 장씩 넣으면:

1. 이미지마다 손톱 1개를 자동 세그멘테이션해서 **오른쪽 미리보기**에 표시
2. **실측값이 표기된 세그멘테이션 이미지**를 `results/<날짜시간>/`에 저장
   (`top.png`, `side.png` + 수치 데이터 `measurements.json`)
3. 사진 속 **ChArUco 체커보드를 자동 인식**하면 손톱 길이/폭을 mm로 실측해
   이미지 위에 표기 (보드가 없으면 픽셀 단위로만 표기)

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
   - **체커보드 실측**: 체크 시 사진 속 ChArUco 보드를 자동 인식해 mm 계산
3. **[세그멘테이션 실행]** 클릭 → 미리보기 확인, 저장 경로는 상태줄에 표시

이미지 한 장에는 손가락(손톱) 하나만 나오게 촬영한다. 여러 개가 검출되면
가장 확실한(신뢰도/면적 최대) 손톱 하나만 사용한다.

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
   CC BY 4.0)로 손톱 마스크를 얻는다. 검출이 여러 개면 신뢰도/면적이 가장
   큰 1개만 사용.
2. **체커보드 보정**: 사진 속 ChArUco 보드의 마커를 자동 검출해
   이미지 픽셀 ↔ 보드 mm 좌표 호모그래피를 계산한다 (보드 일부만 나와도 동작,
   재투영 RMS가 0.5mm를 넘으면 경고).
3. **실측**: 마스크의 세로(y) 범위를 길이로, 길이선 중간 높이의 가로 폭을
   폭으로 재서 호모그래피로 mm 변환한다.

## 체커보드(ChArUco) 스펙

`charuco_calibration.py` 상단 상수와 **인쇄된 보드가 반드시 일치해야 한다**:

- 18 x 26칸, 한 칸 10mm, 마커 7mm, `DICT_4X4_250`

인쇄 후 실측값이 다르면 `SQUARES_X/Y`, `SQUARE_MM`, `MARKER_MM`을 수정할 것.
보드 스펙이 틀리면 mm 실측값이 통째로 배수만큼 틀어진다.

## 문제 해결

| 증상 | 해결 |
|---|---|
| 손톱 검출 실패 | conf를 0.1 이하로 낮춤, 조명/초점 확인 |
| 체커보드 미검출 | 보드가 사진에 충분히 나오는지, 반사/그림자 확인 |
| macOS에서 실행 중 죽음 (OpenMP) | `KMP_DUPLICATE_LIB_OK=TRUE` 붙여서 실행 |
| `cv2.aruco` 없음 오류 | `pip3 uninstall opencv-python` 후 `opencv-contrib-python` 재설치 |
| tkinter 없음 오류 | macOS: `brew install python-tk@3.14`, 리눅스: `apt install python3-tk` |
