# nail_analysis

Jetson Nano B01 + 듀얼 CSI 카메라(정면=바닥 내려다보기, 측면=수직 거치)로
손가락을 한 개씩 지그(jig)에 놓고 촬영해서, 손톱 팁(tip) 제조를 위한
세 가지 값을 자동으로 계산하는 프로그램이다.

1. **바디 기장** (Long / Medium / Short) — 정면 세로/가로 비율
2. **손톱 유형** (P / S / B / C) — 정면 사다리꼴 + 측면 다이아몬드 판정 트리
3. **곡면 길이(mm) → 팁 사이즈 매칭** — 팁이 감싸야 하는 곡선의 실제 길이를
   기하 공식으로 계산하고, 사이즈 테이블에서 가장 가까운 번호를 찾는다

`nail_measure/`(같은 저장소의 별도 프로젝트, 손톱 길이만 측정)와는 목적과
파이프라인이 달라서 완전히 분리된 프로젝트로 새로 만들었다.

## 목차

1. [사용 장비](#사용-장비)
2. [프로젝트 구조](#프로젝트-구조)
3. [설치](#설치)
4. [ChArUco 보드 준비 및 정렬](#charuco-보드-준비-및-정렬)
5. [전체 사용 흐름](#전체-사용-흐름)
6. [서브커맨드](#서브커맨드)
7. [판정 로직 요약](#판정-로직-요약)
8. [자주 발생하는 오류](#자주-발생하는-오류)
9. [향후 로드맵: 고전 CV → AI 세그멘테이션](#향후-로드맵-고전-cv--ai-세그멘테이션)

---

## 사용 장비

- Jetson Nano B01 (CSI 포트 2개)
- IMX219 계열 카메라(라즈베리파이 카메라 V2 호환) 2대
  - CAM0(sensor-id 0) = **정면**: 손가락 위에서 내려다보는 위치, 바닥 평면
  - CAM1(sensor-id 1) = **측면**: 손가락 옆에서 수평으로 보는 위치, 수직 거치대
- 손가락 고정용 지그(한 번에 손가락 하나씩 정해진 위치/각도로 고정)
- ChArUco 보드 인쇄물 2장 (정면 바닥용 1장, 측면 수직 거치용 1장 — `generate_board.py`로 생성)
- 카메라/지그가 흔들리지 않는 고정 거치대, 조명

## 프로젝트 구조

```text
nail_analysis/
├── main.py                  # CLI 엔트리: calibrate / measure / analyze / collect
├── config.yaml               # 모든 임계값/상수/사이즈 테이블 (여기만 고치면 대부분의 튜닝이 끝남)
├── config_loader.py          # config.yaml 로더
├── camera.py                 # 듀얼 CSI(GStreamer) 캡처 + 정지 이미지 폴백
├── calibration.py            # ChArUco 캘리브레이션, 실시간 스케일(ScaleProvider), 평면 오프셋 보정
├── generate_board.py         # 인쇄용 ChArUco 보드 PNG 생성
├── visualize.py               # 키포인트/측정선/판정 결과 오버레이(영문 텍스트)
├── collect.py                 # 학습용 원본 이미지/마스크/키포인트 저장 (SAM 라벨링 준비용)
├── detection/
│   ├── base.py                 # NailSegmenter / KeypointDetector 인터페이스
│   ├── contour.py               # 고전 CV 세그멘테이션 + 마스크→키포인트 + 수동 보정 UI
│   └── dl_segmenter.py           # 향후 ONNX/TensorRT 세그멘테이션 스텁 (로드맵 참고)
├── analysis/
│   ├── body_length.py           # 바디 기장 분류
│   ├── nail_type.py              # 손톱 유형(P/S/B/C) 판정
│   ├── curve.py                  # 곡면 길이 공식
│   └── tip_match.py              # 곡면 길이 → 팁 사이즈 매칭
├── tests/
│   └── test_curve.py             # curve.py 공식 검증 (실측 케이스 + PDF 원문 표기 동치 증명)
├── requirements.txt
├── data/
│   ├── captured/                 # (필요 시) 원본 촬영본
│   ├── results/                   # calib_front.json, calib_side.json, measure_*.json, analyze 결과
│   │   └── debug/                  # --debug 세그멘테이션 중간 단계 이미지
│   └── collect/                   # collect 모드 / measure --save-data 저장 위치
└── README.md
```

각 파일 맨 위 docstring에 "1.역할 / 2.실행 명령어 / 3.어디서 실행 /
4.정상 화면 / 5.오류 대처" 형식으로 자세한 설명이 있다.

## 설치

```bash
cd nail_analysis
pip3 install -r requirements.txt
```

Jetson Nano에서는 **시스템 OpenCV(JetPack 번들)를 그대로 쓰는 것을 강력히
권장**한다. pip으로 opencv-contrib-python을 새로 깔면 GStreamer(CSI 카메라)
지원이 빠진 버전으로 덮어써질 수 있다 — 실제로 이 저장소의 `nail_measure`
프로젝트에서 이 문제로 카메라가 전혀 안 열리는 사고가 있었다. 설치 전에
반드시 아래로 먼저 확인하고, requirements.txt 안의 안내 주석을 따를 것.

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
python3 -c "import cv2; print(hasattr(cv2, 'aruco'))"
```

Python 3.6에서도 그대로 동작하도록 작성했다 (dataclass/walrus/match 미사용).

## ChArUco 보드 준비 및 정렬

```bash
python3 generate_board.py --output charuco_board.png --dpi 300
```

- 인쇄 시 프린터 설정을 **"실제 크기(100%)"**로 해야 한다. "페이지에 맞추기"로
  인쇄하면 모든 mm 측정값이 통째로 틀어진다. 인쇄 후 자로 한 칸 크기를 재서
  `config.yaml`의 `charuco_board.square_length_mm`과 일치하는지 확인할 것.
- 같은 보드를 2장 인쇄해서 **정면용**은 손가락이 놓이는 바닥(지그 주변)에,
  **측면용**은 카메라 옆에 수직으로 영구 고정한다.

**측면 보드 정렬이 특히 중요하다.** 측면 카메라는 보드가 항상 카메라와 같은
거리에 고정돼 있지만, 손가락은 지그 위 다른 위치(다른 거리)에 놓인다.
호모그래피 기반 스케일은 "보드가 있는 평면" 기준으로 계산되므로, 보드
평면과 손가락 평면 사이에 거리 차이가 있으면 mm 환산에 체계적인 오차가
생긴다 (스케일은 카메라로부터의 거리에 반비례한다).

이 오차는 `config.yaml`의 `calibration.side_plane_offset_mm`로 보정한다:

```yaml
calibration:
  side_plane_offset_mm: 0.0   # 보드 평면과 손가락 평면의 거리 차이(mm)
```

- 보드를 손가락과 정확히 같은 평면에 세웠다면 `0.0`
- 보드가 손가락보다 카메라에서 **더 멀리** 있으면 그 차이(양수, mm)를 입력
- 보드가 손가락보다 카메라에 **더 가까이** 있으면 음수로 입력

내부적으로 `calibration.py`의 `ScaleProvider`가 매 프레임 `cv2.solvePnP`로
보드까지의 실제 거리(z)를 추정하고, `보정계수 = board_distance / (board_distance + side_plane_offset_mm)`를
호모그래피에 곱해서 손가락 평면 기준 스케일로 맞춘다. 실측 검증 후 오차가
계속 한쪽으로 치우치면 이 값을 조금씩 조정하면 된다.

## 전체 사용 흐름

```bash
# 1) 카메라별로 1회 캘리브레이션 (내부 파라미터 + 기준 호모그래피 저장)
python3 main.py calibrate --camera front
python3 main.py calibrate --camera side

# 2) 손가락 10개 순차 측정 (지그에 손가락을 놓고 스페이스바로 촬영)
python3 main.py measure

# (선택) 측정과 동시에 학습용 데이터도 저장
python3 main.py measure --save-data
```

카메라가 없는 노트북/PC에서 로직만 검증하려면:

```bash
python3 main.py analyze --front finger_front.jpg --side finger_side.jpg
```

## 서브커맨드

### `calibrate --camera front|side`

ChArUco 보드를 여러 각도로 촬영(스페이스바로 캡처, 최소 2장, 기본 15장)해서
카메라 내부 파라미터(camera_matrix/dist_coeffs)를 구하고, 마지막 촬영본을
기준으로 이미지↔mm 호모그래피를 계산해 `data/results/calib_front.json` 또는
`calib_side.json`에 저장한다. `--images board1.jpg board2.jpg ...`로 이미
찍어둔 사진들을 넣어 정지 이미지 모드로도 실행할 수 있다.

### `measure`

`config.yaml`의 `fingers.order`(오른손 엄지→소지, 왼손 엄지→소지) 순서로
한 손가락씩 진행한다.

1. 라이브 프리뷰에서 스페이스바로 정면+측면 동시 촬영 (화면에 `SCALE: live` /
   `held` / `fallback` / `UNAVAILABLE` 상태가 표시됨 — `live`가 아니면 오차가
   커질 수 있으니 가능하면 보드가 보이는 상태에서 촬영할 것)
2. 자동 키포인트 검출 → 실패하거나 신뢰할 수 없으면 수동 보정 UI(마우스 드래그)
3. 결과 오버레이(Body/Type/Curve/Tip, 영문)를 보고 **Enter=확정 / r=재촬영 /
   s=건너뛰기 / q=중단**
4. 전체 손가락이 끝나면 `data/results/measure_YYYYMMDD_HHMMSS.json`에 요약 저장
   (사용된 config 스냅샷 포함)

### `analyze --front A.jpg --side B.jpg`

이미 찍어둔 정면/측면 이미지 한 쌍으로 카메라 없이 로직을 검증한다. 라이브
보드 검출이 불가능하므로 저장된 `calib_front.json`/`calib_side.json`의 기준
호모그래피를 그대로 사용한다(반드시 먼저 `calibrate`를 실행해야 함). 결과는
`data/results/<finger-id>_overlay.jpg`, `<finger-id>.json`으로 저장된다.

### `collect --session NAME`

`measure`와 같은 촬영 루프를 쓰지만 판정 계산 없이 원본 이미지 + (가능하면)
자동 마스크/키포인트만 `data/collect/session_.../`에 저장한다. 향후 AI
세그멘테이션 모델 학습 데이터를 모으기 위한 모드다. `measure --save-data`도
내부적으로 같은 `collect.CollectSession`을 공유한다.

## 판정 로직 요약

- **바디 기장**: `R = H_front / W_front` → `R>=1.5` Long, `1.0<=R<1.5` Medium, `R<1.0` Short
- **손톱 유형**: 정면 사다리꼴 기울기로 P/C 후보 vs S/B 후보를 먼저 힌트로 잡고,
  측면 판정 트리(점3/점4 높이 → S/C 분기 또는 P/B 분기)로 최종 유형을 정한다.
  두 결과 계열이 어긋나면(`mismatch`) 측면 트리를 우선하고 신뢰도를 낮춰 표시한다.
  원본 PDF의 P/B 분기 문구가 일부 모호해서(`analysis/nail_type.py` 상단 주석 참고)
  기본 해석을 구현하고 `config.yaml`의 `nail_type.side_pb_invert`로 방향을
  뒤집을 수 있게 해뒀다 — 실측 검증 후 반대로 나오면 이 값을 바꿀 것.
- **곡면 길이**: `flat = W_front * flat_ratio`, `a = (W_front-flat)/2`,
  `c = sqrt(a^2+W_side^2)`, `h = c * h_ratio`, `r = c^2/(8h) + h/2`,
  `theta = 2*asin(c/(2r))`, `곡면길이 = 2*r*theta + flat`. 유형별 상수는
  `config.yaml`의 `curve_constants`. `tests/test_curve.py`에서 C형
  실측 케이스(W_front=8.3mm, W_side=4.4mm → 약 13.59mm, ±0.15mm 이내)로 검증됨.
- **팁 사이즈**: `config.yaml`의 `tip_size_table`(현재 placeholder)에서
  계산된 곡면 길이와 가장 가까운 번호를 고른다. 실제 제품 사이즈가 확정되면
  이 테이블만 교체하면 된다.

## 자주 발생하는 오류

| 메시지 | 원인/대처 |
|---|---|
| `cv2.aruco 모듈이 없습니다` | OpenCV가 contrib 없이 빌드됨. requirements.txt 안내 주석 참고 |
| `카메라(...)를 열 수 없습니다` | CSI 케이블/장치 확인, `ls /dev/video*`, 카메라 없으면 정지 이미지 모드 사용 |
| `calib_*.json이 없거나 불완전합니다` | `calibrate --camera front`/`--camera side`를 먼저 실행 |
| `mm 스케일을 알 수 없습니다` | 보드가 화면에 안 보임 + 저장된 캘리브레이션도 없음. 보드를 보이게 하고 재촬영 |
| 화면에 `SCALE: held`/`fallback` 계속 표시 | 보드가 손/그림자에 가려짐. 조명/배치 조정, 그래도 안 되면 캘리브레이션 최신화 |
| 유형 판정이 실측과 계속 반대로 나옴 | `config.yaml`의 `nail_type.side_pb_invert` 또는 각종 threshold 조정 |

## 향후 로드맵: 고전 CV → AI 세그멘테이션

지금 `detection/contour.py`의 `ThresholdNailSegmenter`(CLAHE→적응형
이진화→모폴로지→최대 컨투어)는 조명/배경에 민감해서 수동 보정 UI가 사실상
필수다. 정확도를 높이기 위한 다음 단계는 다음과 같이 계획한다.

1. **데이터 수집**: `main.py collect`(또는 `measure --save-data`)로 실제
   사용 환경에서 정면/측면 원본 이미지, 자동(또는 수동 보정된) 마스크,
   키포인트를 계속 쌓는다.
2. **반자동 라벨링**: PC에서 Segment Anything(SAM)에 위 마스크/키포인트를
   프롬프트로 넣어 초기 세그멘테이션 후보를 만들고, 사람이 검수/수정해서
   정답(ground truth) 마스크 데이터셋을 구축한다.
3. **모델 학습**: 가벼운 세그멘테이션 모델(YOLOv8n-seg 또는 U-Net 계열)을
   위 데이터셋으로 학습한다. Jetson Nano에서 실시간으로 돌아갈 수 있는
   가벼운 백본을 우선한다.
4. **ONNX 변환 및 검증**: 학습된 모델을 ONNX로 변환해 `detection/dl_segmenter.py`의
   `OnnxNailSegmenter`를 채워 넣고, 노트북/PC의 ONNX Runtime으로 먼저 정확도를 검증한다.
5. **TensorRT 배포**: 검증된 ONNX 모델을 `trtexec --onnx=model.onnx --saveEngine=model.engine`로
   변환해 `TensorRTNailSegmenter`를 채우고, Jetson Nano에서 실시간 추론으로 교체한다.

`detection/base.py`의 `NailSegmenter` 인터페이스가 "이미지→마스크"만 담당하고
마스크→키포인트 로직(`extract_front_keypoints`/`extract_side_keypoints`)은
마스크 출처와 무관하게 공용이므로, 위 로드맵이 끝나면 `main.py`에서
세그멘터 구현체만 교체하면 된다(다운스트림 분석 코드는 수정 불필요).
