# 공부 가이드 — 이 프로젝트 코드 이해하기

seg_gui.py의 주석과 함께 읽으면 좋은 전체 그림 정리.

## 1. 큰 그림: 데이터가 흐르는 순서

```
사진 파일 (jpg/png)
   │  cv2.imread()
   ▼
numpy 배열 (H x W x 3, BGR 색순서)          ← OpenCV의 이미지 = 그냥 숫자 배열
   │  YOLO 세그멘테이션 (backends/dl_yolo.py)
   ▼
마스크들 (H x W 배열, 손톱=255 / 배경=0)     ← "손톱이 어디인지"를 픽셀로 표시
   │  체커보드 인식 (charuco_calibration.py)
   ▼
호모그래피 (3x3 행렬)                        ← "픽셀 좌표 → mm 좌표" 변환기
   │  측정 (backends/measure_from_mask.py)
   ▼
길이/폭 mm 값
   │  build_overlay() (seg_gui.py)
   ▼
결과 이미지 (윤곽선 + 측정선 + mm 텍스트)
```

## 2. 파일별 역할과 읽는 순서 (추천)

| 순서 | 파일 | 왜 이 순서인가 |
|---|---|---|
| ① | `backends/common.py` | 30줄짜리. `NailMask`(마스크 1개)와 백엔드 인터페이스 정의 — 모든 코드의 공통 언어 |
| ② | `backends/dl_yolo.py` | YOLO 모델로 마스크를 얻는 과정. `segment()` 함수가 핵심 |
| ③ | `backends/measure_from_mask.py` | 마스크에서 세로 끝점/가로 폭 찾기 + mm 변환 |
| ④ | `utils.py` | `measure_length_mm()` — 호모그래피로 두 점 사이 mm 거리 계산 |
| ⑤ | `charuco_calibration.py` | 체커보드 인식. 길지만 `calibrate()` 함수 위주로 |
| ⑥ | `seg_gui.py` | 위 부품들을 조립한 GUI. 1부(파이프라인)→2부(위젯)→3부(App) 순서로 |
| ⑦ | `backends/classical.py` | (선택) YOLO 없을 때의 규칙 기반 폴백. HSV 색공간 공부용으로 좋음 |

## 3. 꼭 이해해야 할 핵심 개념 5가지

### ① 이미지 = numpy 배열
- `image.shape` = (세로, 가로, 3). 각 픽셀은 [B, G, R] 숫자 3개 (0~255).
- OpenCV는 RGB가 아니라 **BGR 순서**다. `(0,0,255)`는 파랑이 아니라 **빨강**.

### ② 마스크(mask)란
- 원본과 같은 크기의 흑백 배열. 손톱인 픽셀만 255, 나머지는 0.
- "세그멘테이션한다" = 이 마스크를 만든다는 뜻.
- `np.nonzero(mask)` → 손톱 픽셀들의 좌표 목록 → 중심/최소/최대 계산에 사용.

### ③ 세그멘테이션 백엔드 구조 (다형성)
- `common.py`의 `SegmentationBackend`가 규격: "BGR 이미지 → NailMask 리스트".
- YOLO든 classical이든 같은 규격이라 GUI는 어느 쪽이 왔는지 몰라도 된다.
- 이렇게 하면 나중에 새 모델로 갈아껴도 GUI 코드는 안 바꿔도 됨.

### ④ 호모그래피 (mm 실측의 원리)
- ChArUco 보드는 "한 칸 10mm"라는 실제 크기를 이미 알고 있는 물체.
- 사진에서 보드의 코너들을 찾으면 "이 픽셀 = 보드 위 (x, y)mm 지점"이라는
  대응점이 수십 개 생긴다 → 이걸로 3x3 변환 행렬(호모그래피)을 푼다.
- 이후 손톱의 두 점을 이 행렬로 mm 좌표로 옮겨서 거리를 재면 실측값.
- RMS(재투영 오차) = 이 변환이 얼마나 정확한지. 0.5mm 이하면 양호.

### ⑤ GUI와 스레드
- tkinter는 `mainloop()`라는 무한 반복문이 클릭/그리기를 처리한다.
- YOLO 추론(수 초)을 메인 스레드에서 돌리면 그동안 창이 얼어버림
  → `threading.Thread`로 백그라운드에서 처리.
- 단, **위젯 수정은 메인 스레드만 가능** → `root.after(0, 함수)`로
  "메인 스레드에 예약"하는 패턴을 쓴다. (seg_gui.py `_set_status` 주석 참고)

## 4. 직접 해보면 좋은 실험

1. `DEFAULT_CONF`를 0.1 / 0.5로 바꿔서 검출 개수가 어떻게 변하는지 보기
2. `build_overlay()`의 색 `(0, 255, 0)`을 바꿔서 윤곽선 색 바꿔보기
3. `process_image()`에 `print(nail_masks[0].mask.shape)` 넣어서 마스크 크기 확인
4. 터미널에서 파이프라인만 실행해보기 (GUI 없이):
   ```python
   # segmentation 폴더에서 ./.venv/bin/python 실행 후
   import cv2, seg_gui
   img = cv2.imread("사진경로.jpg")
   backend, _ = seg_gui.load_backend(0.25)
   res = seg_gui.process_image(img, backend, use_charuco=True)
   print(res["count"], seg_gui.format_measure(res["measures"][0]))
   cv2.imwrite("out.png", res["overlay"])
   ```
5. `measurements.json`을 열어서 GUI 결과 카드와 값이 같은지 대조해보기

## 5. 자주 헷갈리는 것들

- **좌표 순서**: numpy는 `배열[y, x]` (세로 먼저), OpenCV 함수는 `(x, y)` (가로 먼저).
  `np.nonzero`가 `(ys, xs)` 순서로 주는 것도 이 때문.
- **이미지 복사**: `overlay = image.copy()` 없이 그리면 원본까지 바뀐다
  (numpy 배열은 대입해도 복사가 아니라 같은 데이터를 가리킴).
- **PhotoImage 참조**: tkinter 이미지는 파이썬 변수로 붙잡아두지 않으면
  가비지 컬렉션돼서 화면에서 사라진다 (`self._photos`에 저장하는 이유).
