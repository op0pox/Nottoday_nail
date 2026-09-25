# Nottoday_nail

네일 팁 제작을 위한 손톱 데이터 수집·학습·측정·분류 프로젝트입니다.

촬영한 손톱 이미지로 세그멘테이션 모델을 학습하고, 웹/API에서 길이를 재거나 손톱 형태(P/S/B/C)를 비교합니다.

```text
Nottoday_nail/
├── Fast Api/              웹·API (측정 / 형태 분류)
├── Training/              YOLO 학습·가중치·데이터셋
├── utils/                 데이터 전처리·오토라벨·수집 현황
├── Latest_Data/           최신 원본 촬영 데이터
├── Latest_Data_label/     오토라벨 결과 (이미지 + LabelMe json)
├── nail_measure/          Jetson 촬영·실측 파이프라인
├── nail_analysis/         정면/측면 분석 + 웹 대시보드
├── curvature/             곡면 길이 계산 GUI
├── 손톱데이터셋목록.xlsx   사람별 실측·데이터 목록 (git 제외)
└── 손톱데이터셋목록.csv
```

---

## Fast Api/

측정 API와 손톱 형태 비교 프론트엔드입니다. Docker Compose로 API와 프론트를 같이 띄울 수 있습니다.

| 경로 | 역할 |
|---|---|
| `server/` | FastAPI 백엔드. `main.py`에서 라우터를 연결합니다. |
| `server/api/api.py` | 이미지 업로드 → ChArUco 보정 → YOLO 세그 → mm 측정 |
| `server/api/compare.py` | LabelMe json 업로드 → 카탈로그와 형태 비교 (P/S 등) |
| `server/segmentation/` | YOLOv8-seg, 윤곽을 `TARGET_POINTS`개로 균등 재샘플 |
| `server/classification/` | 컨투어/XOR 거리로 템플릿과 비교 |
| `front/` | React + Vite. 측정 화면, 분류 화면 |
| `nail_shape/` | 형태 비교용 손가락 crop / `shapes.json` |
| `docker-compose.yml` | API(8000) + 프론트. 학습 가중치는 `Training/Train_model`을 읽기 전용 마운트 |

실행 환경 변수는 `Fast Api/.env.example`을 참고합니다. 보드 칸 수, 칸 크기(mm), 카메라 높이 등이 들어갑니다.

---

## Training/

손톱 세그멘테이션·분류 학습과 실험 산출물입니다. 이미지 데이터 대부분은 `.gitignore`로 제외하고, **가중치 `weights/*.pt`와 학습·테스트 스크립트만** 커밋합니다.

| 경로 | 역할 |
|---|---|
| `seg_train.py` | HuggingFace 사전학습 모델로 YOLO 세그 학습. 기본 데이터는 `TrainDataset/YOLODataset_white` |
| `seg_test.py` | 두 세그 모델 예측을 GT와 나란히 비교해 `seg_results/`에 저장 |
| `cls_train.py` | 정면+측면 2-view로 P/S 분류. 데이터 폴더 하나만 넣으면 80/20을 나누고 thumb/other를 따로 학습한 뒤 테스트까지 저장 |
| `cls_test.py` | 방금 학습 폴더의 hold-out으로 지표 CSV와 비교 이미지를 다시 만듦. `cls_train.py`가 끝나면 자동으로 호출 |
| `Train_model/` | 실험별 가중치. 예: `nail_segmentation_33people/weights/best.pt` |
| `TrainDataset/` | YOLO yaml 학습셋 |
| `RawDataset/` | 체커보드 원본 풀 |
| `TestDataset/` | 테스트 분할 |
| `CropedDataset/`, `Crop+White/` | 크롭·흰 배경 전처리본 |
| `AutoLabel/` | 예전 오토라벨 출력 |
| `seg_results/` | `seg_test.py` 비교 이미지 |
| `cls_results/` | 분류 실험. `{YYYYMMDD_HHMMSS}/` 아래 가중치·분할·테스트 시각화 |

현재 오토라벨 기본 모델은 `Train_model/nail_segmentation_33people`입니다.

### 분류 실험 결과

`python Training/cls_train.py` 한 번이면 손가락 쌍 기준 80/20 분할, thumb/other 따로 학습, hold-out 테스트까지 이어집니다. 결과는 `Training/cls_results/{YYYYMMDD_HHMMSS}/`에 새로 생깁니다.

- `split.csv` — 어떤 손가락 쌍이 학습인지, 테스트인지
- `thumb/`, `other/` — `final_model.pt`, 교차검증 json/csv
- `test/` — 학습에 넣지 않은 데이터로 본 지표, 예측 CSV, 비교 이미지
- `20260925_full/` — 전체를 학습에 넣고 먼저 돌려 본 참고 모델

---

## utils/

촬영본을 라벨링하고, 이름을 고치고, 수집 현황을 보는 스크립트입니다. 배치 스크립트는 파일 상단의 `ORIGIN_*` / `TARGET_*` 경로를 바꾼 뒤 실행합니다.

| 파일 | 누가 쓰나 | 하는 일 |
|---|---|---|
| `AutoLabel.py` | 라벨 작업 | 하위 폴더까지 찾아 YOLO 세그 → LabelMe json. 윤곽은 둘레를 `TARGET_POINTS`개로 균등 배치 |
| `ReviewImage.py` | 라벨 확인 | 한 폴더의 이미지와 같은 이름 json을 보면서 회전·이름 변경 |
| `ConvertToJpg.py` | 정리 | 원본은 두고 `{폴더명}_to_jpg`에 jpg 복사. heic·확장자 없는 이미지도 변환 |
| `CheckData.py` | 수집 확인 | 사람별 데이터 유무 GUI. `data/dataset_original.csv` / `dataset_dev.csv` |
| `Crop_image.py` | 전처리 | 라벨 박스 기준 손톱 크롭 |
| `Trans_json.py` | 전처리 | LabelMe json의 `imagePath`와 해상도를 이미지에 맞춤 |
| `data/` | 수집 확인 | 순번, 이름, 체커보드 정면/측면, 흰색배경, 3D 유무 CSV |

### 오토라벨

- 입력: `Latest_Data/New체커보드/{사람}/`
- 출력: `Latest_Data_label/{사람}/` 이미지 + 같은 이름의 `.json`
- 원본 폴더는 수정하지 않습니다.

### 이미지와 json 확인

```powershell
python utils/ReviewImage.py
```

연 폴더 **바로 안**의 파일만 봅니다. 하위 폴더는 들어가지 않으므로, 사람 폴더를 직접 선택합니다. 같은 이름인 `사진.jpg`와 `사진.json`을 한 쌍으로 묶습니다. json은 LabelMe 형식입니다. `shapes[].points`가 다각형이고, `imageWidth` / `imageHeight`가 그 점의 기준 크기입니다.

화면에는 사진 위에 그 다각형과 라벨 글자를 같이 그립니다. LabelMe에서 보는 윤곽과 같은 위치인지 여기서 확인할 수 있습니다.

회전은 세 가지 중 하나를 고릅니다. 기본은 **둘 다**입니다.

| 선택 | 저장되는 것 |
|---|---|
| 둘 다 | 사진 픽셀과 json 점을 같은 방향으로 돌립니다. 가로·세로 크기도 json에 다시 적습니다. |
| 이미지만 | 사진만 돌립니다. 점은 그대로입니다. |
| json만 | 점과 json에 적힌 가로·세로만 바꿉니다. 사진 파일은 그대로입니다. |

왼쪽 90° / 오른쪽 90° / 180° 모두 위 선택을 따릅니다. 이름 변경은 사진과 json 파일명을 같이 바꾸고, json의 `imagePath`를 새 사진 이름으로 고칩니다.

저장은 그 폴더에 바로 덮어씁니다. 되돌리는 복사본은 만들지 않습니다. 짝이 없는 파일은 있는 쪽만 바꿉니다. 이미 켜 둔 LabelMe는 파일을 자동으로 다시 읽지 않으므로, 저장 뒤에 LabelMe에서 그 파일을 다시 열어야 합니다.

---

## Latest_Data/ · Latest_Data_label/

| 폴더 | 역할 |
|---|---|
| `Latest_Data/New체커보드/` | 최신 체커보드 원본. 사람 폴더 아래 정면·측면(`*_side`) jpg |
| `Latest_Data/흰색배경/` | 흰 배경 원본 (비어 있을 수 있음) |
| `Latest_Data_label/` | `AutoLabel.py` 결과. 사람 폴더 구조 유지, 이미지 옆에 json |

원본은 수정하지 않고, 라벨본만 따로 둡니다.

---

## nail_measure/

Jetson Nano + 카메라 실측 파이프라인입니다. ChArUco로 mm 호모그래피를 잡고, YOLO-seg 또는 수동 클릭으로 손톱 길이를 잽니다. 곡면 너비와 P/S/B/C 분류까지 포함합니다.

자세한 실행 순서는 [`nail_measure/README.md`](nail_measure/README.md), [`nail_measure/RUNBOOK.md`](nail_measure/RUNBOOK.md)를 봅니다.

---

## nail_analysis/

정면/측면 CSI 카메라로 손가락을 찍고, 바디 기장·손톱 유형·팁 사이즈를 계산합니다. 읽기 전용 웹 대시보드가 있습니다.

자세한 내용은 [`nail_analysis/README.md`](nail_analysis/README.md)를 봅니다.

---

## curvature/

정면 너비, 측면 너비, 유형(P/B/S/C)을 넣으면 팁 제작용 곡면 길이를 계산하는 데스크톱 GUI입니다.

자세한 내용은 [`curvature/README.md`](curvature/README.md)를 봅니다.

---

## 데이터 목록

`손톱데이터셋목록.xlsx` / `.csv`는 사람(순번·이름)별 체커보드 실측과 3D 스캔 실측입니다. Git에는 올리지 않습니다.

수집 여부만 보려면 `utils/CheckData.py`와 `utils/data/` CSV를 사용합니다.

---

## Git에서 빠지는 것

- `Training/` 안 학습 이미지·중간 산출물 (가중치 `.pt`는 예외)
- `Fast Api/nail_shape/` 중 `fingers/`, `shapes.json` 외
- `손톱데이터셋목록.xlsx`, `.csv`

---

## 실행 환경

- Python 3.9+ (conda 환경 권장)
- Fast Api 프론트는 Node + Vite
- 하위 폴더의 `requirements.txt`를 먼저 확인
- macOS에서 OpenMP 충돌이 나면 `KMP_DUPLICATE_LIB_OK=TRUE`를 앞에 붙여 실행
