# 손톱 세그멘테이션 모델 파인튜닝 가이드 (문서만, 코드 구현 없음)

`segmentation/dl_yolo.py`가 쓰는 공개 사전학습 모델
(`mnemic/nails_seg_yolov8`, YOLOv8s-seg)은 우리 촬영 환경(수직 촬영,
ChArUco 보정판 배경, 29.5cm 고정 높이)으로 학습된 모델이 아니다.
`--conf`를 낮춰도 검출이 5개가 안 맞거나 마스크 경계가 부정확하면,
우리가 직접 찍은 사진으로 이 모델을 파인튜닝해서 정확도를 높일 수 있다.

이 문서는 **절차 설명만** 담고 있다. 실제 파인튜닝 코드/스크립트는
이번 버전에 포함하지 않았다 — 라벨링 데이터가 충분히 쌓인 뒤 필요할 때
별도로 구현한다.

## 언제 파인튜닝을 고려해야 하는가

- `measure_auto.py --backend yolo`를 여러 사진에 돌려봐도 계속
  5개가 아닌 개수가 검출된다 (conf를 0.05~0.5 사이로 다 바꿔봐도 마찬가지).
- 검출은 5개가 맞는데 마스크 경계가 손톱을 벗어나거나 너무 안쪽만
  잡아서, `manual`(수동 클릭) 백엔드와 오차가 계속 크게 난다
  (`python3 main.py --mode compare`로 확인).
- 촬영 환경(조명, 배경, 손 각도)이 고정적이라 "이 환경 전용"으로
  모델을 맞추는 게 이득인 경우.

## 절차 개요

1. **데이터 수집**: 다양한 조명/손/각도로 최소 수십~수백 장의
   ChArUco 보드 위 손 사진을 `data/captured/`에 모은다. 다양성이
   많을수록(사람, 손 크기, 매니큐어 유무 등) 일반화가 잘 된다.

2. **라벨링**: 아래 둘 중 하나로 손톱 영역을 폴리곤으로 라벨링한다.
   - [labelme](https://github.com/wkentaro/labelme): 로컬 GUI 도구,
     설치 후 `labelme data/captured/`로 실행해서 사진마다 손톱
     폴리곤을 그리고 저장한다.
   - [Roboflow](https://roboflow.com/): 웹 기반, 여러 명이 나눠서
     라벨링하기 편하고 증강(augmentation)/train-val 분할도 자동으로
     해준다. 내보낼 때 포맷을 "YOLOv8 (segmentation)"으로 선택한다.

3. **데이터셋 포맷 변환**: `ultralytics`가 요구하는 YOLO 세그멘테이션
   포맷(이미지 + 폴리곤 좌표가 담긴 `.txt` 라벨, `data.yaml`)으로
   맞춘다. labelme로 라벨링했다면 `labelme2yolo` 같은 변환 도구가
   필요하고, Roboflow는 내보내기 시 이미 이 포맷으로 나온다.

4. **초기 가중치로 공개 모델 사용**: 처음부터 학습하지 않고, 지금
   쓰고 있는 `models/nails_seg_s_yolov8_v1.pt`를 시작점으로 삼아
   우리 데이터로 이어서 학습한다(전이학습). 다음은 예시 명령이다
   (실제 실행 전 `ultralytics` 버전에 맞는 옵션인지 확인할 것).

   ```bash
   pip3 install ultralytics

   yolo segment train \
     data=path/to/data.yaml \
     model=models/nails_seg_s_yolov8_v1.pt \
     epochs=100 \
     imgsz=640 \
     project=training/runs \
     name=nail_finetune_v1
   ```

5. **평가**: 학습이 끝나면 `training/runs/nail_finetune_v1/weights/best.pt`
   가 생긴다. 이 가중치로 `measure_auto.py`를 돌려서 `manual` 백엔드
   대비 오차가 실제로 줄었는지 `python3 main.py --mode compare`로 확인한다.

6. **모델 교체**: 정확도가 만족스러우면 `best.pt`를
   `models/nails_seg_s_yolov8_v1.pt` 자리에 덮어쓰거나, 새 파일명으로
   두고 `measure_auto.py --backend yolo`에 새 경로를 쓸 수 있도록
   `segmentation/dl_yolo.py`의 `DEFAULT_MODEL_PATH`(또는
   `YoloNailBackend(model_path=...)` 인자)를 그 경로로 바꾼다.

## 주의사항

- 파인튜닝은 데이터가 충분하지 않으면 오히려 공개 모델보다 나빠질 수
  있다(과적합). 최소 수십 장, 가능하면 수백 장 이상을 목표로 한다.
- 학습/검증 데이터는 서로 다른 촬영 세션(다른 날, 다른 조명)에서
  나온 사진을 섞어야 실제 사용 환경에서의 정확도를 신뢰할 수 있다.
- Jetson Nano에서는 학습하지 않는다 (연산량이 너무 크다). 파인튜닝은
  노트북 또는 GPU가 있는 다른 환경에서 진행하고, 완성된 가중치
  파일(.pt)만 노트북/Nano의 `models/` 폴더로 가져온다.
