# Nottoday_nail

네일 팁 제작 자동화를 위한 손톱 촬영·측정·곡면 계산 프로젝트 모음.

## 하위 프로젝트

### [`nail_measure/`](nail_measure/README.md)
Jetson Nano + 카메라로 손톱 사진을 찍고, ChArUco 보정판을 자동 인식해서
이미지↔mm 호모그래피를 계산한 뒤 5개 손톱의 길이(뿌리~끝)를 mm 단위로
측정한다. ChArUco 자동 보정, YOLOv8-seg 자동 세그멘테이션(또는 수동 클릭),
곡면 너비 측정과 P/S/B/C 유형 자동 분류까지 포함한 가장 오래되고 성숙한
파이프라인.

### [`nail_analysis/`](nail_analysis/README.md)
Jetson Nano B01 듀얼 CSI 카메라(정면/측면)로 손가락을 한 개씩 촬영해서
바디 기장(Long/Medium/Short), 손톱 유형(P/S/B/C), 곡면 길이 기반 팁
사이즈를 자동으로 계산하는 프로그램. `nail_measure/`와는 목적·파이프라인이
달라 별도 프로젝트로 새로 시작했고, 촬영/결과를 브라우저에서 볼 수 있는
읽기 전용 웹 대시보드도 포함한다.

### [`curvature/`](curvature/README.md)
손톱의 정면 너비·측면 너비 실측값과 곡면 유형(P/B/S/C)을 입력하면
네일 팁 제작에 필요한 곡면 길이를 계산해주는 데스크톱 GUI 계산기.

### `capture.py`
Jetson/노트북에서 정지 이미지를 빠르게 캡처하기 위한 간단한 CLI 스크립트.

## 실행 환경 공통 참고

- Python 3.9+, miniconda 환경 권장
- macOS에서 OpenMP 충돌 오류가 나면 `KMP_DUPLICATE_LIB_OK=TRUE`를 앞에 붙여 실행
- 각 하위 프로젝트의 `requirements*.txt`와 README를 먼저 확인할 것
