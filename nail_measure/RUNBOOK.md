# RUNBOOK

혼자 실행할 때 참고하는 전체 워크플로 정리. 각 단계를 어느 기기(젯슨 나노 vs 노트북)에서, GUI가 필요한지 여부와 함께 정리했다.

## 0. 사전 준비 (이미 되어 있음, 참고용)

- 노트북 ↔ 젯슨 나노: SSH 키 인증 설정됨 (`ssh comso@172.30.1.67`, 비밀번호 불필요)
- sshfs 마운트: 젯슨 홈 디렉토리가 노트북의 `/home/xmfos/jetson/`에 마운트되어 있음
  - `systemd --user` 서비스(`jetson-mount.service`)로 등록되어 있어 노트북 재부팅 시 자동 마운트됨
  - 상태 확인: `systemctl --user status jetson-mount.service`
- 젯슨 카메라: CSI 카메라(IMX219, Raspberry Pi Camera Module 2 계열)로 설정되어 있음 (`jetson-io`로 전환 완료)
- 젯슨 aruco 전용 가상환경: `~/aruco_venv` (numpy는 시스템 것 그대로 사용, `opencv-contrib-python`만 pip 설치, `OPENBLAS_CORETYPE=ARMV8`가 activate 스크립트에 자동 포함되어 있어 Jetson Nano의 구형 ARM 코어에서도 죽지 않음)
- 노트북 YOLO 패키지: `numpy`, `opencv-contrib-python`, `ultralytics`를 시스템 파이썬에 `pip3 install --user --break-system-packages`로 설치함 (venv 없이 사용자 계정에 바로 설치)

## 1. 촬영 (젯슨 나노, GUI 필요)

사진 창이 뜨고 키 입력을 받아야 하므로, 반드시 **본인 데스크톱의 실제 터미널**에서 `-X` 옵션으로 접속해서 실행한다 (헤드리스 SSH만으로는 안 됨).

```bash
ssh -X comso@172.30.1.67
cd ~/Nottoday_nail/nail_measure
python3 capture_image.py
```

- 화면이 뜨면 `s`로 저장, `q`로 종료
- 저장 위치: `data/captured/capture_YYYYMMDD_HHMMSS.jpg`

체크리스트:
- ChArUco 보드는 딱딱하고 평평한 곳(책상 등)에 놓기 — 상자 위처럼 휘어지는 곳 금지
- 손가락 5개(엄지 포함) 전부 프레임 안에, 잘리지 않게
- 가능하면 손을 프레임 중앙에 모아서 촬영 (렌즈 왜곡 영향 최소화)

## 2. ChArUco 보정 (젯슨 나노, GUI 불필요)

순수 SSH 접속만으로 실행 가능하다.

```bash
ssh comso@172.30.1.67
source ~/aruco_venv/bin/activate
cd ~/Nottoday_nail/nail_measure
python3 charuco_calibration.py --image data/captured/파일명.jpg
```

- 재투영 RMS가 0.5mm를 넘으면 경고가 뜬다 → 보드 평탄도/카메라 각도를 다시 확인
- 결과: `data/results/파일명_charuco.json`

## 3. YOLO 자동 측정 (노트북, GUI 불필요)

노트북에서 sshfs 마운트 경로로 직접 접근해서 실행한다 (scp로 옮길 필요 없음).

```bash
cd /home/xmfos/jetson/Nottoday_nail/nail_measure
python3 measure_auto.py \
  --image data/captured/파일명.jpg \
  --calib data/results/파일명_charuco.json \
  --hand right
```

- `--hand right` / `--hand left`: 촬영한 손 방향에 맞게 지정
- 출력: 손가락별 **길이(세로) + 폭(가로)** 둘 다 출력됨
- 결과는 `data/results/measurements.csv`에 누적 저장됨
- 디버그 이미지: `data/results/debug/파일명_yolo_debug.jpg`
  - 빨간 선 = 세로 길이(뿌리~끝)
  - 파란 선 = 가로 폭 (세로선의 중간 높이 기준)

## 결과 확인

- 사진: `/home/xmfos/jetson/Nottoday_nail/nail_measure/data/results/debug/` 폴더를 파일 탐색기로 바로 열어보면 됨
- 수치: `data/results/measurements.csv`

## 오늘(2026-07-11) 적용한 코드 수정 사항

`segmentation/measure_from_mask.py`, `measure_auto.py`, `utils.py`에 반영됨. 아직 git 커밋은 안 되어 있음.

1. **세로축 고정 측정**: 기존에는 PCA로 마스크의 "장축"을 찾아 뿌리/끝을 정했는데, 세그멘테이션 마스크가 손톱 일부만 잡는 경우가 많아 부정확했다. 지금은 5개 손가락 전부 마스크의 세로(y) 범위를 그대로 뿌리~끝으로 쓴다 (x좌표는 마스크 중심 x로 고정).
2. **가로 폭 측정 추가**: 세로선의 중간 y좌표에서 마스크의 좌우 끝을 찾아 폭을 잰다. `measure_nail_from_mask()`가 `width_mm` 등을 추가로 반환하고, CSV에도 `width_left_x/y`, `width_right_x/y`, `width_pixel_distance`, `width_mm` 컬럼이 추가됨.
3. **좌우 손 라벨링 버그 수정**: `label_fingers_by_x_order()`에서 `hand="right"`일 때 엄지/소지가 반대로 배정되던 버그를 고쳤다. 실측 비교로 검증함.

## 알려진 한계

- **가로 폭 ≠ 둘레**: `measure_auto.py`가 재는 "폭"은 위에서 내려다본 평면 사진 속 직선거리(현, chord)다. 실제로 종이 등을 감아서 잰 둘레(호, arc)와는 다른 값이며, 손가락이 통통하고 둥글수록(특히 엄지) 차이가 크게 난다. 이건 버그가 아니라 카메라 한 대로 위에서만 찍은 사진의 구조적 한계다.
- **약지는 다른 손가락보다 오차가 좀 더 큰 편**: YOLO 세그멘테이션 마스크가 약지에서 실제 손톱보다 약간 넓게 잡히는 경향이 있음. 원인 미해결.
- 이 프로젝트가 쓰는 YOLO 모델은 공개 사전학습 모델이라 이 프로젝트의 촬영 환경(다크 매니큐어, ChArUco 배경, 이 조명)에 최적화되어 있지 않다. 정확도를 더 높이려면 `training/finetune_guide.md`를 참고해 직접 파인튜닝이 필요하다.
