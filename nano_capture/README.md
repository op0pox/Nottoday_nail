# nano_capture — Jetson Nano 손톱 촬영 장치

Jetson Nano + CSI 카메라 2대(위/측면)를 장치에 고정해두고,
노트북에서 X11 포워딩으로 미리보기를 보면서 버튼 클릭으로
위/측면 사진 2장을 동시에 촬영·저장하는 도구 모음.

```
nano_capture/
├── dual_capture.py         # 카메라 2대 미리보기 + 버튼 클릭 촬영 (나노에서 실행)
├── distance_calibration.py # 고정 거리(cm) 입력 → 픽셀↔mm 보정 JSON 생성
└── README.md
```

## 하드웨어 구성

- Jetson Nano (JetPack 4.x)
- 라즈베리파이 카메라 v2 (IMX219) x 2 — CAM0/CAM1 포트
  - **v3(IMX708)는 나노에서 지원 안 됨** (JetPack 4.x에 드라이버 없음)
- 고정 거리: 위 카메라 10.5cm, 측면 카메라 7cm (변경 시 `distance_calibration.py`의 `CAMERA_PRESETS` 수정)
- 전원: 5V/4A 어댑터(배럴잭 + J48 점퍼) 권장
  - 마이크로 USB 전원으로 카메라 2대를 돌리면 전력 부족으로 나노가 꺼질 수 있음

주의: CSI 카메라는 **부팅할 때만 인식**된다. 카메라는 반드시 나노 전원을 끈 상태에서
연결하고, 연결 후 부팅해야 `/dev/video0`, `/dev/video1`이 생긴다.

## 노트북(맥) 사전 준비 — 최초 1회

```bash
# X11 서버 설치
brew install --cask xquartz     # 설치 후 로그아웃/로그인 필요

# (선택) 비밀번호 없이 접속하도록 SSH 키 등록
ssh-copy-id home@192.168.55.1
```

나노를 마이크로 5핀 USB 케이블로 노트북에 연결하면 `192.168.55.1`로 접속된다.

## 촬영 사용법

```bash
# 1. 노트북에서 X11 포워딩으로 나노 접속
ssh -Y -C home@192.168.55.1

# 2. 나노 터미널에서 실행
python3 dual_capture.py
```

노트북에 "Nail Capture" 창이 뜨면:

| 동작 | 방법 |
|---|---|
| 촬영 | 화면 아래 초록 **CAPTURE 버튼 클릭** 또는 스페이스/`c` 키 |
| 종료 | `q` 또는 ESC |

촬영하면 나노의 `~/captures/` 폴더에 1920x1080 사진 2장이 저장된다 (번호 자동 증가):

```
capture_0001_1_top.jpg    # 위 카메라
capture_0001_2_side.jpg   # 측면 카메라
```

자주 쓰는 옵션:

```bash
python3 dual_capture.py --preview-width 480   # 미리보기 끊기면 작게
python3 dual_capture.py --flip-top 2          # 위 카메라 180도 회전
python3 dual_capture.py --flip-side 2         # 측면 카메라 180도 회전
python3 dual_capture.py --outdir ~/my_photos  # 저장 폴더 변경
```

찍은 사진을 노트북으로 가져오려면 (노트북 터미널에서):

```bash
scp home@192.168.55.1:"~/captures/*.jpg" ./
```

## 픽셀↔mm 보정 (카메라 고정 후 1회)

카메라가 장치에 고정되어 있으므로 사진마다 체커보드를 찍을 필요 없이,
렌즈~대상 거리(cm)만 입력하면 보정 JSON이 생성된다.

```bash
python3 distance_calibration.py --camera top    # 위 카메라 (기본 10.5cm)
python3 distance_calibration.py --camera side   # 측면 카메라 (기본 7.0cm)

# 거리를 다시 잰 경우
python3 distance_calibration.py --camera top --distance-cm 11.2
```

결과는 `data/results/top_distance_calib.json` 형태로 저장되며, ChArUco 보정
JSON과 같은 형식(homography + camera_height_mm)이라 기존 측정 스크립트의
`--calib` 류 인자에 그대로 넣을 수 있다.

- 정확도는 보통 실제와 2~3% 이내. 더 정밀하게 하려면 모눈판을 한 번 찍어
  실측 배율을 구한 뒤 `--correction` 옵션으로 넣는다 (파일 상단 주석 [6] 참고).
- 거리 재는 기준: 렌즈 표면 ~ 손톱 표면. 받침면까지 쟀다면 측정 시
  `--nail-height`로 손톱 높이를 보정한다 (주석 [7] 참고).

## 문제 해결

| 증상 | 해결 |
|---|---|
| 창이 안 뜸 | 노트북에서 XQuartz 실행 확인, `ssh -Y`로 접속했는지 확인, 나노에서 `echo $DISPLAY`가 비어있지 않은지 확인 |
| 양쪽 다 NO SIGNAL | `ls /dev/video*`로 장치 확인. 없으면 전원 끄고 카메라 연결 후 재부팅 |
| 장치는 있는데 안 열림 (Argus 에러) | `sudo systemctl restart nvargus-daemon` 후 재시도 |
| 카메라 인식 실패 (dmesg에 i2c -121) | CSI 케이블 재장착: 걸쇠 완전히 올리고, 금속 접촉면이 방열판 쪽을 향하게 끝까지 삽입 |
| 나노가 갑자기 꺼짐 | 전원 부족. 5V/4A 어댑터(배럴잭 + J48 점퍼) 사용 |
