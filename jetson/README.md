# jetson

Jetson Nano에서 CSI 두 대(측면 sensor 0, 정면 sensor 1)로 한 프레임씩 찍어 노트북 브라우저에 넘깁니다. 세그·측정은 노트북 FastAPI가 합니다.

이 보드에는 이 폴더만 받습니다.

```bash
git sparse-checkout init --cone
git sparse-checkout set jetson
```

USB로 노트북에 붙인 뒤 SSH는 `nvidia@192.168.55.1` 입니다.

```bash
python3 capture_server.py            # 서버 + HDMI 화면 창
python3 capture_server.py --no-gui   # 화면 없이 서버만
```

실행하면 HDMI 모니터에 사용자 화면 창(Tkinter, 전체화면)이 바로 뜬다. 브라우저는 필요 없다.
SSH 로 켜도 `DISPLAY` 가 없으면 `:0`(보드에 꽂힌 HDMI)으로 띄운다. 창을 못 열면 서버만 돈다.
처음 한 번 설치가 필요할 수 있다.

```bash
sudo apt-get install python3-tk fonts-noto-cjk
```

- 화면: 위에 측면/정면 촬영 사진 + 손톱 윤곽선, 아래에 측면 전처리 | 형태 | 정면 전처리 와 각 실측값.
- 관리자 페이지에서 촬영을 시작하면 결과가 올 때까지 '촬영중...' 만 보이고 재시도 사진은 안 보인다. 30초 안에 결과가 없으면 이전 화면으로 돌아간다.
- 키: `Esc` 전체화면 해제, `F11` 전체화면 전환, `Ctrl+Q` 종료.

- `GET http://192.168.55.1:8080/health`
- `GET http://192.168.55.1:8080/shot` → `{ "front": "<jpeg base64>", "side": "<jpeg base64>" }`
- `GET http://192.168.55.1:8080/preview` → 왼쪽 측면, 오른쪽 정면을 보여주는 MJPEG. 화면 가운데 Laplacian 분산이 `focus` 숫자로 찍힌다. 같은 장면을 두고 렌즈를 돌려 숫자가 가장 클 때가 초점이다. 시연 중 `/shot`이 느리면 이 탭은 닫는다.
- `POST /display` → 관리자 페이지가 측정 결과(전처리 이미지, 실측값, 형태, 윤곽선)를 보낸다. 화면 창이 이걸 그린다. `GET /display` 로 마지막 값을 볼 수 있다.

전송 이미지는 IMX219 mode 2(1920x1080)를 30fps로 받습니다. 첫 프레임이 들어오면 터미널에 `준비완료`가 찍힙니다. 포트 8080이 막혀 있으면 열어 둡니다.
