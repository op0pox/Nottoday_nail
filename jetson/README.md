# jetson

Jetson Nano에서 CSI 두 대(측면 sensor 0, 정면 sensor 1)로 한 프레임씩 찍어 노트북 브라우저에 넘깁니다. 세그·측정은 노트북 FastAPI가 합니다.

이 보드에는 이 폴더만 받습니다.

```bash
git sparse-checkout init --cone
git sparse-checkout set jetson
```

USB로 노트북에 붙인 뒤 SSH는 `nvidia@192.168.55.1` 입니다.

```bash
python3 capture_server.py
```

- `GET http://192.168.55.1:8080/health`
- `GET http://192.168.55.1:8080/shot` → `{ "front": "<jpeg base64>", "side": "<jpeg base64>" }`
- `GET http://192.168.55.1:8080/preview` → 왼쪽 측면, 오른쪽 정면을 보여주는 MJPEG. 화면 가운데 Laplacian 분산이 `focus` 숫자로 찍힌다. 같은 장면을 두고 렌즈를 돌려 숫자가 가장 클 때가 초점이다. 시연 중 `/shot`이 느리면 이 탭은 닫는다.

전송 이미지는 IMX219 mode 2(1920x1080)를 30fps로 받습니다. 첫 프레임이 들어오면 터미널에 `준비완료`가 찍힙니다. 포트 8080이 막혀 있으면 열어 둡니다.
