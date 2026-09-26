# jetson

Jetson Nano에서 CSI 두 대(정면 sensor 0, 측면 sensor 1)로 한 프레임씩 찍어 노트북 브라우저에 넘깁니다. 세그·측정은 노트북 FastAPI가 합니다.

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

전송 이미지는 1280x720, 30fps입니다. 카메라 두 대를 동시에 열고, 첫 프레임이 들어오면 터미널에 `준비완료`가 찍힙니다. GStreamer `nvarguscamerasrc`입니다. 포트 8080이 막혀 있으면 열어 둡니다.
