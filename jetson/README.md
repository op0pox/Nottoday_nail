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

센서와 전송 이미지는 IMX219 mode 4(1280x720, 60fps)입니다. GStreamer `nvarguscamerasrc`입니다. 포트 8080이 막혀 있으면 열어 둡니다.
