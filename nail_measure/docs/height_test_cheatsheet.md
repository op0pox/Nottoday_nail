# 높이별/카메라별 반복 테스트 치트시트

높이(10, 13.5, 15, 20, 25, 30cm...)나 카메라를 바꿔가며 같은 과정을
계속 반복할 때 쓰는 명령어 모음이다. 매번 이 순서(1~5)를 그대로
따라 하면 된다. `$IMG`, `$CALIB`는 매 촬영마다 실제 파일명으로
바꿔서 셸 변수에 담아두면 이후 명령어를 짧게 재사용할 수 있다.

```bash
cd ~/Nottoday_nail/nail_measure
```

## 촬영 1회 사이클 (높이/카메라 바꿀 때마다 반복)

```bash
# 1) 촬영 (CSI 카메라 기준. USB면 --camera usb)
python3 capture_image.py --camera csi
# 창에서 's' = 저장, 'q' = 종료. 터미널에 찍힌 [SAVED] 경로를 아래에 붙여넣기
IMG=data/captured/capture_YYYYMMDD_HHMMSS.jpg

# 2) 모눈 보정 (모눈 한 칸 5mm 기준 10칸. 손가락 근처 모눈을 클릭할 것)
python3 grid_calibration.py --image "$IMG" --cells 10
# 터미널에 찍힌 [SAVED] 경로를 아래에 붙여넣기
CALIB=data/results/YYYYMMDD_HHMMSS_calib.json

# 3) 손톱 5개 측정 (엄지->검지->중지->약지->소지, 각 뿌리->끝 클릭)
#    사진에 안 나온 손가락은 그 순서에서 'q'로 건너뛰면 된다
python3 manual_measure.py --image "$IMG" --calib "$CALIB"

# 4) 실제 자로 잰 값과 비교 (반드시 mm 단위! cm면 x10 해서 입력)
#    사진에 없어서 측정 안 한 손가락은 --actual에서 그냥 빼면 된다
python3 compare_actual.py --image "$IMG" \
  --actual thumb=<mm> index=<mm> middle=<mm> ring=<mm> pinky=<mm>

# 5) 촬영 조건 기록 (카메라 모델/높이/조명/초점은 매번 그 회차 값으로)
python3 height_experiment.py log --image "$IMG" \
  --camera-model "Raspberry Pi Camera Module 2" \
  --height 13.5 \
  --lighting "책상 스탠드" \
  --focus "auto" \
  --note "1차 테스트"
```

`--height`, `--lighting`, `--focus`, `--note` 값은 그 회차 촬영 상황에 맞게
매번 바꿔서 입력한다. `--camera-model`은 카메라를 바꾸기 전까지는 계속
같은 문자열로 통일해야 나중에 `summary`에서 카메라별로 제대로 묶인다.

## 모든 높이/카메라 테스트가 끝난 뒤: 높이별 평균 오차 요약

```bash
# 특정 카메라 모델 기준으로 높이별 평균 오차 + 최적 높이 출력
python3 height_experiment.py summary --camera-model "Raspberry Pi Camera Module 2"

# 카메라를 여러 개 테스트했다면 모델별로 각각 실행
python3 height_experiment.py summary --camera-model "Arducam 16MP AF"

# 전체(모든 카메라 통틀어) 요약
python3 height_experiment.py summary
```

## 참고

- 카메라를 바꾸면 그 뒤로는 `--camera-model` 값만 새 카메라 이름으로
  바꿔서 위 1~5번을 그대로 반복하면 된다.
- 매 촬영마다 모눈판/손 위치가 조금씩 달라질 수 있으므로, 2번(모눈 보정)은
  사진마다 매번 새로 해야 한다 (이전 calib.json을 재사용하지 말 것).
- 결과 파일 위치: `data/results/measurements.csv`,
  `data/results/comparison.csv`, `data/results/experiment_log.csv`
