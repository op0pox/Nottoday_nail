# -*- coding: utf-8 -*-
"""
cls_test.py
cls_train.py 에서 저장한 thumb / other 2-view 분류 모델을 평가하고
정확도·혼동행렬·샘플별 확률을 CSV와 비교 이미지로 남긴다.

폴더 구조는 학습과 동일:
    data/
      thumb/P/  *_front_nail.jpg  *_side_nail.jpg
      thumb/S/
      other/P/
      other/S/

실행:
    python cls_test.py
    # RUN_DIR 이 비어 있으면 Training/cls_results 에서 split.csv 있는 최신 폴더를 사용
    # cls_train.py 가 끝나면 같은 함수를 자동 호출한다
"""
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cls_train import (  # noqa: E402
    CLASSES,
    FINGER_SPLITS,
    NailPairDataset,
    TwoViewNet,
    build_transforms,
    get_device,
)

# ----------------------------------------------------------------------------
# 설정 (CLI 대신 여기만 수정)
# ----------------------------------------------------------------------------
RUN_DIR = None  # 예: SCRIPT_DIR / "cls_results" / "20260925_021500"  / None 이면 최신 실험
BATCH_SIZE = 16
THRESH = 0.5  # P(S) >= 이 값이면 S

CELL_W = 560
CELL_H = 280
GRID_COLS = 4
MAX_GRID_ROWS = 8
MAX_JPEG_DIM = 65000
FONT_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")


def load_font(size):
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def open_bgr_as_rgb(path):
    img = Image.open(path).convert("RGB")
    return img


def letterbox_rgb(img, cell_w, cell_h, fill=(255, 255, 255)):
    w, h = img.size
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), fill)
    canvas.paste(resized, ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
    return canvas


def draw_text(draw, xy, text, font, fill, outline=(0, 0, 0)):
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text(xy, text, font=font, fill=fill)


def load_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    a = SimpleNamespace(**ckpt["args"])
    model = TwoViewNet(a.backbone, False, a.dropout, "none").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, a


@torch.no_grad()
def predict_probs(model, samples, img_size, device):
    tf = build_transforms(img_size, False)
    loader = DataLoader(
        NailPairDataset(samples, tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    probs = []
    for front, side, _ in loader:
        logits = model(front.to(device), side.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y, p, thr=THRESH):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(
        n=int(len(y)),
        n_P=int((y == 0).sum()),
        n_S=int((y == 1).sum()),
        acc=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        specificity=float(tn / max(tn + fp, 1)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out, pred


def load_cv_summary(run_dir, split_name):
    path = Path(run_dir) / split_name / "cv_results.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    return data.get("summary")


def latest_run_dir(runs_root=None):
    root = Path(runs_root or (SCRIPT_DIR / "cls_results"))
    if not root.is_dir():
        raise FileNotFoundError(f"cls_results 폴더가 없습니다: {root}")
    cands = [p for p in root.iterdir() if p.is_dir() and (p / "split.csv").is_file()]
    if not cands:
        raise FileNotFoundError(f"split.csv 가 있는 실험 폴더가 없습니다: {root}")
    return max(cands, key=lambda p: p.name)


def samples_from_split_csv(run_dir, finger, usage="test"):
    path = Path(run_dir) / "split.csv"
    if not path.is_file():
        raise FileNotFoundError(f"분할 기록이 없습니다: {path}")
    samples = []
    with open(path, encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["finger"] != finger or row["usage"] != usage:
                continue
            label = CLASSES.index(row["label"]) if row["label"] in CLASSES else int(row["label"])
            samples.append(dict(
                id=row["id"],
                front=row["front"],
                side=row["side"],
                label=label,
                group=row.get("person", row["id"]),
            ))
    return samples


def make_pair_cell(sample, pred_name, prob_s, ok):
    front = open_bgr_as_rgb(sample["front"])
    side = open_bgr_as_rgb(sample["side"])
    half_w = CELL_W // 2
    img_h = CELL_H - 64
    left = letterbox_rgb(front, half_w, img_h)
    right = letterbox_rgb(side, half_w, img_h)
    cell = Image.new("RGB", (CELL_W, CELL_H), (255, 255, 255))
    cell.paste(left, (0, 0))
    cell.paste(right, (half_w, 0))
    draw = ImageDraw.Draw(cell)
    draw.line([(half_w, 0), (half_w, img_h)], fill=(210, 210, 210), width=2)
    border = (34, 177, 76) if ok else (237, 28, 36)
    draw.rectangle([(1, 1), (CELL_W - 2, CELL_H - 2)], outline=border, width=4)

    sid = sample["id"]
    gt_name = CLASSES[sample["label"]]
    color = (34, 177, 76) if ok else (237, 28, 36)
    font = load_font(16)
    small = load_font(14)
    draw_text(draw, (10, img_h + 6), f"{sid}", font, (30, 30, 30), (255, 255, 255))
    mark = "OK" if ok else "WRONG"
    draw_text(
        draw,
        (10, img_h + 32),
        f"GT {gt_name}  Pred {pred_name}  P(S)={prob_s:.3f}  {mark}",
        small,
        color,
        (255, 255, 255),
    )
    draw_text(draw, (12, 8), "front", small, (80, 80, 80), (255, 255, 255))
    draw_text(draw, (half_w + 12, 8), "side", small, (80, 80, 80), (255, 255, 255))
    return cell


def build_grid(cells, cols=GRID_COLS):
    if not cells:
        canvas = Image.new("RGB", (CELL_W, CELL_H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw_text(draw, (20, 120), "표시할 샘플 없음", load_font(22), (0, 128, 0))
        return canvas
    rows = math.ceil(len(cells) / cols)
    canvas = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (255, 255, 255))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        canvas.paste(cell, (c * CELL_W, r * CELL_H))
    return canvas


def build_header(split_name, metrics, cv_summary, n_wrong, width):
    h = 220
    canvas = Image.new("RGB", (width, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title = load_font(28)
    body = load_font(18)
    draw_text(draw, (24, 16), f"P/S 2-view 분류 테스트  [{split_name}]", title, (20, 20, 20))
    lines = [
        f"샘플 {metrics['n']}세트  (P={metrics['n_P']}, S={metrics['n_S']})   "
        f"threshold={THRESH}   오분류 {n_wrong}개",
        f"acc={metrics['acc']:.3f}   f1={metrics['f1']:.3f}   auc={metrics['auc']:.3f}   "
        f"precision={metrics['precision']:.3f}   recall={metrics['recall']:.3f}   "
        f"specificity={metrics['specificity']:.3f}",
        f"혼동행렬  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}  TP={metrics['tp']}"
        f"   (행=실제 P/S, 열=예측 P/S)",
        "초록 테두리=정답 / 빨강 테두리=오분류   왼쪽=정면  오른쪽=측면",
    ]
    if cv_summary:
        lines.append(
            f"참고: 학습 시 K-fold CV  acc={cv_summary['acc']['mean']:.3f}±{cv_summary['acc']['std']:.3f}  "
            f"f1={cv_summary['f1']['mean']:.3f}±{cv_summary['f1']['std']:.3f}  "
            f"auc={cv_summary['auc']['mean']:.3f}±{cv_summary['auc']['std']:.3f}"
        )
    lines.append("이 점수는 학습에 넣지 않은 hold-out 테스트셋 기준")
    y = 58
    for line in lines:
        draw_text(draw, (24, y), line, body, (40, 40, 40))
        y += 24
    draw.line([(0, h - 3), (width, h - 3)], fill=(0, 0, 0), width=3)
    return canvas


def build_table_image(rows, width):
    """샘플별 GT/Pred/확률 표. 오분류를 위에 둔다."""
    font = load_font(16)
    header_h, row_h = 44, 28
    max_rows = min(len(rows), 40)
    show = rows[:max_rows]
    h = header_h + row_h * (len(show) + 1) + 16
    canvas = Image.new("RGB", (width, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cols = [
        (16, "id"),
        (int(width * 0.42), "GT"),
        (int(width * 0.52), "Pred"),
        (int(width * 0.64), "P(S)"),
        (int(width * 0.78), "결과"),
    ]
    draw.rectangle([(0, 0), (width, header_h)], fill=(240, 240, 240))
    for x, name in cols:
        draw_text(draw, (x, 12), name, font, (0, 0, 0))
    draw.line([(0, header_h), (width, header_h)], fill=(0, 0, 0), width=2)
    for i, r in enumerate(show):
        y = header_h + i * row_h
        color = (34, 177, 76) if r["correct"] else (237, 28, 36)
        vals = [
            r["id"],
            r["gt"],
            r["pred"],
            f"{r['prob_S']:.3f}",
            "OK" if r["correct"] else "WRONG",
        ]
        for (x, _), val in zip(cols, vals):
            draw_text(draw, (x, y + 6), str(val), font, color)
        draw.line([(0, y + row_h), (width, y + row_h)], fill=(230, 230, 230), width=1)
    extra = len(rows) - len(show)
    if extra > 0:
        y = header_h + len(show) * row_h
        draw_text(draw, (16, y + 6), f"... 나머지 {extra}행은 CSV 참고", font, (80, 80, 80))
    return canvas


def save_jpeg(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    w, h = rgb.size
    scale = min(MAX_JPEG_DIM / max(w, 1), MAX_JPEG_DIM / max(h, 1), 1.0)
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    rgb.save(path, quality=92)


def evaluate_split(split_name, samples, ckpt_path, device, out_dir, cv_summary=None):
    if not samples:
        print(f"[{split_name}] hold-out 샘플이 없습니다.")
        return [], None, cv_summary, []
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"가중치가 없습니다: {ckpt_path}")
    model, args = load_model(ckpt_path, device)
    probs = predict_probs(model, samples, args.img_size, device)
    y = np.array([s["label"] for s in samples])
    metrics, pred = compute_metrics(y, probs)

    print(f"\n======== {split_name} hold-out ========")
    print(f"checkpoint: {ckpt_path}")
    print(f"n={metrics['n']}  P={metrics['n_P']}  S={metrics['n_S']}")
    print(f"acc={metrics['acc']:.3f}  f1={metrics['f1']:.3f}  auc={metrics['auc']:.3f}")
    print(f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"specificity={metrics['specificity']:.3f}")
    print("confusion [행=실제 P,S / 열=예측 P,S]")
    print(np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]))

    rows = []
    cells = []
    for s, pred_i, prob in zip(samples, pred, probs):
        gt_name = CLASSES[s["label"]]
        pred_name = CLASSES[int(pred_i)]
        ok = int(pred_i) == s["label"]
        rows.append(dict(
            split=split_name,
            id=s["id"],
            gt=gt_name,
            pred=pred_name,
            prob_S=float(prob),
            correct=int(ok),
            front=s["front"],
            side=s["side"],
        ))
        cells.append((not ok, make_pair_cell(s, pred_name, float(prob), ok)))
    cells.sort(key=lambda x: x[0], reverse=True)
    cells = [c for _, c in cells]
    n_wrong = sum(1 for r in rows if not r["correct"])
    rows_sorted = sorted(rows, key=lambda r: (r["correct"], -abs(r["prob_S"] - 0.5)))

    width = CELL_W * GRID_COLS
    header = build_header(split_name, metrics, cv_summary, n_wrong, width)
    table = build_table_image(rows_sorted, width)
    max_grid_h = MAX_JPEG_DIM - header.size[1] - table.size[1] - 8
    rows_per_page = min(MAX_GRID_ROWS, max(1, max_grid_h // CELL_H))
    cells_per_page = rows_per_page * GRID_COLS

    image_paths = []
    for page, start in enumerate(range(0, len(cells), cells_per_page), start=1):
        page_cells = cells[start:start + cells_per_page]
        grid = build_grid(page_cells)
        out_img = Image.new(
            "RGB",
            (width, header.size[1] + grid.size[1] + table.size[1]),
            (255, 255, 255),
        )
        y = 0
        out_img.paste(header, (0, y)); y += header.size[1]
        out_img.paste(grid, (0, y)); y += grid.size[1]
        out_img.paste(table, (0, y))
        path = out_dir / f"{split_name}_p{page}.jpg"
        save_jpeg(path, out_img)
        image_paths.append(path)
        print(f"비교 이미지: {path}")

    return rows, metrics, cv_summary, image_paths


def evaluate_run(run_dir, device=None):
    run_dir = Path(run_dir)
    device = device or get_device()
    out_dir = run_dir / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("run:", run_dir)
    print("hold-out 결과:", out_dir)

    all_rows = []
    metric_rows = []
    for split_name in FINGER_SPLITS:
        samples = samples_from_split_csv(run_dir, split_name, "test")
        ckpt_path = run_dir / split_name / "final_model.pt"
        cv_summary = load_cv_summary(run_dir, split_name)
        rows, metrics, cv_summary, _ = evaluate_split(
            split_name, samples, ckpt_path, device, out_dir, cv_summary
        )
        if metrics is None:
            continue
        all_rows.extend(rows)
        rec = dict(split=split_name, **metrics)
        if cv_summary:
            rec["cv_acc"] = cv_summary["acc"]["mean"]
            rec["cv_f1"] = cv_summary["f1"]["mean"]
            rec["cv_auc"] = cv_summary["auc"]["mean"]
        else:
            rec["cv_acc"] = rec["cv_f1"] = rec["cv_auc"] = ""
        metric_rows.append(rec)

    pred_csv = out_dir / "predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["split", "id", "gt", "pred", "prob_S", "correct", "front", "side"])
        w.writeheader()
        w.writerows(all_rows)

    metrics_csv = out_dir / "metrics.csv"
    fields = [
        "split", "n", "n_P", "n_S", "acc", "f1", "auc", "precision", "recall",
        "specificity", "tn", "fp", "fn", "tp", "cv_acc", "cv_f1", "cv_auc",
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(metric_rows)

    print(f"\n샘플 CSV: {pred_csv}")
    print(f"지표 CSV: {metrics_csv}")
    return out_dir


def main():
    device = get_device()
    run_dir = Path(RUN_DIR) if RUN_DIR else latest_run_dir()
    print("device:", device)
    evaluate_run(run_dir, device)


if __name__ == "__main__":
    main()
