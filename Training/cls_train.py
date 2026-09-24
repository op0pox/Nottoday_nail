# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:39:54 2026

@author: user

nail_type_classification.py
손톱 정면/측면 2-view 이진분류 (P형 vs S형) + Stratified K-Fold 교차검증

폴더 구조:
    data/
      thumb/P/  *_front_nail.jpg  *_side_nail.jpg
      thumb/S/
      other/P/
      other/S/
    -> thumb(엄지) / other(그 외 손가락) 각각 따로 학습
    -> "<샘플ID>_front_nail.*" 와 "<샘플ID>_side_nail.*" 가 한 세트

실행:
    python cls_train.py
    # 상단 DATA_DIR 을 바꾸거나, 비워 두면 실행 중 경로를 입력받음
    # 앞단에서 train 80% / test 20% 로 나누고, 학습이 끝나면 hold-out 테스트까지 저장
    # 산출물: Training/cls_results/{YYYYMMDD_HHMMSS}/

필요 패키지: torch, torchvision(>=0.13), scikit-learn(>=1.0), pillow, numpy

===== 교차검증 요약 (fold 평균 ± 표준편차) =====
  acc : 0.814 ± 0.096
  f1  : 0.862 ± 0.081
  auc : 0.840 ± 0.094
결과 저장: cls_results/cv_results.json, cls_results/oof_predictions.csv
"""
import csv
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
CLASSES = ["P", "S"]  # label: P=0, S=1
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet 정규화값

# ----------------------------------------------------------------------------
# 설정 (CLI 대신 여기만 수정)
# ----------------------------------------------------------------------------
DATA_DIR = r"C:\Users\USER\Downloads\9-24_최종본"  # thumb/, other/ 아래에 P/, S/
FINGER_SPLITS = ("thumb", "other")  # 엄지 / 그 외 손가락을 따로 학습
FRONT_TAG = "_front_nail"
SIDE_TAG = "_side_nail"
GROUP_REGEX = None  # 예: r"^([^_]+)_"  지정 시 StratifiedGroupKFold
TEST_RATIO = 0.20  # 손가락 쌍 단위 hold-out. 남은 80%로 CV + 최종 학습
RUN_TEST_AFTER_TRAIN = True

MODE = "finetune"  # "linear" | "finetune"
BACKBONE = "resnet18"
NO_PRETRAINED = False
FREEZE = "partial"  # "all" | "partial" | "none"
K = 5
REPEATS = 3
EPOCHS = 30
BATCH_SIZE = 8
IMG_SIZE = 224
LR = 1e-3
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
C = 0.1  # linear 모드 로지스틱 회귀 규제 강도
WORKERS = 0
SEED = 42
OUT_DIR = str(SCRIPT_DIR / "cls_results")
TRAIN_FINAL = True

PREDICT_FRONT = None
PREDICT_SIDE = None
CHECKPOINT = None  # 예측 모드일 때 가중치 경로. None 이면 최신 cls_results/*/thumb/final_model.pt


# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_data_dir(path):
    data_dir = Path(path).expanduser()
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    else:
        data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {data_dir}")
    return str(data_dir)


def collect_samples(root, front_tag, side_tag, group_regex=None):
    """정면/측면 이미지를 한 세트로 묶어 샘플 목록 생성."""
    samples = []
    for label, cls in enumerate(CLASSES):
        d = Path(root) / cls
        if not d.is_dir():
            raise FileNotFoundError(f"클래스 폴더가 없습니다: {d}")
        files = {f.stem: f for f in d.iterdir() if f.suffix.lower() in IMG_EXT}
        for stem, fpath in sorted(files.items()):
            if not stem.endswith(front_tag):
                continue
            sid = stem[: -len(front_tag)]
            spath = files.get(sid + side_tag)
            if spath is None:
                print(f"[경고] 측면 이미지 없음 -> 제외: {fpath}")
                continue
            group = sid
            if group_regex:
                m = re.search(group_regex, sid)
                group = m.group(1) if m else sid
            samples.append(dict(id=f"{cls}/{sid}", front=str(fpath), side=str(spath),
                                label=label, group=group))
    if not samples:
        raise RuntimeError("샘플을 찾지 못했습니다. 폴더 구조/파일명 태그를 확인하세요.")
    labels = [s["label"] for s in samples]
    print(f"총 {len(samples)}세트  (P={labels.count(0)}, S={labels.count(1)}), "
          f"그룹 수={len(set(s['group'] for s in samples))}")
    return samples


def person_id_from_sample(sample):
    sid = sample["id"].split("/", 1)[-1]
    return sid.split("_")[0] if sid else sid


def split_bucket(pairs, test_ratio, rng):
    """손가락 쌍을 섞어서 test_ratio 만큼 hold-out. 같은 사람이 train/test에 같이 들어가도 됨."""
    items = list(pairs)
    rng.shuffle(items)
    n = len(items)
    if n <= 1:
        return items, []
    n_test = int(round(n * test_ratio))
    n_test = min(max(n_test, 1), n - 1)
    return items[n_test:], items[:n_test]


def split_train_test(samples, test_ratio, rng):
    by_cls = defaultdict(list)
    for sample in samples:
        by_cls[sample["label"]].append(sample)
    train, test = [], []
    for cls in sorted(by_cls):
        tr, te = split_bucket(by_cls[cls], test_ratio, rng)
        train.extend(tr)
        test.extend(te)
    return train, test


def write_split_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=[
            "finger", "usage", "id", "label", "person", "front", "side",
        ])
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------
# 전처리 / 데이터셋
# ----------------------------------------------------------------------------
class PadToSquare:
    """종횡비를 유지한 채 검은 배경으로 정사각형 패딩 (손톱 모양 왜곡 방지)."""

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        new = Image.new(img.mode, (s, s), self.fill)
        new.paste(img, ((s - w) // 2, (s - h) // 2))
        return new


def build_transforms(img_size, train):
    t = [PadToSquare(0), transforms.Resize((img_size, img_size))]
    if train:
        t += [
            transforms.RandomHorizontalFlip(),  # 좌/우 손 대칭 가정. 모양 기준이 비대칭이면 제거
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    t += [
        transforms.Grayscale(num_output_channels=3),  # 사전학습 모델은 3채널 입력
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    return transforms.Compose(t)


class NailPairDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        f = self.transform(Image.open(s["front"]).convert("L"))
        sd = self.transform(Image.open(s["side"]).convert("L"))
        return f, sd, torch.tensor(s["label"], dtype=torch.float32)


# ----------------------------------------------------------------------------
# 모델: 정면/측면이 백본을 공유(파라미터 절약) -> 특징 연결 -> 분류
# ----------------------------------------------------------------------------
def build_backbone(name, pretrained):
    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=w)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        partial_prefixes = ("layer4",)
    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=w)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        partial_prefixes = ("features.7", "features.8")
    else:
        raise ValueError(name)
    return net, dim, partial_prefixes


class TwoViewNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, dropout=0.3, freeze="partial"):
        super().__init__()
        self.backbone, dim, partial = build_backbone(backbone, pretrained)
        for n, p in self.backbone.named_parameters():
            if freeze == "all":
                p.requires_grad = False
            elif freeze == "partial":
                p.requires_grad = n.startswith(partial)
            else:  # "none"
                p.requires_grad = True
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim * 2, 1))

    def features(self, front, side):
        return torch.cat([self.backbone(front), self.backbone(side)], dim=1)

    def forward(self, front, side):
        return self.head(self.features(front, side)).squeeze(1)


def freeze_bn(model):
    """배치가 작으면 BN 통계가 불안정 -> 사전학습된 BN 통계를 고정."""
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ----------------------------------------------------------------------------
# 학습 / 평가
# ----------------------------------------------------------------------------
def train_model(train_samples, args, device):
    model = TwoViewNet(args.backbone, not args.no_pretrained, args.dropout, args.freeze).to(device)
    loader = DataLoader(NailPairDataset(train_samples, build_transforms(args.img_size, True)),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=len(train_samples) > args.batch_size)

    # 클래스 불균형 보정
    y = np.array([s["label"] for s in train_samples])
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32,
                              device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": model.head.parameters(), "lr": args.lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.lr * args.backbone_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        freeze_bn(model)
        tot, n = 0.0, 0
        for f, sd, lab in loader:
            f, sd, lab = f.to(device), sd.to(device), lab.to(device)
            opt.zero_grad()
            loss = criterion(model(f, sd), lab)
            loss.backward()
            opt.step()
            tot += loss.item() * lab.size(0)
            n += lab.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep + 1:3d}/{args.epochs}  train_loss={tot / max(n, 1):.4f}")
    return model


@torch.no_grad()
def predict_proba(model, samples, args, device):
    model.eval()
    loader = DataLoader(NailPairDataset(samples, build_transforms(args.img_size, False)),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    probs = []
    for f, sd, _ in loader:
        probs.append(torch.sigmoid(model(f.to(device), sd.to(device))).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def extract_features(samples, args, device):
    """linear 모드: 고정된 사전학습 백본으로 특징 추출 (라벨을 쓰지 않으므로 누수 없음)."""
    model = TwoViewNet(args.backbone, not args.no_pretrained, 0.0, "all").to(device).eval()
    loader = DataLoader(NailPairDataset(samples, build_transforms(args.img_size, False)),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    feats = [model.features(f.to(device), sd.to(device)).cpu().numpy() for f, sd, _ in loader]
    return np.concatenate(feats)


def compute_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    m = dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0))
    m["auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return m


def get_splitter(args, seed, use_groups):
    if use_groups:
        return StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=args.k, shuffle=True, random_state=seed)


def run_cv(samples, args, device):
    y = np.array([s["label"] for s in samples])
    groups = np.array([s["group"] for s in samples])
    use_groups = args.group_regex is not None
    feats = extract_features(samples, args, device) if args.mode == "linear" else None

    fold_metrics, oof_rows = [], []
    for r in range(args.repeats):
        seed = args.seed + r
        splitter = get_splitter(args, seed, use_groups)
        oof = np.zeros(len(samples))
        split_iter = splitter.split(np.zeros(len(y)), y, groups if use_groups else None)
        for fold, (tr, te) in enumerate(split_iter):
            set_seed(seed * 100 + fold)
            print(f"[repeat {r + 1}/{args.repeats}] fold {fold + 1}/{args.k}  "
                  f"train={len(tr)}  test={len(te)}  (test S비율={y[te].mean():.2f})")
            if args.mode == "linear":
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=args.C, class_weight="balanced",
                                                       max_iter=5000))
                clf.fit(feats[tr], y[tr])
                p = clf.predict_proba(feats[te])[:, 1]
            else:
                model = train_model([samples[i] for i in tr], args, device)
                p = predict_proba(model, [samples[i] for i in te], args, device)
            oof[te] = p
            m = compute_metrics(y[te], p)
            m.update(repeat=r, fold=fold)
            fold_metrics.append(m)
            print(f"    -> acc={m['acc']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}")
        for i, s in enumerate(samples):
            oof_rows.append(dict(repeat=r, id=s["id"], label=s["label"], prob_S=oof[i]))
        om = compute_metrics(y, oof)
        print(f"[repeat {r + 1}] OOF 전체: acc={om['acc']:.3f} f1={om['f1']:.3f} auc={om['auc']:.3f}")
        print("    confusion matrix [행=실제 P,S / 열=예측 P,S]:\n",
              confusion_matrix(y, (oof >= 0.5).astype(int)))
    return fold_metrics, oof_rows


def summarize(fold_metrics):
    out = {}
    for k in ["acc", "f1", "auc"]:
        v = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)))
    return out


# ----------------------------------------------------------------------------
# 예측 모드
# ----------------------------------------------------------------------------
def predict_single(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    a = SimpleNamespace(**ckpt["args"])
    model = TwoViewNet(a.backbone, False, a.dropout, "none").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = build_transforms(a.img_size, False)
    f = tf(Image.open(args.predict[0]).convert("L")).unsqueeze(0).to(device)
    s = tf(Image.open(args.predict[1]).convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(f, s)).item()
    print(f"P(S형)={p:.3f}  ->  예측: {'S' if p >= 0.5 else 'P'}형")


# ----------------------------------------------------------------------------
def build_args():
    return SimpleNamespace(
        front_tag=FRONT_TAG,
        side_tag=SIDE_TAG,
        group_regex=GROUP_REGEX,
        mode=MODE,
        backbone=BACKBONE,
        no_pretrained=NO_PRETRAINED,
        freeze=FREEZE,
        k=K,
        repeats=REPEATS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        lr=LR,
        backbone_lr_mult=BACKBONE_LR_MULT,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        C=C,
        workers=WORKERS,
        seed=SEED,
        test_ratio=TEST_RATIO,
        out=OUT_DIR,
        train_final=TRAIN_FINAL,
        predict=None if PREDICT_FRONT is None else [PREDICT_FRONT, PREDICT_SIDE],
        checkpoint=CHECKPOINT,
    )


def run_split(split_name, samples, args, device, out_root):
    out = Path(out_root) / split_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n======== {split_name} ========")
    fold_metrics, oof_rows = run_cv(samples, args, device)
    summary = summarize(fold_metrics)
    print(f"\n===== [{split_name}] 교차검증 요약 (fold 평균 ± 표준편차) =====")
    for k, v in summary.items():
        print(f"  {k:4s}: {v['mean']:.3f} ± {v['std']:.3f}")

    args_dump = vars(args).copy()
    args_dump["split"] = split_name
    with open(out / "cv_results.json", "w", encoding="utf-8") as fp:
        json.dump(dict(args=args_dump, summary=summary, folds=fold_metrics), fp,
                  ensure_ascii=False, indent=2, default=str)
    with open(out / "oof_predictions.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["repeat", "id", "label", "prob_S"])
        w.writeheader()
        w.writerows(oof_rows)
    print(f"결과 저장: {out / 'cv_results.json'}, {out / 'oof_predictions.csv'}")

    if not TRAIN_FINAL:
        return
    if args.mode != "finetune":
        print("[안내] TRAIN_FINAL 은 finetune 모드에서만 모델을 저장합니다.")
        return
    print(f"\n[{split_name}] 학습 세트(hold-out 제외)로 최종 모델 학습...")
    set_seed(args.seed)
    model = train_model(samples, args, device)
    torch.save(dict(state_dict=model.state_dict(), args=args_dump), out / "final_model.pt")
    print(f"최종 모델 저장: {out / 'final_model.pt'}")


def main():
    args = build_args()
    device = get_device()
    print("device:", device)
    if PREDICT_FRONT and PREDICT_SIDE:
        args.predict = [PREDICT_FRONT, PREDICT_SIDE]
        if not args.checkpoint:
            raise ValueError("예측 모드에는 CHECKPOINT 경로가 필요합니다.")
        predict_single(args, device)
        return

    data_dir = DATA_DIR
    if not data_dir:
        data_dir = input("데이터 폴더 경로: ").strip().strip('"')
    data_dir = resolve_data_dir(data_dir)
    print("data:", data_dir)

    set_seed(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    args.out = str(run_dir)
    print("run:", run_dir)

    rng = random.Random(SEED)
    split_rows = []
    for split_name in FINGER_SPLITS:
        split_dir = Path(data_dir) / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{split_name} 폴더가 없습니다: {split_dir}")
        samples = collect_samples(split_dir, args.front_tag, args.side_tag, args.group_regex)
        train_samples, test_samples = split_train_test(samples, TEST_RATIO, rng)
        print(f"  hold-out  train={len(train_samples)}  test={len(test_samples)}  "
              f"(test {len(test_samples) / max(len(samples), 1):.0%})")
        for usage, group in (("train", train_samples), ("test", test_samples)):
            for s in group:
                split_rows.append(dict(
                    finger=split_name,
                    usage=usage,
                    id=s["id"],
                    label=CLASSES[s["label"]],
                    person=person_id_from_sample(s),
                    front=s["front"],
                    side=s["side"],
                ))
        run_split(split_name, train_samples, args, device, run_dir)

    write_split_csv(run_dir / "split.csv", split_rows)
    meta = dict(
        created=stamp,
        data_dir=data_dir,
        test_ratio=TEST_RATIO,
        seed=SEED,
        fingers=list(FINGER_SPLITS),
        n_train=sum(1 for r in split_rows if r["usage"] == "train"),
        n_test=sum(1 for r in split_rows if r["usage"] == "test"),
    )
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    print(f"분할 기록: {run_dir / 'split.csv'}")

    if RUN_TEST_AFTER_TRAIN:
        import cls_test
        cls_test.evaluate_run(run_dir, device)


if __name__ == "__main__":
    main()
