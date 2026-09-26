# 정면·측면 그레이 크롭을 P/S 분류 모델에 넣는다.
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ("P", "S")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class _PadToSquare:
    def __call__(self, img):
        width, height = img.size
        side = max(width, height)
        canvas = Image.new(img.mode, (side, side), 0)
        canvas.paste(img, ((side - width) // 2, (side - height) // 2))
        return canvas


def _build_backbone(name):
    if name == "resnet18":
        net = models.resnet18(weights=None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim
    if name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, dim
    raise ValueError(name)


class TwoViewNet(nn.Module):
    def __init__(self, backbone="resnet18", dropout=0.3):
        super().__init__()
        self.backbone, dim = _build_backbone(backbone)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim * 2, 1))

    def forward(self, front, side):
        features = torch.cat([self.backbone(front), self.backbone(side)], dim=1)
        return self.head(features).squeeze(1)


class ShapeClassifier:
    def __init__(self, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("분류 모델이 없습니다: %s" % path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved = checkpoint.get("args") or {}
        self.img_size = int(saved.get("img_size", 224))
        self.model = TwoViewNet(saved.get("backbone", "resnet18"), float(saved.get("dropout", 0.3)))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            _PadToSquare(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def _tensor(self, gray):
        image = Image.fromarray(np.asarray(gray, dtype=np.uint8)).convert("L")
        return self.transform(image).unsqueeze(0)

    def predict(self, front_gray, side_gray):
        with torch.no_grad():
            prob_s = float(torch.sigmoid(self.model(self._tensor(front_gray), self._tensor(side_gray))).item())
        return CLASSES[1] if prob_s >= 0.5 else CLASSES[0]
