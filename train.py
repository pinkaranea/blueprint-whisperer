import csv
import json
import random
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

SIZE = 96
SEED = 42


class BlueprintMatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 64),
        )

    def forward(self, x):
        return self.encoder(x)


class PairDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        self.augment = augment

        path = self.root / f"pairs_{split}.csv"
        with path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError(f"No pairs in {path}")

        self.cache = {}
        unique_paths = set()

        for row in self.rows:
            unique_paths.add(row["img1"])
            unique_paths.add(row["img2"])

        for rel in sorted(unique_paths):
            path = self.root / rel.replace("\\", "/")
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise FileNotFoundError(path)

            if img.shape != (SIZE, SIZE):
                img = cv2.resize(img, (SIZE, SIZE))

            self.cache[rel] = img

        print(
            f"{split}: {len(self.rows)} pairs, "
            f"{len(self.cache)} unique patches",
            flush=True,
        )

    def __len__(self):
        return len(self.rows)

    def tensor(self, rel):
        img = self.cache[rel]

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img)
            if random.random() < 0.5:
                img = np.flipud(img)

        img = np.ascontiguousarray(img)
        ink = (255.0 - img.astype(np.float32)) / 255.0
        return torch.from_numpy(ink[None, :, :])

    def __getitem__(self, index):
        row = self.rows[index]
        a = self.tensor(row["img1"])
        b = self.tensor(row["img2"])
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return a, b, y


def similarity(model, a, b):
    za = F.normalize(model(a), dim=1)
    zb = F.normalize(model(b), dim=1)
    return (za * zb).sum(dim=1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores, labels = [], []

    for a, b, y in loader:
        a, b = a.to(device), b.to(device)
        sim = similarity(model, a, b)

        scores.extend(sim.cpu().numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    best = {"f1": -1, "threshold": 0.65,
            "precision": 0, "recall": 0}

    for threshold in np.linspace(-0.1, 0.99, 110):
        pred = scores >= threshold

        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(
            1e-9, precision + recall
        )

        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best


def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    print("DEVICE:", device, flush=True)

    train_ds = PairDataset(args.data, "train", augment=True)
    val_ds = PairDataset(args.data, "val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch,
        shuffle=False, num_workers=0
    )

    model = BlueprintMatchNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "blueprint_match.pt"

    best_f1 = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for a, b, y in train_loader:
            a = a.to(device)
            b = b.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            sim = similarity(model, a, b)
            logits = 12.0 * (sim - 0.55)
            loss = F.binary_cross_entropy_with_logits(
                logits, y
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            count += len(y)

        metrics = validate(model, val_loader, device)
        train_loss = total_loss / max(1, count)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"pair_F1={metrics['f1']:.3f} | "
            f"precision={metrics['precision']:.3f} | "
            f"recall={metrics['recall']:.3f}",
            flush=True,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                **metrics,
            }, checkpoint)
            print("Saved:", checkpoint, flush=True)

    saved = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )

    encoder = BlueprintMatchNet()
    encoder.load_state_dict(saved["model"])
    encoder.eval()

    metadata = {
        "model": "BlueprintMatchNet",
        "input_size": SIZE,
        "embedding_size": 64,
        "threshold": saved["threshold"],
        "validation_pair_f1": saved["f1"],
        "validation_pair_precision": saved["precision"],
        "validation_pair_recall": saved["recall"],
        "epoch": saved["epoch"],
        "note": "Pair classification metrics, not detection metrics",
    }

    (out / "blueprint_match.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    onnx_path = out / "blueprint_match.onnx"

    try:
        torch.onnx.export(
            encoder,
            torch.zeros(1, 1, SIZE, SIZE),
            str(onnx_path),
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={
                "input": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=13,
            dynamo=False,
        )
        print("ONNX:", onnx_path)

        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        net.setInput(
            np.zeros((2, 1, SIZE, SIZE), dtype=np.float32)
        )
        result = net.forward()
        print("ONNX OpenCV test:", result.shape)

    except Exception as exc:
        print("ONNX export or check failed:", exc)
        print("PyTorch checkpoint is still saved.")

    print("TRAINING FINISHED")
    print("Best validation pair F1:", round(best_f1, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data_stoa/patch_dataset_v5"
    )
    parser.add_argument(
        "--out", default="weights"
    )
    parser.add_argument(
        "--epochs", type=int, default=8
    )
    parser.add_argument(
        "--batch", type=int, default=128
    )
    parser.add_argument(
        "--lr", type=float, default=0.001
    )
    train(parser.parse_args())
