import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Stala kolejnosc zgodna ze scan_v2.py.
CLASSES = ["window", "door", "sink", "toilet", "_background"]
TARGETS = ["door", "sink", "toilet"]
SIZE = 96
SEED = 42
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_files(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def original_names(folder):
    manifest = folder / "rename_map.csv"
    mapping = {}

    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                mapping[row["new_name"]] = row["original_name"]

    return mapping


def plan_group(filename):
    match = re.match(r"^(\d+)_", filename)
    return match.group(1) if match else None


def collect_data(root):
    items = []
    counts = Counter()
    unknown_groups = 0

    for cls in TARGETS:
        folder = root / cls
        files = image_files(folder)

        if len(files) < 5:
            raise RuntimeError(
                f"Za malo probek w {folder}: {len(files)}"
            )

        mapping = original_names(folder)

        for path in files:
            old_name = mapping.get(path.name, path.name)
            plan = plan_group(old_name)

            if plan is None:
                unknown_groups += 1

            items.append({
                "path": path,
                "class": cls,
                "group": (
                    f"plan:{plan}" if plan is not None
                    else f"file:{cls}:{path.name}"
                ),
            })
            counts[cls] += 1

    background = root / "_background"

    if not image_files(background):
        fallback = Path(
            "data_stoa/patch_dataset_v5/"
            "symbols/train/_background"
        )

        if image_files(fallback):
            background = fallback
            print("Background fallback:", fallback, flush=True)

    bg_files = image_files(background)

    if bg_files:
        rng = random.Random(SEED)
        rng.shuffle(bg_files)
        bg_files = bg_files[:150]
        mapping = original_names(background)

        for path in bg_files:
            old_name = mapping.get(path.name, path.name)
            plan = plan_group(old_name)

            items.append({
                "path": path,
                "class": "_background",
                "group": (
                    f"plan:{plan}" if plan is not None
                    else f"bg:{path.name}"
                ),
            })
            counts["_background"] += 1
    else:
        print(
            "WARNING: No background examples. "
            "False positives may remain.",
            flush=True
        )

    print("DATASET:", dict(counts), flush=True)

    if unknown_groups:
        print(
            f"WARNING: {unknown_groups} samples have no "
            "recoverable plan ID. Group leakage cannot "
            "be excluded for those images.",
            flush=True
        )

    return items


def split_data(items, fraction=0.20):
    groups = defaultdict(list)

    for item in items:
        groups[item["group"]].append(item)

    keys = list(groups.keys())
    rng = random.Random(SEED)
    classes = set(x["class"] for x in items)
    totals = Counter(x["class"] for x in items)
    target_groups = max(1, round(len(keys) * fraction))

    best = None
    best_score = float("inf")

    for _ in range(1000):
        chosen = set(rng.sample(keys, target_groups))

        val = [
            x for x in items if x["group"] in chosen
        ]
        train = [
            x for x in items if x["group"] not in chosen
        ]

        tr = Counter(x["class"] for x in train)
        va = Counter(x["class"] for x in val)

        if any(tr[c] < 2 or va[c] < 2 for c in classes):
            continue

        score = sum(
            abs(va[c] / totals[c] - fraction)
            for c in classes
        )

        if score < best_score:
            best = (train, val)
            best_score = score

    if best is None:
        raise RuntimeError(
            "Nie mozna wykonac bezpiecznego podzialu "
            "po planach. Potrzeba wiecej roznych planow "
            "w kazdej klasie."
        )

    train, val = best
    print(
        "TRAIN:", dict(Counter(x["class"] for x in train)),
        flush=True
    )
    print(
        "VAL:", dict(Counter(x["class"] for x in val)),
        flush=True
    )

    return train, val


def read_image(path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Cannot read: {path}")

    return cv2.resize(
        image, (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    )


def transform(image, augment=False):
    if augment:
        image = np.rot90(image, random.randrange(4))

        if random.random() < 0.5:
            image = np.fliplr(image)

        dx, dy = random.randint(-3, 3), random.randint(-3, 3)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        image = cv2.warpAffine(
            np.ascontiguousarray(image),
            matrix, (SIZE, SIZE),
            borderValue=255
        )

    image = np.ascontiguousarray(image)
    ink = (255.0 - image.astype(np.float32)) / 255.0

    return torch.from_numpy(ink[None])


class Triplets(Dataset):
    def __init__(self, items):
        self.items = items
        self.images = [
            read_image(item["path"]) for item in items
        ]

        self.by_class = defaultdict(list)

        for idx, item in enumerate(items):
            self.by_class[item["class"]].append(idx)

        self.anchors = [
            idx for idx, item in enumerate(items)
            if item["class"] in TARGETS
        ]

        counts = Counter(
            items[idx]["class"] for idx in self.anchors
        )

        self.weights = [
            1.0 / counts[items[idx]["class"]]
            for idx in self.anchors
        ]

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor_id = self.anchors[index]
        anchor = self.items[anchor_id]
        cls = anchor["class"]

        candidates = [
            idx for idx in self.by_class[cls]
            if self.items[idx]["group"] != anchor["group"]
        ]

        if not candidates:
            candidates = [
                idx for idx in self.by_class[cls]
                if idx != anchor_id
            ]

        positive_id = random.choice(candidates)

        choices = [
            c for c in TARGETS
            if c != cls and self.by_class[c]
        ]

        if (
            self.by_class["_background"]
            and random.random() < 0.5
        ):
            negative_class = "_background"
        else:
            negative_class = random.choice(choices)

        negative_id = random.choice(
            self.by_class[negative_class]
        )

        return (
            transform(self.images[anchor_id], True),
            transform(self.images[positive_id], True),
            transform(self.images[negative_id], True),
            CLASSES.index(cls),
            CLASSES.index(negative_class),
        )


class BlueprintMatchNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        blocks = []
        channels = [1, 16, 32, 64, 96]

        for i in range(4):
            blocks.extend([
                nn.Conv2d(
                    channels[i], channels[i+1],
                    3, padding=1, bias=False
                ),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])

        self.features = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )

        self.embedding = nn.Linear(96 * 3 * 3, 64)
        self.classifier = nn.Linear(64, len(CLASSES))

    def forward(self, x):
        features = self.features(x)

        embedding = F.normalize(
            self.embedding(features), dim=1
        )

        logits = self.classifier(embedding * 8.0)

        # 64D embedding + 5 logits, for scan_v2.py.
        return torch.cat([embedding, logits], dim=1)


def unpack(output):
    return output[:, :64], output[:, 64:]


@torch.no_grad()
def validate(model, items, device):
    model.eval()

    vectors = []
    predictions = []
    truth = []

    for start in range(0, len(items), 32):
        block = items[start:start+32]
        data = torch.stack([
            transform(read_image(x["path"]))
            for x in block
        ]).to(device)

        emb, logits = unpack(model(data))
        vectors.extend(emb.cpu().numpy())
        predictions.extend(
            logits.argmax(dim=1).cpu().tolist()
        )
        truth.extend(
            CLASSES.index(x["class"]) for x in block
        )

    vectors = np.asarray(vectors)
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)

    metrics = {}
    f1s = []

    for cls in sorted(set(x["class"] for x in items)):
        idx = CLASSES.index(cls)

        tp = int(np.sum(
            (predictions == idx) & (truth == idx)
        ))
        fp = int(np.sum(
            (predictions == idx) & (truth != idx)
        ))
        fn = int(np.sum(
            (predictions != idx) & (truth == idx)
        ))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        metrics[cls] = round(f1, 4)
        f1s.append(f1)

    # Fixed validation pairs, independent of training batches.
    rng = random.Random(SEED)
    similarities = []
    labels = []

    by_class = defaultdict(list)

    for i, item in enumerate(items):
        by_class[item["class"]].append(i)

    for i, item in enumerate(items):
        if item["class"] not in TARGETS:
            continue

        positive = [
            j for j in by_class[item["class"]]
            if j != i
        ]
        negative = [
            j for j, other in enumerate(items)
            if other["class"] != item["class"]
        ]

        if not positive or not negative:
            continue

        p = rng.choice(positive)
        n = rng.choice(negative)

        similarities.extend([
            float(np.dot(vectors[i], vectors[p])),
            float(np.dot(vectors[i], vectors[n]))
        ])
        labels.extend([1, 0])

    best_pair_f1 = 0.0
    best_threshold = 0.75

    for threshold in np.linspace(0.3, 0.99, 70):
        pred = np.asarray(similarities) >= threshold
        y = np.asarray(labels) == 1

        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        fn = int(np.sum(~pred & y))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        if f1 > best_pair_f1:
            best_pair_f1 = f1
            best_threshold = float(threshold)

    return {
        "macro_f1": round(float(np.mean(f1s)), 4),
        "pair_f1": round(best_pair_f1, 4),
        "pair_threshold": round(best_threshold, 4),
        "per_class_f1": metrics,
    }


def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cpu":
        torch.set_num_threads(
            min(6, torch.get_num_threads())
        )

    print("DEVICE:", device, flush=True)

    items = collect_data(Path(args.data))
    train_items, val_items = split_data(items)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "group", "split"])

        for split_name, group in (
            ("train", train_items),
            ("val", val_items)
        ):
            for item in group:
                writer.writerow([
                    str(item["path"]),
                    item["class"],
                    item["group"],
                    split_name
                ])

    dataset = Triplets(train_items)

    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=max(len(dataset), 256),
        replacement=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=0
    )

    model = BlueprintMatchNetV2().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0

        for anchor, positive, negative, ya, yn in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            ya = ya.to(device)
            yn = yn.to(device)

            optimizer.zero_grad()

            za, ca = unpack(model(anchor))
            zp, cp = unpack(model(positive))
            zn, cn = unpack(model(negative))

            classification = (
                F.cross_entropy(ca, ya)
                + F.cross_entropy(cp, ya)
                + F.cross_entropy(cn, yn)
            ) / 3.0

            triplet = F.triplet_margin_loss(
                za, zp, zn, margin=0.3
            )

            positive_similarity = (za * zp).sum(dim=1)
            negative_similarity = (za * zn).sum(dim=1)

            separation = F.relu(
                0.2
                + negative_similarity
                - positive_similarity
            ).mean()

            loss = (
                classification
                + 1.0 * triplet
                + 0.4 * separation
            )

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 3.0
            )
            optimizer.step()

            total_loss += loss.item() * len(ya)
            samples += len(ya)

        metrics = validate(model, val_items, device)

        score = (
            0.7 * metrics["macro_f1"]
            + 0.3 * metrics["pair_f1"]
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={total_loss/max(1,samples):.4f} | "
            f"macro_F1={metrics['macro_f1']:.3f} | "
            f"pair_F1={metrics['pair_f1']:.3f}",
            flush=True
        )
        print(
            "Classes:", metrics["per_class_f1"],
            flush=True
        )

        if score > best_score:
            best_score = score
            torch.save(
                model.cpu().state_dict(),
                out / "blueprint_match_clean.pt"
            )
            model.to(device)
            print("BEST CHECKPOINT SAVED", flush=True)

    # Export best checkpoint, not merely the last epoch.
    final_model = BlueprintMatchNetV2()
    final_model.load_state_dict(
        torch.load(
            out / "blueprint_match_clean.pt",
            map_location="cpu",
            weights_only=True
        )
    )
    final_model.eval()

    onnx_file = out / "blueprint_match_clean.onnx"

    torch.onnx.export(
        final_model,
        torch.zeros(2, 1, SIZE, SIZE),
        str(onnx_file),
        input_names=["input"],
        output_names=["features_and_logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "features_and_logits": {0: "batch"}
        },
        opset_version=13,
        dynamo=False
    )

    metadata = {
        "classes": CLASSES,
        "trained_classes": TARGETS,
        "input_size": SIZE,
        "embedding_size": 64,
        "validation": "held-out samples",
        "best_score": best_score,
    }

    (out / "model_info.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nTRAINING FINISHED", flush=True)
    print("ONNX:", onnx_file.resolve(), flush=True)
    print("Best validation score:", round(best_score, 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pre_data")
    parser.add_argument(
        "--out", default="weights/clean_v2"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.001)
    main(parser.parse_args())
