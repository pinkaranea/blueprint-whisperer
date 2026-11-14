
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("door", "sink", "toilet")


def read_image(path, mode=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, mode)

    if image is None:
        raise FileNotFoundError(path)

    return image


def save_image(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot save {path}")
    data.tofile(str(path))



def descriptor(image):
    image = cv2.resize(
        image, (96, 96),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    # Gradienty obrazu
    gy, gx = np.gradient(image)

    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    cell_size = 8
    bins_count = 9
    features = []

    # Histogram kierunkow krawedzi w komorkach 8x8
    for y in range(0, 96, cell_size):
        for x in range(0, 96, cell_size):
            angles = angle[
                y:y+cell_size, x:x+cell_size
            ]
            magnitudes = magnitude[
                y:y+cell_size, x:x+cell_size
            ]

            bins = np.floor(
                angles / 20.0
            ).astype(np.int32)

            histogram = np.bincount(
                bins.ravel(),
                weights=magnitudes.ravel(),
                minlength=bins_count
            )

            features.extend(histogram)

    features = np.asarray(features, dtype=np.float32)

    norm = np.linalg.norm(features)
    return features / max(norm, 1e-8)



def build_shape_bank(manifest):
    bank = defaultdict(list)

    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["split"] != "train":
            continue

        cls = row["class"]
        if cls not in CLASSES:
            continue

        path = Path(row["path"])
        image = read_image(path)

        # Rotations and mirror images cover symbol orientation.
        for k in range(4):
            rotated = np.rot90(image, k)

            bank[cls].append(descriptor(rotated))
            bank[cls].append(
                descriptor(np.fliplr(rotated))
            )

    result = {}

    for cls, vectors in bank.items():
        result[cls] = np.stack(vectors)

        print(
            f"{cls}: {len(vectors)} shape templates",
            flush=True
        )

    for cls in ("door", "sink", "toilet"):
        if cls not in result:
            raise RuntimeError(f"Missing class: {cls}")

    return result


def best_similarity(vector, prototypes):
    scores = prototypes @ vector
    k = min(3, len(scores))
    return float(np.partition(scores, -k)[-k:].mean())


def evaluate_shape(image, box, bank, target):
    h, w = image.shape

    x1 = max(0, int(box["x1"]))
    y1 = max(0, int(box["y1"]))
    x2 = min(w, int(box["x2"]))
    y2 = min(h, int(box["y2"]))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return 0.0, -1.0, {}

    features = descriptor(crop)

    scores = {
        cls: best_similarity(features, prototypes)
        for cls, prototypes in bank.items()
    }

    positive = scores[target]
    negative = max(
        value for cls, value in scores.items()
        if cls != target
    )

    margin = positive - negative

    return positive, margin, scores


def main(args):
    report_path = Path(args.report)

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    target = report["target_class"]

    if target not in CLASSES:
        raise ValueError(f"Unsupported target: {target}")

    image = read_image(
        report["image"],
        mode=cv2.IMREAD_COLOR
    )
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    bank = build_shape_bank(args.manifest)

    accepted = []
    audited = []

    audit_image = image.copy()
    filtered_image = image.copy()

    for detection in report["detections"]:
        score, margin, scores = evaluate_shape(
            gray, detection, bank, target
        )

        keep = (
            score >= args.min_score
            and margin >= args.margin
        )

        entry = dict(detection)
        entry["shape_similarity"] = round(score, 4)
        entry["shape_margin"] = round(margin, 4)
        entry["shape_scores"] = {
            cls: round(v, 4)
            for cls, v in scores.items()
        }
        entry["shape_passed"] = bool(keep)

        audited.append(entry)

        x1, y1 = entry["x1"], entry["y1"]
        x2, y2 = entry["x2"], entry["y2"]

        color = (0, 190, 0) if keep else (0, 0, 255)

        cv2.rectangle(
            audit_image, (x1, y1), (x2, y2),
            color, 3
        )

        cv2.putText(
            audit_image,
            f"{entry['id']} M={margin:+.2f}",
            (x1, max(18, y1-5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2
        )

        if keep:
            accepted.append(entry)
            cv2.rectangle(
                filtered_image,
                (x1, y1), (x2, y2),
                (0, 190, 0), 3
            )

        print(
            f"#{entry['id']:02} | "
            f"shape={score:.3f} | "
            f"margin={margin:+.3f} | "
            f"{'KEEP' if keep else 'REJECT'}",
            flush=True
        )

    output = report_path.parent

    save_image(output / "shape_audit.png", audit_image)
    save_image(
        output / "shape_candidates.png",
        filtered_image
    )

    audit_report = dict(report)
    audit_report["detections"] = audited
    audit_report["shape_filter"] = {
        "min_score": args.min_score,
        "min_margin": args.margin,
    }

    (output / "shape_audit.json").write_text(
        json.dumps(audit_report, indent=2),
        encoding="utf-8"
    )

    final_report = dict(audit_report)
    final_report["detections"] = accepted
    final_report["candidate_count"] = len(accepted)
    final_report["note"] = (
        "Geometry-filtered candidates; "
        "not human-verified detections."
    )

    (output / "shape_candidates.json").write_text(
        json.dumps(final_report, indent=2),
        encoding="utf-8"
    )

    print()
    print("CNN candidates:", len(audited))
    print("After shape filter:", len(accepted))
    print("Output:", output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        default="data_stoa/demo_clean_doors/detections.json"
    )
    parser.add_argument(
        "--manifest",
        default="weights/clean_v2/split_manifest.csv"
    )
    parser.add_argument(
        "--margin", type=float, default=0.0
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0
    )

    main(parser.parse_args())
