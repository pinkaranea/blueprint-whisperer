import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["window", "door", "sink", "toilet", "_background"]
SIZE = 96


def load_image(path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(path)

    return image


def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"Cannot encode image: {path}")

    encoded.tofile(str(path))


def make_patch(gray, box):
    x, y, w, h = map(int, box)

    crop = gray[y:y+h, x:x+w]

    if crop.size == 0:
        raise ValueError("Empty candidate crop")

    return cv2.resize(
        crop,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    )


def infer(net, patches):
    batch = np.stack(patches).astype(np.float32)

    # Same normalization as train_clean.py
    batch = ((255.0 - batch) / 255.0)[:, None]
    batch = np.ascontiguousarray(batch)

    net.setInput(batch)

    output = net.forward()
    output = output.reshape(len(patches), -1)

    if output.shape[1] != 69:
        raise RuntimeError(
            f"Expected 69 outputs, got {output.shape}"
        )

    # 64D feature embedding
    embeddings = output[:, :64]

    norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    embeddings = embeddings / np.maximum(norms, 1e-8)

    # Classification head: 5 classes
    logits = output[:, 64:]

    logits = logits - logits.max(
        axis=1, keepdims=True
    )

    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(
        axis=1, keepdims=True
    )

    return embeddings, probabilities


def grid_positions(length, window, step):
    if window > length:
        return []

    positions = list(
        range(0, length - window + 1, step)
    )

    last = length - window

    if not positions:
        positions = [last]
    elif positions[-1] != last:
        positions.append(last)

    return positions


def calculate_iou(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])

    x2 = min(
        a["x"] + a["w"],
        b["x"] + b["w"]
    )

    y2 = min(
        a["y"] + a["h"],
        b["y"] + b["h"]
    )

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]

    union = area_a + area_b - intersection

    return intersection / max(1, union)


def non_maximum_suppression(
    candidates,
    threshold=0.30,
    limit=30
):
    selected = []

    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    for candidate in candidates:
        if all(
            calculate_iou(candidate, chosen) < threshold
            for chosen in selected
        ):
            selected.append(candidate)

        if len(selected) >= limit:
            break

    return selected


def evaluate_candidates(
    net,
    gray,
    boxes,
    reference,
    class_index,
    batch_size
):
    results = []

    for start in range(0, len(boxes), batch_size):
        block = boxes[start:start + batch_size]

        patches = [
            make_patch(gray, box)
            for box in block
        ]

        embeddings, probabilities = infer(
            net, patches
        )

        similarities = embeddings @ reference

        for box, similarity, probability in zip(
            block,
            similarities,
            probabilities[:, class_index]
        ):
            similarity = float(similarity)
            probability = float(probability)

            score = (
                0.65 * similarity +
                0.35 * probability
            )

            results.append({
                "x": int(box[0]),
                "y": int(box[1]),
                "w": int(box[2]),
                "h": int(box[3]),
                "similarity": similarity,
                "class_prob": probability,
                "score": score,
            })

        if start % (batch_size * 100) == 0:
            print(
                f"Evaluated: "
                f"{min(start + batch_size, len(boxes))}"
                f"/{len(boxes)}",
                flush=True
            )

    return results


def search(net, gray, roi, target_index, args):
    image_h, image_w = gray.shape

    reference_patch = make_patch(gray, roi)

    reference_embedding, reference_probabilities = infer(
        net, [reference_patch]
    )

    reference = reference_embedding[0]

    print(
        "Reference class probabilities:",
        {
            name: round(float(probability), 3)
            for name, probability in zip(
                CLASSES,
                reference_probabilities[0]
            )
        },
        flush=True
    )

    reference_w = roi[2]
    reference_h = roi[3]

    scales = [0.70, 0.85, 1.0, 1.15, 1.35]

    windows = []

    for scale in scales:
        w = max(16, round(reference_w * scale))
        h = max(16, round(reference_h * scale))

        if w > image_w or h > image_h:
            continue

        step_x = max(8, round(w * args.stride))
        step_y = max(8, round(h * args.stride))

        windows.append((w, h, step_x, step_y))

    if not windows:
        raise RuntimeError("Reference ROI is too large")

    estimated = sum(
        len(grid_positions(image_w, w, sx)) *
        len(grid_positions(image_h, h, sy))
        for w, h, sx, sy in windows
    )

    adjustment = max(
        1.0,
        math.sqrt(
            estimated / max(1, args.max_candidates)
        )
    )

    boxes = []

    for w, h, step_x, step_y in windows:
        step_x = max(8, round(step_x * adjustment))
        step_y = max(8, round(step_y * adjustment))

        xs = grid_positions(
            image_w, w, step_x
        )
        ys = grid_positions(
            image_h, h, step_y
        )

        for y in ys:
            for x in xs:
                crop = gray[y:y+h, x:x+w]

                ink_ratio = float(
                    np.mean(crop < 220)
                )

                if 0.008 <= ink_ratio <= 0.85:
                    boxes.append((x, y, w, h))

    print(
        "Coarse candidates:",
        len(boxes),
        flush=True
    )

    coarse_results = evaluate_candidates(
        net,
        gray,
        boxes,
        reference,
        target_index,
        args.batch
    )

    # Refine the strongest regions locally.
    strongest = sorted(
        coarse_results,
        key=lambda item: item["score"],
        reverse=True
    )[:args.refine]

    refined_boxes = []
    visited = set()

    for candidate in strongest:
        w = candidate["w"]
        h = candidate["h"]

        for offset_x in (-0.22, 0.0, 0.22):
            for offset_y in (-0.22, 0.0, 0.22):
                x = round(
                    candidate["x"] + offset_x * w
                )
                y = round(
                    candidate["y"] + offset_y * h
                )

                x = max(0, min(image_w - w, x))
                y = max(0, min(image_h - h, y))

                key = (x, y, w, h)

                if key not in visited:
                    visited.add(key)
                    refined_boxes.append(key)

    print(
        "Refinement candidates:",
        len(refined_boxes),
        flush=True
    )

    if refined_boxes:
        refined_results = evaluate_candidates(
            net,
            gray,
            refined_boxes,
            reference,
            target_index,
            args.batch
        )
    else:
        refined_results = []

    combined = coarse_results + refined_results

    # Both conditions must be fulfilled.
    filtered = [
        candidate
        for candidate in combined
        if (
            candidate["similarity"] >= args.similarity
            and candidate["class_prob"] >= args.class_prob
        )
    ]

    print(
        "Passed similarity + class filters:",
        len(filtered),
        flush=True
    )

    return non_maximum_suppression(
        filtered,
        threshold=args.nms,
        limit=args.max_results
    )


def main(args):
    print("Loading model:", args.model, flush=True)

    net = cv2.dnn.readNetFromONNX(
        str(args.model)
    )

    original = load_image(args.image)

    original_h, original_w = original.shape[:2]

    scale = min(
        1.0,
        args.max_side / max(original_h, original_w)
    )

    view_w = max(1, round(original_w * scale))
    view_h = max(1, round(original_h * scale))

    view = cv2.resize(
        original,
        (view_w, view_h),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        view,
        cv2.COLOR_BGR2GRAY
    )

    if args.roi:
        x, y, w, h = map(
            int, args.roi.split(",")
        )

        roi = (
            round(x * view_w / original_w),
            round(y * view_h / original_h),
            max(1, round(w * view_w / original_w)),
            max(1, round(h * view_h / original_h))
        )

    else:
        print(
            "Select ONE symbol and press ENTER",
            flush=True
        )

        roi = tuple(
            map(
                int,
                cv2.selectROI(
                    "Blueprint Whisperer - Select Symbol",
                    view,
                    showCrosshair=True
                )
            )
        )

        cv2.destroyAllWindows()

    x, y, w, h = roi

    if w < 8 or h < 8:
        raise ValueError("Invalid ROI selection")

    if (
        x < 0 or y < 0
        or x + w > view_w
        or y + h > view_h
    ):
        raise ValueError("ROI outside image")

    if w > view_w * 0.4 or h > view_h * 0.4:
        raise ValueError(
            "Select one symbol, not an entire room"
        )

    print("Reference ROI:", roi, flush=True)
    print("Target class:", args.target, flush=True)
    print("Searching...", flush=True)

    target_index = CLASSES.index(args.target)

    results = search(
        net,
        gray,
        roi,
        target_index,
        args
    )

    result_image = original.copy()

    sx = original_w / view_w
    sy = original_h / view_h

    detections = []

    for index, candidate in enumerate(results, 1):
        x1 = round(candidate["x"] * sx)
        y1 = round(candidate["y"] * sy)

        x2 = round(
            (candidate["x"] + candidate["w"]) * sx
        )

        y2 = round(
            (candidate["y"] + candidate["h"]) * sy
        )

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (220, 35, 230),
            3
        )

        label = (
            f"{index} "
            f"{candidate['similarity']:.2f} "
            f"P={candidate['class_prob']:.2f}"
        )

        cv2.putText(
            result_image,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 35, 230),
            2
        )

        detections.append({
            "id": index,
            "class": args.target,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "similarity": round(
                candidate["similarity"], 5
            ),
            "class_prob": round(
                candidate["class_prob"], 5
            ),
            "reviewed": False
        })

    reference_box = {
        "x1": round(x * sx),
        "y1": round(y * sy),
        "x2": round((x + w) * sx),
        "y2": round((y + h) * sy)
    }

    cv2.rectangle(
        result_image,
        (
            reference_box["x1"],
            reference_box["y1"]
        ),
        (
            reference_box["x2"],
            reference_box["y2"]
        ),
        (255, 220, 0),
        3
    )

    output_folder = Path(args.out)
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "image": str(Path(args.image).resolve()),
        "model": str(Path(args.model).resolve()),
        "target_class": args.target,
        "reference_roi": reference_box,
        "candidate_count": len(detections),
        "review_complete": False,
        "detections": detections,
        "note": (
            "Unreviewed detection candidates; "
            "not verified object counts."
        )
    }

    (output_folder / "detections.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    with (output_folder / "detections.csv").open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fieldnames = [
            "id", "class",
            "x1", "y1", "x2", "y2",
            "similarity",
            "class_prob",
            "reviewed"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(detections)

    save_png(
        output_folder / "detections.png",
        result_image
    )

    print(
        "FINAL CANDIDATES:",
        len(detections),
        flush=True
    )

    print(
        "OUTPUT:",
        output_folder.resolve(),
        flush=True
    )

    if not args.no_show:
        preview = cv2.resize(
            result_image,
            (view_w, view_h),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(
            "Blueprint Whisperer V2 - Detections",
            preview
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    parser.add_argument(
        "--model",
        default="weights/clean_v2/blueprint_match_clean.onnx"
    )

    parser.add_argument(
        "--target",
        choices=["door", "sink", "toilet"],
        default="door"
    )

    parser.add_argument("--roi")

    parser.add_argument(
        "--out",
        default="data_stoa/demo_clean"
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2200
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=25
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2
    )

    parser.add_argument(
        "--stride",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--similarity",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--class-prob",
        type=float,
        default=0.55
    )

    parser.add_argument(
        "--refine",
        type=int,
        default=25
    )

    parser.add_argument(
        "--nms",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--no-show",
        action="store_true"
    )

    main(parser.parse_args())
