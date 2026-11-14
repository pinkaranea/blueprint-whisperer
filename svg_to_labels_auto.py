import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Uses the existing, image-sized SVG registration module.
# Put this file in the SAME directory as svg_to_labels.py.
import svg_to_labels as core


CLASS_THRESHOLDS = {
    "window": 0.30,
    "door": 0.23,
    "sink": 0.27,
    "toilet": 0.27,
}


def sample_geometry(points, max_points=900):
    """Sample object outlines to compare them with raster edges."""
    xy = np.asarray(points, dtype=np.float32)
    if len(xy) < 2:
        return xy

    if len(xy) > 1:
        span = np.ptp(xy, axis=0)
        maximum_jump = max(80.0, float(max(span)) * 1.35)
    else:
        maximum_jump = 80.0

    samples = [xy]

    for a, b in zip(xy[:-1], xy[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1 or length > maximum_jump:
            continue

        steps = min(80, max(2, int(length / 2) + 1))
        t = np.linspace(0, 1, steps, dtype=np.float32)
        samples.append(
            a[None, :] * (1 - t[:, None])
            + b[None, :] * t[:, None]
        )

    output = np.vstack(samples)

    if len(output) > max_points:
        indices = np.linspace(
            0, len(output) - 1, max_points
        ).astype(int)
        output = output[indices]

    return output


def create_evidence_maps(image):
    gray = core.grayscale(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # No dependence on ink color.
    edges = cv2.Canny(blurred, 35, 110)

    radius = max(
        3, int(round(max(gray.shape) * 0.003))
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1)
    )

    nearby_edges = cv2.dilate(edges, kernel)

    # Robust relative threshold for dark drawing strokes.
    background = float(np.median(gray))
    if background < 127:
        work = cv2.bitwise_not(gray)
        background = float(np.median(work))
    else:
        work = gray

    threshold = max(35, min(220, background - 35))
    ink = (work < threshold).astype(np.uint8)

    return nearby_edges, ink


def verify_object(obj, image_shape, near_edges, ink,
                  support_override=None):
    """Produce conservative geometric evidence, not AI confidence."""
    h, w = image_shape[:2]
    bbox = obj["bbox"]
    x1, y1, x2, y2 = bbox

    points = sample_geometry(obj["points"])
    if len(points) == 0:
        return {
            "accepted": False,
            "reason": "missing_geometry",
            "line_support": 0.0,
            "ink_fraction": 0.0,
        }

    xx = np.rint(points[:, 0]).astype(int)
    yy = np.rint(points[:, 1]).astype(int)

    inside = (
        (xx >= 0) & (xx < w)
        & (yy >= 0) & (yy < h)
    )

    if inside.mean() < 0.75:
        return {
            "accepted": False,
            "reason": "outside_image",
            "line_support": 0.0,
            "ink_fraction": 0.0,
        }

    xx = xx[inside]
    yy = yy[inside]
    support = float((near_edges[yy, xx] > 0).mean())

    extra = max(3, int(0.08 * max(x2-x1, y2-y1)))
    ax1 = max(0, int(math.floor(x1)) - extra)
    ay1 = max(0, int(math.floor(y1)) - extra)
    ax2 = min(w, int(math.ceil(x2)) + extra)
    ay2 = min(h, int(math.ceil(y2)) + extra)

    region = ink[ay1:ay2, ax1:ax2]
    ink_fraction = (
        float(region.mean()) if region.size else 0.0
    )

    required = (
        float(support_override)
        if support_override is not None
        else CLASS_THRESHOLDS[obj["class"]]
    )

    if support < required:
        reason = "weak_symbol_edge_support"
    elif ink_fraction < 0.008:
        reason = "insufficient_visible_ink"
    else:
        reason = "passed"

    return {
        "accepted": reason == "passed",
        "reason": reason,
        "line_support": round(support, 4),
        "ink_fraction": round(ink_fraction, 4),
        "minimum_support": required,
    }


def draw_quality_overlay(image, accepted, rejected):
    result = image.copy()

    for obj in accepted:
        x1, y1, x2, y2 = map(
            lambda n: int(round(n)), obj["bbox"]
        )
        color = core.COLORS[obj["class"]]
        cv2.rectangle(
            result, (x1, y1), (x2, y2), color, 2
        )
        cv2.putText(
            result,
            f'{obj["class"]} {obj["quality"]["line_support"]:.2f}',
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .42, color, 2
        )

    for obj in rejected:
        x1, y1, x2, y2 = map(
            lambda n: int(round(n)), obj["bbox"]
        )
        color = (20, 20, 220)
        cv2.rectangle(
            result, (x1, y1), (x2, y2), color, 2
        )
        cv2.putText(
            result, "CHECK " + obj["class"],
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .45, color, 2
        )

    return result


def export_clean_samples(image, accepted, all_boxes,
                         folder, size=96,
                         negative_ratio=1.0, seed=42):
    """
    Positive symbols: accepted objects only.
    Negative regions: avoid ALL SVG annotations,
    including suspicious annotations.
    """
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    gray = core.prepare_gray(image)
    height, width = gray.shape

    examples = []
    positive_boxes = []

    for i, item in enumerate(accepted):
        box = core.padded_box(
            item["bbox"], gray.shape
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        patch = core.letterbox_crop(gray, box, size)
        name = (
            f"samples/{item['class']}/"
            f"positive_{i:05d}.png"
        )
        core.save_png(folder / name, patch)

        examples.append({
            "file": name,
            "kind": "positive",
            "class": item["class"],
            "bbox": item["bbox"],
            "quality": item["quality"],
        })
        positive_boxes.append(box)

    goal = math.ceil(
        len(positive_boxes) * max(0, negative_ratio)
    )

    neg_count = 0
    attempts = 0
    while (
        positive_boxes
        and neg_count < goal
        and attempts < max(1500, goal * 90)
    ):
        attempts += 1
        ref = rng.choice(positive_boxes)
        bw = max(12, ref[2] - ref[0])
        bh = max(12, ref[3] - ref[1])

        if bw >= width or bh >= height:
            continue

        x = rng.randint(0, width - bw)
        y = rng.randint(0, height - bh)
        candidate = [x, y, x + bw, y + bh]

        if any(
            core.overlap_ratio(candidate, b) > 0.01
            for b in all_boxes
        ):
            continue

        region = gray[y:y+bh, x:x+bw]
        ink_fraction = float((region < 205).mean())
        if not 0.025 <= ink_fraction <= 0.75:
            continue

        patch = core.letterbox_crop(
            gray, candidate, size
        )
        name = (
            f"samples/background/"
            f"negative_{neg_count:05d}.png"
        )
        core.save_png(folder / name, patch)
        examples.append({
            "file": name,
            "kind": "negative",
            "class": "background",
            "bbox": candidate,
        })
        neg_count += 1

    groups = {}
    for example in examples:
        groups.setdefault(
            example["class"], []
        ).append(example["file"])

    pairs = []
    for cls, names in groups.items():
        if cls == "background" or len(names) < 2:
            continue
        count = min(48, len(names) * 3)
        for _ in range(count):
            a, b = rng.sample(names, 2)
            pairs.append({
                "a": a, "b": b,
                "same": 1, "class": cls
            })

    target = len(pairs)
    attempts = 0
    while (
        len(examples) >= 2
        and len(pairs) < target * 2
        and attempts < target * 30 + 100
    ):
        attempts += 1
        a, b = rng.sample(examples, 2)
        if a["class"] == b["class"]:
            continue
        pairs.append({
            "a": a["file"], "b": b["file"],
            "same": 0,
            "class_a": a["class"],
            "class_b": b["class"],
        })

    rng.shuffle(pairs)

    (folder / "samples.jsonl").write_text(
        "".join(
            json.dumps(e, ensure_ascii=False) + "\n"
            for e in examples
        ), encoding="utf-8"
    )
    (folder / "pairs.jsonl").write_text(
        "".join(
            json.dumps(p, ensure_ascii=False) + "\n"
            for p in pairs
        ), encoding="utf-8"
    )

    core.save_json(
        folder / "samples_manifest.json", {
            "positive_count": len(positive_boxes),
            "negative_count": neg_count,
            "pair_count": len(pairs),
            "template_size": size,
            "split_policy": "split_by_plan_before_training",
        }
    )


def run(args):
    required = [
        "read_svg", "canvas_matrix",
        "write_transformed_svg", "find_alignment",
        "alignment_matrix", "extract_annotations",
    ]
    missing = [x for x in required if not hasattr(core, x)]
    if missing:
        raise RuntimeError(
            "Existing svg_to_labels.py is not the latest "
            "image-sized version. Missing: "
            + ", ".join(missing)
        )

    image_path = Path(args.image)
    svg_path = (
        Path(args.svg)
        if args.svg
        else image_path.with_name("model.svg")
    )

    if not svg_path.is_file():
        raise FileNotFoundError(svg_path)

    image = core.load_image(image_path)
    h, w = image.shape[:2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    viewbox, _, source_walls = core.read_svg(svg_path)

    canvas_path = out / "01_svg_canvas.svg"
    core.write_transformed_svg(
        svg_path, canvas_path, image.shape,
        core.canvas_matrix(viewbox, image.shape, 0)
    )

    params, alignment_error = core.find_alignment(
        image, viewbox, source_walls, args.rotation
    )

    aligned_path = out / "02_svg_aligned.svg"
    core.write_transformed_svg(
        svg_path, aligned_path, image.shape,
        core.alignment_matrix(
            viewbox, image.shape, params
        )
    )

    _, objects, walls = core.read_svg(aligned_path)
    raw_labels = core.extract_annotations(
        objects, w, h
    )

    edges, ink = create_evidence_maps(image)

    accepted, rejected = [], []

    for item in raw_labels:
        source = objects[item["id"]]
        quality = verify_object(
            {**item, "points": source["points"]},
            image.shape, edges, ink,
            args.min_support
        )
        result = {
            **item,
            "quality": quality,
        }
        if quality["accepted"]:
            accepted.append(result)
        else:
            rejected.append(result)

    core.save_png(
        out / "wall_alignment.png",
        core.make_wall_preview(image, walls)
    )
    core.save_png(
        out / "preview.png",
        core.make_preview(image, accepted)
    )
    core.save_png(
        out / "preview_all.png",
        draw_quality_overlay(
            image, accepted, rejected
        )
    )

    core.save_json(
        out / "annotations.json", accepted
    )
    core.save_json(
        out / "quarantine.json", rejected
    )

    core.save_yolo(
        out / "labels.txt", accepted, w, h
    )

    global_pass = alignment_error <= args.max_error

    report = {
        "image": str(image_path.resolve()),
        "svg": str(svg_path.resolve()),
        "image_size": [w, h],
        "alignment": {
            "rotation": params[0],
            "scale_factor": params[1],
            "dx": params[2],
            "dy": params[3],
            "error": alignment_error,
            "passed": global_pass,
        },
        "total_svg_objects": len(raw_labels),
        "accepted": len(accepted),
        "quarantined": len(rejected),
        "accepted_by_class": dict(
            Counter(x["class"] for x in accepted)
        ),
        "quarantined_by_class": dict(
            Counter(x["class"] for x in rejected)
        ),
        "automatic_quality_filter": True,
        "quality_thresholds": CLASS_THRESHOLDS,
        "note": (
            "Edge-support heuristics may reject valid "
            "objects and accept incorrect class labels."
        ),
    }

    export_status = "skipped"
    if global_pass and accepted and not args.no_export:
        export_clean_samples(
            image, accepted,
            [x["bbox"] for x in raw_labels],
            out / "siamese", args.size,
            args.negative_ratio, args.seed
        )
        export_status = "exported"
    else:
        # Avoid leaving sample files from an older, accepted run.
        old_samples = out / "siamese"
        if old_samples.exists():
            shutil.rmtree(old_samples)

    report["samples_status"] = export_status
    core.save_json(out / "quality_report.json", report)

    print("\nBLUEPRINT WHISPERER - AUTO QUALITY")
    print("--------------------------------")
    print("Global alignment:", round(alignment_error, 3))
    print("Global gate:", "PASS" if global_pass else "REJECT")
    print("SVG annotations:", len(raw_labels))
    print("Accepted:", len(accepted))
    print("Quarantine:", len(rejected))
    print("Samples:", export_status)
    print("Preview:", out / "preview_all.png")
    print("Report:", out / "quality_report.json")


def main():
    parser = argparse.ArgumentParser(
        description="Blueprint Whisperer automatic SVG QA"
    )
    parser.add_argument(
        "--image", type=Path, required=True
    )
    parser.add_argument("--svg", type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/annotations/auto")
    )
    parser.add_argument(
        "--rotation",
        choices=["auto", "0", "90", "180", "270"],
        default="auto",
    )
    parser.add_argument(
        "--max-error", type=float, default=7.0
    )
    parser.add_argument(
        "--min-support", type=float
    )
    parser.add_argument(
        "--size", type=int, default=96
    )
    parser.add_argument(
        "--negative-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--no-export", action="store_true"
    )

    args = parser.parse_args()
    if args.min_support is not None:
        if not 0 <= args.min_support <= 1:
            parser.error("--min-support must be 0..1")
    if args.size < 32:
        parser.error("--size must be at least 32")

    run(args)


if __name__ == "__main__":
    main()
