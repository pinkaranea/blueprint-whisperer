import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


SIZE = 96


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot load: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IOError(path)
    encoded.tofile(str(path))


def letterbox(image, bbox):
    x, y, w, h = [int(v) for v in bbox]
    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        raise ValueError("Empty template or candidate")

    h, w = crop.shape
    scale = min(SIZE / w, SIZE / h)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    crop = cv2.resize(
        crop, (nw, nh), interpolation=interpolation
    )

    result = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    x0 = (SIZE - nw) // 2
    y0 = (SIZE - nh) // 2
    result[y0:y0+nh, x0:x0+nw] = crop
    return result


def encode(net, patches):
    batch = np.stack(patches).astype(np.float32)
    batch = (255.0 - batch) / 255.0
    batch = batch[:, None, :, :]

    net.setInput(batch)
    vectors = np.asarray(net.forward(), dtype=np.float32)
    vectors = vectors.reshape(len(patches), -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def positions(length, window, step):
    values = list(range(0, length-window+1, step))
    if not values:
        return []
    last = length - window
    if values[-1] != last:
        values.append(last)
    return values


def iou(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"]+a["w"], b["x"]+b["w"])
    y2 = min(a["y"]+a["h"], b["y"]+b["h"])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    return intersection / max(1, area_a+area_b-intersection)


def nms(boxes, threshold=0.25, limit=60):
    selected = []
    for box in sorted(
        boxes, key=lambda b: b["score"], reverse=True
    )[:1000]:
        if all(iou(box, chosen) < threshold
               for chosen in selected):
            selected.append(box)
        if len(selected) >= limit:
            break
    return selected


def search(net, gray, roi, threshold,
           stride=0.55, batch_size=64,
           max_candidates=12000, max_results=60):

    height, width = gray.shape
    rx, ry, rw, rh = roi

    reference = encode(
        net, [letterbox(gray, roi)]
    )[0]

    scales = [0.75, 1.0, 1.3]
    sizes = []

    for factor in scales:
        ww = max(16, round(rw * factor))
        hh = max(16, round(rh * factor))

        if ww >= width or hh >= height:
            continue

        sx = max(8, round(ww * stride))
        sy = max(8, round(hh * stride))
        sizes.append((ww, hh, sx, sy))

    estimate = sum(
        (math.ceil((width - ww) / sx) + 1)
        * (math.ceil((height - hh) / sy) + 1)
        for ww, hh, sx, sy in sizes
    )
    boost = max(
        1.0, math.sqrt(estimate / max(1, max_candidates))
    )

    candidates = []
    patches = []
    boxes = []
    total = 0

    def flush():
        if not patches:
            return

        vectors = encode(net, patches)
        scores = vectors @ reference

        for box, score in zip(boxes, scores):
            if float(score) >= threshold:
                candidates.append({
                    **box, "score": round(float(score), 5)
                })

        patches.clear()
        boxes.clear()

    for ww, hh, sx, sy in sizes:
        sx = max(8, round(sx * boost))
        sy = max(8, round(sy * boost))

        for y in positions(height, hh, sy):
            for x in positions(width, ww, sx):
                crop = gray[y:y+hh, x:x+ww]
                ink = float(np.mean(crop < 220))

                if not 0.008 <= ink <= 0.85:
                    continue

                patches.append(
                    letterbox(gray, (x, y, ww, hh))
                )
                boxes.append({
                    "x": x, "y": y, "w": ww, "h": hh
                })
                total += 1

                if len(patches) >= batch_size:
                    flush()

    flush()

    print("Evaluated windows:", total, flush=True)
    print("Above threshold:", len(candidates), flush=True)

    return nms(
        candidates,
        threshold=0.25,
        limit=max_results
    )


def main(args):
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    meta_path = model_path.with_suffix(".json")
    meta = {}

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    threshold = args.threshold
    if threshold is None:
        threshold = max(
            0.45, float(meta.get("threshold", 0.7)) - 0.08
        )

    print("Loading ONNX:", model_path, flush=True)
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    original = load_image(args.image)
    oh, ow = original.shape[:2]

    scale = min(1.0, args.max_side / max(oh, ow))
    vw = max(1, round(ow * scale))
    vh = max(1, round(oh * scale))

    view = cv2.resize(
        original, (vw, vh), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

    if args.roi:
        raw = [int(x) for x in args.roi.split(",")]
        if len(raw) != 4:
            raise ValueError("--roi expects x,y,w,h")

        roi = (
            round(raw[0] * vw / ow),
            round(raw[1] * vh / oh),
            max(1, round(raw[2] * vw / ow)),
            max(1, round(raw[3] * vh / oh)),
        )
    else:
        roi = cv2.selectROI(
            "Select ONE reference symbol and press ENTER",
            view,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyAllWindows()

    roi = tuple(map(int, roi))
    x, y, w, h = roi

    if w < 4 or h < 4:
        raise ValueError("Reference symbol not selected")

    if x < 0 or y < 0 or x+w > vw or y+h > vh:
        raise ValueError("ROI outside the image")

    print("Reference ROI:", roi)
    print("Similarity threshold:", round(threshold, 3))
    print("Searching...", flush=True)

    results = search(
        net, gray, roi, threshold,
        stride=args.stride,
        batch_size=args.batch,
        max_candidates=args.max_candidates,
        max_results=args.max_results,
    )

    sx = ow / vw
    sy = oh / vh

    output = original.copy()
    detections = []

    for i, box in enumerate(results, start=1):
        x1 = round(box["x"] * sx)
        y1 = round(box["y"] * sy)
        x2 = round((box["x"]+box["w"]) * sx)
        y2 = round((box["y"]+box["h"]) * sy)

        cv2.rectangle(
            output, (x1, y1), (x2, y2), (208, 65, 224), 3
        )
        cv2.putText(
            output,
            f"{i}: {box['score']:.2f}",
            (x1, max(22, y1-7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (208, 65, 224), 2
        )

        detections.append({
            "id": i,
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "similarity": box["score"],
            "reviewed": False,
        })

    ref_box = (
        round(x*sx), round(y*sy),
        round((x+w)*sx), round((y+h)*sy)
    )
    cv2.rectangle(
        output, ref_box[:2], ref_box[2:],
        (255, 220, 0), 3
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    save_image(out / "detections.png", output)

    report = {
        "image": str(Path(args.image).resolve()),
        "model": str(model_path.resolve()),
        "reference_roi": {
            "x1": ref_box[0], "y1": ref_box[1],
            "x2": ref_box[2], "y2": ref_box[3]
        },
        "threshold": threshold,
        "candidate_count": len(detections),
        "note": (
            "Unreviewed similarity candidates. "
            "Counts are not verified object counts."
        ),
        "detections": detections,
    }

    (out / "detections.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    with (out / "detections.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "id", "x1", "y1", "x2", "y2",
                "similarity", "reviewed"
            ]
        )
        writer.writeheader()
        writer.writerows(detections)

    print("CANDIDATES:", len(detections))
    print("OUTPUT:", out.resolve())

    if not args.no_show:
        preview_scale = min(
            1.0, 1400 / max(oh, ow)
        )
        preview = cv2.resize(
            output, None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA
        )
        cv2.imshow(
            "Blueprint Whisperer - Evidence Preview",
            preview
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model", default="weights/blueprint_match.onnx"
    )
    parser.add_argument(
        "--out", default="data_stoa/demo"
    )
    parser.add_argument("--roi")
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--max-side", type=int, default=1600
    )
    parser.add_argument(
        "--stride", type=float, default=0.55
    )
    parser.add_argument(
        "--batch", type=int, default=64
    )
    parser.add_argument(
        "--max-candidates", type=int, default=12000
    )
    parser.add_argument(
        "--max-results", type=int, default=60
    )
    parser.add_argument("--no-show", action="store_true")
    main(parser.parse_args())
