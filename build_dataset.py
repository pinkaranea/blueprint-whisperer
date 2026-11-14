import os
import re
import csv
import json
import math
import random
import hashlib
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

CLASS_KEYWORDS = {
    "window": ["window"],
    "door": ["door"],
    "sink": ["sink", "washbasin", "basin"],
    "toilet": ["toilet", "wc"],
}

IMAGE_CANDIDATES = [
    "F1_scaled.png",
    "F1_original.png",
    "F1.png",
    "image.png",
    "image.jpg",
    "image.jpeg",
]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# UTILS
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        x = str(x).replace("px", "").strip()
        return float(x)
    except Exception:
        return default


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha1_array(arr):
    return hashlib.sha1(arr.tobytes()).hexdigest()


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def strip_ns(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip().lower()


def detect_class_from_attrs(attrs):
    blob = " ".join([normalize_text(v) for v in attrs if v is not None])
    for cls_name, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in blob:
                return cls_name
    return None


# =========================================================
# SVG PARSING
# =========================================================

_transform_cmd = re.compile(r"([a-zA-Z]+)\(([^)]*)\)")


def parse_transform(transform_str):
    """
    Obsługa najważniejszych transformacji:
    translate, scale
    Zwraca macierz 3x3.
    """
    M = np.eye(3, dtype=np.float32)

    if not transform_str:
        return M

    for cmd, args_str in _transform_cmd.findall(transform_str):
        args = [safe_float(x) for x in re.split(r"[,\s]+", args_str.strip()) if x.strip()]
        cmd = cmd.lower()

        if cmd == "translate":
            tx = args[0] if len(args) > 0 else 0.0
            ty = args[1] if len(args) > 1 else 0.0
            T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
            M = M @ T

        elif cmd == "scale":
            sx = args[0] if len(args) > 0 else 1.0
            sy = args[1] if len(args) > 1 else sx
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)
            M = M @ S

        # rotate/skew/matrix pomijamy dla prostoty MVP
    return M


def apply_transform_to_points(points, M):
    if len(points) == 0:
        return points
    pts = np.array(points, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    out = (pts_h @ M.T)[:, :2]
    return out.tolist()


def bbox_from_points(points):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def union_bbox(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def parse_points_string(points_str):
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", points_str or "")
    nums = [float(v) for v in vals]
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append([nums[i], nums[i + 1]])
    return pts


def parse_path_points(d):
    """
    Uproszczony parser współrzędnych z path.
    Nie odtwarza krzywych dokładnie, ale do bbox MVP wystarczy.
    """
    if not d:
        return []
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", d)
    nums = [float(v) for v in vals]
    pts = []
    for i in range(0, len(nums) - 1, 2):
        pts.append([nums[i], nums[i + 1]])
    return pts


def element_bbox(elem, inherited_transform=None):
    if inherited_transform is None:
        inherited_transform = np.eye(3, dtype=np.float32)

    own_transform = parse_transform(elem.attrib.get("transform"))
    M = inherited_transform @ own_transform

    tag = strip_ns(elem.tag).lower()

    if tag == "rect":
        x = safe_float(elem.attrib.get("x"))
        y = safe_float(elem.attrib.get("y"))
        w = safe_float(elem.attrib.get("width"))
        h = safe_float(elem.attrib.get("height"))
        pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        pts = apply_transform_to_points(pts, M)
        return bbox_from_points(pts)

    elif tag == "line":
        x1 = safe_float(elem.attrib.get("x1"))
        y1 = safe_float(elem.attrib.get("y1"))
        x2 = safe_float(elem.attrib.get("x2"))
        y2 = safe_float(elem.attrib.get("y2"))
        pts = apply_transform_to_points([[x1, y1], [x2, y2]], M)
        return bbox_from_points(pts)

    elif tag in ("polyline", "polygon"):
        pts = parse_points_string(elem.attrib.get("points"))
        pts = apply_transform_to_points(pts, M)
        return bbox_from_points(pts)

    elif tag == "circle":
        cx = safe_float(elem.attrib.get("cx"))
        cy = safe_float(elem.attrib.get("cy"))
        r = safe_float(elem.attrib.get("r"))
        pts = [[cx - r, cy - r], [cx + r, cy + r]]
        pts = apply_transform_to_points(pts, M)
        return bbox_from_points(pts)

    elif tag == "ellipse":
        cx = safe_float(elem.attrib.get("cx"))
        cy = safe_float(elem.attrib.get("cy"))
        rx = safe_float(elem.attrib.get("rx"))
        ry = safe_float(elem.attrib.get("ry"))
        pts = [[cx - rx, cy - ry], [cx + rx, cy + ry]]
        pts = apply_transform_to_points(pts, M)
        return bbox_from_points(pts)

    elif tag == "path":
        pts = parse_path_points(elem.attrib.get("d"))
        pts = apply_transform_to_points(pts, M)
        return bbox_from_points(pts)

    # Jeśli grupa lub inny kontener: sumujemy dzieci
    child_union = None
    for child in list(elem):
        b = element_bbox(child, M)
        child_union = union_bbox(child_union, b)
    return child_union


def extract_viewbox_size(svg_root):
    viewbox = svg_root.attrib.get("viewBox")
    if viewbox:
        vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", viewbox)
        if len(vals) >= 4:
            _, _, w, h = map(float, vals[:4])
            if w > 0 and h > 0:
                return w, h

    w = safe_float(svg_root.attrib.get("width"), 0)
    h = safe_float(svg_root.attrib.get("height"), 0)
    if w > 0 and h > 0:
        return w, h

    return None, None


def collect_labeled_objects(svg_path, image_w, image_h):
    """
    Szuka elementów/grup oznaczonych jako window / door / sink / toilet
    na podstawie class/id/label/inkscape:label.
    Następnie liczy bbox i mapuje na rozmiar obrazu.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    svg_w, svg_h = extract_viewbox_size(root)
    if not svg_w or not svg_h:
        svg_w, svg_h = image_w, image_h

    sx = image_w / float(svg_w)
    sy = image_h / float(svg_h)

    objects = []

    def walk(elem, inherited_transform=None):
        if inherited_transform is None:
            inherited_transform = np.eye(3, dtype=np.float32)

        own_transform = parse_transform(elem.attrib.get("transform"))
        M = inherited_transform @ own_transform

        attrs_for_class = [
            elem.attrib.get("id"),
            elem.attrib.get("class"),
            elem.attrib.get("label"),
            elem.attrib.get("{http://www.inkscape.org/namespaces/inkscape}label"),
        ]
        cls_name = detect_class_from_attrs(attrs_for_class)

        if cls_name is not None:
            b = element_bbox(elem, inherited_transform)
            if b is not None:
                x1, y1, x2, y2 = b
                x1 *= sx
                x2 *= sx
                y1 *= sy
                y2 *= sy

                if x2 > x1 and y2 > y1:
                    objects.append({
                        "class": cls_name,
                        "bbox": [x1, y1, x2, y2]
                    })

        for child in list(elem):
            walk(child, M)

    walk(root)

    # Usuwanie duplikatów / bardzo podobnych ramek
    dedup = []
    for obj in objects:
        keep = True
        for other in dedup:
            if obj["class"] == other["class"] and iou_xyxy(obj["bbox"], other["bbox"]) > 0.90:
                keep = False
                break
        if keep:
            dedup.append(obj)

    return dedup


# =========================================================
# SPLITS / FILE DISCOVERY
# =========================================================

def load_split_list(split_path):
    items = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(s)
    return items


def build_sample_folder_candidates(root, item):
    """
    Obsługa różnych formatów linii:
    - 11397
    - colorful/11397
    - cubicasa5k/colorful/11397
    """
    item = item.replace("\\", "/").strip().strip('"').strip("'").lstrip("/")
    base = os.path.join(root, item)
    cands = [base]

    # jeśli samo ID
    if "/" not in item:
        cands.extend([
            os.path.join(root, "colorful", item),
            os.path.join(root, "high_quality", item),
            os.path.join(root, "high_quality_architectural", item),
        ])
    else:
        # jeśli ma prefiks cubicasa5k/
        parts = item.split("/")
        if parts[0].lower() == "cubicasa5k":
            cands.append(os.path.join(root, *parts[1:]))

    # usuń duplikaty z zachowaniem kolejności
    out = []
    seen = set()
    for p in cands:
        p = os.path.normpath(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def resolve_sample_folder(root, item):
    for cand in build_sample_folder_candidates(root, item):
        if os.path.isdir(cand):
            return cand
    return None


def find_image_file(sample_dir):
    for name in IMAGE_CANDIDATES:
        p = os.path.join(sample_dir, name)
        if os.path.isfile(p):
            return p

    # fallback: pierwszy png/jpg
    for fn in os.listdir(sample_dir):
        low = fn.lower()
        if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg"):
            return os.path.join(sample_dir, fn)
    return None


def find_svg_file(sample_dir):
    for fn in os.listdir(sample_dir):
        if fn.lower().endswith(".svg"):
            return os.path.join(sample_dir, fn)
    return None


# =========================================================
# PATCH EXTRACTION
# =========================================================

def crop_with_context(image, bbox, context_ratio=0.20):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * context_ratio
    pad_y = bh * context_ratio

    nx1 = int(clamp(math.floor(x1 - pad_x), 0, w - 1))
    ny1 = int(clamp(math.floor(y1 - pad_y), 0, h - 1))
    nx2 = int(clamp(math.ceil(x2 + pad_x), 1, w))
    ny2 = int(clamp(math.ceil(y2 + pad_y), 1, h))

    crop = image[ny1:ny2, nx1:nx2]
    return crop, [nx1, ny1, nx2, ny2]


def is_good_patch(patch, min_size=12, min_ink_ratio=0.005):
    if patch is None or patch.size == 0:
        return False

    h, w = patch.shape[:2]
    if min(h, w) < min_size:
        return False

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    # "ink" = coś ciemniejszego od tła
    ink = np.mean(gray < 220)
    if ink < min_ink_ratio:
        return False

    return True


def save_patch(patch, out_path, patch_size=96):
    resized = letterbox_resize(patch, patch_size)
    cv2.imwrite(out_path, resized)
    return resized


def sample_background_boxes(image_w, image_h, positive_boxes, how_many=20, patch_size=96):
    boxes = []
    tries = 0
    max_tries = how_many * 30

    while len(boxes) < how_many and tries < max_tries:
        tries += 1

        bw = random.randint(32, 96)
        bh = random.randint(32, 96)
        x1 = random.randint(0, max(0, image_w - bw - 1))
        y1 = random.randint(0, max(0, image_h - bh - 1))
        x2 = x1 + bw
        y2 = y1 + bh

        cand = [x1, y1, x2, y2]

        bad = False
        for pb in positive_boxes:
            if (min(cand[2], pb[2]) > max(cand[0], pb[0])
                and min(cand[3], pb[3]) > max(cand[1], pb[1])):

                bad = True
                break

        if not bad:
            boxes.append(cand)

    return boxes


# =========================================================
# PAIR GENERATION
# =========================================================

def build_pairs_for_split(index_rows, out_csv, negatives_per_positive=1):
    by_class = defaultdict(list)
    for row in index_rows:
        by_class[row["class"]].append(row)

    classes = [c for c in by_class.keys() if c != "_background"]

    pairs = []

    # Positive pairs
    for cls in classes:
        items = by_class[cls]
        if len(items) < 2:
            continue

        shuffled = items[:]
        random.shuffle(shuffled)

        for i in range(0, len(shuffled) - 1, 2):
            a = shuffled[i]
            b = shuffled[i + 1]
            pairs.append({
                "img1": a["patch_path"],
                "img2": b["patch_path"],
                "label": 1,
                "class1": a["class"],
                "class2": b["class"],
                "pair_type": "positive"
            })

    # Negative pairs
    all_rows = index_rows[:]
    bg_items = by_class.get("_background", [])

    for p in [x for x in pairs if x["label"] == 1]:
        img1 = p["img1"]
        class1 = p["class1"]

        anchor = None
        for r in all_rows:
            if r["patch_path"] == img1:
                anchor = r
                break
        if anchor is None:
            continue

        for _ in range(negatives_per_positive):
            # 50% inna klasa, 50% background
            if bg_items and random.random() < 0.5:
                other = random.choice(bg_items)
            else:
                other_class_choices = [c for c in classes if c != class1 and len(by_class[c]) > 0]
                if not other_class_choices:
                    continue
                oc = random.choice(other_class_choices)
                other = random.choice(by_class[oc])

            pairs.append({
                "img1": anchor["patch_path"],
                "img2": other["patch_path"],
                "label": 0,
                "class1": anchor["class"],
                "class2": other["class"],
                "pair_type": "negative"
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img1", "img2", "label", "class1", "class2", "pair_type"]
        )
        writer.writeheader()
        writer.writerows(pairs)

    return len(pairs)


# =========================================================
# MAIN DATASET BUILD
# =========================================================


def collect_labeled_objects_auto(svg_path, image, max_error=7.0):
    import svg_to_labels as align
    import svg_to_labels_auto as quality

    viewbox, objects, walls = align.read_svg(svg_path)
    if not walls:
        raise ValueError("No wall polygons in SVG")

    if hasattr(align, "find_alignment") and hasattr(align, "alignment_matrix"):
        params, error = align.find_alignment(
            image, viewbox, walls, orientation="auto"
        )
        matrix = align.alignment_matrix(viewbox, image.shape, params)
        transformed = [
            {
                **obj,
                "points": align.transform_points(obj["points"], matrix)
            }
            for obj in objects
        ]
        labels = align.extract_annotations(
            transformed, image.shape[1], image.shape[0]
        )

    elif hasattr(align, "align_svg") and hasattr(align, "transform_to_image"):
        params, error, thumb_shape = align.align_svg(
            image, viewbox, walls, orientation="auto"
        )
        transformed = [
            {
                **obj,
                "points": align.transform_to_image(
                    obj["points"], viewbox, image.shape,
                    thumb_shape, params
                )
            }
            for obj in objects
        ]
        labels = align.extract_annotations(
            objects, viewbox, image.shape, thumb_shape, params
        )
    else:
        raise RuntimeError("Unsupported svg_to_labels.py API")

    if not np.isfinite(error) or error > max_error:
        raise ValueError(f"Alignment rejected: {error:.3f}")

    near_edges, ink = quality.create_evidence_maps(image)
    accepted = []
    rejected = 0

    for label in labels:
        points = transformed[label["id"]]["points"]
        check = quality.verify_object(
            {**label, "points": points},
            image.shape, near_edges, ink
        )
        if check["accepted"]:
            accepted.append({
                "class": label["class"],
                "bbox": label["bbox"]
            })
        else:
            rejected += 1

    print(
        f"[ALIGN] error={error:.3f} "
        f"accepted={len(accepted)} rejected={rejected}",
        flush=True
    )
    return accepted, labels



def letterbox_resize(image, size=96):
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Empty image patch")

    scale = min(size / w, size / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    if image.ndim == 2:
        output = np.full((size, size), 255, dtype=np.uint8)
    else:
        output = np.full(
            (size, size, image.shape[2]),
            255,
            dtype=np.uint8
        )

    x = (size - new_w) // 2
    y = (size - new_h) // 2

    output[y:y + new_h, x:x + new_w] = resized
    return output



def process_split(root, split_name, split_items, out_root, patch_size, context_ratio,
                  max_samples=None, backgrounds_per_image=8):
    index_rows = []
    class_counts = defaultdict(int)
    reject_counts = defaultdict(int)

    symbols_dir = os.path.join(out_root, "symbols", split_name)
    ensure_dir(symbols_dir)

    processed = 0

    for attempt_index, item in enumerate(split_items):
        if max_samples is not None and attempt_index >= max_samples:
            break

        sample_dir = resolve_sample_folder(root, item)
        if sample_dir is None:
            reject_counts["missing_sample_dir"] += 1
            continue

        image_path = find_image_file(sample_dir)
        svg_path = find_svg_file(sample_dir)

        if image_path is None:
            reject_counts["missing_image"] += 1
            continue
        if svg_path is None:
            reject_counts["missing_svg"] += 1
            continue

        image = cv2.imread(image_path)
        if image is None:
            reject_counts["unreadable_image"] += 1
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_h, image_w = image.shape[:2]
        sample_id = os.path.basename(sample_dir)

        try:
            objects, raw_annotations = collect_labeled_objects_auto(svg_path, image)
        except Exception as e:
            print(f"[WARN] SVG parse failed for {sample_id}: {e}")
            reject_counts["svg_parse_fail"] += 1
            continue
        

        if len(objects) == 0:
            reject_counts["no_objects"] += 1
            continue

        positive_boxes = [ obj["bbox"] for obj in raw_annotations ]

        per_image_counter = defaultdict(int)
        per_image_hashes = set()

        for obj in objects:
            cls_name = obj["class"]
            bbox = obj["bbox"]

            patch, used_box = crop_with_context(image, bbox, context_ratio=context_ratio)

            if not is_good_patch(patch):
                reject_counts[f"bad_patch_{cls_name}"] += 1
                continue

            resized = letterbox_resize(patch, patch_size)
            patch_hash = sha1_array(resized)

            if patch_hash in per_image_hashes:
                reject_counts[f"duplicate_patch_{cls_name}"] += 1
                continue
            per_image_hashes.add(patch_hash)

            class_dir = os.path.join(symbols_dir, cls_name)
            ensure_dir(class_dir)

            idx = per_image_counter[cls_name]
            per_image_counter[cls_name] += 1

            out_name = f"{sample_id}_{cls_name}_{idx:03d}.png"
            out_path = os.path.join(class_dir, out_name)
            cv2.imwrite(out_path, resized)

            index_rows.append({
                "split": split_name,
                "sample_id": sample_id,
                "class": cls_name,
                "patch_path": os.path.relpath(out_path, out_root).replace("\\", "/"),
                "image_path": os.path.relpath(image_path, out_root).replace("\\", "/")
                              if image_path.startswith(out_root) else image_path,
                "svg_path": os.path.relpath(svg_path, out_root).replace("\\", "/")
                            if svg_path.startswith(out_root) else svg_path,
                "bbox_x1": round(float(bbox[0]), 2),
                "bbox_y1": round(float(bbox[1]), 2),
                "bbox_x2": round(float(bbox[2]), 2),
                "bbox_y2": round(float(bbox[3]), 2),
                "crop_x1": used_box[0],
                "crop_y1": used_box[1],
                "crop_x2": used_box[2],
                "crop_y2": used_box[3],
                "image_w": image_w,
                "image_h": image_h
            })

            class_counts[cls_name] += 1

        # Background patches (hard negatives)
        if backgrounds_per_image > 0 and len(positive_boxes) > 0:
            bg_dir = os.path.join(symbols_dir, "_background")
            ensure_dir(bg_dir)

            bg_boxes = sample_background_boxes(image_w, image_h, positive_boxes, how_many=backgrounds_per_image)

            for i, bg in enumerate(bg_boxes):
                patch, used_box = crop_with_context(image, bg, context_ratio=0.05)
                if not is_good_patch(patch, min_ink_ratio=0.001):
                    continue

                resized = letterbox_resize(patch, patch_size)
                out_name = f"{sample_id}_background_{i:03d}.png"
                out_path = os.path.join(bg_dir, out_name)
                cv2.imwrite(out_path, resized)

                index_rows.append({
                    "split": split_name,
                    "sample_id": sample_id,
                    "class": "_background",
                    "patch_path": os.path.relpath(out_path, out_root).replace("\\", "/"),
                    "image_path": image_path,
                    "svg_path": svg_path,
                    "bbox_x1": round(float(bg[0]), 2),
                    "bbox_y1": round(float(bg[1]), 2),
                    "bbox_x2": round(float(bg[2]), 2),
                    "bbox_y2": round(float(bg[3]), 2),
                    "crop_x1": used_box[0],
                    "crop_y1": used_box[1],
                    "crop_x2": used_box[2],
                    "crop_y2": used_box[3],
                    "image_w": image_w,
                    "image_h": image_h
                })

                class_counts["_background"] += 1

        processed += 1
        if processed % 25 == 0:
            print(f"[{split_name}] processed: {processed}")

    return index_rows, class_counts, reject_counts, processed


def write_index_csv(rows, out_path):
    if not rows:
        return
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build patch dataset for Blueprint Whisperer / BlueprintMatchNet")
    parser.add_argument("--root", required=True, help="Path to cubicasa5k root, e.g. C:/.../cubicasa5k/cubicasa5k")
    parser.add_argument("--out", required=True, help="Output folder, e.g. data/patch_dataset")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--context", type=float, default=0.20, help="Context margin around bbox")
    parser.add_argument("--max-train", type=int, default=100)
    parser.add_argument("--max-val", type=int, default=30)
    parser.add_argument("--max-test", type=int, default=30)
    parser.add_argument("--backgrounds-per-image", type=int, default=6)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    args = parser.parse_args()

    root = os.path.normpath(args.root)
    out_root = os.path.normpath(args.out)

    ensure_dir(out_root)

    split_files = {
        "train": os.path.join(root, "train.txt"),
        "val": os.path.join(root, "val.txt"),
        "test": os.path.join(root, "test.txt"),
    }

    for name, path in split_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing split file: {path}")

    train_items = load_split_list(split_files["train"])
    val_items = load_split_list(split_files["val"])
    test_items = load_split_list(split_files["test"])

    print("========================================")
    print("BLUEPRINT WHISPERER - DATASET BUILDER")
    print("========================================")
    print(f"Root: {root}")
    print(f"Out : {out_root}")
    print(f"Train items: {len(train_items)}")
    print(f"Val items  : {len(val_items)}")
    print(f"Test items : {len(test_items)}")
    print("Classes:", list(CLASS_KEYWORDS.keys()))
    print()

    all_summary = {}
    all_rows = {}

    split_cfg = {
        "train": (train_items, args.max_train),
        "val": (val_items, args.max_val),
        "test": (test_items, args.max_test),
    }

    for split_name, (items, mx) in split_cfg.items():
        rows, class_counts, reject_counts, processed = process_split(
            root=root,
            split_name=split_name,
            split_items=items,
            out_root=out_root,
            patch_size=args.patch_size,
            context_ratio=args.context,
            max_samples=mx,
            backgrounds_per_image=args.backgrounds_per_image,
        )

        all_rows[split_name] = rows

        index_path = os.path.join(out_root, f"index_{split_name}.csv")
        write_index_csv(rows, index_path)

        all_summary[split_name] = {
            "processed_samples": processed,
            "patches_total": len(rows),
            "class_counts": dict(class_counts),
            "reject_counts": dict(reject_counts),
        }

        print(f"[{split_name}] processed_samples = {processed}")
        print(f"[{split_name}] patches_total    = {len(rows)}")
        print(f"[{split_name}] class_counts     = {dict(class_counts)}")
        print(f"[{split_name}] reject_counts    = {dict(reject_counts)}")
        print()

    # Pairs only for train + val
    pair_train_count = build_pairs_for_split(
        all_rows["train"],
        os.path.join(out_root, "pairs_train.csv"),
        negatives_per_positive=args.negatives_per_positive,
    )
    pair_val_count = build_pairs_for_split(
        all_rows["val"],
        os.path.join(out_root, "pairs_val.csv"),
        negatives_per_positive=args.negatives_per_positive,
    )

    all_summary["pairs"] = {
        "train_pairs": pair_train_count,
        "val_pairs": pair_val_count
    }

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("DONE.")
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()