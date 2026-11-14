
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return image


def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        raise IOError(f"Cannot save: {path}")

    path.write_bytes(encoded.tobytes())


def rotate_image(image, degrees):
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    if degrees not in rotations:
        raise ValueError("Invalid rotation")

    if degrees == 0:
        return image.copy()

    return cv2.rotate(image, rotations[degrees])


def to_grayscale(image):
    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def analyze_image(gray):
    small = gray

    height, width = gray.shape
    scale = min(1.0, 1200.0 / max(width, height))

    if scale < 1.0:
        small = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    p1, p99 = np.percentile(small, [1, 99])

    residual = cv2.absdiff(
        small,
        cv2.medianBlur(small, 3),
    )

    return {
        "width": width,
        "height": height,
        "contrast": float(p99 - p1),
        "brightness": float(np.mean(small)),
        "noise_estimate": float(np.mean(residual)),
    }


def select_profile(metrics):
    if metrics["contrast"] < 95:
        return "clahe"

    if metrics["noise_estimate"] > 12:
        return "denoise"

    return "original"


def apply_clahe(gray):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def create_variants(gray):
    return {
        "original": gray.copy(),
        "clahe": apply_clahe(gray),
        "denoise": cv2.medianBlur(gray, 3),
    }


def diagnostic_mask(gray):
    """
    Binary ink mask for geometric analysis.

    Supports dark lines on light backgrounds and
    light lines on dark backgrounds.
    """
    background = float(np.median(gray))

    if background < 120:
        work = cv2.bitwise_not(gray)
    else:
        work = gray

    blurred = cv2.GaussianBlur(work, (3, 3), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    return mask


def detect_wall_lines(gray):
    """
    Extract long horizontal and vertical strokes.

    No colour assumptions.
    These are wall candidates, not semantic wall labels.
    """
    mask = diagnostic_mask(gray)

    height, width = gray.shape

    horizontal_length = max(
        11, int(width * 0.012)
    )
    vertical_length = max(
        11, int(height * 0.012)
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length),
    )

    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    return cv2.bitwise_or(horizontal, vertical)


def boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return (
        min(ax2, bx2) > max(ax1, bx1)
        and min(ay2, by2) > max(ay1, by1)
    )


def merge_overlapping_boxes(boxes):
    """
    Merge overlapping candidate boxes.

    Separate floor plans remain separate when their
    bounding rectangles do not intersect.
    """
    boxes = [list(map(int, box)) for box in boxes]

    changed = True

    while changed:
        changed = False
        result = []

        while boxes:
            current = boxes.pop(0)
            remaining = []

            for other in boxes:
                if boxes_intersect(current, other):
                    current = [
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    ]
                    changed = True
                else:
                    remaining.append(other)

            result.append(current)
            boxes = remaining

        boxes = result

    return boxes


def detect_floorplan_regions(
    gray,
    margin=24,
    max_regions=12,
):
    """
    Find large connected groups of candidate wall lines.

    Returns ROIs in full-image pixel coordinates.
    Falls back to the whole image if detection is uncertain.
    """
    height, width = gray.shape

    fallback = [{
        "id": "roi_000",
        "bbox": [0, 0, width, height],
        "fallback": True,
    }]

    lines = detect_wall_lines(gray)

    connection_size = max(
        9,
        min(41, int(min(width, height) * 0.018)),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (connection_size, connection_size),
    )

    connected = cv2.morphologyEx(
        lines,
        cv2.MORPH_CLOSE,
        kernel,
    )

    connected = cv2.dilate(
        connected,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            connected,
            connectivity=8,
        )
    )

    candidates = []

    for component_id in range(1, count):
        x, y, w, h, area = map(
            int, stats[component_id]
        )

        if w < max(40, int(width * 0.09)):
            continue

        if h < max(40, int(height * 0.09)):
            continue

        if w * h < width * height * 0.008:
            continue

        # Find actual line pixels belonging to this component.
        component = (
            labels[y:y + h, x:x + w] == component_id
        )

        line_crop = lines[y:y + h, x:x + w] > 0

        yy, xx = np.where(component & line_crop)

        if len(xx) < 80:
            continue

        x1 = x + int(xx.min())
        y1 = y + int(yy.min())
        x2 = x + int(xx.max()) + 1
        y2 = y + int(yy.max()) + 1

        candidates.append([x1, y1, x2, y2])

    if not candidates:
        return fallback

    candidates = merge_overlapping_boxes(candidates)

    candidates.sort(
        key=lambda b: (
            (b[2] - b[0]) * (b[3] - b[1])
        ),
        reverse=True,
    )

    largest_area = (
        (candidates[0][2] - candidates[0][0])
        * (candidates[0][3] - candidates[0][1])
    )

    candidates = [
        box for box in candidates
        if (
            (box[2] - box[0])
            * (box[3] - box[1])
            >= largest_area * 0.12
        )
    ]

    if not candidates or len(candidates) > max_regions:
        return fallback

    expanded = []

    for x1, y1, x2, y2 in candidates:
        extra = max(
            int(margin),
            int(0.012 * max(x2 - x1, y2 - y1)),
        )

        expanded.append([
            max(0, x1 - extra),
            max(0, y1 - extra),
            min(width, x2 + extra),
            min(height, y2 + extra),
        ])

    expanded = merge_overlapping_boxes(expanded)

    # Check how many wall-line pixels the ROIs retain.
    coverage = np.zeros_like(lines)

    for x1, y1, x2, y2 in expanded:
        coverage[y1:y2, x1:x2] = 255

    total = cv2.countNonZero(lines)

    retained = cv2.countNonZero(
        cv2.bitwise_and(lines, coverage)
    )

    if total == 0 or retained / total < 0.70:
        return fallback

    expanded.sort(key=lambda b: (b[1], b[0]))

    return [
        {
            "id": f"roi_{i:03d}",
            "bbox": box,
            "fallback": False,
        }
        for i, box in enumerate(expanded)
    ]


def crop_region(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def axis_positions(length, tile_size, overlap):
    if tile_size <= 0:
        raise ValueError("Tile size must be positive")

    if not 0 <= overlap < tile_size:
        raise ValueError("Invalid overlap")

    if length <= tile_size:
        return [0]

    stride = tile_size - overlap

    positions = list(
        range(0, length - tile_size + 1, stride)
    )

    last = length - tile_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def generate_tiles(image, tile_size=512, overlap=96):
    height, width = image.shape[:2]

    xs = axis_positions(width, tile_size, overlap)
    ys = axis_positions(height, tile_size, overlap)

    tile_id = 0

    for y in ys:
        for x in xs:
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)

            crop = image[y:y2, x:x2]

            valid_height, valid_width = crop.shape[:2]

            pad_bottom = tile_size - valid_height
            pad_right = tile_size - valid_width

            if pad_bottom or pad_right:
                tile = cv2.copyMakeBorder(
                    crop,
                    0, pad_bottom,
                    0, pad_right,
                    cv2.BORDER_CONSTANT,
                    value=255,
                )
            else:
                tile = crop.copy()

            yield {
                "id": tile_id,
                "x": int(x),
                "y": int(y),
                "x2": int(x2),
                "y2": int(y2),
                "valid_width": int(valid_width),
                "valid_height": int(valid_height),
                "tile_size": int(tile_size),
            }, tile

            tile_id += 1


def draw_review(image, regions, tiles):
    preview = image.copy()

    if preview.ndim == 2:
        preview = cv2.cvtColor(
            preview, cv2.COLOR_GRAY2BGR
        )

    # Pink: tiles.
    for tile in tiles:
        cv2.rectangle(
            preview,
            (tile["x"], tile["y"]),
            (tile["x2"] - 1, tile["y2"] - 1),
            (215, 65, 230),
            1,
        )

    # Blue: region crops.
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (255, 140, 35),
            3,
        )

        cv2.putText(
            preview,
            region["id"],
            (x1 + 5, min(height_of(preview) - 8, y1 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 140, 35),
            2,
        )

    return preview


def height_of(image):
    return image.shape[0]


def process_image(
    input_path,
    output_dir,
    profile="auto",
    tile_size=512,
    overlap=96,
    rotation=0,
    margin=24,
    regions_mode="auto",
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_image(input_path)
    oriented = rotate_image(source, rotation)

    # Every image goes through grayscale.
    gray = to_grayscale(oriented)

    metrics = analyze_image(gray)
    variants = create_variants(gray)

    selected_profile = (
        select_profile(metrics)
        if profile == "auto"
        else profile
    )

    processed = variants[selected_profile]

    if regions_mode == "full":
        height, width = gray.shape

        regions = [{
            "id": "roi_000",
            "bbox": [0, 0, width, height],
            "fallback": True,
        }]
    else:
        regions = detect_floorplan_regions(
            gray,
            margin=margin,
        )

    save_png(output_dir / "original_color.png", oriented)
    save_png(output_dir / "original.png", gray)

    for name, variant in variants.items():
        save_png(
            output_dir / f"variant_{name}.png",
            variant,
        )

    mask = diagnostic_mask(gray)
    wall_lines = detect_wall_lines(gray)

    save_png(
        output_dir / "diagnostic_mask.png",
        mask,
    )

    save_png(
        output_dir / "wall_candidates.png",
        wall_lines,
    )

    tile_metadata = []
    global_tile_id = 0

    for region in regions:
        region_id = region["id"]
        x1, y1, x2, y2 = region["bbox"]

        region_folder = (
            output_dir / "regions" / region_id
        )

        original_crop = crop_region(
            gray, region["bbox"]
        )

        processed_crop = crop_region(
            processed, region["bbox"]
        )

        save_png(
            region_folder / "original.png",
            original_crop,
        )

        save_png(
            region_folder / "processed.png",
            processed_crop,
        )

        region_tile_count = 0

        for tile_info, tile in generate_tiles(
            processed_crop,
            tile_size=tile_size,
            overlap=overlap,
        ):
            filename = (
                f"tile_{global_tile_id:05d}.png"
            )

            save_png(
                output_dir / "tiles" / filename,
                tile,
            )

            local_x = tile_info["x"]
            local_y = tile_info["y"]

            tile_info.update({
                "id": global_tile_id,
                "region_id": region_id,
                "local_x": local_x,
                "local_y": local_y,
                "x": x1 + local_x,
                "y": y1 + local_y,
                "x2": x1 + tile_info["x2"],
                "y2": y1 + tile_info["y2"],
                "file": f"tiles/{filename}",
            })

            tile_metadata.append(tile_info)

            global_tile_id += 1
            region_tile_count += 1

        region["tile_count"] = region_tile_count

    preview = draw_review(
        oriented,
        regions,
        tile_metadata,
    )

    save_png(
        output_dir / "tile_grid.png",
        preview,
    )

    manifest = {
        "source_path": str(input_path.resolve()),
        "source_width": int(source.shape[1]),
        "source_height": int(source.shape[0]),
        "rotation_clockwise": rotation,
        "coordinate_space": "oriented_original_pixels",
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "colour_mode": "grayscale",
        "requested_profile": profile,
        "selected_profile": selected_profile,
        "image_metrics": metrics,
        "regions_mode": regions_mode,
        "margin_pixels": margin,
        "regions": regions,
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(tile_metadata),
        "tiles": tile_metadata,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nBLUEPRINT WHISPERER")
    print("---------------------------")
    print(f"Image: {input_path.name}")
    print(f"Size: {gray.shape[1]}x{gray.shape[0]}")
    print(f"Profile: {selected_profile}")
    print(f"Regions: {len(regions)}")
    print(f"Tiles: {len(tile_metadata)}")
    print(f"Output: {output_dir.resolve()}")

    if any(r["fallback"] for r in regions):
        print(
            "NOTE: Full-image fallback was used. "
            "Review the crop before training."
        )

    return manifest



def grayscale(image):
    return to_grayscale(image)


def prepare_gray(image, profile="auto"):
    gray = to_grayscale(image)
    selected = (
        select_profile(analyze_image(gray))
        if profile == "auto" else profile
    )
    if selected == "original":
        result = gray.copy()
    elif selected == "clahe":
        result = apply_clahe(gray)
    elif selected == "denoise":
        result = cv2.medianBlur(gray, 3)
    else:
        raise ValueError(f"Unknown profile: {selected}")
    return result, selected


def main():
    parser = argparse.ArgumentParser(
        description="Blueprint Whisperer preprocessing"
    )

    parser.add_argument("image", type=Path)

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--profile",
        choices=[
            "auto",
            "original",
            "clahe",
            "denoise",
        ],
        default="auto",
    )

    parser.add_argument(
        "--tile", type=int, default=512
    )

    parser.add_argument(
        "--overlap", type=int, default=96
    )

    parser.add_argument(
        "--margin", type=int, default=24
    )

    parser.add_argument(
        "--regions",
        choices=["auto", "full"],
        default="auto",
    )

    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
    )

    args = parser.parse_args()

    process_image(
        input_path=args.image,
        output_dir=args.out,
        profile=args.profile,
        tile_size=args.tile,
        overlap=args.overlap,
        rotation=args.rotate,
        margin=args.margin,
        regions_mode=args.regions,
    )


if __name__ == "__main__":
    main()
