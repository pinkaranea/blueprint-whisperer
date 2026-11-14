
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    """Read images, including Windows paths with Unicode characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return image


def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise IOError(f"Cannot encode image: {path}")

    path.write_bytes(encoded.tobytes())


def rotate_image(image, degrees):
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    if degrees not in rotations:
        raise ValueError("Rotation must be 0, 90, 180 or 270")

    if degrees == 0:
        return image.copy()

    return cv2.rotate(image, rotations[degrees])


def analyze_image(image):
    """
    Estimate image properties.
    Heuristics are initial rules, not a learned quality model.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Analyze a smaller copy to limit computation.
    scale = min(1.0, 1200.0 / max(width, height))

    if scale < 1.0:
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

    p1, p99 = np.percentile(gray, [1, 99])
    contrast = float(p99 - p1)

    median = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, median)
    noise = float(np.mean(residual))

    brightness = float(np.mean(gray))
    dark_fraction = float(np.mean(gray < 180))

    return {
        "contrast": round(contrast, 3),
        "noise_estimate": round(noise, 3),
        "brightness": round(brightness, 3),
        "dark_fraction": round(dark_fraction, 4),
        "width": width,
        "height": height,
    }


def select_profile(metrics):
    """
    Initial deterministic routing policy.
    Thresholds must be validated on held-out floor plans.
    """
    if metrics["contrast"] < 95:
        return "clahe"

    if metrics["noise_estimate"] > 12:
        return "denoise"

    return "original"


def apply_clahe(image):
    """
    CLAHE on LAB luminance preserves colour information
    better than replacing the RGB image with a binary mask.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    improved_l = clahe.apply(l_channel)

    improved_lab = cv2.merge(
        (improved_l, a_channel, b_channel)
    )

    return cv2.cvtColor(
        improved_lab, cv2.COLOR_LAB2BGR
    )


def create_variants(image):
    """
    The original is always retained.
    Denoising may remove thin lines, so it is optional.
    """
    return {
        "original": image.copy(),
        "clahe": apply_clahe(image),
        "denoise": cv2.medianBlur(image, 3),
    }


def diagnostic_mask(image):
    """
    Diagnostic binary mask only.
    Do not feed it to the detector unless explicitly tested.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Highlight strokes relative to dominant background.
    background = float(np.median(gray))

    mode = (
        cv2.THRESH_BINARY_INV
        if background >= 127
        else cv2.THRESH_BINARY
    )

    _, mask = cv2.threshold(
        gray, 0, 255, mode | cv2.THRESH_OTSU
    )

    return mask


def axis_positions(length, tile_size, overlap):
    """
    Generate tile origins that cover the entire axis.
    Include the rightmost/bottommost boundary.
    """
    if tile_size <= 0:
        raise ValueError("Tile size must be positive")

    if not 0 <= overlap < tile_size:
        raise ValueError(
            "Overlap must satisfy 0 <= overlap < tile_size"
        )

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
    """
    Yield (metadata, tile).

    Every tile uses coordinates from the full oriented image.
    Smaller boundary tiles are padded to a fixed input size.
    """
    height, width = image.shape[:2]

    xs = axis_positions(width, tile_size, overlap)
    ys = axis_positions(height, tile_size, overlap)

    tile_id = 0

    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)

            crop = image[y:y2, x:x2]

            valid_h, valid_w = crop.shape[:2]
            pad_right = tile_size - valid_w
            pad_bottom = tile_size - valid_h

            if pad_right or pad_bottom:
                tile = cv2.copyMakeBorder(
                    crop,
                    0, pad_bottom,
                    0, pad_right,
                    cv2.BORDER_REPLICATE
                )
            else:
                tile = crop.copy()

            metadata = {
                "id": tile_id,
                "x": int(x),
                "y": int(y),
                "x2": int(x2),
                "y2": int(y2),
                "valid_width": int(valid_w),
                "valid_height": int(valid_h),
                "tile_size": int(tile_size),
            }

            yield metadata, tile
            tile_id += 1


def draw_tile_grid(image, tiles):
    preview = image.copy()

    for tile in tiles:
        x1, y1 = tile["x"], tile["y"]
        x2, y2 = tile["x2"], tile["y2"]

        cv2.rectangle(
            preview,
            (x1, y1), (x2 - 1, y2 - 1),
            (210, 40, 235), 2
        )

        cv2.putText(
            preview,
            str(tile["id"]),
            (x1 + 8, min(y1 + 25, y2 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (210, 40, 235), 2
        )

    return preview


def process_image(
    input_path,
    output_dir,
    profile="auto",
    tile_size=512,
    overlap=96,
    rotation=0,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and orient document.
    source = load_image(input_path)
    oriented = rotate_image(source, rotation)

    # 2. Analyze the document.
    metrics = analyze_image(oriented)

    # 3. Prepare different variants.
    variants = create_variants(oriented)

    selected_profile = (
        select_profile(metrics)
        if profile == "auto"
        else profile
    )

    if selected_profile not in variants:
        raise ValueError(f"Unknown profile: {selected_profile}")

    processed = variants[selected_profile]

    # 4. Save original and variants.
    save_png(output_dir / "original.png", oriented)

    for name, variant in variants.items():
        save_png(
            output_dir / f"variant_{name}.png",
            variant
        )

    save_png(
        output_dir / "diagnostic_mask.png",
        diagnostic_mask(processed)
    )

    # 5. Generate tiles with global coordinates.
    tile_metadata = []

    for metadata, tile in generate_tiles(
        processed,
        tile_size=tile_size,
        overlap=overlap
    ):
        filename = f"tile_{metadata['id']:05d}.png"

        save_png(tiles_dir / filename, tile)

        metadata["file"] = (
            f"tiles/{filename}"
        )

        tile_metadata.append(metadata)

    # 6. Produce evidence preview.
    grid = draw_tile_grid(
        oriented, tile_metadata
    )

    save_png(
        output_dir / "tile_grid.png",
        grid
    )

    # 7. Store transformations for later detection fusion.
    manifest = {
        "source_path": str(input_path.resolve()),
        "source_width": int(source.shape[1]),
        "source_height": int(source.shape[0]),
        "rotation_clockwise": rotation,
        "coordinate_space": "oriented_original_pixels",
        "width": int(processed.shape[1]),
        "height": int(processed.shape[0]),
        "requested_profile": profile,
        "selected_profile": selected_profile,
        "image_metrics": metrics,
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(tile_metadata),
        "tiles": tile_metadata,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nBLUEPRINT WHISPERER - PREPROCESSING")
    print("--------------------------------")
    print(f"Input:       {input_path.name}")
    print(
        f"Dimensions:  "
        f"{processed.shape[1]}x{processed.shape[0]}"
    )
    print(f"Profile:     {selected_profile}")
    print(f"Contrast:    {metrics['contrast']}")
    print(f"Noise:       {metrics['noise_estimate']}")
    print(f"Tiles:       {len(tile_metadata)}")
    print(f"Output:      {output_dir.resolve()}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Blueprint Whisperer preprocessing"
    )

    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/processed")
    )
    parser.add_argument(
        "--profile",
        choices=["auto", "original", "clahe", "denoise"],
        default="auto"
    )
    parser.add_argument(
        "--tile", type=int, default=512
    )
    parser.add_argument(
        "--overlap", type=int, default=96
    )
    parser.add_argument(
        "--rotate", type=int,
        choices=[0, 90, 180, 270],
        default=0
    )

    args = parser.parse_args()

    process_image(
        input_path=args.image,
        output_dir=args.out,
        profile=args.profile,
        tile_size=args.tile,
        overlap=args.overlap,
        rotation=args.rotate
    )


if __name__ == "__main__":
    main()
