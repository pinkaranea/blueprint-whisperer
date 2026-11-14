import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def grayscale(image):
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rotate_image(image, degrees):
    codes = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if degrees not in codes:
        raise ValueError("Rotation must be 0/90/180/270")
    return image.copy() if degrees == 0 else cv2.rotate(image, codes[degrees])


def image_metrics(image):
    gray = grayscale(image)
    h, w = gray.shape
    scale = min(1.0, 1200.0 / max(h, w))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    p1, p99 = np.percentile(gray, (1, 99))
    residual = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
    return {
        "contrast": round(float(p99-p1), 3),
        "noise_estimate": round(float(residual.mean()), 3),
        "brightness": round(float(gray.mean()), 3),
        "dark_fraction": round(float((gray < 180).mean()), 5)
    }


def prepare_gray(image, profile="auto"):
    """
    Shared training/inference preprocessing.
    Returns (uint8 grayscale image, selected profile).
    """
    gray = grayscale(image)
    metrics = image_metrics(gray)
    selected = profile
    if profile == "auto":
        if metrics["contrast"] < 95:
            selected = "clahe"
        elif metrics["noise_estimate"] > 12:
            selected = "denoise"
        else:
            selected = "original"

    if selected == "original":
        processed = gray.copy()
    elif selected == "clahe":
        processed = cv2.createCLAHE(clipLimit=2.0,
                                   tileGridSize=(8, 8)).apply(gray)
    elif selected == "denoise":
        processed = cv2.medianBlur(gray, 3)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    return processed, selected


def diagnostic_mask(gray):
    gray = grayscale(gray)
    source = cv2.bitwise_not(gray) if np.median(gray) < 127 else gray
    source = cv2.GaussianBlur(source, (3, 3), 0)
    _, mask = cv2.threshold(source, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask


def axis_positions(length, tile_size, overlap):
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    if length <= tile_size:
        return [0]
    result = list(range(0, length-tile_size+1, tile_size-overlap))
    if result[-1] != length-tile_size:
        result.append(length-tile_size)
    return result


def generate_tiles(image, tile_size=512, overlap=96, origin=(0, 0)):
    """
    Tiles in original oriented-image coordinates.
    origin=(x_crop,y_crop) supports optional manual cropping.
    """
    h, w = image.shape[:2]
    item_id = 0
    for y in axis_positions(h, tile_size, overlap):
        for x in axis_positions(w, tile_size, overlap):
            x2, y2 = min(w, x+tile_size), min(h, y+tile_size)
            patch = image[y:y2, x:x2]
            valid_h, valid_w = patch.shape[:2]
            patch = cv2.copyMakeBorder(
                patch, 0, tile_size-valid_h,
                0, tile_size-valid_w,
                cv2.BORDER_CONSTANT, value=255
            )
            metadata = {
                "id": item_id,
                "local_x": x,
                "local_y": y,
                "x": origin[0]+x,
                "y": origin[1]+y,
                "x2": origin[0]+x2,
                "y2": origin[1]+y2,
                "valid_width": valid_w,
                "valid_height": valid_h,
                "tile_size": tile_size
            }
            yield metadata, patch
            item_id += 1


def process_image(input_path, output_dir, profile="auto",
                  tile_size=512, overlap=96, rotation=0, crop=None):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_image(input_path)
    oriented = rotate_image(source, rotation)
    gray = grayscale(oriented)
    processed, selected = prepare_gray(gray, profile)
    metrics = image_metrics(gray)

    h, w = gray.shape
    if crop is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = map(int, crop)
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError("Crop must be inside the oriented image")

    work = processed[y1:y2, x1:x2]
    save_png(output_dir/"original_color.png", oriented)
    save_png(output_dir/"gray.png", gray)
    save_png(output_dir/"processed.png", processed)
    save_png(output_dir/"processed_crop.png", work)
    save_png(output_dir/"diagnostic_mask.png", diagnostic_mask(gray))

    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tile_metadata = []
    for meta, tile in generate_tiles(
        work, tile_size=tile_size, overlap=overlap, origin=(x1, y1)
    ):
        name = f"tiles/tile_{meta['id']:05d}.png"
        save_png(output_dir/name, tile)
        meta["file"] = name
        tile_metadata.append(meta)
        cv2.rectangle(preview, (meta["x"], meta["y"]),
                      (meta["x2"]-1, meta["y2"]-1), (220, 50, 235), 1)

    cv2.rectangle(preview, (x1, y1), (x2-1, y2-1), (255, 100, 30), 2)
    save_png(output_dir/"tile_grid.png", preview)

    manifest = {
        "version": "blueprint-whisperer-siamese-prep-v1",
        "source_path": str(input_path.resolve()),
        "source_size": [int(source.shape[1]), int(source.shape[0])],
        "oriented_size": [w, h],
        "rotation_clockwise": rotation,
        "coordinate_space": "oriented_source_pixels",
        "image_mode": "grayscale_uint8",
        "crop_bbox": [x1, y1, x2, y2],
        "profile_requested": profile,
        "profile_selected": selected,
        "image_metrics": metrics,
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(tile_metadata),
        "tiles": tile_metadata
    }
    (output_dir/"manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("BLUEPRINT WHISPERER PREP")
    print(f"Image: {w}x{h} | profile: {selected}")
    print(f"Crop: {[x1, y1, x2, y2]}")
    print(f"Tiles: {len(tile_metadata)}")
    print(f"Output: {output_dir.resolve()}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Blueprint Whisperer grayscale preprocessing")
    p.add_argument("image", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--profile", choices=["auto", "original", "clahe", "denoise"],
                   default="auto")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=96)
    p.add_argument("--rotate", type=int, choices=[0,90,180,270], default=0)
    p.add_argument("--crop", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"))
    a = p.parse_args()
    process_image(a.image, a.out, a.profile, a.tile, a.overlap, a.rotate, a.crop)


if __name__ == "__main__":
    main()
