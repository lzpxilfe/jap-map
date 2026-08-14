"""Build a georeferenced visual check from a connected-sheet pilot manifest.

This developer utility is intentionally outside the QGIS plugin runtime. It
requires OpenCV, NumPy, Pillow, and Rasterio and writes only ignored derived
files. Registration JSON remains the authoritative portable artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Pilot manifest JSON")
    parser.add_argument("--output", type=Path, default=Path("data/derived/registered_pilot_preview.tif"))
    parser.add_argument("--width", type=int, default=3200, help="Output mosaic width in pixels")
    return parser.parse_args()


def main():
    try:
        import cv2
        import numpy as np
        import rasterio
        from PIL import Image
        from rasterio.transform import from_bounds
    except ImportError as error:
        raise SystemExit(f"Missing preview dependency: {error.name}") from error

    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else repository / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for sheet in manifest["sheets"]:
        registration_path = Path(sheet["registration_path"])
        if not registration_path.is_absolute():
            registration_path = repository / registration_path
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        image_path = Path(sheet["image_path"])
        if not image_path.is_absolute():
            image_path = repository / image_path
        if not image_path.exists():
            raise SystemExit(f"Raw scan not found: {image_path}")
        records.append((sheet, registration, image_path))

    crs_values = {registration["crs_authid"] for _, registration, _ in records}
    if len(crs_values) != 1:
        raise SystemExit("All sheets in one preview must use the same CRS")
    map_points = [point for _, registration, _ in records for point in registration["gcps"]]
    west, east = min(point["map_x"] for point in map_points), max(point["map_x"] for point in map_points)
    south, north = min(point["map_y"] for point in map_points), max(point["map_y"] for point in map_points)
    mean_latitude = (south + north) / 2
    output_width = args.width
    x_resolution = (east - west) / output_width
    y_resolution = x_resolution * math.cos(math.radians(mean_latitude))
    output_height = round((north - south) / y_resolution)
    mosaic = np.full((output_height, output_width), 255, dtype=np.uint8)

    for sheet, registration, image_path in records:
        image = np.asarray(Image.open(image_path).convert("L"))
        source = np.array([[point["pixel_x"], point["pixel_y"]] for point in registration["gcps"]], dtype=np.float32)
        destination = np.array(
            [
                [(point["map_x"] - west) / (east - west) * (output_width - 1), (north - point["map_y"]) / (north - south) * (output_height - 1)]
                for point in registration["gcps"]
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(image, transform, (output_width, output_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        neatline_mask = np.zeros((output_height, output_width), dtype=np.uint8)
        cv2.fillConvexPoly(neatline_mask, np.round(destination).astype(np.int32), 255)
        warped[neatline_mask == 0] = 255
        mosaic = np.minimum(mosaic, warped)
        print(f"Warped {sheet['sheet_id']}")

    output = args.output if args.output.is_absolute() else repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(west, south, east, north, output_width, output_height)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=output_width,
        height=output_height,
        count=1,
        dtype="uint8",
        crs=next(iter(crs_values)),
        transform=transform,
        nodata=255,
        compress="deflate",
        tiled=True,
    ) as dataset:
        dataset.write(mosaic, 1)
    png_path = output.with_suffix(".png")
    Image.fromarray(mosaic).save(png_path)
    print(output)
    print(png_path)


if __name__ == "__main__":
    main()
