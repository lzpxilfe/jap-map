"""Create ignored, georeferenced annotation tiles from pilot specifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from histcontour_core.registration import SheetRegistration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tile_spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/annotation_package"))
    return parser.parse_args()


def resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def main():
    try:
        import cv2
        import numpy as np
        import rasterio
        from PIL import Image, ImageDraw
        from rasterio.transform import from_bounds
    except ImportError as error:
        raise SystemExit(f"Missing annotation dependency: {error.name}") from error

    args = parse_args()
    repository = REPOSITORY
    manifest = json.loads(resolve(repository, args.manifest).read_text(encoding="utf-8"))
    tile_spec = json.loads(resolve(repository, args.tile_spec).read_text(encoding="utf-8"))
    output_dir = resolve(repository, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = {sheet["sheet_id"]: sheet for sheet in manifest["sheets"]}
    contact_tiles = []
    generated_index = {"version": "1", "source_tile_spec": str(args.tile_spec), "tiles": []}

    for tile in tile_spec["tiles"]:
        sheet = sheets[tile["sheet_id"]]
        registration = SheetRegistration.read_json(resolve(repository, sheet["registration_path"]))
        image = Image.open(resolve(repository, sheet["image_path"])).convert("L")
        x, y, width, height = tile["pixel_bounds"]
        if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
            raise SystemExit(f"Tile falls outside its scan: {tile['tile_id']}")
        crop = np.asarray(image.crop((x, y, x + width, y + height)))
        source = np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype=np.float32)
        map_corners = tuple(registration.transform(pixel_x, pixel_y) for pixel_x, pixel_y in ((x, y), (x + width - 1, y), (x + width - 1, y + height - 1), (x, y + height - 1)))
        west, east = min(point.x for point in map_corners), max(point.x for point in map_corners)
        south, north = min(point.y for point in map_corners), max(point.y for point in map_corners)
        destination = np.array(tuple(((point.x - west) / (east - west) * (width - 1), (north - point.y) / (north - south) * (height - 1)) for point in map_corners), dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(crop, transform_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        output_path = output_dir / f"{tile['tile_id']}.tif"
        with rasterio.open(output_path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8", crs=registration.crs_authid, transform=from_bounds(west, south, east, north, width, height), nodata=255, compress="deflate", tiled=True) as dataset:
            dataset.write(warped, 1)
            dataset.update_tags(sheet_id=tile["sheet_id"], tile_id=tile["tile_id"], split=tile["split"], scene_type=tile["scene_type"])
        thumbnail = Image.fromarray(warped).resize((384, 384))
        contact_tiles.append((tile, thumbnail))
        generated_index["tiles"].append({**tile, "raster_path": str(output_path.relative_to(repository)).replace("\\", "/"), "bounds": [west, south, east, north], "crs_authid": registration.crs_authid})
        print(output_path)

    contact_width, contact_height = 4 * 384, 3 * 420
    contact = Image.new("L", (contact_width, contact_height), 255)
    draw = ImageDraw.Draw(contact)
    for index, (tile, thumbnail) in enumerate(contact_tiles):
        column, row = index % 4, index // 4
        left, top = column * 384, row * 420
        contact.paste(thumbnail, (left, top))
        draw.rectangle((left, top + 384, left + 383, top + 419), fill=255)
        draw.text((left + 6, top + 390), f"{tile['tile_id']} [{tile['split']}]", fill=0)
    contact.save(output_dir / "contact_sheet.png")
    (output_dir / "index.json").write_text(json.dumps(generated_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir / "contact_sheet.png")
    print(output_dir / "index.json")


if __name__ == "__main__":
    main()
