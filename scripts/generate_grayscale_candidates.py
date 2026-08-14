"""Generate reviewable grayscale line-candidate overlays for annotation tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.grayscale import GrayscaleCandidateSettings, grayscale_line_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="annotation_package/index.json")
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/annotation_package/candidates"))
    return parser.parse_args()


def main():
    try:
        import numpy as np
        import rasterio
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise SystemExit(f"Missing candidate-generation dependency: {error.name}") from error

    args = parse_args()
    index_path = args.index if args.index.is_absolute() else REPOSITORY / args.index
    index = json.loads(index_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPOSITORY / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = GrayscaleCandidateSettings()
    result_index = {"version": "1", "backend": "grayscale_ridge_v1", "settings": settings.__dict__, "tiles": []}
    contact_tiles = []

    for tile in index["tiles"]:
        raster_path = Path(tile["raster_path"])
        if not raster_path.is_absolute():
            raster_path = REPOSITORY / raster_path
        with rasterio.open(raster_path) as source:
            gray = source.read(1)
            result = grayscale_line_candidates(gray, settings)
            rgba = np.zeros((4, source.height, source.width), dtype=np.uint8)
            rgba[0] = 225
            rgba[1] = 29
            rgba[2] = 72
            rgba[3] = np.where(result.candidate_mask, np.maximum(result.likelihood, 96), 0)
            output_path = output_dir / f"{tile['tile_id']}-candidates.tif"
            profile = source.profile.copy()
            profile.update(driver="GTiff", count=4, dtype="uint8", nodata=None, compress="deflate", tiled=True, photometric="RGB")
            with rasterio.open(output_path, "w", **profile) as destination:
                destination.write(rgba)
                destination.update_tags(
                    backend="grayscale_ridge_v1",
                    candidate_kind="review_only_linework",
                    ink_fraction=result.ink_fraction,
                    candidate_fraction=result.candidate_fraction,
                    ridge_threshold=result.ridge_threshold,
                )
            preview = np.repeat(gray[..., None], 3, axis=2)
            preview[result.candidate_mask] = (preview[result.candidate_mask] * 0.35 + np.array((225, 29, 72)) * 0.65).astype(np.uint8)
            contact_tiles.append((tile, Image.fromarray(preview).resize((384, 384))))
        result_index["tiles"].append(
            {
                "tile_id": tile["tile_id"],
                "sheet_id": tile["sheet_id"],
                "split": tile["split"],
                "candidate_raster_path": str(output_path.relative_to(REPOSITORY)).replace("\\", "/"),
                "ink_fraction": result.ink_fraction,
                "candidate_fraction": result.candidate_fraction,
                "ridge_threshold": result.ridge_threshold,
            }
        )
        print(output_path)
    index_output = output_dir / "candidate_index.json"
    index_output.write_text(json.dumps(result_index, ensure_ascii=False, indent=2), encoding="utf-8")
    contact = Image.new("RGB", (4 * 384, 3 * 420), "white")
    draw = ImageDraw.Draw(contact)
    for index, (tile, preview) in enumerate(contact_tiles):
        column, row = index % 4, index // 4
        left, top = column * 384, row * 420
        contact.paste(preview, (left, top))
        draw.text((left + 6, top + 390), f"{tile['tile_id']} [{tile['split']}]", fill="black")
    contact_output = output_dir / "candidate_contact_sheet.png"
    contact.save(contact_output)
    print(index_output)
    print(contact_output)


if __name__ == "__main__":
    main()
