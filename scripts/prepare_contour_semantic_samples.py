#!/usr/bin/env python3
"""Freeze a score-blind, length-stratified sample of actual development lines.

The pages show source ink and a target line, never model predictions. Labels
are left blank; later AI interpretation must remain separate from human truth.
Existing annotation GeoPackages and held-out rasters are never opened.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.provenance import sha256_file
from scripts.compare_contour_vector_outputs import pixel_proposals


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sample_indices(features, *, seed, per_bin):
    if type(per_bin) is not int or per_bin < 1:
        raise ValueError("per_bin must be a positive integer")
    buckets = {"short": [], "medium": [], "long": []}
    identities = set()
    for index, feature in enumerate(features):
        length = feature["properties"]["pixel_length"]
        if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length < 18:
            raise ValueError("this predeclared sample requires finite candidate lengths of at least 18 native pixels")
        bucket = "short" if length < 28 else ("medium" if length < 60 else "long")
        identity = feature["properties"]["segment_uid"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("candidate identities must be non-empty and unique")
        identities.add(identity)
        key = hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()
        buckets[bucket].append((key, index))
    if any(len(rows) < per_bin for rows in buckets.values()):
        raise ValueError("every predeclared length bin needs enough candidates")
    return [(bucket, index) for bucket, rows in buckets.items() for _, index in sorted(rows)[:per_bin]]


def target_crop(source, points, box):
    from PIL import ImageDraw
    crop = source.crop(box).convert("RGB")
    draw = ImageDraw.Draw(crop)
    relative = [(x-box[0], y-box[1]) for x, y in points]
    draw.line(relative, fill=(225, 15, 65), width=2)
    first, last = relative[0], relative[-1]
    for x, y in (first, last):
        draw.ellipse((x-2, y-2, x+2, y+2), outline=(0, 95, 230), width=1)
    return crop


def prepare(index_path, vector_path, output, *, seed=20260912, per_bin=6):
    from PIL import Image, ImageDraw, ImageFont
    index, vectors = read(index_path), read(vector_path)
    tiles = {row["tile_id"]: row for row in index["tiles"] if row["split"] == "development" and row["sheet_id"] != "178-gongju"}
    if len(tiles) != 9 or {row["tile_id"] for row in vectors["tiles"]} != set(tiles) or vectors.get("holdout_included"):
        raise ValueError("semantic sampling is restricted to the nine development tiles")
    if per_bin < 1:
        raise ValueError("per_bin must be positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    records, panels = [], []
    for entry in vectors["tiles"]:
        tile = tiles[entry["tile_id"]]
        raster_path = resolve(tile["raster_path"])
        digest = sha256_file(raster_path)
        if digest != entry["source_raster_sha256"]:
            raise ValueError("sample raster no longer matches the candidate run")
        collection_path = resolve(entry["ink_vector_path"])
        collection = read(collection_path)
        proposals = pixel_proposals(tile, collection)
        with Image.open(raster_path) as image:
            source = image.convert("RGB")
        for bucket, index_in_collection in sample_indices(collection["features"], seed=seed, per_bin=per_bin):
            feature = collection["features"][index_in_collection]
            proposal = proposals[index_in_collection]
            points = proposal.points
            xs, ys = zip(*points)
            box = [max(0, math.floor(min(xs))-32), max(0, math.floor(min(ys))-32),
                   min(source.width, math.ceil(max(xs))+33), min(source.height, math.ceil(max(ys))+33)]
            # At least 128 px of source context around a short stroke.
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            box = [max(0, math.floor(min(box[0], cx-64))), max(0, math.floor(min(box[1], cy-64))),
                   min(source.width, math.ceil(max(box[2], cx+64))), min(source.height, math.ceil(max(box[3], cy+64)))]
            sid = f"S{len(records)+1:03}"
            crop = target_crop(source, points, box)
            source_crop = source.crop(box)
            crop.save(output / "samples" / f"{sid}-target.png")
            source_crop.save(output / "samples" / f"{sid}-source.png")
            panel = Image.new("RGB", (360, 330), "white")
            draw = ImageDraw.Draw(panel)
            draw.text((8, 5), f"{sid}  {tile['tile_id']}", font=font, fill="black")
            draw.text((8, 26), f"source box {box}; {bucket}; target red", font=small, fill="black")
            crop.thumbnail((344, 275), Image.Resampling.LANCZOS)
            # Show small-source details large enough to judge the actual stroke.
            scale = min(344/crop.width, 275/crop.height)
            crop = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.BICUBIC)
            panel.paste(crop, ((360-crop.width)//2, 50))
            panels.append(panel)
            records.append({"sample_id": sid, "tile_id": tile["tile_id"], "sheet_id": tile["sheet_id"],
                            "scene_type": tile["scene_type"], "length_bin": bucket,
                            "segment_uid": feature["properties"]["segment_uid"],
                            "segment_geometry_id": feature["properties"]["segment_geometry_id"],
                            "source_raster_sha256": digest, "source_vector_sha256": sha256_file(collection_path),
                            "pixel_length": proposal.pixel_length, "pixel_points": points, "pixel_box": box})
    for start in range(0, len(panels), 6):
        page = Image.new("RGB", (720, 990), "#bbbbbb")
        for offset, panel in enumerate(panels[start:start+6]):
            page.paste(panel, ((offset%2)*360, (offset//2)*330))
        page.save(output / f"page-{start//6+1:02}.png")
    manifest = {"schema": "jap-map-contour-semantic-samples/1", "seed": seed, "per_length_bin_per_tile": per_bin,
                "selection": f"SHA256(seed:segment_uid), {'six' if per_bin == 6 else per_bin} per [18,28), [28,60), [60,infinity) native-pixel-length bin per development tile; no model scores used",
                "source_index_sha256": sha256_file(index_path), "vector_index_sha256": sha256_file(vector_path),
                "samples": records, "human_approvals": 0, "holdout_used": False}
    write(output / "manifest.json", manifest)
    write(output / "labels.template.json", {"schema": "jap-map-contour-semantic-labels/1", "sample_manifest_sha256": sha256_file(output / "manifest.json"),
                                            "reference_origin": "unset", "human_approved": False,
                                            "items": [{"sample_id": row["sample_id"], "class": None, "note": ""} for row in records]})
    print(f"Frozen {len(records)} actual development segments in {(len(records)+5)//6} score-blind pages. No labels or models generated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--per-bin", type=int, default=6)
    args = parser.parse_args()
    prepare(args.index, args.vectors, args.output, seed=args.seed, per_bin=args.per_bin)


if __name__ == "__main__":
    main()
