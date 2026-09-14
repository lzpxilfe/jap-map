#!/usr/bin/env python3
"""Source-first, pixel-exact diagnostic crops; no automatic truth or approval.

The baseline is the pinned, already assembled vector network. This packet is
development material, not a new independent evaluation set. Crop coordinates
are boundaries; vector coordinates put the top-left pixel centre at (0, 0).
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.map_text_detection import inspect_raster, validate_report
from histcontour_core.provenance import sha256_file

SCENES = {"digits", "text_overlap", "faint", "valley", "dense_parallel", "hydro_symbol"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_regions(manifest, report):
    tiles = {t["tile_id"]: t for t in validate_report(report)}
    if (manifest.get("schema") != "jap-map-contour-reconstruction-regions/1"
            or manifest.get("dataset_role") != "development_diagnostics_not_independent_evaluation"
            or any(manifest.get(k) is not False for k in ("human_approved", "training_eligible", "holdout_used"))):
        raise ValueError("expected unapproved development-only diagnostic manifest")
    rows = manifest.get("regions", [])
    if len(rows) != 24 or Counter(r.get("scene_hint") for r in rows) != Counter({k: 4 for k in SCENES}):
        raise ValueError("need exactly four regions in each of the six scene types")
    ids = set()
    for row in rows:
        rid, tid, box = row.get("region_id"), row.get("tile_id"), row.get("box")
        if not isinstance(rid, str) or not re.fullmatch(r"R[0-9]{3}", rid) or rid in ids or tid not in tiles:
            raise ValueError("invalid/duplicate region or undeclared tile")
        ids.add(rid)
        width, height = tiles[tid]["pixel_bounds"][2:]
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError("crop bounds must be nonempty integer boundaries inside the source")
    return tiles


def intersects(points, box):
    if len(points) < 2:
        return False
    xs, ys = zip(*points)
    # Vector centres may sit half a pixel either side of the crop boundaries.
    if not (max(xs) >= box[0]-.5 and min(xs) <= box[2]-.5 and max(ys) >= box[1]-.5 and min(ys) <= box[3]-.5):
        return False
    # Liang–Barsky interval clipping: a loop enclosing the crop does not
    # intersect it merely because its bounding rectangle does.
    for first, second in zip(points, points[1:]):
        low, high = 0., 1.
        for axis, minimum, maximum in ((0, box[0]-.5, box[2]-.5), (1, box[1]-.5, box[3]-.5)):
            delta = second[axis]-first[axis]
            if abs(delta) < 1e-12:
                if not minimum <= first[axis] <= maximum:
                    low, high = 1., 0.
                    break
            else:
                entry, leave = sorted(((minimum-first[axis])/delta, (maximum-first[axis])/delta))
                low, high = max(low, entry), min(high, leave)
        if low <= high:
            return True
    return False


def render_panel(source, box, *, lines=(), polygons=(), colour="#d96400", scale=2):
    """Nearest-neighbour source display, independent antialiased polylines."""
    from PIL import Image, ImageDraw
    l, t, r, b = box
    crop = source.crop(box).convert("RGBA").resize(((r-l)*scale, (b-t)*scale), Image.Resampling.NEAREST)
    aa = 4
    overlay = Image.new("RGBA", (crop.width*aa, crop.height*aa), (0, 0, 0, 0))
    pen = ImageDraw.Draw(overlay)
    def xy(p):
        return ((p[0]-l+.5)*scale*aa, (p[1]-t+.5)*scale*aa)
    for line in lines:
        if len(line) >= 2:
            pen.line([xy(p) for p in line], fill=colour, width=aa, joint="curve")
    for polygon in polygons:
        pen.line([xy(p) for p in [*polygon, polygon[0]]], fill="#a027bb", width=aa)
    return Image.alpha_composite(crop, overlay.resize(crop.size, Image.Resampling.LANCZOS)).convert("RGB")


def run(packet, manifest_path, baseline_path, output, *, text_detection_path=None, candidate_path=None, region_ids=None):
    from PIL import Image, ImageDraw, ImageFont
    packet, output = Path(packet).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("review output must be new")
    if output == packet or packet in output.parents:
        raise ValueError("output must be outside the source packet")
    report_path = packet/"drawing-report.json"
    manifest, report = read(manifest_path), read(report_path)
    tiles = validate_regions(manifest, report)
    regions = manifest["regions"]
    if region_ids is not None:
        regions = [r for r in regions if r["region_id"] in region_ids]
        if not regions or len(set(region_ids)) != len(region_ids) or set(region_ids) != {r["region_id"] for r in regions}:
            raise ValueError("requested regions must be distinct IDs from the fixed manifest")
    inputs = [Path(manifest_path), report_path, Path(baseline_path)]
    for path, key in ((report_path, "drawing_report_sha256"), (baseline_path, "baseline_sha256")):
        if sha256_file(path) != manifest.get(key):
            raise ValueError(f"pinned {key} differs")
    baseline = read(baseline_path)
    crs = {"type": "name", "properties": {"name": next(iter(tiles.values()))["crs_authid"]}}
    if baseline.get("crs") != crs:
        raise ValueError("baseline native CRS differs")
    series = {"baseline": baseline}
    if candidate_path:
        series["candidate"] = read(candidate_path)
        if series["candidate"].get("crs") != crs:
            raise ValueError("candidate native CRS differs")
        inputs.append(Path(candidate_path))
    detections = {}
    if text_detection_path:
        text = read(text_detection_path)
        if text.get("schema") != "jap-map-map-text-detection/1" or text.get("holdout_used") is not False:
            raise ValueError("expected development-only text evidence")
        for tile in text["tiles"]:
            source = tiles.get(tile["tile_id"])
            if source is None or tile["split"] != "development" or any(tile[k] != source[k] for k in (
                    "source_raster_sha256", "bounds", "pixel_bounds", "crs_authid")):
                raise ValueError("text evidence source identity differs")
            detections[tile["tile_id"]] = tile["detection"]["polygons_on_source_pixel_center_grid"]
        inputs.append(Path(text_detection_path))
    paths, metadata = {}, {}
    for tid, tile in tiles.items():
        relative = Path(tile["raster_path"])
        path = (packet/relative).resolve()
        if relative.is_absolute() or packet not in path.parents:
            raise ValueError("source raster path escapes packet")
        metadata[tid] = inspect_raster(path, tile)
        paths[tid] = path
        inputs.append(path)
    before = {str(p.resolve()): sha256_file(p) for p in inputs}
    pixel_lines = {}
    for name, collection in series.items():
        pixel_lines[name] = {tid: [] for tid in tiles}
        for f in collection["features"]:
            tid = f["properties"]["tile_id"]
            if tid not in tiles or f["geometry"]["type"] != "LineString":
                raise ValueError("vector geometry or tile unsupported")
            pixel_lines[name][tid].append(map_to_pixel(tiles[tid], f["geometry"]["coordinates"]))
    output.mkdir(parents=True)
    font = ImageFont.load_default(size=16)
    rows = []
    for region in regions:
        rid, tid, box = region["region_id"], region["tile_id"], region["box"]
        folder = output/rid
        folder.mkdir()
        with Image.open(paths[tid]) as source:
            source.crop(box).save(folder/"source.png")
            panels = [("Original source", render_panel(source, box))]
            for name in series:
                lines = [p for p in pixel_lines[name][tid] if intersects(p, box)]
                panels.append((name.title()+" | unapproved", render_panel(source, box, lines=lines,
                               colour="#d96400" if name == "baseline" else "#078357")))
            if text_detection_path:
                polygons = [p for p in detections.get(tid, []) if intersects([*p, p[0]], box)]
                panels.append(("Text search regions | not erase masks", render_panel(source, box, polygons=polygons)))
            width, height = panels[0][1].size
            board = Image.new("RGB", (len(panels)*(width+16)+16, height+88), "white")
            pen = ImageDraw.Draw(board)
            pen.text((16, 8), f"{rid}  {tid}  {region['scene_hint']} (scene hint, not truth)", fill="black", font=font)
            for index, (label, panel) in enumerate(panels):
                x = 16+index*(width+16)
                pen.text((x, 36), label, fill="black", font=font)
                board.paste(panel, (x, 64))
            board.save(folder/"comparison.png")
        gt = metadata[tid]["geotransform"]
        l, t, r, b = box
        row = {**region, "source_raster_sha256": tiles[tid]["source_raster_sha256"],
               "source_crop_sha256": sha256_file(folder/"source.png"), "crs_authid": tiles[tid]["crs_authid"],
               "source_geotransform": gt, "crop_geotransform": [gt[0]+l*gt[1], gt[1], 0., gt[3]+t*gt[5], 0., gt[5]],
               "crop_size": [r-l, b-t], "source_pixel_values_unchanged": True,
               "human_semantics": "unknown", "human_route": "unknown", "human_shape": "unknown",
               "human_approved": False, "training_eligible": False,
               "source_image": f"{rid}/source.png", "comparison_image": f"{rid}/comparison.png"}
        rows.append(row)
        write(folder/"region.json", row)
    after = {str(p.resolve()): sha256_file(p) for p in inputs}
    if after != before:
        raise ValueError("input changed during packet generation")
    result = {"schema": "jap-map-contour-reconstruction-review/1", "regions": rows,
              "fixed_manifest_region_count": len(manifest["regions"]),
              "dataset_role": manifest["dataset_role"], "holdout_used": False, "human_approvals": 0,
              "provenance": {"input_sha256": before, "inputs_unchanged": True,
                  "implementation_sha256": sha256_file(__file__)},
              "warning": "Scene hints/boxes are not labels. Unmarked pixels, inferred gaps and question marks are unknown. This is development material, not independent accuracy evidence."}
    write(output/"review.json", result)
    body = "\n".join(f'<h2>{html.escape(r["region_id"]+" — "+r["scene_hint"])}</h2><img src="{r["comparison_image"]}" style="max-width:100%">' for r in rows)
    with (output/"index.html").open("x", encoding="utf-8") as stream:
        stream.write('<!doctype html><meta charset="utf-8"><title>Contour reconstruction diagnostics</title>'
                     f'<h1>{len(rows)} of 24 development regions — unreviewed</h1><p>'+html.escape(result["warning"])+"</p>"+body)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-detection", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--regions", nargs="+")
    args = parser.parse_args()
    result = run(args.packet, args.manifest, args.baseline, args.output,
                 text_detection_path=args.text_detection, candidate_path=args.candidate, region_ids=args.regions)
    print(json.dumps({"regions": len(result["regions"]), "output": str(args.output), "human_approvals": 0}))


if __name__ == "__main__":
    main()
