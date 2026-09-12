#!/usr/bin/env python3
"""Compare exported contour-linework candidates on the same development pixels.

Region probes are explicitly provisional AI visual selections, not approved
human contour truth. This command never opens the held-out rasters or writes
to an annotation GeoPackage. All outputs go to a new comparison directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.provenance import sha256_file
from histcontour_core.vectorization import PixelLineProposal
from scripts.generate_ink_centerline_candidates import proposal_mask

METHODS = ("before_18", "corner_safe_18", "corner_safe_48")
LABELS = ("수정 전: 18 px", "모서리 수정: 18 px", "수정 + 48 px 길이 필터")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def selected_sources(index, probes):
    if probes.get("schema") != "jap-map-contour-region-probes/1" or probes.get("human_approved") is not False:
        raise ValueError("probes must be explicitly provisional region selections")
    tiles = {row["tile_id"]: row for row in index["tiles"] if row["split"] == "development"}
    if len(tiles) != 9:
        raise ValueError("comparison expects the nine development tiles; no holdout access")
    seen = set()
    for region in probes["regions"]:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", region["id"]) or region["id"] in seen
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", region["tile_id"])):
            raise ValueError("probe identifiers must be unique short local names")
        seen.add(region["id"])
        if region["tile_id"] not in probes["source_sha256"]:
            raise ValueError("every probe source needs a frozen raster hash")
        if region["tile_id"] not in tiles or region["kind"] not in {"contour", "text", "road_river"}:
            raise ValueError("unknown, held-out, or unclassified probe")
        box = region["pixel_box"]
        width, height = tiles[region["tile_id"]]["pixel_bounds"][2:]
        if (len(box) != 4 or any(type(value) is not int for value in box)
                or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError("probe leaves its native source pixels")
    for sid, expected in probes["source_sha256"].items():
        if sid not in tiles or sha256_file(resolve(tiles[sid]["raster_path"])) != expected:
            raise ValueError("probe source differs from its frozen raster")
    return tiles


def pixel_proposals(tile, collection):
    west, south, east, north = tile["bounds"]
    width, height = tile["pixel_bounds"][2:]
    result = []
    for feature in collection["features"]:
        points = tuple(((x-west)*width/(east-west)-.5, (north-y)*height/(north-south)-.5)
                       for x, y in feature["geometry"]["coordinates"])
        props = feature["properties"]
        result.append(PixelLineProposal(props["proposal_id"], points, props["pixel_length"], float(props.get("confidence", 1))))
    return result


def overlay(gray, mask, *, enlarge=False):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation
    rgb = np.repeat(gray[..., None], 3, axis=2).copy()
    if enlarge:
        mask = binary_dilation(mask, iterations=1)
    rgb[mask] = rgb[mask]*.2 + np.array([230, 28, 80])*.8
    return Image.fromarray(rgb.astype("uint8"))


def inspect_sources(index_path, probes_path, output):
    from PIL import Image, ImageDraw
    index, probes = read(index_path), read(probes_path)
    tiles = selected_sources(index, probes)
    output.mkdir(parents=True, exist_ok=False)
    contact = Image.new("RGB", (4*270, 3*225), "white")
    draw = ImageDraw.Draw(contact)
    for number, region in enumerate(probes["regions"]):
        with Image.open(resolve(tiles[region["tile_id"]]["raster_path"])) as source:
            crop = source.crop(region["pixel_box"]).convert("RGB")
            crop.save(output / f"{region['id']}-source.png")
        crop.thumbnail((250, 180))
        factor = min(250/crop.width, 180/crop.height)
        crop = crop.resize((round(crop.width*factor), round(crop.height*factor)))
        x, y = number % 4 * 270, number // 4 * 225
        contact.paste(crop, (x, y+35))
        draw.text((x+3, y+3), region["id"], fill="black")
        draw.text((x+3, y+18), str(region["pixel_box"]), fill="black")
    contact.save(output / "source-probes.png")
    write(output / "probes.json", probes)


def compare(index_path, probes_path, before_path, after_path, output, *, filtered_path=None):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    index, probes = read(index_path), read(probes_path)
    tiles = selected_sources(index, probes)
    before, after = read(before_path), read(after_path)
    method_labels = LABELS if after.get("junction_policy", "split") == "split" else (
        LABELS[0], "모서리 수정 + 접선 묶기: 18 px", "접선 묶기 + 48 px 길이 필터")
    methods = list(METHODS)
    filtered, filtered_tiles = None, {}
    if filtered_path is not None:
        filtered = read(filtered_path)
        if (filtered.get("source_ink_index_sha256") != sha256_file(after_path)
                or filtered.get("source_index_sha256") != sha256_file(index_path)
                or filtered.get("holdout_included")
                or sha256_file(filtered_path.parent / filtered["model_snapshot"]) != filtered["model_sha256"]):
            raise ValueError("filtered scores do not match this source/vector run and archived model")
        filtered_tiles = {row["tile_id"]: row for row in filtered["tiles"]}
        if set(filtered_tiles) != set(tiles):
            raise ValueError("filtered candidates must cover the same development tiles")
        methods[2] = "context_filter"
        method_labels = (*method_labels[:2], f"문맥 점수 ≥ {filtered['filter_threshold']}")
    by_method = [{row["tile_id"]: row for row in value["tiles"]} for value in (before, after)]
    if any(set(rows) != set(tiles) for rows in by_method) or before.get("holdout_included") or after.get("holdout_included"):
        raise ValueError("both vector runs must cover the same nine development tiles only")
    output.mkdir(parents=True, exist_ok=False)
    reports, region_reports, html_tiles, html_regions = [], [], [], []
    for tile_id, tile in tiles.items():
        with Image.open(resolve(tile["raster_path"])) as source:
            gray = np.asarray(source.convert("L")).copy()
        digest = sha256_file(resolve(tile["raster_path"]))
        entries = [rows[tile_id] for rows in by_method]
        if any(row["source_raster_sha256"] != digest for row in entries):
            raise ValueError("before/after do not use the same original raster")
        collections = [read(resolve(row["ink_vector_path"])) for row in entries]
        old, new = [pixel_proposals(tile, collection) for collection in collections]
        if filtered is not None:
            selected = read(resolve(filtered_tiles[tile_id]["retained_path"]))
            original_by_id = {feature["properties"]["segment_uid"]: feature["geometry"] for feature in collections[1]["features"]}
            if any(original_by_id.get(feature["properties"]["segment_uid"]) != feature["geometry"] for feature in selected["features"]):
                raise ValueError("filtered output is not an unchanged geometry subset")
            alternative = pixel_proposals(tile, selected)
        else:
            alternative = [line for line in new if line.pixel_length >= 48]
        candidates = [old, new, alternative]
        masks = [proposal_mask(lines, gray.shape) for lines in candidates]
        distances = [distance_transform_edt(~mask) if mask.any() else np.full(gray.shape, np.inf) for mask in masks]
        row = {"tile_id": tile_id, "sheet_id": tile["sheet_id"], "scene_type": tile["scene_type"], "source_sha256": digest,
               "methods": {name: {"line_count": len(lines), "median_line_length_px": statistics.median(line.pixel_length for line in lines) if lines else None,
                                  "total_line_length_px": sum(line.pixel_length for line in lines), "vector_pixels": int(mask.sum()),
                                  "open_line_endpoints": 2*sum(line.points[0] != line.points[-1] for line in lines)}
                           for name, lines, mask in zip(methods, candidates, masks)}}
        reports.append(row)
        tile_images = []
        for name, mask in zip(methods, masks):
            filename = f"{tile_id}-{name}.png"
            overlay(gray, mask).save(output / filename)
            tile_images.append(f'<figure><figcaption>{html.escape(name)}</figcaption><img src="{filename}"></figure>')
        html_tiles.append(f'<h2>{html.escape(tile_id)}</h2><div class="comparison">{"".join(tile_images)}</div>')
        # Preserve the original v3 geometry/identity; this is an explicit
        # review-only subset, never a semantic contour approval.
        if filtered is None:
            selected = {**collections[1], "name": f"corner_safe_min48_{tile_id}",
                        "selection": {"minimum_length_px": 48, "review_only": True},
                        "features": [feature for feature in collections[1]["features"] if feature["properties"]["pixel_length"] >= 48]}
        write(output / f"{tile_id}-alternative-review.geojson", selected)
        for region in [item for item in probes["regions"] if item["tile_id"] == tile_id]:
            x1, y1, x2, y2 = region["pixel_box"]
            # Interior pixels avoid counting a contour just outside the ROI.
            ink = gray[y1:y2, x1:x2] <= 176
            ink[:2] = ink[-2:] = False
            ink[:, :2] = ink[:, -2:] = False
            count = int(ink.sum())
            values = {name: int(((distance[y1:y2, x1:x2] <= 2.0) & ink).sum()) for name, distance in zip(methods, distances)}
            region_reports.append({**region, "reference_dark_pixels": count, "covered_dark_pixels": values,
                                   "coverage": {name: covered/count if count else None for name, covered in values.items()}})
            filenames = []
            filename = f"{region['id']}-source.png"
            Image.fromarray(gray[y1:y2, x1:x2]).save(output / filename)
            filenames.append(filename)
            for name, mask in zip(methods, masks):
                filename = f"{region['id']}-{name}.png"
                overlay(gray[y1:y2, x1:x2], mask[y1:y2, x1:x2]).save(output / filename)
                filenames.append(filename)
            html_regions.append(f'<h3>{html.escape(region["id"])} · {html.escape(region["kind"])}</h3><div class="probe">'
                                + "".join(f'<figure><figcaption>{label}</figcaption><img src="{filename}"></figure>'
                                          for label, filename in zip(("원본", *method_labels), filenames)) + "</div>")
        print(f"{tile_id}: before {len(old)} → corrected {len(new)} lines", flush=True)
    aggregates = {}
    for kind in ("contour", "text", "road_river"):
        chosen = [row for row in region_reports if row["kind"] == kind]
        denominator = sum(row["reference_dark_pixels"] for row in chosen)
        aggregates[kind] = {"regions": len(chosen), "reference_dark_pixels": denominator,
                            "visible_ink_coverage": {name: sum(row["covered_dark_pixels"][name] for row in chosen)/denominator if denominator else None for name in methods}}
    report = {"schema": "jap-map-contour-vector-comparison/1", "reference_origin": probes["reference_origin"],
              "human_approved": False, "probes_sha256": sha256_file(probes_path), "before_index_sha256": sha256_file(before_path),
              "after_index_sha256": sha256_file(after_path), "source_index_sha256": sha256_file(index_path),
              "methods": dict(zip(methods, method_labels)),
              "filtered_model": None if filtered is None else {"index_sha256": sha256_file(filtered_path), "model_sha256": filtered["model_sha256"], "threshold": filtered["filter_threshold"], "scope": "synthetic-trained review candidate, not calibrated historical-map accuracy"},
              "after_vectorization": {key: after.get(key) for key in ("graph_version", "diagonal_policy", "junction_policy", "minimum_length_px", "simplify_tolerance_px")},
              "tiles": reports, "regions": region_reports, "probe_aggregates": aggregates,
              "holdout_used": False, "ground_truth_modified": False, "promotion_passed": False,
              "interpretation": "AI-selected region spot checks only. Dark-pixel coverage within 2px is not contour precision/recall. Higher contour coverage is desirable; higher text/road coverage is undesirable. Full-map false joins and omissions need independent contour truth. No invented gap connections are made."}
    write(output / "comparison.json", report)
    table = "<table><tr><th>AI 영역 점검</th>" + "".join(f"<th>{label}</th>" for label in method_labels) + "</tr>"
    for kind, values in aggregates.items():
        table += f"<tr><td>{kind} ({values['regions']}곳)</td>" + "".join(
            "<td>n/a</td>" if values["visible_ink_coverage"][name] is None else f"<td>{values['visible_ink_coverage'][name]:.1%}</td>"
            for name in methods) + "</tr>"
    table += "</table>"
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8">'
                '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'">'
                '<title>등고선 벡터 추출 전후 비교</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}.comparison,.probe{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%;height:auto}figcaption{padding:8px}h2,h3{margin-top:32px}.probe img{width:auto;max-width:100%;min-width:140px;image-rendering:pixelated}</style>'
                '<h1>실제 지도: 출력 벡터 전후 비교</h1><p>빨강은 실제 GeoJSON으로 내보낸 선입니다. 검은 원본 획 전체가 아닙니다. 모든 출력은 검수용입니다.</p>'
                '<p>아래 수치는 AI가 원본에서 고른 작은 영역의 잉크 피복률입니다. 정식 등고선 정확도가 아닙니다. contour는 높을수록, text·road_river는 낮을수록 좋습니다.</p>'
                + table + '<h2>원본 확대 점검</h2>' + "".join(html_regions) + '<h2>9개 개발 타일 전체</h2>' + "".join(html_tiles) + '</html>')
    with (output / "report.html").open("x", encoding="utf-8") as handle:
        handle.write(document)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--filtered-index", type=Path, help="optional immutable score-filtered candidate index; replaces the length-only alternative")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.inspect_only:
            inspect_sources(args.index, args.probes, args.output)
        else:
            if args.before is None or args.after is None:
                raise ValueError("comparison needs both --before and --after indexes")
            compare(args.index, args.probes, args.before, args.after, args.output, filtered_path=args.filtered_index)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Comparison stopped: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
