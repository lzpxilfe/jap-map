"""Generate review-only contour completion candidates on development tiles.

This script never promotes Ink linework to contours.  Existing ridge-vector
endpoints provide anchors; Ink supports short visible routes, while tightly
aligned long anchors may produce a lower-confidence Hermite completion across
text, symbols, or blank gaps.  Holdout tiles require explicit opt-in.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.completion import (
    COMPLETION_BACKEND_ID,
    ContourCompletionSettings,
    anchors_from_polyline,
    propose_contour_completions,
    rasterize_polylines,
)
from histcontour_core.ink import ARCHAEOTRACE_UPSTREAM_COMMIT, ink_centerline_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="annotation_package/index.json")
    parser.add_argument(
        "--ridge-vector-index",
        type=Path,
        default=Path("data/derived/annotation_package/candidate_vectors/candidate_vector_index.json"),
    )
    parser.add_argument(
        "--review-geopackage",
        type=Path,
        default=Path("data/derived/annotation_package/contour_annotations.gpkg"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/annotation_package/contour_completion_candidates"),
    )
    parser.add_argument(
        "--anchor-status",
        choices=("all", "contour"),
        default="contour",
        help="Use human-reviewed contours by default; 'all' is an explicit heuristic research mode",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--include-holdout", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY / value


def _portable(path: Path) -> str:
    try:
        value = path.resolve().relative_to(REPOSITORY)
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _review_statuses(package_path: Path) -> dict[str, str]:
    if not package_path.exists():
        return {}
    with sqlite3.connect(package_path) as connection:
        try:
            rows = connection.execute("SELECT proposal_uid, review_status FROM proposal_review").fetchall()
        except sqlite3.DatabaseError:
            return {}
    return {str(identifier): str(status) for identifier, status in rows}


def select_tiles(index: dict, available_tile_ids, include_holdout: bool = False) -> list[dict]:
    """Exclude holdout tiles unless the caller makes the leak explicit."""

    available = set(available_tile_ids)
    return [
        tile for tile in index["tiles"]
        if tile["tile_id"] in available and (include_holdout or tile["split"] != "holdout_test")
    ]


def _pixel_point(tile: dict, coordinate) -> tuple[float, float]:
    west, south, east, north = (float(value) for value in tile["bounds"])
    width, height = (int(value) for value in tile["pixel_bounds"][2:])
    return (
        (float(coordinate[0]) - west) * width / (east - west) - 0.5,
        (north - float(coordinate[1])) * height / (north - south) - 0.5,
    )


def _map_point(tile: dict, point) -> list[float]:
    west, south, east, north = (float(value) for value in tile["bounds"])
    width, height = (int(value) for value in tile["pixel_bounds"][2:])
    return [
        west + (float(point[0]) + 0.5) * (east - west) / width,
        north - (float(point[1]) + 0.5) * (north - south) / height,
    ]


def _load_ridge_lines(tile: dict, vector_path: Path, statuses: dict[str, str], anchor_status: str, settings):
    collection = json.loads(vector_path.read_text(encoding="utf-8"))
    polylines = []
    anchors = []
    properties_by_proposal = {}
    for feature in collection["features"]:
        properties = feature["properties"]
        proposal_id = str(properties["proposal_id"])
        review_status = statuses.get(f"{tile['tile_id']}:{proposal_id}", "unreviewed")
        points = tuple(_pixel_point(tile, coordinate) for coordinate in feature["geometry"]["coordinates"])
        polylines.append(points)
        properties_by_proposal[proposal_id] = {**properties, "review_status": review_status}
        if anchor_status == "contour" and review_status != "contour":
            continue
        anchors.extend(
            anchors_from_polyline(
                proposal_id,
                points,
                source_length_px=float(properties["pixel_length"]),
                source_confidence=float(properties.get("confidence", 1.0)),
                tangent_lookback_px=settings.tangent_lookback_px,
            )
        )
    return tuple(polylines), tuple(anchors), properties_by_proposal


def _feature(tile, candidate, source_properties, anchor_status) -> dict:
    first = source_properties[candidate.first_proposal_id]
    second = source_properties[candidate.second_proposal_id]
    return {
        "type": "Feature",
        "properties": {
            "completion_id": candidate.completion_id,
            "tile_id": tile["tile_id"],
            "sheet_id": tile["sheet_id"],
            "split": tile["split"],
            "proposal_kind": "contour_completion_review_only",
            "backend": COMPLETION_BACKEND_ID,
            "ink_upstream_commit": ARCHAEOTRACE_UPSTREAM_COMMIT,
            "mode": candidate.mode,
            "first_proposal_id": candidate.first_proposal_id,
            "second_proposal_id": candidate.second_proposal_id,
            "first_review_status": first["review_status"],
            "second_review_status": second["review_status"],
            "anchor_scope": anchor_status,
            "direct_gap_px": round(candidate.direct_gap_px, 3),
            "path_length_px": round(candidate.path_length_px, 3),
            "tangent_error_deg": round(candidate.maximum_tangent_error_degrees, 3),
            "missing_fraction": round(candidate.missing_fraction, 4),
            "maximum_missing_run_px": candidate.maximum_missing_run_px,
            "ink_support_fraction": round(candidate.ink_support_fraction, 4),
            "junction_pixels": candidate.junction_pixels,
            "score": round(candidate.score, 4),
            "review_status": "unreviewed",
            "auto_apply": False,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [_map_point(tile, point) for point in candidate.points],
        },
    }


def _preview(source, candidates, output_path: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    values = np.asarray(source)
    if values.ndim == 2:
        rgb = np.repeat(values[..., None], 3, axis=2)
    else:
        rgb = values[..., :3]
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((512, 512))
    draw = ImageDraw.Draw(image)
    colours = {
        "ink_path": "#16a34a",
        "hermite_occlusion": "#f97316",
        "hermite_gap": "#9333ea",
    }
    scale_x = 512 / values.shape[1]
    scale_y = 512 / values.shape[0]
    for candidate in candidates:
        points = [(x * scale_x, y * scale_y) for x, y in candidate.points]
        draw.line(points, fill=colours[candidate.mode], width=3)
        for x, y in (points[0], points[-1]):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#ffffff", outline=colours[candidate.mode])
    image.save(output_path)


def process_tile(task) -> dict:
    tile, vector_record, output_directory, settings, statuses, anchor_status = task
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(f"Missing completion dependency: {error.name}") from error

    raster_path = _resolve(tile["raster_path"])
    vector_path = _resolve(vector_record["candidate_vector_path"])
    with Image.open(raster_path) as image:
        source = np.asarray(image).copy()
    height, width = source.shape[:2]
    polylines, anchors, source_properties = _load_ridge_lines(
        tile, vector_path, statuses, anchor_status, settings
    )
    existing = rasterize_polylines((height, width), polylines)
    ink = ink_centerline_candidates(source, tile_origin=tuple(tile["pixel_bounds"][:2]))
    result = propose_contour_completions(
        anchors,
        ink.centerline,
        ink.center_score,
        existing_linework=existing,
        settings=settings,
    )
    output_dir = Path(output_directory)
    vector_output = output_dir / f"{tile['tile_id']}-contour-completions.geojson"
    preview_output = output_dir / f"{tile['tile_id']}-contour-completions.png"
    collection = {
        "type": "FeatureCollection",
        "name": f"contour_completion_candidates_{tile['tile_id']}",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{tile['crs_authid'].replace(':', '::')}"}},
        "features": [_feature(tile, candidate, source_properties, anchor_status) for candidate in result.candidates],
    }
    _atomic_json(vector_output, collection)
    _preview(source, result.candidates, preview_output)
    mode_counts = {}
    for candidate in result.candidates:
        mode_counts[candidate.mode] = mode_counts.get(candidate.mode, 0) + 1
    return {
        "tile_id": tile["tile_id"],
        "sheet_id": tile["sheet_id"],
        "split": tile["split"],
        "scene_type": tile["scene_type"],
        "completion_vector_path": _portable(vector_output),
        "completion_preview_path": _portable(preview_output),
        "completion_count": len(result.candidates),
        "mode_counts": dict(sorted(mode_counts.items())),
        "anchor_count": result.anchor_count,
        "eligible_anchor_count": result.eligible_anchor_count,
        "evaluated_pair_count": result.evaluated_pair_count,
        "rejection_counts": result.rejection_counts,
    }


def _contact_sheet(results: list[dict], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    columns = 3
    rows = (len(results) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 512, rows * 548), "white")
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        left, top = (index % columns) * 512, (index // columns) * 548
        with Image.open(_resolve(result["completion_preview_path"])) as source:
            preview = source.convert("RGB")
        sheet.paste(preview, (left, top))
        draw.text(
            (left + 6, top + 518),
            f"{result['tile_id']} | {result['completion_count']} links | {result['mode_counts']}",
            fill="black",
        )
    sheet.save(output_path)


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    index = json.loads(_resolve(args.index).read_text(encoding="utf-8"))
    vector_index = json.loads(_resolve(args.ridge_vector_index).read_text(encoding="utf-8"))
    vector_by_tile = {record["tile_id"]: record for record in vector_index["tiles"]}
    selected = select_tiles(index, vector_by_tile, args.include_holdout)
    if not selected:
        raise SystemExit("No matching tiles selected")
    all_statuses = _review_statuses(_resolve(args.review_geopackage))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = ContourCompletionSettings()
    tasks = []
    for tile in selected:
        tile_statuses = {
            key: value for key, value in all_statuses.items()
            if key.startswith(f"{tile['tile_id']}:")
        }
        tasks.append(
            (tile, vector_by_tile[tile["tile_id"]], str(output_dir), settings, tile_statuses, args.anchor_status)
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_tile, tasks))

    result_index = {
        "version": "1",
        "backend": COMPLETION_BACKEND_ID,
        "source_ridge_backend": vector_index.get("backend"),
        "ink_upstream_commit": ARCHAEOTRACE_UPSTREAM_COMMIT,
        "review_only": True,
        "auto_apply": False,
        "anchor_status": args.anchor_status,
        "holdout_included": bool(args.include_holdout),
        "settings": asdict(settings),
        "tiles": results,
    }
    index_output = output_dir / "completion_candidate_index.json"
    contact_output = output_dir / "completion_candidate_contact_sheet.png"
    _atomic_json(index_output, result_index)
    _contact_sheet(results, contact_output)
    for result in results:
        print(f"{result['completion_vector_path']} ({result['completion_count']} completions; {result['mode_counts']})")
    print(index_output)
    print(contact_output)


if __name__ == "__main__":
    main()
