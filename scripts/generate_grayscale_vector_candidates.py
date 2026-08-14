"""Vectorize raster review candidates into per-tile GeoJSON line proposals.

The output is deliberately review-only: it contains visible historical-map
linework, not asserted contour geometry.  Each feature retains its tile,
confidence, and backend provenance so a reviewer can adopt or reject it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.vectorization import mask_to_pixel_line_proposals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="candidates/candidate_index.json")
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/annotation_package/candidate_vectors"))
    parser.add_argument("--minimum-length-px", type=float, default=18.0)
    parser.add_argument("--simplify-tolerance-px", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Independent tile workers (default: up to 4)")
    return parser.parse_args()


def geojson_feature(tile: dict, proposal, transform) -> dict:
    coordinates = [transform * (x + 0.5, y + 0.5) for x, y in proposal.points]
    return {
        "type": "Feature",
        "properties": {
            "proposal_id": proposal.proposal_id,
            "tile_id": tile["tile_id"],
            "sheet_id": tile["sheet_id"],
            "split": tile["split"],
            "proposal_kind": "visible_linework_review_only",
            "backend": "grayscale_ridge_v1_skeleton_v1",
            "pixel_length": round(proposal.pixel_length, 3),
            "point_count": len(proposal.points),
            "confidence": round(proposal.confidence, 4),
            "review_status": "unreviewed",
        },
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


def process_tile(task: tuple[dict, str, float, float]) -> dict:
    """Process one tile in an isolated worker to keep large scans bounded."""
    tile, output_directory, minimum_length_px, simplify_tolerance_px = task
    import rasterio

    raster_path = Path(tile["candidate_raster_path"])
    if not raster_path.is_absolute():
        raster_path = REPOSITORY / raster_path
    with rasterio.open(raster_path) as source:
        alpha = source.read(4)
        proposals = mask_to_pixel_line_proposals(
            alpha > 0,
            alpha,
            minimum_length_px=minimum_length_px,
            simplify_tolerance_px=simplify_tolerance_px,
        )
        feature_collection = {
            "type": "FeatureCollection",
            "name": f"candidate_proposals_{tile['tile_id']}",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::5132"}},
            "features": [geojson_feature(tile, proposal, source.transform) for proposal in proposals],
        }
    output_path = Path(output_directory) / f"{tile['tile_id']}-proposals.geojson"
    output_path.write_text(json.dumps(feature_collection, ensure_ascii=False), encoding="utf-8")
    return {
        "tile_id": tile["tile_id"],
        "sheet_id": tile["sheet_id"],
        "split": tile["split"],
        "candidate_vector_path": str(output_path.relative_to(REPOSITORY)).replace("\\", "/"),
        "proposal_count": len(proposals),
        "output_path": str(output_path),
    }


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    index_path = args.index if args.index.is_absolute() else REPOSITORY / args.index
    index = json.loads(index_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPOSITORY / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_index = {
        "version": "1",
        "backend": "grayscale_ridge_v1_skeleton_v1",
        "minimum_length_px": args.minimum_length_px,
        "simplify_tolerance_px": args.simplify_tolerance_px,
        "tiles": [],
    }

    tasks = [(tile, str(output_dir), args.minimum_length_px, args.simplify_tolerance_px) for tile in index["tiles"]]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for tile_result in executor.map(process_tile, tasks):
            result_index["tiles"].append({key: value for key, value in tile_result.items() if key != "output_path"})
            print(f"{tile_result['output_path']} ({tile_result['proposal_count']} proposals)")
    output_path = output_dir / "candidate_vector_index.json"
    output_path.write_text(json.dumps(result_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
