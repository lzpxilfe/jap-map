#!/usr/bin/env python3
"""Freeze 1–3 exact observed-line drafts for non-training human review."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.human_feedback import geometry_digest, map_to_pixel, validate_pixel_points
from histcontour_core.map_text_detection import inspect_raster
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_reconstruction_review import read, write, validate_regions, intersects
from scripts.render_assisted_refinement import render_comparison


def run(packet, manifest_path, reconstruction_dir, output, cases):
    packet, reconstruction_dir, output = (Path(p).resolve() for p in (packet, reconstruction_dir, output))
    if output.exists():
        raise FileExistsError("line review output must be new")
    if packet == output or packet in output.parents or reconstruction_dir == output or reconstruction_dir in output.parents:
        raise ValueError("review must be separate from source and reconstruction snapshots")
    if not 1 <= len(cases) <= 3 or len({c[0] for c in cases}) != len(cases):
        raise ValueError("one to three distinct question IDs are required")
    report_path = packet/"drawing-report.json"
    report, manifest = read(report_path), read(manifest_path)
    tiles = validate_regions(manifest, report)
    if sha256_file(report_path) != manifest["drawing_report_sha256"]:
        raise ValueError("source report is not the fixed diagnostic packet")
    regions = {r["region_id"]: r for r in manifest["regions"]}
    reconstruction = read(reconstruction_dir/"reconstruction-report.json")
    if reconstruction.get("status") != "experimental_not_applied" or reconstruction.get("holdout_used") is not False:
        raise ValueError("expected unapproved source reconstruction")
    for path, digest in reconstruction["provenance"]["input_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError("reconstruction input changed")
    raw_path = reconstruction_dir/"raw-observed.geojson"
    candidate_path = reconstruction_dir/"candidate-observed.geojson"
    raw, candidate = read(raw_path), read(candidate_path)
    if raw["crs"] != candidate["crs"]:
        raise ValueError("raw and fitted CRS differ")
    expected_crs = {"type": "name", "properties": {"name": next(iter(tiles.values()))["crs_authid"]}}
    if candidate["crs"] != expected_crs:
        raise ValueError("question vectors do not use the native source CRS")
    inputs = [report_path, Path(manifest_path), raw_path, candidate_path, reconstruction_dir/"reconstruction-report.json"]
    paths_by_kind = {}
    for key, collection in (("raw", raw), ("candidate", candidate)):
        paths_by_kind[key] = {(f["properties"]["tile_id"], f["properties"]["path_id"]): f for f in collection["features"]}
        if len(paths_by_kind[key]) != len(collection["features"]):
            raise ValueError("duplicate tile/path identities")
    selected = []
    for question_id, region_id, path_id in cases:
        if not re.fullmatch(r"N[0-9]{3}", question_id) or region_id not in regions:
            raise ValueError("invalid question or development region")
        region = regions[region_id]
        tid = region["tile_id"]
        before = paths_by_kind["raw"][(tid, path_id)]
        after = paths_by_kind["candidate"][(tid, path_id)]
        if before["geometry"]["type"] != "LineString" or after["geometry"]["type"] != "LineString":
            raise ValueError("question geometry must be a line")
        if any(after["properties"].get(k) is not False for k in ("human_approved", "training_eligible", "contour_semantics_assigned")):
            raise ValueError("only explicitly unapproved, non-training source drafts may be asked")
        tile = tiles[tid]
        source = (packet/tile["raster_path"]).resolve()
        if Path(tile["raster_path"]).is_absolute() or packet not in source.parents:
            raise ValueError("source raster escapes the fixed packet")
        inspect_raster(source, tile)
        inputs.append(source)
        old, new = map_to_pixel(tile, before["geometry"]["coordinates"]), map_to_pixel(tile, after["geometry"]["coordinates"])
        validate_pixel_points(old, tile)
        validate_pixel_points(new, tile)
        if not intersects(new, region["box"]):
            raise ValueError("question path does not intersect its context region")
        row = {"question_id": question_id, "proposal_id": question_id, "region_id": region_id,
               "tile_id": tid, "path_id": path_id, "pixel_box": region["box"], "pixel_points": old,
               "candidate_pixel_points": new, "candidate_geometry_sha256": geometry_digest(after["geometry"]),
               "raw_geometry_sha256": geometry_digest(before["geometry"]),
               "source_raster_sha256": tile["source_raster_sha256"], "crs_authid": tile["crs_authid"],
               "human_semantics": "unknown", "human_shape": "unknown", "human_approved": False,
               "training_eligible": False, "dataset_role": "review_only_not_training",
               "question_scope": "Only the highlighted observed fragment and its exact draft shape; not a new gap connection, whole region, or neighbouring line.",
               "comparison_image": question_id+".png"}
        selected.append((row, after, source))
    hashes = {str(p): sha256_file(p) for p in inputs}
    output.mkdir(parents=True)
    features = []
    for row, feature, source in selected:
        render_comparison(source, row, row["candidate_pixel_points"], output/row["comparison_image"],
                          title=row["question_id"]+" | observed fragment, NOT a gap connection | unapproved")
        item = copy.deepcopy(feature)
        item["properties"]["question_id"] = row["question_id"]
        features.append(item)
    if hashes != {str(p): sha256_file(p) for p in inputs}:
        raise ValueError("input changed while preparing human questions")
    result = {"schema": "jap-map-observed-line-questions/1", "questions": [row for row, _, _ in selected],
              "human_approvals": 0, "training_eligible": False, "holdout_used": False,
              "selection_basis": "Post-output low-support examples for human diagnosis, not an unbiased evaluation sample.",
              "source_inputs_sha256": hashes, "inputs_unchanged": True,
              "implementation_sha256": {str(Path(__file__).resolve()): sha256_file(__file__),
                  str(ROOT/"scripts"/"render_assisted_refinement.py"): sha256_file(ROOT/"scripts"/"render_assisted_refinement.py")}}
    write(output/"questions.json", result)
    write(output/"question-geometries.geojson", {"type": "FeatureCollection", "crs": candidate["crs"], "features": features})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("reconstruction", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--case", action="append", required=True, help="N001:R009:observed-000001")
    args = p.parse_args()
    cases = [tuple(c.split(":")) for c in args.case]
    if any(len(c) != 3 for c in cases):
        p.error("each --case must have question:region:path")
    result = run(args.packet, args.manifest, args.reconstruction, args.output, cases)
    print({"questions": len(result["questions"]), "human_approvals": 0, "output": str(args.output)})


if __name__ == "__main__":
    main()
