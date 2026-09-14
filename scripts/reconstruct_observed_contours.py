#!/usr/bin/env python3
"""Source-first experimental linework + bounded curve drafts in NEW files.

This is not an approved contour map. It exposes short/faint observed fragments,
suspected text and uncertainty separately. It never modifies the old network,
human ledger, source rasters or approved connectors. Regional blank reconnection
and a validated contour-vs-hydro semantic stage remain separate work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import importlib.metadata
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.map_text_detection import inspect_raster, validate_report
from histcontour_core.observed_linework import extract_observed_linework
from histcontour_core.observed_curve import refine_observed_curve
from histcontour_core.vectorization import _length
from histcontour_core.provenance import sha256_file
from histcontour_core.text_ownership import derive_text_ownership
from scripts.prepare_contour_reconstruction_review import read, write, validate_regions, intersects


def smooth_path(gray, skeleton, points, allowed_boxes, *, locked_source_spans=()):
    """Fit within a source crop; keep all other skeleton lines as clearance."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    p = np.asarray(points, dtype=float)
    left, top = np.maximum(0, np.floor(p.min(axis=0)).astype(int)-8)
    right, bottom = np.minimum([gray.shape[1], gray.shape[0]], np.ceil(p.max(axis=0)).astype(int)+9)
    local = p-[left, top]
    membership = np.array([[(l-.5 <= x <= r-.5 and t-.5 <= y <= b-.5) for l, t, r, b in allowed_boxes]
                           for x, y in p], dtype=bool)
    inside = membership.any(axis=1)
    # A partially outside edge is wholly locked. The observed-curve module
    # keeps its original polyline, not merely the vertices at either end.
    locked_spans = [(i, i+1) for i in range(len(p)-1) if not (membership[i] & membership[i+1]).any()]
    locked_spans.extend(locked_source_spans)
    others = skeleton[top:bottom, left:right].copy()
    indices = np.rint(local).astype(int)
    others[indices[:, 1], indices[:, 0]] = False
    clearance = distance_transform_edt(~others) if others.any() else None
    result = refine_observed_curve(gray[top:bottom, left:right], local, clearance=clearance,
                                   locked_indices=np.flatnonzero(~inside), locked_spans=locked_spans)
    audit = result["audit"]
    accepted = audit["changed"] and audit["after"]["total_turn_degrees"] <= audit["before"]["total_turn_degrees"]+1e-6
    # Source fitting is not necessarily smoothness improvement. Do not quietly
    # adopt a more wrinkled result; preserve it in the audit as an abstention.
    if not accepted:
        return copy.deepcopy(points), {**audit, "adopted_in_draft": False,
                                     "draft_rejection": "turning_increased" if audit["changed"] else "no_change"}
    output = (np.asarray(result["points"])+[left, top]).tolist()
    original_distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    original_samples = np.column_stack([np.interp(result["point_source_distances"], original_distances, p[:, a]) for a in (0, 1)])
    # Reject a fit that moves an originally inside sample outside the union
    # of requested regions. Outside/partially crossing spans remain locked.
    for i, q in enumerate(output):
        old = original_samples[i]
        if np.linalg.norm(np.asarray(q)-old) > 1e-8 and not any(l-.5 <= q[0] <= r-.5 and t-.5 <= q[1] <= b-.5 for l, t, r, b in allowed_boxes):
            return copy.deepcopy(points), {**audit, "adopted_in_draft": False, "draft_rejection": "moved_outside_requested_region"}
    output[0], output[-1] = list(points[0]), list(points[-1])
    return output, {**audit, "adopted_in_draft": True, "spans": result["spans"],
                    "original_vertex_indices": result["original_vertex_indices"]}


def run(packet, manifest_path, detection_path, approved_path, output, region_ids):
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    packet, output = Path(packet).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("reconstruction output must be new")
    if packet == output or packet in output.parents:
        raise ValueError("output must be outside source packet")
    if not region_ids or len(set(region_ids)) != len(region_ids):
        raise ValueError("provide distinct, nonempty predeclared region IDs")
    implementation = [Path(__file__), ROOT/"scripts"/"prepare_contour_reconstruction_review.py",
                      *[ROOT/"histcontour_core"/name for name in ("observed_linework.py", "observed_curve.py",
                         "text_ownership.py", "ink.py", "vectorization.py", "contours.py", "map_text_detection.py",
                         "provenance.py", "human_feedback.py", "margin_ocr.py", "paddle_margin_ocr.py")]]
    code_before = {str(p.resolve()): sha256_file(p) for p in implementation}
    report_path = packet/"drawing-report.json"
    report, manifest, detections, approved = map(read, (report_path, manifest_path, detection_path, approved_path))
    tiles = validate_regions(manifest, report)
    if sha256_file(report_path) != manifest["drawing_report_sha256"]:
        raise ValueError("source report differs from fixed development manifest")
    if detections.get("schema") != "jap-map-map-text-detection/1" or detections.get("holdout_used") is not False:
        raise ValueError("expected development-only raw text evidence")
    text_by_tile = {t["tile_id"]: t for t in detections["tiles"]}
    regions = [r for r in manifest["regions"] if r["region_id"] in region_ids]
    if set(region_ids) != {r["region_id"] for r in regions}:
        raise ValueError("only predeclared diagnostic regions are allowed")
    tids = sorted({r["tile_id"] for r in regions})
    sources, metadata = {}, {}
    paths = [report_path, Path(manifest_path), Path(detection_path), Path(approved_path)]
    for tid in tids:
        tile = tiles[tid]
        path = (packet/tile["raster_path"]).resolve()
        if Path(tile["raster_path"]).is_absolute() or packet not in path.parents:
            raise ValueError("source raster must remain inside packet")
        metadata[tid] = inspect_raster(path, tile)
        text = text_by_tile[tid]
        if any(text[k] != tile[k] for k in ("split", "source_raster_sha256", "pixel_bounds", "bounds", "crs_authid")):
            raise ValueError("text evidence source mismatch")
        sources[tid] = path
        paths.append(path)
    crs = {"type": "name", "properties": {"name": tiles[tids[0]]["crs_authid"]}}
    if approved.get("crs") != crs or any(f["geometry"]["type"] != "LineString" for f in approved["features"]):
        raise ValueError("approved geometry must have matching native CRS and line types")
    before = {str(p.resolve()): sha256_file(p) for p in paths}
    output.mkdir(parents=True)
    collections = {k: {"type": "FeatureCollection", "crs": crs, "features": []}
                   for k in ("raw-observed", "candidate-observed", "suspected-text", "short-context")}
    tile_rows, audits = [], []
    for tid in tids:
        tick = time.perf_counter()
        with Image.open(sources[tid]) as image:
            gray = np.asarray(image.convert("L"))
        text = text_by_tile[tid]["detection"]
        ownership = derive_text_ownership(gray, text["polygons_on_source_pixel_center_grid"], text["dt_scores"])
        reconstruction = extract_observed_linework(gray, text_avoidance_score=ownership.soft_text_avoidance,
                                                   tile_origin=tuple(tiles[tid]["pixel_bounds"][:2]))
        folder = output/tid
        folder.mkdir()
        for name, mask in reconstruction["masks"].items():
            Image.fromarray(np.uint8(mask)*255).save(folder/(name+".png"))
        Image.fromarray(np.uint8(ownership.soft_text_avoidance*255)).save(folder/"soft-text-avoidance.png")
        write(folder/"text-ownership.json", {"regions": list(ownership.region_evidence),
                                           "provenance": ownership.provenance})
        gt = metadata[tid]["geotransform"]
        def feature(path, points):
            props = {k: v for k, v in path.items() if k not in ("points", "length_px", "mean_continuous_support", "text_avoidance_fraction")}
            props.update(length_px=_length(points), source_trace_length_px=path["length_px"],
                         source_trace_mean_continuous_support=path.get("mean_continuous_support"),
                         source_trace_text_avoidance_fraction=path.get("text_avoidance_fraction"),
                         evidence_measurement_scope="support/text fractions belong to original raster graph, not refitted geometry")
            return {"type": "Feature", "properties": props | {
                "tile_id": tid, "source_raster_sha256": tiles[tid]["source_raster_sha256"],
                "dataset_role": "review_only_not_training", "whole_line_semantics_approved": False},
                "geometry": {"type": "LineString", "coordinates": [[gt[0]+(x+.5)*gt[1], gt[3]+(y+.5)*gt[5]] for x, y in points]}}
        stats = Counter()
        for path in reconstruction["paths"]:
            role = path["role"]
            stats[role] += 1
            if role != "observed_linework_review":
                name = "suspected-text" if role == "suspected_text_review" else "short-context"
                collections[name]["features"].append(feature(path, path["points"]))
                continue
            collections["raw-observed"]["features"].append(feature(path, path["points"]))
            points = path["points"]
            if path["length_px"] >= 12 and any(r["tile_id"] == tid and intersects(points, r["box"]) for r in regions):
                points, audit = smooth_path(gray, reconstruction["masks"]["skeleton"], points,
                    [r["box"] for r in regions if r["tile_id"] == tid])
                audits.append({"tile_id": tid, "path_id": path["path_id"], **audit})
                stats["curve_drafts_adopted"] += bool(audit["adopted_in_draft"])
                stats["curve_fits_attempted"] += 1
            collections["candidate-observed"]["features"].append(feature(path, points))
        tile_rows.append({"tile_id": tid, "stage_counts": reconstruction["stage_counts"],
                          "extraction_config": reconstruction["config"],
                          "path_counts": dict(stats), "wall_seconds": time.perf_counter()-tick})
        print(tid, dict(stats), flush=True)
    for name, collection in collections.items():
        write(output/(name+".geojson"), collection)
    # This is a separate, byte-identical layer. No fresh draft inherits these
    # approvals and no human geometry is substituted by raster tracing.
    shutil.copyfile(approved_path, output/"approved-connections-unchanged.geojson")
    if sha256_file(approved_path) != sha256_file(output/"approved-connections-unchanged.geojson"):
        raise ValueError("approved copy changed")
    after = {str(p.resolve()): sha256_file(p) for p in paths}
    if before != after:
        raise ValueError("input changed during source reconstruction")
    if code_before != {str(p.resolve()): sha256_file(p) for p in implementation}:
        raise ValueError("implementation changed during source reconstruction")
    result = {"schema": "jap-map-observed-reconstruction-run/1", "status": "experimental_not_applied",
              "regions": list(region_ids), "tiles": tile_rows, "curve_audits": audits,
              "provenance": {"input_sha256": before, "inputs_unchanged": True,
                  "implementation_sha256": code_before, "implementation_unchanged": True,
                  "runtime_versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "Pillow", "scikit-image")}},
              "holdout_used": False, "human_approvals": 0, "approved_coordinates_modified": False,
              "old_network_modified": False, "model_fitted": False, "wall_seconds": time.perf_counter()-started,
              "limitations": ["Fresh observed linework is not semantically equivalent to the old contour-selected baseline.",
                  "No claim of digit recall, contour accuracy or independent validation.",
                  "Only selected regions receive bounded curve drafts; other observed paths stay raw.",
                  "No blank-gap inference or global topology integration has been performed."]}
    write(output/"reconstruction-report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("detection", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", default=["R004", "R009", "R015"])
    args = parser.parse_args()
    run(args.packet, args.manifest, args.detection, args.approved, args.output, args.regions)


if __name__ == "__main__":
    main()
