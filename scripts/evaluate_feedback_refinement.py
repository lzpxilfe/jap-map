#!/usr/bin/env python3
"""Development-only regression against exact approved revisions; not accuracy.

The same feedback informed this algorithm. No held-out test, retraining, or
human approval of a new algorithm output is implied by these measurements.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.assisted_review import build_review_outputs
from histcontour_core.gap_refinement import GapRefinementConfig, shape_audit
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import read, write
from scripts.refine_assisted_drawing_from_feedback import geometry_draft
from scripts.render_assisted_refinement import render_comparison


def gap_span_error(original, candidate, reference):
    """Mean/max perpendicular separation on the COMMON original-gap span.

    Do not penalize a correct extended tail for having different endpoints.
    Nonmonotone curves need a different metric and are not force-projected.
    """
    import numpy as np
    a, b = np.asarray(original, dtype=float)[[0,-1]]
    gap = np.linalg.norm(b-a); axis = (b-a)/gap; normal = np.array([-axis[1],axis[0]])
    positions = np.linspace(0,gap,201)
    def ordinate(points):
        p = np.asarray(points,dtype=float)-a; u, v = p@axis,p@normal
        if np.any(np.diff(u) < -1e-6) or u[0] > 1e-5 or u[-1] < gap-1e-5:
            raise ValueError("geometry cannot be compared over the original forward gap span")
        return np.interp(positions,u,v)
    separation = abs(ordinate(candidate)-ordinate(reference))
    return {"mean_normal_error_pixels":float(separation.mean()), "max_normal_error_pixels":float(separation.max())}


def evaluate(packet, ledger_path, revisions_path, refinement_output, output):
    import numpy as np
    from PIL import Image
    packet, ledger_path, revisions_path, refinement_output, output = map(Path,(packet,ledger_path,revisions_path,refinement_output,output))
    if output.exists():
        raise FileExistsError("development evaluation output must be new")
    if output.resolve().is_relative_to(packet.resolve()):
        raise ValueError("never write evaluation into original packet")
    report, originals, base, ledger, revisions, drafts = [read(p) for p in (
        packet/"drawing-report.json",packet/"ai-proposals.geojson",packet/"base-lines.geojson",ledger_path,revisions_path,refinement_output/"refinement-report.json")]
    paths = {"source_report_sha256":packet/"drawing-report.json", "source_proposals_sha256":packet/"ai-proposals.geojson",
             "source_base_lines_sha256":packet/"base-lines.geojson", "ledger_sha256":ledger_path, "revision_collection_sha256":revisions_path}
    for key,path in paths.items():
        actual = sha256_file(path)
        if actual != drafts["inputs_sha256"].get(key) or (key != "ledger_sha256" and actual != ledger.get(key)):
            raise ValueError("evaluation inputs differ from refinement and human snapshots")
    for name,digest in drafts["implementation_sha256"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError("implementation changed since refinement run")
    reviewed = build_review_outputs(report,originals,ledger,revisions,base_lines=base)
    expected_locked = set(reviewed["queue"]["responded_ids_not_reasked"])
    actual_locked = {d["proposal_id"] for d in drafts["decisions"] if d["status"] == "locked_human_review"}
    new_features = read(refinement_output/"proposed-revisions.geojson")["features"]
    if actual_locked != expected_locked or any(f["properties"]["parent_proposal_id"] in expected_locked for f in new_features):
        raise ValueError("human-reviewed case escaped its lock")
    tiles = {t["tile_id"]:t for t in report["tiles"]}; rows = {r["proposal_id"]:r for r in report["proposals"]}
    bases = {f["properties"]["segment_uid"]:f for f in base["features"]}; config = GapRefinementConfig(**drafts["config"]["settings"])
    images = {}
    for tid,tile in tiles.items():
        source = (packet/tile["raster_path"]).resolve()
        if not source.is_relative_to(packet.resolve()) or sha256_file(source) != tile["source_raster_sha256"]:
            raise ValueError("source raster changed")
        with Image.open(source) as image:
            images[tid] = np.asarray(image.convert("L"),dtype=float)
    records, pictures = [], []
    specs = {r["revision_id"]:r for r in ledger["revisions"]}
    for feature in revisions["features"]:
        rid = feature["properties"]["proposal_id"]; spec = specs[rid]; sid = spec["parent_proposal_id"]
        row = rows[sid]; tile = tiles[row["tile_id"]]
        sources = [map_to_pixel(tile,bases[row[key]]["geometry"]["coordinates"]) for key in ("source_uid","target_uid")]
        result = geometry_draft(images[row["tile_id"]],*sources,row["pixel_points"],config,
                                competing_pair=row.get("competing_endpoint_pair", False))
        reference = map_to_pixel(tile,feature["geometry"]["coordinates"])
        candidate = result["points"]
        entry = {"proposal_id":sid,"approved_reference_id":rid,"qualification":spec.get("qualification",""),
                 "new_geometry_human_approved":False,"production_geometry_stays_locked":True,
                 "diagnostic_status":result["status"],"diagnostic_kind":result["kind"],
                 "before_shape":shape_audit(row["pixel_points"]),"after_shape":shape_audit(candidate)}
        try:
            entry.update(before=gap_span_error(row["pixel_points"],row["pixel_points"],reference),
                         after=gap_span_error(row["pixel_points"],candidate,reference))
        except ValueError as error:
            entry["metric_unavailable"] = str(error)
        records.append(entry); pictures.append((row,tile,candidate,rid))
    unqualified = [r for r in records if not r["qualification"] and "before" in r]
    before = float(np.mean([r["before"]["mean_normal_error_pixels"] for r in unqualified])) if unqualified else None
    after = float(np.mean([r["after"]["mean_normal_error_pixels"] for r in unqualified])) if unqualified else None
    summary = {"schema":"jap-map-feedback-refinement-development-evaluation/1",
               "scope":"same-feedback development regression, not independent accuracy",
               "formal_accuracy":None,"holdout_used":False,"classifier_fitted":False,
               "unqualified_reference_count":len(unqualified),"qualified_reference_count":sum(bool(r["qualification"]) for r in records),
               "mean_of_case_mean_normal_errors_pixels":{"before":before,"after":after},
               "human_reviewed_lock_count":len(actual_locked),"human_reviewed_cases_reproposed":0,
               "production_approved_revision_bytes_changed":False,
               "refinement_report_sha256":sha256_file(refinement_output/"refinement-report.json"),
               "candidate_collection_sha256":sha256_file(refinement_output/"proposed-revisions.geojson"),
               "evaluation_implementation_sha256":sha256_file(Path(__file__)),
               "reference_collection_sha256":sha256_file(revisions_path),"records":records}
    output.mkdir(parents=True,exist_ok=False)
    write(output/"development-evaluation.json",summary)
    for row,tile,candidate,rid in pictures:
        render_comparison(packet/tile["raster_path"],row,candidate,output/(rid+"-diagnostic.png"),
                          title=rid+" | diagnostic only; approved reference stays unchanged")
    print({k:v for k,v in summary.items() if k not in ("records",)})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet",type=Path); parser.add_argument("ledger",type=Path)
    parser.add_argument("revisions",type=Path); parser.add_argument("refinement_output",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    try:
        evaluate(args.packet,args.ledger,args.revisions,args.refinement_output,args.output)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(2,f"Development evaluation stopped: {error}\n")
