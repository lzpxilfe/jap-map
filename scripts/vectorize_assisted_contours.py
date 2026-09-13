#!/usr/bin/env python3
"""Apply verified feedback refinements to a NEW candidate vector network.

Preserve a reviewed-connection-only alternative; automatic connections are
only a separate review scenario. No original source or judgment is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from histcontour_core.assisted_drawing import validate_machine_drawing_report
from histcontour_core.assisted_review import build_review_outputs, _tail_replacements
from histcontour_core.contour_network import merge_parts
from histcontour_core.human_feedback import geometry_digest, map_to_pixel
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import read, write
from scripts.refine_assisted_drawing_from_feedback import owner_mask, touches_other


def part(feature, kind, identity):
    result = copy.deepcopy(feature)
    result["properties"].update(part_id=identity,part_kind=kind,training_eligible=False,
                                dataset_role="review_only_not_training")
    if kind != "approved_connection":
        result["properties"].update(human_approved=False,whole_line_semantics_approved=False)
    return result


def assemble(report, originals, base, reviewed, drafts, candidates):
    """Pure assembly after file-level provenance checks by the CLI."""
    import numpy as np
    validate_machine_drawing_report(report)
    crs = originals["crs"]
    if base.get("crs") != crs or candidates.get("crs") != crs:
        raise ValueError("all network components must use the same explicit source CRS")
    if (drafts.get("schema") != "jap-map-feedback-gap-refinement/1" or drafts.get("holdout_used") is not False
            or drafts.get("human_approvals") != 0 or drafts.get("automatic_promotion") is not False):
        raise ValueError("expected unapproved development-only refinement scenario")
    tiles = {t["tile_id"]:t for t in report["tiles"]}
    rows = {r["proposal_id"]:r for r in report["proposals"]}
    bases = {f["properties"]["segment_uid"]:copy.deepcopy(f) for f in base["features"]}
    if len(bases) != len(base["features"]):
        raise ValueError("duplicate source line identity")
    original_by_id = {f["properties"]["proposal_id"]:f for f in originals["features"]}
    specs = {r["revision_id"]:r for r in drafts["revisions"]}
    changes = {f["properties"]["proposal_id"]:f for f in candidates["features"]}
    states = {r["proposal_id"]:r for r in drafts["decisions"]}
    if (len(specs) != len(drafts["revisions"]) or len(changes) != len(candidates["features"])
            or set(specs) != set(changes) or len(states) != len(drafts["decisions"]) or set(states) != set(rows)):
        raise ValueError("refinement identities do not match source report")
    locked = set(reviewed["queue"]["responded_ids_not_reasked"])
    deferred = set(reviewed["queue"]["deferred_ids"])
    if ({sid for sid,s in states.items() if s["status"] == "locked_human_review"} != locked
            or {sid for sid,s in states.items() if s["status"] == "deferred_context"} != deferred):
        raise ValueError("refinement locks differ from human review")
    approved = reviewed["collections"]["approved_contour"]["features"]
    approved_ids = {f["properties"]["proposal_id"] for f in approved}
    approved_tails = [f for f in reviewed["source_tail_replacements"]["features"]
                      if f["properties"]["local_patch_revision_id"] in approved_ids]
    changed_uids = set(); tail_features = []
    for feature in approved_tails:
        uid = feature["properties"]["segment_uid"]
        if uid in changed_uids or geometry_digest(bases[uid]["geometry"]) != feature["properties"]["replaced_source_geometry_sha256"]:
            raise ValueError("approved source-tail identity changed")
        bases[uid] = copy.deepcopy(feature); changed_uids.add(uid)
        tail_features.append(copy.deepcopy(feature))
    for feature in approved:
        props = feature["properties"]; tile = tiles[props["tile_id"]]
        pixels = map_to_pixel(tile,feature["geometry"]["coordinates"])
        for side,index in (("source_uid",0),("target_uid",-1)):
            line = bases[props[side]]["geometry"]["coordinates"]
            ends = map_to_pixel(tile,[line[0],line[-1]])
            if sum(math.dist(p,pixels[index]) <= 1e-5 for p in ends) != 1:
                raise ValueError("approved connection no longer joins the retained source endpoints")
    reviewed_parts = [part(f,"retained_source","base:"+uid) for uid,f in bases.items()]
    reviewed_parts += [part(f,"approved_connection","human:"+f["properties"]["proposal_id"]) for f in approved]
    automatic, omitted, extra_parts = [], [], []
    # Validate every proposed revision, including ones not applied by policy.
    by_parent = {}
    for rid,spec in specs.items():
        feature = changes[rid]; props = feature["properties"]; sid = spec["parent_proposal_id"]
        if (sid in by_parent or sid in locked|deferred or sid not in rows or spec.get("approved") is not False
                or props.get("human_approved") is not False or props.get("training_eligible") is not False
                or props.get("parent_proposal_id") != sid or geometry_digest(feature["geometry"]) != spec["geometry_sha256"]
                or states[sid].get("revision_id") != rid or states[sid].get("geometry_sha256") != spec["geometry_sha256"]
                or states[sid]["status"] != "proposed"):
            raise ValueError("a candidate revision is stale, reviewed, duplicated or modified")
        if spec.get("revision_kind") != props.get("revision_kind"):
            raise ValueError("refinement geometry contract differs")
        row = rows[sid]
        for key in ("tile_id","source_uid","target_uid"):
            if props.get(key) != row[key]:
                raise ValueError("refinement changed its source identity")
        if spec["revision_kind"] == "local_tail_replacement":
            _tail_replacements(spec,feature,original_by_id[sid]["geometry"]["coordinates"],row,tiles[row["tile_id"]],base,crs)
        elif spec["revision_kind"] == "fixed_endpoints":
            old,new = original_by_id[sid]["geometry"]["coordinates"],feature["geometry"]["coordinates"]
            if old[0] != new[0] or old[-1] != new[-1]:
                raise ValueError("fixed-endpoint revision moved a join")
        else:
            raise ValueError("unknown refinement geometry contract")
        by_parent[sid] = (feature,spec)
    for tid,tile in tiles.items():
        shape = tuple(reversed(tile["pixel_bounds"][2:]))
        uids = [uid for uid,f in bases.items() if f["properties"]["tile_id"] == tid]
        identities = {uid:i for i,uid in enumerate(uids,1)}
        source_owner = owner_mask(shape,[(identities[uid],map_to_pixel(tile,bases[uid]["geometry"]["coordinates"])) for uid in uids])
        accepted = [f for f in reviewed["reviewed"]["features"] if f["properties"]["tile_id"] == tid and f["properties"]["human_geometry_accepted"]]
        accepted_owner = owner_mask(shape,[(i,map_to_pixel(tile,f["geometry"]["coordinates"])) for i,f in enumerate(accepted,1)])
        used_owner = np.zeros(shape,dtype=np.int32)
        protected_uids = {rows[sid][k] for sid in locked|deferred for k in ("source_uid","target_uid")}
        choices = [r for r in report["proposals"] if r["tile_id"] == tid and r["proposal_id"] not in locked|deferred]
        choices.sort(key=lambda r:(states[r["proposal_id"]]["status"] != "proposed",r["proposal_id"]))
        for row in choices:
            sid = row["proposal_id"]; state = states[sid]
            if state["status"] == "proposed":
                feature,spec = by_parent[sid]
                if state.get("anchor_review_required") or feature["properties"].get("anchor_review_required"):
                    omitted.append({"proposal_id":sid,"reason":"weak_anchor_evidence_not_assembled"}); continue
            elif state["status"] == "unchanged" and state.get("reasons") == ["current_tips_and_shape_are_supported"]:
                feature = copy.deepcopy(original_by_id[sid]); spec = None
            else:
                omitted.append({"proposal_id":sid,"reason":"context_review_required"}); continue
            proposed_tails = []
            if spec and spec["revision_kind"] == "local_tail_replacement":
                if {row["source_uid"],row["target_uid"]} & (changed_uids|protected_uids):
                    omitted.append({"proposal_id":sid,"reason":"shared_tail_requires_combined_contract"}); continue
                proposed_tails = _tail_replacements(spec,feature,original_by_id[sid]["geometry"]["coordinates"],row,tile,base,crs)
            effective = {f["properties"]["segment_uid"]:f for f in proposed_tails}
            pixels = map_to_pixel(tile,feature["geometry"]["coordinates"])
            bad = False
            for key,index in (("source_uid",0),("target_uid",-1)):
                line = effective.get(row[key],bases[row[key]])["geometry"]["coordinates"]
                endpoints = map_to_pixel(tile,[line[0],line[-1]])
                if sum(math.dist(p,pixels[index]) <= 1e-5 for p in endpoints) != 1:
                    bad = True
            if bad:
                omitted.append({"proposal_id":sid,"reason":"endpoint_consumed_or_relocated"}); continue
            allowed = [identities[row[k]] for k in ("source_uid","target_uid")]
            if (touches_other(pixels,source_owner,allowed) or touches_other(pixels,accepted_owner)
                    or touches_other(pixels,used_owner)):
                omitted.append({"proposal_id":sid,"reason":"whole_path_topology_conflict"}); continue
            for replacement in proposed_tails:
                uid = replacement["properties"]["segment_uid"]
                replacement["properties"]["tail_application_scope"] = "machine_candidate_scenario_not_approved"
                bases[uid] = replacement; changed_uids.add(uid); tail_features.append(replacement)
            item = part(feature,"automatic_connection","auto:"+feature["properties"]["proposal_id"])
            item["properties"].update(parent_proposal_id=sid,application_scope="machine_candidate_scenario_not_approved",
                automatic_method="feedback_refinement" if spec else "supported_original_connection",
                requires_source_tail_replacement=bool(proposed_tails),source_tail_replacement_applied_in_this_scenario=bool(proposed_tails))
            extra_parts.append(item); automatic.append(item)
            from histcontour_core.completion import _line_pixels
            for x,y in _line_pixels(pixels):
                used_owner[y,x] = len(automatic)
    enhanced_parts = [part(f,"retained_source","base:"+uid) for uid,f in bases.items()]
    enhanced_parts += [part(f,"approved_connection","human:"+f["properties"]["proposal_id"]) for f in approved]+extra_parts
    baseline = merge_parts([part(f,"retained_source","base:"+f["properties"]["segment_uid"]) for f in base["features"]],tiles)
    human_network = merge_parts(reviewed_parts,tiles)
    enhanced = merge_parts(enhanced_parts,tiles)
    # A merged feature is never promoted to whole-line contour truth.
    return {"baseline":baseline,"reviewed_network":human_network,"enhanced_network":enhanced,
            "reviewed_parts":reviewed_parts,"enhanced_parts":enhanced_parts,
            "automatic_connections":automatic,"approved_connections":copy.deepcopy(approved),
            "tail_replacements":tail_features,"omitted":omitted,"deferred_ids":sorted(deferred),
            "approved_connection_count":len(approved),"automatic_connection_count":len(automatic),
            "approved_source_tail_replacements":len(approved_tails),
            "automatic_source_tail_replacements":len(tail_features)-len(approved_tails)}


def run(packet,ledger_path,revisions_path,refinement_dir,output):
    started = time.perf_counter()
    packet,ledger_path,revisions_path,refinement_dir,output = [Path(p).resolve() for p in (packet,ledger_path,revisions_path,refinement_dir,output)]
    if output.exists():
        raise FileExistsError("vectorization output must be new")
    if output.is_relative_to(packet) or output.is_relative_to(refinement_dir):
        raise ValueError("never assemble inside source or refinement snapshots")
    paths = {"source_report_sha256":packet/"drawing-report.json","source_proposals_sha256":packet/"ai-proposals.geojson",
             "source_base_lines_sha256":packet/"base-lines.geojson","ledger_sha256":ledger_path,"revision_collection_sha256":revisions_path}
    report,originals,base,ledger,revisions = [read(p) for p in paths.values()]
    drafts = read(refinement_dir/"refinement-report.json"); candidates = read(refinement_dir/"proposed-revisions.geojson")
    hashes = {key:sha256_file(path) for key,path in paths.items()}
    for key,digest in hashes.items():
        if digest != drafts["inputs_sha256"].get(key) or (key != "ledger_sha256" and digest != ledger.get(key)):
            raise ValueError("source or human review differs from refinement snapshot")
    for name,digest in drafts["implementation_sha256"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError("refinement implementation changed; regenerate its candidates first")
    reviewed = build_review_outputs(report,originals,ledger,revisions,base_lines=base)
    for tile in report["tiles"]:
        source = (packet/tile["raster_path"]).resolve()
        if not source.is_relative_to(packet) or sha256_file(source) != tile["source_raster_sha256"]:
            raise ValueError("source raster changed")
        key = "raster_"+tile["tile_id"]; paths[key] = source; hashes[key] = tile["source_raster_sha256"]
    for name in ("refinement-report.json","proposed-revisions.geojson"):
        paths[name] = refinement_dir/name; hashes[name] = sha256_file(paths[name])
    result = assemble(report,originals,base,reviewed,drafts,candidates)
    if any(sha256_file(path) != hashes[key] for key,path in paths.items()):
        raise RuntimeError("source changed during assembly; no original file was restored or overwritten")
    output.mkdir(parents=True,exist_ok=False); (output/"sources").mkdir(); (output/"provenance").mkdir()
    collection = lambda features: {"type":"FeatureCollection","crs":copy.deepcopy(base["crs"]),"features":features}
    for name,features in (("contour-candidates",result["enhanced_network"]["features"]),
            ("reviewed-connection-network",result["reviewed_network"]["features"]),
            ("source-before",base["features"]),("approved-connections",result["approved_connections"]),
            ("automatic-connections",result["automatic_connections"]),
            ("assembled-parts",result["enhanced_parts"]),("source-tail-replacements",result["tail_replacements"])):
        write(output/(name+".geojson"),collection(features))
    for tile in report["tiles"]:
        shutil.copyfile(packet/tile["raster_path"],output/"sources"/(tile["tile_id"]+".tif"))
    for name,value in (("review-ledger.json",ledger),("approved-revisions.geojson",revisions),
                       ("refinement-report.json",drafts),("proposed-revisions.geojson",candidates)):
        write(output/"provenance"/name,value)
    write(output/"network-membership.json",result["enhanced_network"]["memberships"])
    write(output/"application-audit.json",{key:result[key] for key in ("omitted","deferred_ids","approved_source_tail_replacements","automatic_source_tail_replacements")})
    omitted_by_id = {r["proposal_id"]:r["reason"] for r in result["omitted"]}
    omitted_by_id.update({sid:"deferred_human_context" for sid in result["deferred_ids"]})
    pending = []
    for feature in originals["features"]:
        sid = feature["properties"]["proposal_id"]
        if sid in omitted_by_id:
            item = copy.deepcopy(feature)
            item["properties"].update(not_assembled_reason=omitted_by_id[sid],human_approved=False,
                                       training_eligible=False,geometry_role="unapplied_original_reference")
            pending.append(item)
    write(output/"needs-review.geojson",collection(pending))
    counts = {"source_fragments":len(base["features"]),"original_endpoint_merged_lines":result["baseline"]["line_count"],
              "reviewed_connection_network_lines":result["reviewed_network"]["line_count"],
              "enhanced_candidate_lines":result["enhanced_network"]["line_count"],
              "approved_connections":result["approved_connection_count"],"automatic_connections":result["automatic_connection_count"],
              "automatic_refined":sum(f["properties"]["automatic_method"] == "feedback_refinement" for f in result["automatic_connections"]),
              "automatic_supported_original":sum(f["properties"]["automatic_method"] == "supported_original_connection" for f in result["automatic_connections"]),
              "open_endpoints_before":result["baseline"]["open_endpoint_nodes"],"open_endpoints_after":result["enhanced_network"]["open_endpoint_nodes"],
              "source_tail_replacements":len(result["tail_replacements"])}
    counts["unapplied_review_connections"] = len(pending)
    summary = {"schema":"jap-map-feedback-vectorization/1","status":"candidate_network_assembled_not_human_approved",
               "source_inputs_sha256":hashes,"counts":counts,"omission_reasons":dict(Counter(x["reason"] for x in result["omitted"])),
               "native_crs":base["crs"]["properties"]["name"],"tiles":[{**t,"raster_path":"sources/"+t["tile_id"]+".tif"} for t in report["tiles"]],
               "network_topology":{key:result[key]["tiles"] for key in ("baseline","reviewed_network","enhanced_network")},
               "original_source_modified":False,"approved_geometry_modified":False,"new_human_approvals":0,
               "whole_network_human_approved":False,"model_fitted":False,"holdout_used":False,"elevations_assigned":False,
               "scope":"Existing nine development tile vectors plus verified feedback geometry; not whole-sheet re-extraction or independently validated contours.",
               "assembly_rules":["Replace specified source tails in this separate scenario before adding the patch.",
                   "Exclude weak-anchor drafts, context-review cases, semantic-only uncertainty, non-contours and rejected routes from added connections.",
                   "Merge only degree-two endpoint nodes within 1e-5 source pixel round-off; preserve every component vertex and tile boundary.",
                   "Human contour approval belongs only to the exact approved connector, never the whole merged line."],
               "elapsed_seconds":time.perf_counter()-started,
               "implementation_sha256":{name:sha256_file(ROOT/name) for name in ("histcontour_core/contour_network.py","scripts/vectorize_assisted_contours.py")}}
    summary["output_sha256"] = {p.name:sha256_file(p) for p in sorted(output.glob('*.geojson'))}
    write(output/"vectorization-report.json",summary)
    print(json.dumps(counts,ensure_ascii=False),flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet","ledger","revisions","refinements"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    try:
        run(args.packet,args.ledger,args.revisions,args.refinements,args.output)
    except (OSError,ValueError,KeyError) as error:
        parser.exit(2,f"Vectorization stopped: {error}\n")
