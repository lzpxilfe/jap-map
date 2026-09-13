#!/usr/bin/env python3
"""Draft short-gap refinements without changing any human-reviewed geometry.

The ledger is used for locks, not classifier fitting. Outputs are NEW,
unapproved alternatives in the source CRS. Local-tail changes cannot be
appended to a source network: each needs separate approval and its contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import copy
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.assisted_drawing import validate_machine_drawing_report
from histcontour_core.assisted_review import build_review_outputs, _tail_replacements, _trim_at_endpoint
from histcontour_core.completion import _line_pixels
from histcontour_core.gap_refinement import GapRefinementConfig, REFINEMENT_VERSION, propose_gap_refinement, regularize_gap_path
from histcontour_core.human_feedback import geometry_digest, map_to_pixel
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import read, write, world


def owner_mask(shape, paths):
    """A shared pixel keeps a conflict marker, never just the last identity."""
    import numpy as np
    owner = np.zeros(shape, dtype=np.int32)
    for identity, points in paths:
        for x, y in _line_pixels(points):
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                owner[y, x] = identity if owner[y, x] in (0, identity) else -1
    return owner


def touches_other(points, owner, allowed=()):
    """Check the WHOLE path, including joins, against a one-pixel halo."""
    import numpy as np
    accepted = {0, *allowed}
    for x, y in _line_pixels(points):
        if not 0 <= y < owner.shape[0] or not 0 <= x < owner.shape[1]:
            return True
        if any(int(value) not in accepted for value in np.unique(owner[max(0,y-1):y+2, max(0,x-1):x+2])):
            return True
    return False


def geometry_draft(gray, source, target, points, config, *, competing_pair=False):
    """Keep bounded smoothing separate from evidence needed to move tips.

    A subpixel smoothing draft can still be shown if the anchor estimator
    abstains. Its evidence warning is retained; it is not a supported link.
    """
    result = propose_gap_refinement(gray, source, target, points, config=config)
    if competing_pair:
        return {"status": "context_review_required", "kind": "fixed_endpoints",
                "points": [list(p) for p in points], "reasons": ["competing_endpoint_pair_needs_context"],
                "human_approved": False, "model_fitted": False, "requires_source_tail_replacement": False}
    if "local_fit_reverses_existing_bend" in result["reasons"]:
        return result
    if result["status"] == "context_review_required":
        smooth, audit = regularize_gap_path(points, config)
        if audit["changed"]:
            result.update(status="proposed", points=smooth, kind="fixed_endpoints",
                          anchor_review_required=True,
                          reasons=["bounded_smoothing_only_anchor_estimator_abstained", *result["reasons"]])
    return result


def candidate_feature(row, tile, original, bases, result, crs, config_digest):
    coordinates = world(tile, result["points"])
    contract = None
    if result["kind"] == "local_tail_replacement":
        contract = {"schema": "jap-map-local-tail-replacement/1", "join_points_pixels": []}
        for side, end in (("source", 0), ("target", -1)):
            uid = row[side+"_uid"]; feature = bases[uid]
            trim = result[side+"_tail_trim_pixels"]
            cut, _, _ = _trim_at_endpoint(feature["geometry"]["coordinates"],
                                         original["geometry"]["coordinates"][end], trim, tile)
            coordinates[end] = cut  # same native-CRS node as the actual source cut
            contract.update({side+"_uid": uid, side+"_geometry_sha256": geometry_digest(feature["geometry"]),
                             side+"_tail_trim_pixels": trim})
            contract["join_points_pixels"].append(map_to_pixel(tile, [cut])[0])
    else:
        coordinates[0], coordinates[-1] = copy.deepcopy(original["geometry"]["coordinates"][0]), copy.deepcopy(original["geometry"]["coordinates"][-1])
    geometry = {"type": "LineString", "coordinates": coordinates}
    digest = geometry_digest(geometry)
    rid = row["proposal_id"]+"-F1-"+digest[:8]
    feature = {"type": "Feature", "geometry": geometry, "properties": {
        "proposal_id": rid, "parent_proposal_id": row["proposal_id"], "tile_id": row["tile_id"],
        "source_uid": row["source_uid"], "target_uid": row["target_uid"],
        "source_raster_sha256": tile["source_raster_sha256"],
        "original_geometry_sha256": geometry_digest(original["geometry"]), "geometry_sha256": digest,
        "refinement_version": REFINEMENT_VERSION, "config_sha256": config_digest,
        "revision_kind": result["kind"], "human_approved": False, "human_geometry_accepted": False,
        "review_status": "unreviewed_refinement", "semantic_decision": "not_separately_stated",
        "dataset_role": "review_only_not_training", "training_eligible": False,
        "reference_origin": "machine_generated_not_human",
        "requires_source_tail_replacement": contract is not None, "append_only_safe": contract is None,
        "anchor_review_required": result.get("anchor_review_required", False)}}
    spec = {"revision_id": rid, "parent_proposal_id": row["proposal_id"],
            "revision_kind": result["kind"], "geometry_sha256": digest, "approved": False}
    if contract:
        spec["tail_replacement_contract"] = contract
        # Reuse the production exporter's exact cut/hash/CRS/corridor checks.
        _tail_replacements(spec, feature, original["geometry"]["coordinates"], row, tile,
                           {"type": "FeatureCollection", "crs": crs, "features": [bases[row[k]] for k in ("source_uid", "target_uid")]}, crs)
    return feature, spec


def balanced_review_queue(candidates):
    """Round-robin tiles within each evidence/geometry tier; no class labels."""
    groups = defaultdict(list)
    for row in candidates:
        groups[(row.get("anchor_review_required", False), row["kind"] != "local_tail_replacement", row["tile_id"])].append(row)
    result = []
    tiers = sorted({key[:2] for key in groups})
    for tier in tiers:
        queues = [deque(sorted(groups[key], key=lambda row: row["proposal_id"])) for key in sorted(groups) if key[:2] == tier]
        while any(queues):
            for queue in queues:
                if queue:
                    result.append(queue.popleft())
    return result


def refine(packet, ledger_path, output, *, revisions_path=None, config=GapRefinementConfig(), previews=6):
    import numpy as np
    from PIL import Image
    started = time.perf_counter(); config.validate()
    packet, ledger_path, output = Path(packet).resolve(), Path(ledger_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refinement output must be a new directory")
    if output.is_relative_to(packet):
        raise ValueError("never write refinements into the original packet")
    if type(previews) is not int or not 0 <= previews <= 30:
        raise ValueError("previews must be an integer from 0 to 30")
    report, originals, base, ledger = (read(p) for p in (packet/"drawing-report.json", packet/"ai-proposals.geojson", packet/"base-lines.geojson", ledger_path))
    validate_machine_drawing_report(report)
    paths = {"source_report_sha256": packet/"drawing-report.json", "source_proposals_sha256": packet/"ai-proposals.geojson",
             "source_base_lines_sha256": packet/"base-lines.geojson", "ledger_sha256": ledger_path}
    if revisions_path is not None:
        revisions_path = Path(revisions_path).resolve()
        paths["revision_collection_sha256"] = revisions_path
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    for key in paths:
        if key != "ledger_sha256" and hashes[key] != ledger.get(key):
            raise ValueError("source or approved-revision fingerprint differs from ledger: "+key)
    reviewed = build_review_outputs(report, originals, ledger, read(revisions_path) if revisions_path else None, base_lines=base)
    if base.get("crs") != originals["crs"]:
        raise ValueError("base lines must retain the source CRS")
    bases = {f["properties"]["segment_uid"]: f for f in base["features"]}
    if len(bases) != len(base["features"]):
        raise ValueError("duplicate base-line identity")
    original_by_id = {f["properties"]["proposal_id"]: f for f in originals["features"]}
    rows = {r["proposal_id"]: r for r in report["proposals"]}
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", sid) for sid in rows):
        raise ValueError("proposal IDs must be safe artifact names")
    tiles = {t["tile_id"]: t for t in report["tiles"]}
    gray_by_tile, pixels_by_uid = {}, {}
    # Validate every input before creating any output.
    for tid, tile in tiles.items():
        source = (packet/tile["raster_path"]).resolve()
        if not source.is_relative_to(packet) or sha256_file(source) != tile["source_raster_sha256"]:
            raise ValueError("source raster path or fingerprint differs")
        with Image.open(source) as image:
            gray = np.asarray(image.convert("L"), dtype=float)
        if gray.shape != tuple(reversed(tile["pixel_bounds"][2:])):
            raise ValueError("source raster dimensions differ")
        gray_by_tile[tid] = gray
        paths["raster_"+tid] = source; hashes["raster_"+tid] = tile["source_raster_sha256"]
    for uid, feature in bases.items():
        pixels_by_uid[uid] = map_to_pixel(tiles[feature["properties"]["tile_id"]], feature["geometry"]["coordinates"])
    locked = set(reviewed["queue"]["responded_ids_not_reasked"])
    deferred = set(reviewed["queue"]["deferred_ids"])
    protected_uids = {rows[sid][key] for sid in locked|deferred for key in ("source_uid", "target_uid")}
    config_record = {"version": REFINEMENT_VERSION, "settings": asdict(config)}
    config_digest = geometry_digest(config_record)
    decisions, features, specs = [], [], []
    for tid, tile in tiles.items():
        shape = gray_by_tile[tid].shape
        selected_bases = [uid for uid, f in bases.items() if f["properties"]["tile_id"] == tid]
        identities = {uid: i for i, uid in enumerate(selected_bases, 1)}
        source_owner = owner_mask(shape, [(identities[uid], pixels_by_uid[uid]) for uid in selected_bases])
        tile_rows = [r for r in report["proposals"] if r["tile_id"] == tid]
        proposal_ids = {r["proposal_id"]: i for i, r in enumerate(tile_rows, 1)}
        original_owner = owner_mask(shape, [(proposal_ids[r["proposal_id"]], r["pixel_points"]) for r in tile_rows])
        accepted = [f for f in reviewed["reviewed"]["features"] if f["properties"]["tile_id"] == tid and f["properties"]["human_geometry_accepted"]]
        accepted_owner = owner_mask(shape, [(i, map_to_pixel(tile, f["geometry"]["coordinates"])) for i, f in enumerate(accepted, 1)])
        draft_owner = np.zeros(shape, dtype=np.int32); modified_uids = set()
        for row in tile_rows:
            sid = row["proposal_id"]; item = {"proposal_id": sid, "tile_id": tid}
            decisions.append(item)
            if sid in locked|deferred:
                item.update(status="locked_human_review" if sid in locked else "deferred_context", reasons=["existing_human_state_preserved"])
                continue
            source_uid, target_uid = row["source_uid"], row["target_uid"]
            if source_uid == target_uid or any(bases[uid]["properties"]["tile_id"] != tid for uid in (source_uid, target_uid)):
                raise ValueError("proposal needs two distinct source lines from its tile")
            result = geometry_draft(gray_by_tile[tid], pixels_by_uid[source_uid], pixels_by_uid[target_uid], row["pixel_points"], config,
                                    competing_pair=row.get("competing_endpoint_pair", False))
            item.update({key: value for key, value in result.items() if key != "points"})
            if result["status"] != "proposed":
                continue
            reasons = []
            pair_uids = {source_uid, target_uid}
            if result["kind"] == "local_tail_replacement" and pair_uids & (protected_uids|modified_uids):
                reasons.append("shared_source_tail_needs_combined_review_contract")
            if touches_other(result["points"], source_owner, [identities[uid] for uid in pair_uids]):
                reasons.append("touches_third_source_line_including_join_halo")
            if touches_other(result["points"], original_owner, [proposal_ids[sid]]):
                reasons.append("touches_another_original_proposal")
            if touches_other(result["points"], accepted_owner):
                reasons.append("touches_accepted_human_geometry")
            if touches_other(result["points"], draft_owner):
                reasons.append("touches_another_refinement_draft")
            if reasons:
                item.update(status="context_review_required", reasons=reasons+result["reasons"])
                continue
            feature, spec = candidate_feature(row, tile, original_by_id[sid], bases, result, originals["crs"], config_digest)
            features.append(feature); specs.append(spec)
            item.update(revision_id=spec["revision_id"], geometry_sha256=spec["geometry_sha256"], topology_screen_passed=True)
            if result["kind"] == "local_tail_replacement":
                modified_uids.update(pair_uids)
            for x, y in _line_pixels(result["points"]):
                draft_owner[y, x] = len(features)
    if any(sha256_file(path) != hashes[key] for key, path in paths.items()):
        raise RuntimeError("input changed during refinement; nothing was restored or overwritten")
    candidates = [d for d in decisions if d["status"] == "proposed"]
    # Strongly supported anchor corrections first; uncertain smoothing later.
    candidates = balanced_review_queue(candidates)
    counts = dict(Counter(d["status"] for d in decisions))
    summary = {"schema": "jap-map-feedback-gap-refinement/1", "version": REFINEMENT_VERSION,
               "scope": "feedback-informed development drafts; not an independent accuracy evaluation",
               "config": config_record, "config_sha256": config_digest, "inputs_sha256": hashes,
               "implementation_sha256": {name: sha256_file(ROOT/name) for name in (
                   "histcontour_core/gap_refinement.py", "histcontour_core/assisted_review.py",
                   "histcontour_core/human_feedback.py", "histcontour_core/completion.py",
                   "scripts/generate_assisted_contour_drawing.py",
                   "scripts/refine_assisted_drawing_from_feedback.py", "scripts/render_assisted_refinement.py")},
               "counts": counts, "proposed_kind_counts": dict(Counter(d["kind"] for d in candidates)),
               "reason_counts": dict(Counter(reason for d in decisions for reason in d.get("reasons", []))),
               "original_proposal_count": len(rows), "reviewed_locked": len(locked), "deferred_locked": len(deferred),
               "eligible_unreviewed": len(rows)-len(locked)-len(deferred),
               "human_approvals": 0, "automatic_promotion": False, "model_fitted": False, "holdout_used": False,
               "formal_accuracy": None, "original_source_modified": False, "approved_geometries_modified": 0,
               "elapsed_seconds": time.perf_counter()-started,
               "queue_order": "round_robin_tiles_within_evidence_and_geometry_tiers",
               "preview_ids": [d["revision_id"] for d in candidates[:previews]],
               "queue": [d["revision_id"] for d in candidates], "revisions": specs, "decisions": decisions}
    output.mkdir(parents=True, exist_ok=False)
    write(output/"refinement-report.json", summary)
    write(output/"proposed-revisions.geojson", {"type": "FeatureCollection", "crs": copy.deepcopy(originals["crs"]), "features": features})
    write(output/"review-queue.json", {"schema": "jap-map-refinement-review-queue/1", "revision_ids": summary["queue"],
        "locked_original_ids": sorted(locked), "deferred_original_ids": sorted(deferred),
        "question": "연결 상대와 곡선 모양이 맞나요? 등고선인지도 별도로 판단해 주세요.",
        "local_tail_rule": "A local-tail draft must replace its two specified source tails in a separate working copy, then add the patch. Approval of the original does not approve this revision.",
        "human_approved": False})
    if previews:
        from scripts.render_assisted_refinement import render_comparison
        feature_by_id = {f["properties"]["proposal_id"]: f for f in features}
        for candidate in candidates[:previews]:
            row = rows[candidate["proposal_id"]]; tile = tiles[row["tile_id"]]
            new_points = map_to_pixel(tile, feature_by_id[candidate["revision_id"]]["geometry"]["coordinates"])
            render_comparison(packet/tile["raster_path"], row, new_points,
                              output/(candidate["revision_id"]+".png"), title=candidate["revision_id"]+" | UNAPPROVED")
    note = (f"# 피드백 기반 연결 수정 초안\n\n검토 가능한 새 수정안 {len(candidates)}개입니다. 사람 승인은 0개이며 정확도 결과가 아닙니다.\n\n"
            f"기존 응답 {len(locked)}건과 주변 문맥 보류 {len(deferred)}건은 잠갔습니다. 원본 패킷과 승인 도형은 바꾸지 않았습니다.\n\n"
            "도엽별로 번갈아 검수하며, 연결 상대·곡선 모양·등고선 여부를 따로 판단합니다.\n\n"
            "`proposed-revisions.geojson`은 원본 CRS의 수정 후보이고, `refinement-report.json`에 각 도형 해시와 출처·제약·검사 이유가 있습니다. "
            "`local_tail_replacement`는 기존 두 선의 지정된 끝부분을 별도 작업 사본에서 교체한 뒤 추가해야 합니다. 단순 덧붙이기는 금지합니다. "
            "이 보고서의 미승인 계약은 사람 승인을 대신하지 않습니다.\n\n"
            "## 먼저 볼 비교판\n\n왼쪽은 주변 원본, 가운데는 이전 제안, 오른쪽은 새 미승인 제안입니다. "
            "파란 원은 기존 끝점, 초록 원은 새 접속점입니다. 원본 지도 픽셀은 그대로이며 선 표시만 매끄럽게 렌더링했습니다.\n\n"
            + "\n".join(f"- [{sid}]({sid}.png)" for sid in summary["preview_ids"]) + "\n")
    with (output/"START-HERE.md").open("x", encoding="utf-8") as handle:
        handle.write(note)
    print(json.dumps({"counts": counts, "kinds": summary["proposed_kind_counts"], "elapsed_seconds": summary["elapsed_seconds"]}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--revisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=int, default=6)
    args = parser.parse_args()
    try:
        refine(args.packet, args.ledger, args.output, revisions_path=args.revisions, previews=args.previews)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Feedback refinement stopped: {error}\n")


if __name__ == "__main__":
    main()
