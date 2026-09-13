"""Replay explicit assisted-tracing judgments without fitting or inferring labels.

Geometry acceptance, contour semantics, uncertainty, and a revised geometry's
approval are independent. This validates recorded attestations, not identity.
"""

from __future__ import annotations

import copy
import math

from .human_feedback import geometry_digest, map_to_pixel


GEOMETRY_DECISIONS = {
    "accept_original", "reject_original", "unresolved_no_explicit_decision",
    "accept_connection_refine_curvature", "accept_revised_geometry_with_reservation",
    "accept_revised_geometry",
}
SEMANTIC_DECISIONS = {
    "contour", "non_contour", "unsure", "probable_contour", "not_separately_stated",
    "separate_contours_user_indicated", "contour_context_route_unresolved",
}
BUCKETS = ("approved_contour", "semantics_pending", "non_contour", "rejected_route",
           "unresolved_route", "pending_refinement")


def _index(rows, key, description):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {description} IDs")
    return result


def _line(geometry):
    points = geometry.get("coordinates", [])
    if (geometry.get("type") != "LineString" or len(points) < 2
            or any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p) for p in points)
            or sum(math.dist(a, b) for a, b in zip(points, points[1:])) <= 0):
        raise ValueError("expected a nonempty finite 2-D LineString")
    return points


def _case_events(decision, events):
    references = decision.get("evidence_event_ids", [])
    if not references or len(set(references)) != len(references):
        raise ValueError("a decision needs distinct evidence events")
    bound = False
    actions, semantics = set(), set()
    for event_id in references:
        event = events.get(event_id)
        if event is None or event.get("status") == "superseded":
            raise ValueError("missing or superseded human evidence")
        targets = set(event.get("proposal_ids", []))
        if event.get("proposal_id"):
            targets.add(event["proposal_id"])
        if targets and decision["proposal_id"] not in targets:
            raise ValueError("evidence belongs to another proposal")
        if decision["proposal_id"] in targets and event.get("origin") == "explicit_user_chat":
            bound = True
            actions.add(event.get("case_actions", {}).get(decision["proposal_id"]))
            semantics.add(event.get("case_semantics", {}).get(decision["proposal_id"]))
    if not bound:
        raise ValueError("decision lacks case-bound explicit user evidence")
    if decision["geometry_decision"] != "unresolved_no_explicit_decision" and decision["geometry_decision"] not in actions:
        raise ValueError("human event does not attest this geometry action")
    if decision["semantic_decision"] in ("contour", "non_contour") and decision["semantic_decision"] not in semantics:
        raise ValueError("human event does not attest this definite semantic label")


def _trim_at_endpoint(points, endpoint, distance, tile):
    """Return a local cut and the retained line, in its original orientation."""
    if (type(distance) not in (int, float) or not math.isfinite(distance)
            or not 0 < distance <= 16):
        raise ValueError("tail replacement needs a finite local trim of at most 16 pixels")
    matches = [i for i in (0, -1) if math.dist(points[i], endpoint) <= 1e-9]
    if len(matches) != 1:
        raise ValueError("original proposal endpoint is not a unique source-line endpoint")
    reverse = matches[0] == -1
    ordered = list(reversed(points)) if reverse else list(points)
    pixels = map_to_pixel(tile, ordered)
    total = sum(math.dist(a, b) for a, b in zip(pixels, pixels[1:]))
    if distance >= total-1e-6:
        raise ValueError("tail replacement must retain the rest of the original source line")
    remaining = distance
    for i, (a, b) in enumerate(zip(pixels, pixels[1:])):
        length = math.dist(a, b)
        if length <= 1e-12:
            continue
        if remaining <= length:
            fraction = remaining/length
            cut = [ordered[i][j]+fraction*(ordered[i+1][j]-ordered[i][j]) for j in (0, 1)]
            kept = [cut]+copy.deepcopy(ordered[i+1:])
            if len(kept) > 1 and math.dist(kept[0], kept[1]) <= 1e-12:
                kept.pop(1)
            return cut, list(reversed(kept)) if reverse else kept, -1 if reverse else 0
        remaining -= length
    raise ValueError("tail cut could not be located")


def _tail_replacements(spec, candidate, original, row, tile, base_lines, crs):
    contract = spec.get("tail_replacement_contract", {})
    if contract.get("schema") != "jap-map-local-tail-replacement/1":
        raise ValueError("local tail revision needs an explicit replacement contract")
    if base_lines is None or base_lines.get("crs") != crs:
        raise ValueError("local tail revision needs the actual original base lines in the source CRS")
    bases = _index([dict(f, uid=f["properties"]["segment_uid"]) for f in base_lines["features"]],
                   "uid", "original base line")
    points = _line(candidate["geometry"])
    pixels = map_to_pixel(tile, points)
    chord = [pixels[-1][i]-pixels[0][i] for i in (0, 1)]
    length = math.hypot(*chord)
    if length <= 0:
        raise ValueError("local tail patch has zero-length chord")
    progress = [sum((p[i]-pixels[0][i])*chord[i] for i in (0, 1))/length for p in pixels]
    deviations = [abs((p[0]-pixels[0][0])*chord[1]-(p[1]-pixels[0][1])*chord[0])/length for p in pixels]
    if any(b-a <= 1e-9 for a, b in zip(progress, progress[1:])) or max(deviations) > 4:
        raise ValueError("local tail patch doubles back or leaves its local corridor")
    joins = contract.get("join_points_pixels", [])
    if (len(joins) != 2 or any(len(p) != 2 or any(type(v) not in (int, float)
            or not math.isfinite(v) for v in p) for p in joins)):
        raise ValueError("local tail contract needs two finite source-pixel join points")
    result = []
    for side, endpoint_index, join_index in (("source", 0, 0), ("target", -1, 1)):
        uid = contract.get(side+"_uid")
        feature = bases.get(uid)
        if uid != row[side+"_uid"] or feature is None or feature["properties"].get("tile_id") != row["tile_id"]:
            raise ValueError("tail replacement refers to another source identity")
        if contract.get(side+"_geometry_sha256") != geometry_digest(feature["geometry"]):
            raise ValueError("tail replacement source geometry changed")
        cut, retained, retained_end = _trim_at_endpoint(
            _line(feature["geometry"]), original[endpoint_index], contract.get(side+"_tail_trim_pixels"), tile)
        if (math.dist(cut, points[endpoint_index]) > 1e-9
                or math.dist(map_to_pixel(tile, [cut])[0], joins[join_index]) > 1e-5):
            raise ValueError("tail patch endpoint does not match its verified source cut")
        # Eliminate numerical round-off at the shared node without moving it.
        retained[retained_end] = copy.deepcopy(points[endpoint_index])
        changed = {"type": "Feature", "properties": copy.deepcopy(feature["properties"]),
                   "geometry": {"type": "LineString", "coordinates": retained}}
        changed["properties"].update(
            replaced_source_geometry_sha256=contract[side+"_geometry_sha256"],
            local_patch_revision_id=spec["revision_id"], local_tail_trim_pixels=contract[side+"_tail_trim_pixels"],
            geometry_scope="trimmed_base_line_original_semantics_preserved",
            whole_source_line_semantics_approved=False, training_eligible=False)
        result.append(changed)
    if result[0]["properties"]["segment_uid"] == result[1]["properties"]["segment_uid"]:
        raise ValueError("a local patch needs two distinct source lines")
    return result


def build_review_outputs(report, proposals, ledger, revisions=None, *, base_lines=None):
    """Pure replay: return separated GeoJSON collections and a remaining queue."""
    if (ledger.get("schema") != "jap-map-assisted-human-review/1"
            or ledger.get("review_origin") != "explicit_user_chat"
            or ledger.get("dataset_role") != "review_only_not_training"
            or ledger.get("holdout_used") is not False):
        raise ValueError("expected explicit development-only review ledger, not training labels")
    if report.get("schema") != "jap-map-assisted-contour-drawing/1" or report.get("holdout_used") is not False:
        raise ValueError("expected assisted development drawing report")
    tiles = _index(report["tiles"], "tile_id", "tile")
    if not tiles or any(t["split"] != "development" or t["sheet_id"] == "178-gongju" for t in tiles.values()):
        raise ValueError("held-out sources cannot enter assisted review replay")
    crs = proposals.get("crs")
    if (not crs or crs.get("properties", {}).get("name") != ledger.get("native_crs")
            or {t["crs_authid"] for t in tiles.values()} != {ledger["native_crs"]}):
        raise ValueError("source CRS declaration differs from review ledger")
    rows = _index(report["proposals"], "proposal_id", "source proposal")
    features = _index([dict(f, proposal_id=f["properties"]["proposal_id"]) for f in proposals["features"]],
                      "proposal_id", "source feature")
    if set(rows) != set(features) or len(rows) != report["proposal_count"]:
        raise ValueError("source report and feature identities differ")
    for sid, row in rows.items():
        feature = features[sid]
        if row["dataset_role"] != "review_only_not_training" or row.get("human_approved") is not False:
            raise ValueError("original machine source must remain unapproved and review-only")
        if (feature["properties"].get("dataset_role") != "review_only_not_training"
                or feature["properties"].get("human_approved") is not False):
            raise ValueError("original feature cannot already contain a human approval")
        points = _line(feature["geometry"])
        actual = map_to_pixel(tiles[row["tile_id"]], points)
        if len(actual) != len(row["pixel_points"]) or any(math.dist(a, b) > 1e-5 for a, b in zip(actual, row["pixel_points"])):
            raise ValueError("source feature no longer matches its report pixel geometry")
        for key in ("tile_id", "source_uid", "target_uid"):
            if feature["properties"].get(key) != row.get(key):
                raise ValueError("source feature identity changed")
    decisions = _index(ledger["decisions"], "proposal_id", "human decision")
    events = _index(ledger["events"], "event_id", "human event")
    specs = _index(ledger.get("revisions", []), "revision_id", "revision approval")
    if not set(decisions) <= set(rows):
        raise ValueError("review refers to an unknown original proposal")
    if revisions is None:
        revisions = {"type": "FeatureCollection", "crs": crs, "features": []}
    if revisions.get("crs") != crs:
        raise ValueError("revised geometry CRS differs from the source")
    revised = _index([dict(f, revision_id=f["properties"]["proposal_id"]) for f in revisions["features"]],
                     "revision_id", "revision geometry")
    if set(revised) != set(specs):
        raise ValueError("approved revision manifest and provided revision geometry differ")
    grouped = {name: [] for name in BUCKETS}
    reviewed, used_revisions = [], set()
    tail_replacements, tail_revision_ids, replaced_uids = [], [], set()
    for sid, decision in decisions.items():
        source = features[sid]
        action, semantic = decision["geometry_decision"], decision["semantic_decision"]
        if action not in GEOMETRY_DECISIONS or semantic not in SEMANTIC_DECISIONS:
            raise ValueError("unknown geometry or semantic decision")
        if decision.get("original_geometry_sha256") != geometry_digest(source["geometry"]):
            raise ValueError("review refers to different original geometry")
        for key in ("tile_id", "source_uid", "target_uid"):
            if decision.get(key) != rows[sid].get(key):
                raise ValueError("review source identity changed")
        if decision.get("source_raster_sha256") != tiles[rows[sid]["tile_id"]]["source_raster_sha256"]:
            raise ValueError("review source raster identity changed")
        if type(decision.get("human_judgment_received")) is not bool:
            raise ValueError("human judgment flag must be explicit boolean")
        if action != "unresolved_no_explicit_decision" and decision["human_judgment_received"] is not True:
            raise ValueError("an explicit case judgment is needed for acceptance or rejection")
        _case_events(decision, events)
        geometry = source["geometry"]
        effective_id = sid
        if action.startswith("accept_revised_geometry"):
            rid = decision.get("revision_id")
            spec = specs.get(rid)
            candidate = revised.get(rid)
            if not spec or not candidate or spec.get("approved") is not True or spec.get("parent_proposal_id") != sid:
                raise ValueError("revised acceptance needs its exact approved revision")
            event = events.get(spec.get("accepted_event_id"), {})
            if (event.get("origin") != "explicit_user_chat" or event.get("status") == "superseded"
                    or rid not in event.get("accepted_revision_ids", [])
                    or spec["accepted_event_id"] not in decision["evidence_event_ids"]):
                raise ValueError("endpoint approval alone is not revised-geometry approval")
            if (candidate["properties"].get("parent_proposal_id") != sid
                    or spec.get("geometry_sha256") != geometry_digest(candidate["geometry"])):
                raise ValueError("approved revised geometry identity or digest changed")
            a, b = _line(candidate["geometry"]), _line(source["geometry"])
            kind = spec.get("revision_kind", "fixed_endpoints")
            if kind == "local_tail_replacement":
                changed = _tail_replacements(spec, candidate, b, rows[sid], tiles[rows[sid]["tile_id"]], base_lines, crs)
                uids = {f["properties"]["segment_uid"] for f in changed}
                if replaced_uids.intersection(uids):
                    raise ValueError("multiple patches need an explicit combined source-tail contract")
                tail_replacements.extend(changed)
                replaced_uids.update(uids)
                tail_revision_ids.append(rid)
            elif kind == "fixed_endpoints":
                if math.dist(a[0], b[0]) > 1e-9 or math.dist(a[-1], b[-1]) > 1e-9:
                    raise ValueError("curvature revision changed the confirmed endpoints")
            else:
                raise ValueError("unknown revised-geometry contract")
            geometry, effective_id = candidate["geometry"], rid
            used_revisions.add(rid)
        if action == "reject_original":
            bucket = "rejected_route"
        elif action == "unresolved_no_explicit_decision":
            bucket = "unresolved_route"
        elif action == "accept_connection_refine_curvature":
            bucket = "pending_refinement"
        elif semantic == "non_contour":
            bucket = "non_contour"
        elif semantic == "contour":
            bucket = "approved_contour"
        else:
            bucket = "semantics_pending"
        accepted_geometry = action == "accept_original" or action.startswith("accept_revised_geometry")
        properties = {
            "proposal_id": effective_id, "original_proposal_id": sid, "tile_id": rows[sid]["tile_id"],
            "source_uid": rows[sid]["source_uid"], "target_uid": rows[sid]["target_uid"],
            "source_raster_sha256": decision["source_raster_sha256"],
            "original_geometry_sha256": decision["original_geometry_sha256"],
            "geometry_decision": action, "semantic_decision": semantic, "review_bucket": bucket,
            "human_approved": bucket == "approved_contour", "human_geometry_accepted": accepted_geometry,
            "review_origin": "explicit_user_chat", "reviewer_role": "conversation_user",
            "review_qualification": decision.get("review_qualification", ""),
            "review_note": decision.get("review_note", ""),
            "evidence_event_ids": decision["evidence_event_ids"],
            "dataset_role": "review_only_not_training", "training_eligible": False,
            "reference_kind": "inferred_gap_not_observed_ink" if rows[sid]["mode"] == "contextual_gap" else "traced_linework_semantics_reviewed_separately",
        }
        feature = {"type": "Feature", "properties": properties, "geometry": copy.deepcopy(geometry)}
        if effective_id in tail_revision_ids:
            properties.update(reference_kind="inferred_bridge_with_local_source_tail_refinement",
                              requires_source_tail_replacement=True, append_only_safe=False)
        grouped[bucket].append(feature)
        reviewed.append(feature)
    if used_revisions != set(specs):
        raise ValueError("revision approval is not used by its parent decision")
    unreviewed = [sid for sid in rows if sid not in decisions]
    deferred = ledger.get("deferred_original_ids", [])
    if len(set(deferred)) != len(deferred) or not set(deferred) <= set(unreviewed):
        raise ValueError("deferred cases must be distinct unreviewed original IDs")
    queued = [sid for sid in unreviewed if sid not in set(deferred)]
    counts = {name: len(grouped[name]) for name in BUCKETS}
    counts.update(reviewed=len(decisions), unreviewed=len(unreviewed), deferred=len(deferred),
                  accepted_geometry=sum(f["properties"]["human_geometry_accepted"] for f in reviewed),
                  original_proposal_count=len(rows))
    collection = lambda values: {"type": "FeatureCollection", "crs": copy.deepcopy(crs), "features": values}
    return {"collections": {name: collection(values) for name, values in grouped.items()},
            "source_tail_replacements": collection(tail_replacements),
            "reviewed": collection(reviewed), "summary": {
                "schema": "jap-map-assisted-review-export/1", "counts": counts,
                "model_fitted": False, "automatic_training_promotion": False, "formal_accuracy": None,
                "original_source_modified": False, "revision_endpoints_verified": sorted(used_revisions),
                "local_tail_replacement_revision_ids": sorted(tail_revision_ids),
                "source_tail_replacement_feature_count": len(tail_replacements),
                "note": ("Approved local patches require the accompanying source-tail replacements; "
                         "do not append them without trimming the matching original lines."
                         if tail_revision_ids else
                         "Accepted contour additions only; do not relabel whole source lines or inferred gaps as observed ink.")},
            "queue": {"unreviewed_ids": unreviewed, "eligible_ids": queued, "deferred_ids": deferred,
                      "responded_ids_not_reasked": list(decisions)}}
