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


def build_review_outputs(report, proposals, ledger, revisions=None):
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
            if math.dist(a[0], b[0]) > 1e-9 or math.dist(a[-1], b[-1]) > 1e-9:
                raise ValueError("curvature revision changed the confirmed endpoints")
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
            "reviewed": collection(reviewed), "summary": {
                "schema": "jap-map-assisted-review-export/1", "counts": counts,
                "model_fitted": False, "automatic_training_promotion": False, "formal_accuracy": None,
                "original_source_modified": False, "revision_endpoints_verified": sorted(used_revisions),
                "note": "Accepted contour additions only; do not relabel whole source lines or inferred gaps as observed ink."},
            "queue": {"unreviewed_ids": unreviewed, "eligible_ids": queued, "deferred_ids": deferred,
                      "responded_ids_not_reasked": list(decisions)}}
