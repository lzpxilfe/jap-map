#!/usr/bin/env python3
"""Publish a small path-free review ledger, not private screenshot/chat files."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.assisted_review import build_review_outputs
from histcontour_core.human_feedback import geometry_digest
from histcontour_core.provenance import sha256_file


def serialized(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"


def assert_no_private_paths(value):
    if isinstance(value, str) and any(marker in value for marker in (
            "/Users/", "/home/", "/private/", "/var/folders/", "/tmp/", "file://", "C:\\Users\\")):
        raise ValueError("public review ledger contains a private local path; curate the source text first")
    if isinstance(value, dict):
        for child in value.values():
            assert_no_private_paths(child)
    if isinstance(value, list):
        for child in value:
            assert_no_private_paths(child)


def public_event(event):
    human_event_types = {
        "case_judgment", "user_drawn_topology_correction", "source_map_convention",
        "case_judgment_with_supplementary_sketch", "case_rejection_with_tentative_sketch",
        "batch_judgment_with_proposed_redrawing", "batch_uncertainty_with_sketched_reference",
        "single_case_confirmation", "accepted_connection_with_shape_refinement_request",
        "revision_acceptance_with_reservation",
    }
    if event.get("type") not in human_event_types or event.get("origin", "explicit_user_chat") != "explicit_user_chat":
        raise ValueError("cannot publish an AI or unknown event as a human attestation")
    result = {key: copy.deepcopy(event[key]) for key in (
        "event_id", "type", "proposal_id", "proposal_ids", "status", "superseded_by",
        "supersedes", "supplemented_by", "accepted_revision_ids") if key in event}
    result["origin"] = "explicit_user_chat"
    text = event.get("publication_text", event.get("verbatim_text", ""))
    if text and text not in event.get("verbatim_text", ""):
        raise ValueError("public excerpt is not part of the recorded user message")
    result["user_text"] = text
    targets = event.get("proposal_ids", [event["proposal_id"]] if event.get("proposal_id") else [])
    actions, semantics = {}, {}
    if event.get("geometry_decision"):
        actions.update({sid: event["geometry_decision"] for sid in targets})
    if event.get("semantic_decision"):
        semantics.update({sid: event["semantic_decision"] for sid in targets})
    interpretation = event.get("interpretation")
    if isinstance(interpretation, dict):
        actions.update({sid: "accept_original" for sid in interpretation.get("accepted_original_connection_ids", [])})
        actions.update({sid: "reject_original" for sid in interpretation.get("rejected_original_connection_ids", [])})
    result.update(case_actions=actions, case_semantics=semantics)
    return result


def publish(packet, feedback_path, session_path, output):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    feedback, session = read(feedback_path), read(session_path)
    if feedback.get("schema") != "jap-map-assisted-chat-feedback/1" or feedback.get("review_origin") != "explicit_user_chat":
        raise ValueError("expected explicit assisted chat feedback")
    for name, key in (("drawing-report.json", "drawing_report_sha256"), ("ai-proposals.geojson", "ai_proposals_sha256")):
        if sha256_file(packet/name) != feedback["source_packet"][key]:
            raise ValueError("chat feedback does not belong to this packet")
    report, proposals = read(packet/"drawing-report.json"), read(packet/"ai-proposals.geojson")
    originals = {f["properties"]["proposal_id"]: f for f in proposals["features"]}
    source_rows = {p["proposal_id"]: p for p in report["proposals"]}
    ids = [d["proposal_id"] for d in feedback["decisions"]]
    if len(set(ids)) != len(ids) or set(ids) != set(session["reviewed_ids"]):
        raise ValueError("chat review session and case decisions differ")
    decisions, specs, revised = [], [], []
    base_lines = None
    source_base_hash = None
    needed_events = set()
    for decision in feedback["decisions"]:
        sid = decision["proposal_id"]
        for stored, source_key in (("original_pixel_points", "pixel_points"), ("original_start_pixel", "start"),
                                   ("original_end_pixel", "end"), ("original_mode", "mode")):
            if decision.get(stored) != source_rows[sid].get(source_key):
                raise ValueError("chat decision's original geometry or mode drifted from its source")
        clean = {key: copy.deepcopy(decision[key]) for key in (
            "proposal_id", "tile_id", "source_uid", "target_uid", "source_raster_sha256",
            "geometry_decision", "semantic_decision", "human_judgment_received", "evidence_event_ids")}
        clean["original_geometry_sha256"] = geometry_digest(originals[sid]["geometry"])
        clean["review_note"] = decision.get("note", "")
        clean["review_qualification"] = decision.get("correction", {}).get("review_qualification", "")
        needed_events.update(clean["evidence_event_ids"])
        needed_events.update(decision.get("superseded_event_ids", []))
        if decision["geometry_decision"].startswith("accept_revised_geometry"):
            correction = decision["correction"]
            if correction.get("geometry_revision_approved_by_user") is not True:
                raise ValueError("chat record has no explicit revised-geometry approval")
            path = (feedback_path.parent/correction["approved_revised_geometry_file"]).resolve()
            if not path.is_relative_to(feedback_path.parent.parent.resolve()):
                raise ValueError("approved geometry must be in the local versioned review workspace")
            if sha256_file(path) != correction["approved_geometry_file_sha256"]:
                raise ValueError("approved geometry file changed")
            collection = read(path)
            if collection.get("crs") != proposals["crs"] or len(collection["features"]) != 1:
                raise ValueError("expected one approved native-CRS curvature revision")
            feature = copy.deepcopy(collection["features"][0])
            if feature["properties"]["proposal_id"] != correction["revision_id"]:
                raise ValueError("wrong revision file")
            if (feature["properties"].get("human_approved") is not True
                    or feature["properties"].get("geometry_revision_human_approved") is not True):
                raise ValueError("revision snapshot is still unapproved")
            local_tail = correction.get("revision_kind") == "local_tail_replacement"
            if feature["properties"].get("requires_explicit_local_tail_replacement_contract") is True and not local_tail:
                raise ValueError("reanchored revision needs its explicit local-tail contract")
            tail_contract = None
            if local_tail:
                metadata_path = (feedback_path.parent/correction["revision_metadata"]).resolve()
                if (not metadata_path.is_relative_to(feedback_path.parent.parent.resolve())
                        or sha256_file(metadata_path) != correction["revision_metadata_sha256"]):
                    raise ValueError("local-tail metadata path or digest differs")
                metadata = read(metadata_path)
                if (metadata.get("schema") != "jap-map-assisted-anchor-refinement/1"
                        or metadata.get("proposal_id") != correction["revision_id"]
                        or metadata.get("geometry_sha256") != geometry_digest(feature["geometry"])):
                    raise ValueError("local-tail metadata describes different geometry")
                for name, key in (("drawing-report.json", "source_report_sha256"),
                                  ("ai-proposals.geojson", "source_proposals_sha256"),
                                  ("base-lines.geojson", "source_base_lines_sha256")):
                    if metadata.get(key) != sha256_file(packet/name):
                        raise ValueError("local-tail source fingerprint changed")
                contract = metadata["application_contract"]
                tail_contract = {"schema": "jap-map-local-tail-replacement/1", **{
                    key: copy.deepcopy(contract[key]) for key in (
                        "source_uid", "target_uid", "source_geometry_sha256", "target_geometry_sha256",
                        "source_tail_trim_pixels", "target_tail_trim_pixels", "join_points_pixels")}}
                base_lines = read(packet/"base-lines.geojson")
                source_base_hash = sha256_file(packet/"base-lines.geojson")
            feature["properties"] = {key: feature["properties"][key] for key in (
                "proposal_id", "parent_proposal_id", "revision", "review_status", "human_approved",
                "endpoint_pairing_human_approved", "geometry_revision_human_approved", "dataset_role",
                "review_qualification", "review_event_id")}
            if local_tail:
                feature["properties"].update(revision_kind="local_tail_replacement",
                                             requires_source_tail_replacement=True, append_only_safe=False)
            revised.append(feature)
            spec = {"revision_id": correction["revision_id"], "parent_proposal_id": sid,
                    "geometry_sha256": geometry_digest(feature["geometry"]), "approved": True,
                    "accepted_event_id": correction["approved_event_id"],
                    "source_approved_file_sha256": correction["approved_geometry_file_sha256"],
                    "qualification": correction.get("review_qualification", "")}
            specs.append(spec)
            if local_tail:
                spec.update(revision_kind="local_tail_replacement", tail_replacement_contract=tail_contract)
            clean["revision_id"] = spec["revision_id"]
            needed_events.add(spec["accepted_event_id"])
        decisions.append(clean)
    all_events = {e["event_id"]: e for e in feedback["events"]}
    if len(all_events) != len(feedback["events"]):
        raise ValueError("duplicate chat events")
    # Retain the original A0512 correction chain, including retracted opinions.
    while True:
        before = set(needed_events)
        for event_id in before:
            for key in ("superseded_by", "supersedes", "supplemented_by"):
                if all_events[event_id].get(key):
                    needed_events.add(all_events[event_id][key])
        if before == needed_events:
            break
    events = [public_event(e) for e in feedback["events"] if e["event_id"] in needed_events]
    revision_collection = {"type": "FeatureCollection", "crs": proposals["crs"], "features": revised}
    revision_text = serialized(revision_collection)
    ledger = {
        "schema": "jap-map-assisted-human-review/1", "review_origin": "explicit_user_chat",
        "reviewer_role": "conversation_user", "dataset_role": "review_only_not_training",
        "holdout_used": False, "model_fitted": False, "native_crs": proposals["crs"]["properties"]["name"],
        "source_report_sha256": sha256_file(packet/"drawing-report.json"),
        "source_proposals_sha256": sha256_file(packet/"ai-proposals.geojson"),
        "source_feedback_sha256": sha256_file(feedback_path),
        "revision_collection_sha256": hashlib.sha256(revision_text.encode()).hexdigest(),
        "events": events, "decisions": decisions, "revisions": specs,
        "deferred_original_ids": [d["proposal_id"] for d in session["deferred_cases"]],
        "review_protocol": {"batch_size": 1, "uncertainty_is_not_rejection": True,
                            "inferred_gap_is_not_observed_ink": True, "automatic_training_promotion": False},
        "drawing_guidance": {"numeral_label_blanks_may_hide_several_contours": True,
                             "compare_neighbouring_curve_order_before_pairing": True,
                             "subtle_curvature_start_max_pixels": .5,
                             "subtle_curvature_start_max_fraction_of_gap": .05,
                             "curvature_direction_must_be_chosen_from_context": True,
                             "numeric_curvature_policy_origin": "assistant cautious proposal after qualified user acceptance; not a trained model"},
        "annotation_roles": [{"proposal_id": m["proposal_id"], "glyph": m["glyph"], "role": "uncertainty_annotation_not_contour",
                              "uncertain_geographic_extent_supplied": m.get("uncertain_geographic_region") is not None,
                              "unmarked_strokes_not_automatically_approved": True} for m in feedback.get("uncertainty_markers", [])],
        "publication_scope": "Decision ledger and explicitly accepted vector revisions only; no screenshots, rasters, model weights, private paths, or verified personal identity.",
    }
    if base_lines is not None:
        ledger["source_base_lines_sha256"] = source_base_hash
    assert_no_private_paths(ledger)
    assert_no_private_paths(revision_collection)
    build_review_outputs(report, proposals, ledger, revision_collection, base_lines=base_lines)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"review-ledger.json").open("x", encoding="utf-8") as handle:
        handle.write(serialized(ledger))
    with (output/"approved-revisions.geojson").open("x", encoding="utf-8") as handle:
        handle.write(revision_text)
    print({"published_cases": len(decisions), "revisions": len(revised), "private_paths": False})
    return ledger, revision_collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        publish(args.packet, args.feedback, args.session, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(f"Public review ledger stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
