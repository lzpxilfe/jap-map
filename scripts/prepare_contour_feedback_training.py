#!/usr/bin/env python3
"""Prepare human-only training references; optionally fit only after readiness.

QGIS exports plain feedback first. This stage uses NumPy/SciPy/Pillow without
installing anything into QGIS. Evaluation-only cases never enter model fitting,
and corrected paths become sparse geometry supervision, not fabricated labels
for the original whole segment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.contour_chains import PRIOR_CHAIN_FEATURE_NAMES
from histcontour_core.human_feedback import BINARY_CLASSES, rasterize_sparse_training_feedback, validate_packet
from histcontour_core.provenance import sha256_file
from scripts.compare_contour_vector_outputs import pixel_proposals
from scripts.run_contour_chain_development import describe_tile, fitter
from scripts.run_contour_context_experiment import nested_validation, read, write
from scripts.score_ink_segments import load_model


def validate_training_labels(packet, feedback):
    cases, _ = validate_packet(packet)
    if feedback.get("schema") != "jap-map-contour-human-feedback/1" or feedback.get("status") != "validated_human_edit_snapshot":
        raise ValueError("training requires a validated human-edit snapshot, never AI labels")
    decisions = {row["case_id"]: row for row in feedback["decisions"]}
    labels = feedback["training_labels"]
    seen = set()
    for row in labels:
        case = cases.get(row["case_id"])
        decision = decisions.get(row["case_id"])
        if (case is None or decision is None or row["case_id"] in seen or case["dataset_role"] != "training" or row.get("dataset_role") != "training"
                or case["sample_id"].startswith("E") or row.get("human_approved") is not True
                or row.get("reference_origin") != "human_explicitly_approved" or row.get("class") not in BINARY_CLASSES
                or not decision.get("human_approved") or decision.get("review_status") != row["class"]
                or not isinstance(row.get("annotator"), str) or not row["annotator"].strip()):
            raise ValueError("non-human, duplicate, or evaluation-only label entered training")
        if any(row.get(key) != case[key] for key in ("sample_id", "segment_uid", "tile_id", "sheet_id", "source_raster_sha256", "original_geometry_sha256")):
            raise ValueError("human training label lost its exact source identity")
        if row["class"] == "contour" and decision.get("geometry_decision") != "accept_original":
            raise ValueError("a corrected partial path is not a whole-original-segment positive label")
        seen.add(row["case_id"])
    return labels


def prepare(packet_path, feedback_path, output, *, fit_if_ready=False, minimum_per_class=20):
    packet, feedback = read(packet_path), read(feedback_path)
    if feedback.get("packet_sha256") != sha256_file(packet_path):
        raise ValueError("human feedback belongs to a different source packet")
    labels = validate_training_labels(packet, feedback)
    if minimum_per_class < 2:
        raise ValueError("minimum_per_class must be at least two")
    positives = sum(row["class"] == "contour" for row in labels)
    negatives = len(labels)-positives
    groups = sorted({row["sheet_id"] for row in labels})
    ready = positives >= minimum_per_class and negatives >= minimum_per_class and len(groups) >= 3
    output.mkdir(parents=True, exist_ok=False)
    sparse = rasterize_sparse_training_feedback(packet, feedback, output/"sparse-labels")
    sparse.update(packet_sha256=sha256_file(packet_path), feedback_sha256=sha256_file(feedback_path))
    write(output/"sparse-labels"/"index.json", sparse)
    report = {"schema": "jap-map-human-contour-training-preparation/1", "status": "ready_not_fitted" if ready else "waiting_for_human_labels",
              "packet_sha256": sha256_file(packet_path), "feedback_sha256": sha256_file(feedback_path), "fit_requested": fit_if_ready,
              "training_label_count": len(labels), "class_counts": dict(Counter(row["class"] for row in labels)), "positive": positives, "negative": negatives,
              "sheet_groups": groups, "minimum_per_class": minimum_per_class, "evaluation_labels_not_used": len(feedback["evaluation_labels"]),
              "geometry_supervision": "Sparse observed/corrected paths exported separately; inferred gaps and E-series excluded, unmarked pixels remain unknown.",
              "model_fitted": False, "automatic_promotion": False}
    if ready and fit_if_ready:
        import numpy as np
        from PIL import Image
        legacy_path = packet_path.parent/packet["legacy_model_path"]
        if sha256_file(legacy_path) != packet["legacy_model_sha256"]:
            raise ValueError("copied legacy prior changed")
        kind, legacy = load_model(legacy_path)
        if kind != "logistic":
            raise ValueError("human training prior must be the declared legacy logistic model")
        rows = []
        for tile in packet["tiles"]:
            selected = [row for row in labels if row["tile_id"] == tile["tile_id"]]
            if not selected:
                continue
            raster_path, vector_path = packet_path.parent/tile["raster_path"], packet_path.parent/tile["reference_vector_path"]
            if sha256_file(raster_path) != tile["source_raster_sha256"] or sha256_file(vector_path) != tile["source_vector_sha256"]:
                raise ValueError("human training source raster/vector changed")
            with Image.open(raster_path) as image:
                gray = np.asarray(image.convert("L")).copy()
            collection = read(vector_path)
            descriptors, _ = describe_tile(gray, pixel_proposals(tile, collection), legacy)
            lookup = {feature["properties"]["segment_uid"]: descriptor for feature, descriptor in zip(collection["features"], descriptors)}
            for label in selected:
                if label["segment_uid"] not in lookup:
                    raise ValueError("human-labeled original segment disappeared")
                rows.append({**label, "sample_id": label["case_id"], **lookup[label["segment_uid"]]})
        write(output/"human-training-descriptors.json", rows)
        try:
            validation = nested_validation(rows, PRIOR_CHAIN_FEATURE_NAMES, l2=.1, target_recall=1.0, fitter=fitter)
        except ValueError as error:
            report.update(status="waiting_for_sheet_class_balance", reason=str(error))
        else:
            report.update(status="human_label_model_trained_review_only", model_fitted=True, validation=validation,
                          evaluation_scope="Nested leave-one-training-sheet-out on human labels; E-series and inferred gaps were not fitted or used for threshold choice.")
    write(output/"training-readiness.json", report)
    print({key: report[key] for key in ("status", "positive", "negative", "evaluation_labels_not_used", "model_fitted")}, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-if-ready", action="store_true", help="fit only if named approved training labels meet the readiness gate")
    parser.add_argument("--minimum-per-class", type=int, default=20)
    args = parser.parse_args()
    prepare(args.packet, args.feedback, args.output, fit_if_ready=args.fit_if_ready, minimum_per_class=args.minimum_per_class)


if __name__ == "__main__":
    main()
