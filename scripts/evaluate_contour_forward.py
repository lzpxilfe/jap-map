#!/usr/bin/env python3
"""One-way evaluation of a frozen model on line-disjoint E-series labels.

No fitting, threshold adjustment, or label revision is performed. Every source
line survives in an all-score layer; outputs are review evidence, not truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.contour_chains import validate_chain_classifier
from histcontour_core.contour_context import probability
from histcontour_core.provenance import sha256_file
from scripts.compare_contour_vector_outputs import pixel_proposals
from scripts.run_contour_chain_development import describe_tile, validate_separation
from scripts.run_contour_context_experiment import CLASSES, DEVELOPMENT_SHEETS, metrics, read, resolve, write
from scripts.score_ink_segments import load_model


def portable(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def validate_inputs(index_path, vector_path, training_path, forward_path, labels_path, protocol_path, legacy_path):
    training, forward = validate_separation(training_path, forward_path, vector_path)
    protocol, labels = read(protocol_path), read(labels_path)
    if (protocol.get("schema") != "jap-map-contour-forward-protocol/1" or protocol.get("status") != "frozen_before_forward_labeling_and_new_model_predictions"
            or protocol.get("human_approved") is not False or protocol.get("holdout_used") is not False):
        raise ValueError("the forward protocol must have been frozen before evaluation")
    paths = {"source_index_sha256": index_path, "vector_index_sha256": vector_path, "training_manifest_sha256": training_path,
             "forward_manifest_sha256": forward_path, "legacy_model_sha256": legacy_path}
    if any(protocol.get(key) != sha256_file(path) for key, path in paths.items()):
        raise ValueError("forward evaluation inputs changed after model freezing")
    if any(sha256_file(resolve(path)) != digest for path, digest in protocol["implementation_sha256"].items()):
        raise ValueError("frozen model implementation changed; do not silently reuse this evaluation")
    if (labels.get("sample_manifest_sha256") != sha256_file(forward_path) or labels.get("evaluation_protocol_sha256") != sha256_file(protocol_path)
            or labels.get("human_approved") is not False or labels.get("reference_origin") != "ai_visual_provisional"
            or labels.get("status") != "frozen_after_source_visual_review_before_forward_model_scores"):
        raise ValueError("forward source labels must bind the already frozen protocol")
    samples = {row["sample_id"]: row for row in forward["samples"]}
    labels_by_id = {row["sample_id"]: row for row in labels["items"]}
    if (len(samples) != len(forward["samples"]) or len(labels_by_id) != len(labels["items"]) or set(samples) != set(labels_by_id)
            or any(not re.fullmatch(r"E[0-9]{3}", sid) for sid in samples)
            or any(row["class"] not in (*CLASSES, "unsure", "mixed") for row in labels["items"])):
        raise ValueError("forward samples require unique matching E-series labels")
    training_by_id = {row["sample_id"]: row for row in training["samples"]}
    folds = {row["test_sheet"]: row for row in protocol["folds"]}
    if set(folds) != DEVELOPMENT_SHEETS or len(folds) != len(protocol["folds"]):
        raise ValueError("three declared source-excluded models are required")
    for sheet, fold in folds.items():
        validate_chain_classifier(fold["model"])
        if (type(fold.get("threshold")) not in (int, float) or not math.isfinite(fold["threshold"]) or not 0 <= fold["threshold"] <= 1
                or any(sid not in training_by_id or training_by_id[sid]["sheet_id"] == sheet for sid in fold["training_sample_ids"])):
            raise ValueError("invalid threshold or displayed-source training contamination")
    return protocol, forward, labels_by_id, folds


def comparison_gate(rows):
    usable = [row for row in rows if row["class"] in CLASSES]
    previous = [{**row, "retained": row["legacy_retained"]} for row in usable]
    current = [{**row, "retained": row["candidate_retained"]} for row in usable]
    previous_metrics, current_metrics = metrics(previous), metrics(current)
    additional = [row["sample_id"] for row in usable if row["class"] == "contour" and row["legacy_retained"] and not row["candidate_retained"]]
    recovered = [row["sample_id"] for row in usable if row["class"] == "contour" and not row["legacy_retained"] and row["candidate_retained"]]
    both_classes = previous_metrics["by_class"]["contour"]["total"] > 0 and sum(previous_metrics["by_class"][kind]["total"] for kind in CLASSES if kind != "contour") > 0
    passed = bool(both_classes and not additional and current_metrics["noncontours_rejected"] > previous_metrics["noncontours_rejected"])
    return {"legacy": previous_metrics, "candidate": current_metrics, "additional_contour_misses": additional,
            "recovered_legacy_contour_misses": recovered, "criteria_met_on_provisional_forward_samples": passed,
            "automatic_promotion": False, "formal_human_accuracy": None}


def evaluate(index_path, vector_path, training_path, forward_path, labels_path, protocol_path, legacy_path, output):
    import numpy as np
    from PIL import Image
    protocol, forward, labels, folds = validate_inputs(index_path, vector_path, training_path, forward_path, labels_path, protocol_path, legacy_path)
    index, raw = read(index_path), read(vector_path)
    tiles = {row["tile_id"]: row for row in index["tiles"] if row["split"] == "development" and row["sheet_id"] in DEVELOPMENT_SHEETS}
    entries = {row["tile_id"]: row for row in raw["tiles"]}
    if len(tiles) != 9 or len(entries) != len(raw["tiles"]) or set(entries) != set(tiles) or raw.get("holdout_included"):
        raise ValueError("forward output is limited to the same nine development tiles")
    kind, legacy = load_model(legacy_path)
    if kind != "logistic":
        raise ValueError("forward baseline must be the frozen legacy logistic model")
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the exact frozen bytes; reserializing floating point JSON can
    # change a file hash even when every numeric model value is equivalent.
    shutil.copyfile(protocol_path, output/"frozen-protocol.json")
    if sha256_file(output/"frozen-protocol.json") != sha256_file(protocol_path):
        raise ValueError("frozen protocol snapshot changed during copying")
    samples = {row["segment_uid"]: row for row in forward["samples"]}
    predictions, records = [], []
    start = time.perf_counter()
    for tile_id, tile in tiles.items():
        raster_path, collection_path = resolve(tile["raster_path"]), resolve(entries[tile_id]["ink_vector_path"])
        raster_hash, vector_hash = sha256_file(raster_path), sha256_file(collection_path)
        subset = [row for row in forward["samples"] if row["tile_id"] == tile_id]
        if entries[tile_id]["source_raster_sha256"] != raster_hash or any(row["source_raster_sha256"] != raster_hash or row["source_vector_sha256"] != vector_hash for row in subset):
            raise ValueError("forward raster/vector provenance changed")
        with Image.open(raster_path) as image:
            gray = np.asarray(image.convert("L")).copy()
        collection = read(collection_path)
        proposals = pixel_proposals(tile, collection)
        descriptors, audit = describe_tile(gray, proposals, legacy)
        fold = folds[tile["sheet_id"]]
        groups = {"all": [], "legacy": [], "candidate": [], "differences": []}
        for feature, descriptor in zip(collection["features"], descriptors):
            uid = feature["properties"]["segment_uid"]
            current = probability(fold["model"], descriptor)
            previous = descriptor["legacy_score"]
            keep_legacy, keep_candidate = previous >= .1, current >= fold["threshold"]
            properties = {**feature["properties"], "legacy_contour_score": previous, "forward_candidate_score": current,
                          "forward_candidate_threshold": fold["threshold"], "forward_candidate_model": protocol["selected_candidate"],
                          "forward_test_sheet": tile["sheet_id"], "forward_review_only": True, "human_approved": False,
                          "chain_member_count": descriptor["chain_member_count"], "legacy_retained": keep_legacy, "candidate_retained": keep_candidate}
            item = {**feature, "properties": properties}
            groups["all"].append(item)
            if keep_legacy:
                groups["legacy"].append(item)
            if keep_candidate:
                groups["candidate"].append(item)
            if keep_legacy != keep_candidate:
                groups["differences"].append(item)
            if uid in samples:
                sample = samples[uid]
                sid = sample["sample_id"]
                predictions.append({"sample_id": sid, "segment_uid": uid, "sheet_id": tile["sheet_id"], "tile_id": tile_id,
                                    "class": labels[sid]["class"], "legacy_score": previous, "candidate_score": current,
                                    "threshold": fold["threshold"], "legacy_retained": keep_legacy, "candidate_retained": keep_candidate,
                                    "chain_member_count": descriptor["chain_member_count"]})
        record = {"tile_id": tile_id, "sheet_id": tile["sheet_id"], "source_raster_sha256": raster_hash, "source_vector_sha256": vector_hash,
                  "model_excludes_displayed_sheet": True, "threshold": fold["threshold"], "chain_audit": audit, "methods": {}}
        for name, features in groups.items():
            path = output/f"{tile_id}-{name}.geojson"
            write(path, {**collection, "name": f"forward_{name}_{tile_id}", "review_only": True, "features": features})
            record["methods"][name] = {"path": portable(path), "count": len(features), "sha256": sha256_file(path)}
        # Independent disk read, not only the in-memory construction.
        original_geometry = {feature["properties"]["segment_uid"]: feature["geometry"] for feature in collection["features"]}
        for name, entry in record["methods"].items():
            saved = read(resolve(entry["path"]))["features"]
            if (len(saved) != entry["count"] or len({f["properties"]["segment_uid"] for f in saved}) != len(saved)
                    or any(original_geometry.get(f["properties"]["segment_uid"]) != f["geometry"] for f in saved)):
                raise ValueError("forward geometry preservation verification failed")
            if name == "all" and len(saved) != len(collection["features"]):
                raise ValueError("the all-score layer lost an original candidate")
        records.append(record)
        print(f"{tile_id}: all={len(groups['all'])}, legacy={len(groups['legacy'])}, candidate={len(groups['candidate'])}, changed={len(groups['differences'])}", flush=True)
    predictions.sort(key=lambda row: row["sample_id"])
    if len(predictions) != len(samples) or len({row["sample_id"] for row in predictions}) != len(samples):
        raise ValueError("not every forward sample was scored exactly once")
    result = {"schema": "jap-map-contour-forward-evaluation/1", "status": "completed_without_refitting", "human_approved": False,
              "holdout_used": False, "automatic_promotion": False, "formal_human_accuracy": None,
              "protocol_sha256": sha256_file(protocol_path), "forward_manifest_sha256": sha256_file(forward_path),
              "forward_labels_sha256": sha256_file(labels_path), "source_index_sha256": sha256_file(index_path),
              "vector_index_sha256": sha256_file(vector_path), "selected_candidate": protocol["selected_candidate"],
              "class_counts": dict(Counter(row["class"] for row in predictions)), "samples": predictions,
              "comparison": comparison_gate(predictions), "tiles": records,
              "output_counts": {name: sum(row["methods"][name]["count"] for row in records) for name in ("all", "legacy", "candidate", "differences")},
              "all_original_geometries_verified_unchanged": True, "elapsed_seconds": time.perf_counter()-start,
              "limitations": ["AI provisional labels, not human ground truth.", "Line identities are new but source sheets and nearby image contexts are not independent.",
                              "No E labels or E outcomes entered fitting, calibration, or model-family selection.",
                              "This evaluates classification of pre-existing Ink proposals; initial missed ink and existing wrong joins remain outside complete ground-truth coverage."]}
    write(output/"forward_evaluation.json", result)
    print(result["comparison"], flush=True)
    print(output/"forward_evaluation.json", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--training-samples", type=Path, required=True)
    parser.add_argument("--forward-samples", type=Path, required=True)
    parser.add_argument("--forward-labels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.index, args.vectors, args.training_samples, args.forward_samples, args.forward_labels, args.protocol, args.legacy_model, args.output)


if __name__ == "__main__":
    main()
