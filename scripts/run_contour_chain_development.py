#!/usr/bin/env python3
"""Select a research classifier on OLD development labels, never forward labels.

The new E-series manifest is used only to prove separation and bind a later
forward evaluation. This command has no argument for evaluation class labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.contour_context import CONTEXT_FEATURE_NAMES, NeighborhoodFeatures, fit_classifier
from histcontour_core.contour_chains import (
    CHAIN_FEATURE_NAMES, CHAIN_SHAPE_FEATURE_NAMES, PRIOR_CONTEXT_FEATURE_NAMES, PRIOR_CHAIN_FEATURE_NAMES,
    chain_features, fit_chain_classifier, legacy_log_odds,
)
from histcontour_core.provenance import sha256_file
from histcontour_core.segment_review import geometry_features
from scripts.compare_contour_vector_outputs import pixel_proposals
from scripts.run_contour_context_experiment import CLASSES, DEVELOPMENT_SHEETS, metrics, nested_validation, read, resolve, validate_sample_labels, write
from scripts.score_ink_segments import _context_features, load_model

FAMILIES = {"neighborhood": CONTEXT_FEATURE_NAMES, "chain_shape": CHAIN_SHAPE_FEATURE_NAMES, "chain_context": CHAIN_FEATURE_NAMES,
            "legacy_prior_context": PRIOR_CONTEXT_FEATURE_NAMES, "legacy_prior_chain": PRIOR_CHAIN_FEATURE_NAMES}


def validate_separation(training_path, forward_path, vector_path):
    training, forward = read(training_path), read(forward_path)
    if (forward.get("role") != "forward_development_evaluation" or forward.get("holdout_used") is not False
            or forward.get("excluded_manifest_sha256") != sha256_file(training_path)
            or training.get("role") == "forward_development_evaluation"
            or forward.get("source_index_sha256") != training.get("source_index_sha256")
            or forward.get("vector_index_sha256") != sha256_file(vector_path)
            or training.get("vector_index_sha256") != sha256_file(vector_path)):
        raise ValueError("forward evaluation must be excluded and frozen against this exact old development run")
    old_ids = {row["segment_uid"] for row in training["samples"]}
    new_ids = {row["segment_uid"] for row in forward["samples"]}
    if old_ids & new_ids or len(old_ids) != len(training["samples"]) or len(new_ids) != len(forward["samples"]):
        raise ValueError("training and forward identities must be unique and disjoint")
    return training, forward


def describe_tile(gray, proposals, legacy):
    cache = NeighborhoodFeatures(gray)
    local_contexts = [cache.describe(proposal.points) for proposal in proposals]
    extras, chains, audit = chain_features(proposals, cache, local_contexts)
    records = []
    membership = {index: len(chain.member_indices) for chain in chains for index in chain.member_indices}
    for index, proposal in enumerate(proposals):
        local = {**geometry_features(proposal.points), **_context_features(gray, proposal.points), **local_contexts[index]}
        previous = legacy.probability(local)
        records.append({**local, **extras[index], "legacy_log_odds": legacy_log_odds(previous), "legacy_score": previous,
                        "chain_member_count": membership[index]})
    return records, audit


def fitter(records, feature_names, *, l2):
    if tuple(feature_names) == CONTEXT_FEATURE_NAMES:
        return fit_classifier(records, feature_names, l2=l2)
    return fit_chain_classifier(records, feature_names=feature_names, l2=l2)


def select_development_candidate(results, baseline_predictions):
    """Prefer no *additional* old-positive misses, not only an equal total count."""
    baseline_kept = {row["sample_id"] for row in baseline_predictions if row["class"] == "contour" and row["retained"]}
    baseline_negatives = metrics(baseline_predictions)["noncontours_rejected"]
    eligible = []
    for name, result in results.items():
        current_kept = {row["sample_id"] for row in result["outer_predictions"] if row["class"] == "contour" and row["retained"]}
        extra_misses = sorted(baseline_kept-current_kept)
        result["additional_development_contour_misses_vs_legacy"] = extra_misses
        if not extra_misses and result["metrics"]["noncontours_rejected"] > baseline_negatives:
            eligible.append(name)
    if not eligible:
        return None
    return min(eligible, key=lambda name: (-results[name]["metrics"]["noncontours_rejected"],
                                          -results[name]["metrics"]["contours_retained"],
                                          len(results[name]["final_model"]["feature_names"]), name))


def develop(index_path, vector_path, training_path, labels_path, forward_path, baseline_path, output):
    import numpy as np
    from PIL import Image
    training, forward = validate_separation(training_path, forward_path, vector_path)
    _, samples = validate_sample_labels(training_path, labels_path)
    if training["source_index_sha256"] != sha256_file(index_path):
        raise ValueError("source index changed")
    tiles = {row["tile_id"]: row for row in read(index_path)["tiles"] if row["split"] == "development" and row["sheet_id"] in DEVELOPMENT_SHEETS}
    raw_index = read(vector_path)
    entries = {row["tile_id"]: row for row in raw_index["tiles"]}
    if len(tiles) != 9 or set(tiles) != set(entries) or raw_index.get("holdout_included"):
        raise ValueError("only the nine current development tiles may be described")
    kind, legacy = load_model(baseline_path)
    if kind != "logistic":
        raise ValueError("the legacy prior must be the frozen logistic model")
    output.mkdir(parents=True, exist_ok=False)
    records, audits = [], []
    start = time.perf_counter()
    for tile_id, tile in tiles.items():
        path = resolve(tile["raster_path"])
        collection_path = resolve(entries[tile_id]["ink_vector_path"])
        raster_hash, vector_hash = sha256_file(path), sha256_file(collection_path)
        subset = [row for row in samples if row["tile_id"] == tile_id]
        if entries[tile_id]["source_raster_sha256"] != raster_hash or any(row["source_raster_sha256"] != raster_hash or row["source_vector_sha256"] != vector_hash for row in subset):
            raise ValueError("old sample source/vector changed")
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L")).copy()
        collection = read(collection_path)
        proposals = pixel_proposals(tile, collection)
        descriptors, audit = describe_tile(gray, proposals, legacy)
        by_uid = {feature["properties"]["segment_uid"]: descriptor for feature, descriptor in zip(collection["features"], descriptors)}
        for sample in subset:
            records.append({**{key: sample[key] for key in ("sample_id", "tile_id", "sheet_id", "segment_uid", "pixel_length", "class")}, **by_uid[sample["segment_uid"]]})
        audits.append({"tile_id": tile_id, "source_raster_sha256": raster_hash, "source_vector_sha256": vector_hash, **audit})
        print(tile_id, audit, flush=True)
    write(output/"training-descriptors.json", records)
    baseline_predictions = [{"sample_id": row["sample_id"], "class": row["class"], "score": row["legacy_score"], "retained": row["legacy_score"] >= .1}
                            for row in records if row["class"] in CLASSES]
    results = {}
    for family, names in FAMILIES.items():
        for target in (0.95, 1.0):
            name = f"{family}-inner{target:g}"
            result = nested_validation(records, names, l2=0.1, target_recall=target, fitter=fitter)
            result["family"], result["inner_retention_target"] = family, target
            results[name] = result
            print(name, json.dumps(result["metrics"], ensure_ascii=False), flush=True)
    selected = select_development_candidate(results, baseline_predictions)
    report = {"schema": "jap-map-contour-chain-development/1", "status": "development_selection_complete", "human_approved": False,
              "reference_origin": "ai_visual_provisional", "holdout_used": False, "automatic_promotion": False,
              "source_index_sha256": sha256_file(index_path), "vector_index_sha256": sha256_file(vector_path),
              "training_manifest_sha256": sha256_file(training_path), "training_labels_sha256": sha256_file(labels_path),
              "forward_manifest_sha256": sha256_file(forward_path), "legacy_model_sha256": sha256_file(baseline_path),
              "old_training_sample_count": len(records), "new_forward_sample_count": len(forward["samples"]),
              "forward_labels_read": False, "forward_candidate_predictions_exported": False, "forward_evaluation_metrics_produced": False,
              "selection_rule": "On OLD development only: reject more noncontours than legacy with no additional missed baseline-kept AI contour IDs; maximize rejected noncontours, then retained contours, then prefer fewer features.",
              "selected_candidate": selected, "baseline_metrics": metrics(baseline_predictions), "baseline_predictions": baseline_predictions,
              "results": results, "chain_audits": audits, "elapsed_seconds": time.perf_counter()-start,
              "limitations": ["OLD development folds are reused to select a method, not an independent performance claim.",
                              "New E-series class labels are not accepted by this command and cannot fit/calibrate models.",
                              "Chains group existing touching arms for features only; no geometry is changed or gap filled.",
                              "New evaluation still shares three source sheets; it is not external geographic validation."]}
    write(output/"development.json", report)
    print("Selected before new forward labels/predictions:", selected, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--training-samples", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--forward-samples", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    develop(args.index, args.vectors, args.training_samples, args.training_labels, args.forward_samples, args.legacy_model, args.output)


if __name__ == "__main__":
    main()
