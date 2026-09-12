#!/usr/bin/env python3
"""Nested leave-one-development-sheet-out experiment with provisional labels.

This is a research comparison, not an independent human accuracy certificate.
The outer test sheet never enters model fitting, scaling, or threshold choice.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.contour_context import (
    BASE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, CONTEXT_SCHEMA, NeighborhoodFeatures,
    fit_classifier, probability, recall_first_threshold,
)
from histcontour_core.provenance import sha256_file
from histcontour_core.segment_review import geometry_features
from scripts.score_ink_segments import _context_features, load_model

CLASSES = ("contour", "text", "road_river", "symbol")
DEVELOPMENT_SHEETS = {"173-buyeo", "174-cheongyang", "177-nonsan"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def validate_sample_labels(manifest_path, labels_path):
    manifest, labels = read(manifest_path), read(labels_path)
    if manifest.get("role") == "forward_development_evaluation":
        raise ValueError("forward-evaluation-only samples must not enter model fitting or threshold calibration")
    if manifest.get("holdout_used") is not False or labels.get("human_approved") is not False or labels.get("reference_origin") != "ai_visual_provisional":
        raise ValueError("this experiment accepts only explicit non-human development evidence")
    if labels.get("sample_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("labels do not belong to the frozen sample manifest")
    samples, items = manifest["samples"], labels["items"]
    ids = [row["sample_id"] for row in samples]
    label_ids = [row["sample_id"] for row in items]
    if len(set(ids)) != len(ids) or len(set(label_ids)) != len(label_ids) or set(ids) != set(label_ids):
        raise ValueError("sample labels must be an exact one-to-one match")
    if {row["sheet_id"] for row in samples} != DEVELOPMENT_SHEETS or len({row["tile_id"] for row in samples}) != 9:
        raise ValueError("only the nine specified development tiles may be used")
    if len({row["segment_uid"] for row in samples}) != len(samples):
        raise ValueError("duplicate segments may not be weighted as independent samples")
    allowed = set(CLASSES) | {"mixed", "unsure"}
    if any(row["class"] not in allowed for row in items):
        raise ValueError("all samples require an explicit provisional class or exclusion")
    labels_by_id = {row["sample_id"]: row for row in items}
    return manifest, [{**row, "class": labels_by_id[row["sample_id"]]["class"]} for row in samples]


def descriptor_records(index_path, manifest_path, labels_path, legacy_path):
    import numpy as np
    from PIL import Image
    manifest, samples = validate_sample_labels(manifest_path, labels_path)
    if manifest["source_index_sha256"] != sha256_file(index_path):
        raise ValueError("source index changed after sampling")
    index = read(index_path)
    tiles = {row["tile_id"]: row for row in index["tiles"] if row["split"] == "development" and row["sheet_id"] in DEVELOPMENT_SHEETS}
    if set(tiles) != {row["tile_id"] for row in samples}:
        raise ValueError("source index and development sample tiles differ")
    kind, legacy = load_model(legacy_path)
    if kind != "logistic":
        raise ValueError("this baseline must be the frozen legacy logistic model")
    records = []
    for tile_id, tile in tiles.items():
        subset = [row for row in samples if row["tile_id"] == tile_id]
        raster_path = resolve(tile["raster_path"])
        if any(row["source_raster_sha256"] != sha256_file(raster_path) or row["sheet_id"] != tile["sheet_id"] for row in subset):
            raise ValueError("sample source fingerprint or sheet identity changed")
        with Image.open(raster_path) as image:
            gray = np.asarray(image.convert("L")).copy()
        cache = NeighborhoodFeatures(gray)
        for row in subset:
            points = row["pixel_points"]
            descriptor = {**geometry_features(points), **_context_features(gray, points), **cache.describe(points)}
            records.append({**{key: row[key] for key in ("sample_id", "tile_id", "sheet_id", "length_bin", "segment_uid", "class", "pixel_length")},
                            **descriptor, "legacy_score": legacy.probability(descriptor)})
        print(f"Described {tile_id}: {len(subset)} frozen lines", flush=True)
    return records


def metrics(rows):
    tp = sum(row["class"] == "contour" and row["retained"] for row in rows)
    fn = sum(row["class"] == "contour" and not row["retained"] for row in rows)
    fp = sum(row["class"] != "contour" and row["retained"] for row in rows)
    tn = sum(row["class"] != "contour" and not row["retained"] for row in rows)
    return {"contours_retained": tp, "contours_rejected": fn, "noncontours_retained": fp, "noncontours_rejected": tn,
            "provisional_contour_retention": tp/(tp+fn) if tp+fn else None,
            "provisional_noncontour_rejection": tn/(tn+fp) if tn+fp else None,
            "provisional_selected_precision": tp/(tp+fp) if tp+fp else None,
            "by_class": {kind: {"total": sum(row["class"] == kind for row in rows),
                                "retained": sum(row["class"] == kind and row["retained"] for row in rows)} for kind in CLASSES}}


def nested_validation(records, feature_names, *, l2=0.1, target_recall=0.95, fitter=fit_classifier):
    usable = [row for row in records if row["class"] in CLASSES]
    sheets = sorted({row["sheet_id"] for row in usable})
    if len(sheets) != 3:
        raise ValueError("three development sheets are required for nested validation")
    folds, predictions = [], []
    for test_sheet in sheets:
        train = [row for row in usable if row["sheet_id"] != test_sheet]
        test = [row for row in usable if row["sheet_id"] == test_sheet]
        calibration, inner_folds = [], []
        for validation_sheet in sorted({row["sheet_id"] for row in train}):
            inner_train = [row for row in train if row["sheet_id"] != validation_sheet]
            inner_validation = [row for row in train if row["sheet_id"] == validation_sheet]
            model = fitter(inner_train, feature_names, l2=l2)
            calibration.extend({"sample_id": row["sample_id"], "class": row["class"], "score": probability(model, row)} for row in inner_validation)
            inner_folds.append({"training_sample_ids": [row["sample_id"] for row in inner_train],
                                "validation_sample_ids": [row["sample_id"] for row in inner_validation]})
        threshold = recall_first_threshold(calibration, target_recall=target_recall)
        model = fitter(train, feature_names, l2=l2)
        outer_rows = []
        for row in test:
            score = probability(model, row)
            outer_rows.append({"sample_id": row["sample_id"], "sheet_id": test_sheet, "class": row["class"],
                               "score": score, "threshold": threshold, "retained": score >= threshold})
        predictions.extend(outer_rows)
        folds.append({"test_sheet": test_sheet, "training_sample_ids": [row["sample_id"] for row in train],
                      "test_sample_ids": [row["sample_id"] for row in test], "threshold": threshold,
                      "inner_folds": inner_folds, "inner_predictions": calibration,
                      "model": model, "metrics": metrics(outer_rows)})
    final_model = fitter(usable, feature_names, l2=l2)
    # A deployment threshold fitted to OOF scores is not a new independent metric.
    final_threshold = recall_first_threshold(predictions, target_recall=target_recall)
    return {"folds": folds, "outer_predictions": predictions, "metrics": metrics(predictions),
            "final_model": final_model, "final_threshold": final_threshold,
            "final_threshold_scope": "calibrated to pooled outer scores for a later all-development fit; do not report same-data threshold metrics as independent validation"}


def experiment(index_path, manifest_path, labels_path, legacy_path, output, *, l2=0.1, target_recall=0.95):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records = descriptor_records(index_path, manifest_path, labels_path, legacy_path)
    write(output/"descriptors.json", records)
    usable = [row for row in records if row["class"] in CLASSES]
    legacy_predictions = [{"sample_id": row["sample_id"], "sheet_id": row["sheet_id"], "class": row["class"],
                           "score": row["legacy_score"], "threshold": 0.1, "retained": row["legacy_score"] >= 0.1} for row in usable]
    alternatives = {}
    for name, feature_names in (("real_weak_base", BASE_FEATURE_NAMES), ("real_weak_neighborhood", CONTEXT_FEATURE_NAMES)):
        alternatives[name] = nested_validation(records, feature_names, l2=l2, target_recall=target_recall)
        print(name, json.dumps(alternatives[name]["metrics"], ensure_ascii=False), flush=True)
    report = {"schema": "jap-map-contour-context-experiment/1", "status": "completed_research_only", "human_approved": False,
              "reference_origin": "ai_visual_provisional", "formal_human_accuracy": None, "holdout_used": False,
              "automatic_promotion": False, "source_index_sha256": sha256_file(index_path), "sample_manifest_sha256": sha256_file(manifest_path),
              "labels_sha256": sha256_file(labels_path), "legacy_model_sha256": sha256_file(legacy_path),
              "feature_schema": CONTEXT_SCHEMA, "l2_fixed_for_both_models": l2, "target_inner_contour_retention": target_recall,
              "sampling_scope": "equal counts per source tile and length bin; unweighted sample metrics are not whole-map population rates",
              "class_counts": dict(Counter(row["class"] for row in records)), "usable_samples": len(usable),
              "legacy_fixed_010": {"predictions": legacy_predictions, "metrics": metrics(legacy_predictions)},
              "alternatives": alternatives, "elapsed_seconds": time.perf_counter()-start,
              "limitations": ["Only three development sheets; excluded Gongju has not been viewed or tuned.",
                              "AI labels are imperfect, not human approvals or complete contour ground truth.",
                              "This compares classification of existing candidates, not missing contour detection or gap-joining correctness.",
                              "Outer-fold metrics use inner-only thresholds. All-data final model is not independently tested."]}
    write(output/"experiment.json", report)
    print("legacy_fixed_010", json.dumps(report["legacy_fixed_010"]["metrics"], ensure_ascii=False), flush=True)
    print(output/"experiment.json", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--legacy-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--target-recall", type=float, default=0.95, help="inner-validation contour retention target; never fitted on the outer test sheet")
    args = parser.parse_args()
    experiment(args.index, args.samples, args.labels, args.legacy_model, args.output, l2=args.l2, target_recall=args.target_recall)


if __name__ == "__main__":
    main()
