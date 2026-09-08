"""Train and sheet-cross-validate a segment-level Ink contour baseline.

This script never edits review labels or promotes predictions into contour
geometry.  With too few labels it writes a readiness report instead of
pretending to have measured model quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.segment_review import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    binary_metrics,
    labelled_training_records,
    train_logistic_baseline,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels",
        type=Path,
        help="contour_annotations.gpkg or ink_segment_review_candidates.geojson",
    )
    parser.add_argument("--layer", default="ink_segment_review")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/annotation_package/ink_segment_review/ink_segment_baseline.json"),
    )
    parser.add_argument("--minimum-per-class", type=int, default=20)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def read_geojson(path: Path) -> list[dict]:
    collection = json.loads(path.read_text(encoding="utf-8"))
    return [dict(feature["properties"]) for feature in collection["features"]]


def read_gpkg(path: Path, layer: str) -> list[dict]:
    safe_layer = layer.replace('"', '""')
    fields = ("segment_uid", "tile_id", "sheet_id", "split", "review_status", *FEATURE_NAMES)
    with sqlite3.connect(path) as database:
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if layer not in tables:
            raise ValueError(f"GeoPackage has no {layer!r} layer")
        rows = database.execute(
            f'SELECT {", ".join(fields)} FROM "{safe_layer}" ORDER BY segment_uid'
        ).fetchall()
    return [dict(zip(fields, row)) for row in rows]


def read_records(path: Path, layer: str = "ink_segment_review") -> list[dict]:
    if path.suffix.lower() in (".json", ".geojson"):
        return read_geojson(path)
    if path.suffix.lower() == ".gpkg":
        return read_gpkg(path, layer)
    raise ValueError("labels must be GeoJSON or GeoPackage")


def label_counts(records) -> dict[str, int]:
    counts = {status: 0 for status in ("unreviewed", "contour", "text", "road_river", "symbol", "unsure")}
    for record in records:
        status = record.get("review_status")
        counts[status] = counts.get(status, 0) + 1
    return counts


def train_report(records: list[dict], minimum_per_class: int = 20) -> dict:
    if minimum_per_class < 2:
        raise ValueError("minimum_per_class must be at least two")
    counts = label_counts(records)
    contaminated = [record.get("segment_uid") for record in records if record.get("split") == "holdout_test"]
    if contaminated:
        return {
            "status": "blocked_holdout_contamination",
            "total_queue": len(records),
            "label_counts": counts,
            "contaminated_count": len(contaminated),
            "example_segment_uids": contaminated[:10],
            "holdout_policy": "remove holdout_test rows before any training or tuning",
        }
    labelled = labelled_training_records(records)
    contour_count = sum(record["review_status"] == "contour" for record in labelled)
    non_contour_count = len(labelled) - contour_count
    groups = sorted({record["sheet_id"] for record in labelled})
    readiness = {
        "status": "ready" if contour_count >= minimum_per_class and non_contour_count >= minimum_per_class and len(groups) >= 3 else "waiting_for_labels",
        "total_queue": len(records),
        "label_counts": counts,
        "binary_training_count": len(labelled),
        "contour_count": contour_count,
        "non_contour_count": non_contour_count,
        "sheet_groups": groups,
        "minimum_per_class": minimum_per_class,
        "requirements": "at least the minimum in both classes across all three development sheets",
        "holdout_policy": "Gongju is absent; final holdout scoring is a separate explicit step",
    }
    if readiness["status"] != "ready":
        return readiness

    folds = []
    all_labels, all_probabilities = [], []
    for held_out in groups:
        training = [record for record in labelled if record["sheet_id"] != held_out]
        testing = [record for record in labelled if record["sheet_id"] == held_out]
        try:
            model = train_logistic_baseline(training)
        except ValueError as error:
            readiness["status"] = "waiting_for_sheet_balance"
            readiness["reason"] = f"{held_out}: {error}"
            return readiness
        labels = [1 if record["review_status"] == "contour" else 0 for record in testing]
        probabilities = [model.probability(record) for record in testing]
        folds.append({"held_out_sheet": held_out, **binary_metrics(labels, probabilities)})
        all_labels.extend(labels)
        all_probabilities.extend(probabilities)
    final_model = train_logistic_baseline(labelled)
    readiness.update(
        {
            "status": "trained",
            "evaluation": "leave-one-sheet-out cross-validation",
            "folds": folds,
            "aggregate_metrics": binary_metrics(all_labels, all_probabilities),
            "model": final_model.to_dict(),
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "promotion_policy": "review predictions manually; do not replace Ink geometry or auto-create contour_gt",
        }
    )
    return readiness


def main():
    args = parse_args()
    records = read_records(_resolve(args.labels), args.layer)
    report = train_report(records, args.minimum_per_class)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(output)
    print(report["status"], report["label_counts"])


if __name__ == "__main__":
    main()
