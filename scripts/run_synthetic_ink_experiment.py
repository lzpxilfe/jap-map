"""Generate grouped synthetic Ink data and measure a contour-score baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.ink import ink_centerline_candidates
from histcontour_core.segment_review import binary_metrics, geometry_features, train_logistic_baseline
from histcontour_core.synthetic_ink import SYNTHETIC_DATASET_VERSION, iter_synthetic_cases
from histcontour_core.vectorization import skeleton_to_pixel_line_proposals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/synthetic_ink_experiment"))
    parser.add_argument("--terrain-count", type=int, default=120)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--write-cases", action="store_true", help="write PNG inputs and truth masks in addition to the report")
    return parser.parse_args()


def _resolve(value: Path) -> Path:
    return value if value.is_absolute() else REPOSITORY / value


def _samples(points):
    output = []
    for first, second in zip(points, points[1:]):
        count = max(1, int(math.ceil(math.hypot(second[0] - first[0], second[1] - first[1]))))
        output.extend((first[0] + step / count * (second[0] - first[0]), first[1] + step / count * (second[1] - first[1])) for step in range(count))
    return output + [points[-1]]


def _context(image, points):
    import numpy as np

    gray = np.asarray(image[..., :3].mean(axis=2), dtype=np.float32)
    height, width = gray.shape
    samples = _samples(points)
    xs = np.clip(np.rint([point[0] for point in samples]).astype(int), 0, width - 1)
    ys = np.clip(np.rint([point[1] for point in samples]).astype(int), 0, height - 1)
    centre_x, centre_y = int(round(sum(point[0] for point in samples) / len(samples))), int(round(sum(point[1] for point in samples) / len(samples)))
    def crop(radius):
        return gray[max(0, centre_y - radius):min(height, centre_y + radius + 1), max(0, centre_x - radius):min(width, centre_x + radius + 1)]
    near, context = crop(16), crop(48)
    return {
        "line_darkness": float((1.0 - gray[ys, xs] / 255.0).mean()),
        "dark_fraction_near": float((near <= 96).mean()),
        "dark_fraction_context": float((context <= 96).mean()),
        "mid_fraction_context": float((context <= 176).mean()),
        "context_std": float(context.std() / 255.0),
    }


def _truth_at_samples(mask, points):
    import numpy as np
    height, width = mask.shape
    samples = _samples(points)
    xs = np.clip(np.rint([point[0] for point in samples]).astype(int), 0, width - 1)
    ys = np.clip(np.rint([point[1] for point in samples]).astype(int), 0, height - 1)
    return float(mask[ys, xs].mean())


def records_for_case(case):
    evidence = ink_centerline_candidates(case.image)
    proposals = skeleton_to_pixel_line_proposals(evidence.centerline, evidence.center_score, minimum_length_px=12, simplify_tolerance_px=0.5, proposal_prefix="synthetic")
    records = []
    for proposal in proposals:
        visible = _truth_at_samples(case.visible_contour, proposal.points)
        non_contour = _truth_at_samples(case.non_contour, proposal.points)
        complete = _truth_at_samples(case.complete_contour, proposal.points)
        if visible >= 0.45 and non_contour <= 0.20:
            status = "contour"
        elif non_contour >= 0.45 and visible <= 0.15:
            status = "text"
        else:
            status = "unsure"
        records.append({
            "segment_uid": f"{case.case_id}:{proposal.proposal_id}",
            "terrain_id": case.terrain_id,
            "split": case.split,
            "review_status": status,
            "visible_contour_fraction": visible,
            "complete_contour_fraction": complete,
            "non_contour_fraction": non_contour,
            "pixel_points": tuple(proposal.points),
            **geometry_features(proposal.points),
            **_context(case.image, proposal.points),
        })
    return records


def choose_threshold(records, probabilities):
    candidates = []
    labels = [1 if record["review_status"] == "contour" else 0 for record in records]
    for step in range(5, 96, 5):
        threshold = step / 100
        metrics = binary_metrics(labels, probabilities, threshold)
        candidates.append((threshold, metrics))
    qualified = [item for item in candidates if item[1]["recall"] >= 0.95 and item[1]["specificity"] >= 0.20]
    return min(qualified, key=lambda item: item[0]) if qualified else max(candidates, key=lambda item: item[1]["balanced_accuracy"])


def run_experiment(*, terrain_count: int = 120, variants: int = 8, size: int = 192):
    all_records = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for case in iter_synthetic_cases(terrain_count=terrain_count, variants=variants, size=size):
        records = records_for_case(case)
        all_records.extend(records)
        counts[case.split] += 1
    usable = [record for record in all_records if record["review_status"] in {"contour", "text"}]
    train = [record for record in usable if record["split"] == "train"]
    validation = [record for record in usable if record["split"] == "validation"]
    test = [record for record in usable if record["split"] == "test"]
    if not train or not validation or not test:
        raise ValueError("the requested terrain count does not populate train, validation, and test groups")
    model = train_logistic_baseline(train)
    validation_probabilities = [model.probability(record) for record in validation]
    threshold, validation_metrics = choose_threshold(validation, validation_probabilities)
    test_probabilities = [model.probability(record) for record in test]
    test_metrics = binary_metrics([1 if record["review_status"] == "contour" else 0 for record in test], test_probabilities, threshold)
    return {
        "status": "trained",
        "dataset_version": SYNTHETIC_DATASET_VERSION,
        "terrain_count": terrain_count,
        "variants_per_terrain": variants,
        "case_counts": counts,
        "segment_counts": {"all": len(all_records), "eligible": len(usable), "train": len(train), "validation": len(validation), "test": len(test)},
        "ignored_mixed_segments": len(all_records) - len(usable),
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "promotion_criteria": {"minimum_contour_recall": 0.95, "minimum_non_contour_reduction": 0.20},
        "promotion_passed": test_metrics["recall"] >= 0.95 and test_metrics["specificity"] >= 0.20,
        "model": model.to_dict(),
        "feature_schema": model.to_dict()["feature_schema"],
        "scope": "synthetic evidence only; this does not claim historical-map accuracy",
    }


def write_cases(output_dir: Path, *, terrain_count: int, variants: int, size: int) -> None:
    """Write source rasters and separate visible/complete/non-contour masks."""

    from PIL import Image
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in iter_synthetic_cases(terrain_count=terrain_count, variants=variants, size=size):
        stem = case.case_id
        Image.fromarray(case.image).save(case_dir / f"{stem}.png")
        for name, mask in (("visible_contour", case.visible_contour), ("complete_contour", case.complete_contour), ("non_contour", case.non_contour), ("label_gap", case.label_gap)):
            Image.fromarray((mask.astype("uint8") * 255)).save(case_dir / f"{stem}-{name}.png")
        manifest.append({"case_id": stem, "terrain_id": case.terrain_id, "variant_id": case.variant_id, "split": case.split})
    (case_dir / "manifest.json").write_text(json.dumps({"dataset_version": SYNTHETIC_DATASET_VERSION, "cases": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    result = run_experiment(terrain_count=args.terrain_count, variants=args.variants, size=args.size)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_cases:
        write_cases(output_dir, terrain_count=args.terrain_count, variants=args.variants, size=args.size)
    (output_dir / "synthetic_ink_experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir / "synthetic_ink_experiment.json")
    print(f"promotion_passed={result['promotion_passed']} threshold={result['threshold']}")


if __name__ == "__main__":
    main()
