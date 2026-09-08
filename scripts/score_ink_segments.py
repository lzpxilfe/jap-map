"""Apply a compatible contour-score model to every Ink proposal.

Scores are review evidence only.  This command writes a new GeoJSON layer and
never changes the original Ink proposals, review queue, or contour ground
truth.  The caller must provide a model produced by
``train_ink_segment_classifier.py`` with the current feature schema.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.segment_review import LogisticModel, geometry_features
from histcontour_core.onnx_scorer import OnnxScorer, OnnxScorerUnavailable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="annotation_package/index.json")
    parser.add_argument("model", type=Path, help="trained baseline JSON")
    parser.add_argument(
        "--ink-index",
        type=Path,
        default=Path("data/derived/annotation_package/ink_candidate_vectors_evidence_v2/ink_candidate_vector_index.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/annotation_package/ink_scored_vectors"),
    )
    parser.add_argument("--include-holdout", action="store_true")
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _map_to_pixel(tile: dict, coordinates):
    west, south, east, north = (float(value) for value in tile["bounds"])
    width, height = tile["pixel_bounds"][2:]
    return [
        ((float(x) - west) * width / (east - west) - 0.5, (north - float(y)) * height / (north - south) - 0.5)
        for x, y in coordinates
    ]


def _sample_polyline(points):
    samples = []
    for first, second in zip(points, points[1:]):
        count = max(1, int(math.ceil(math.hypot(second[0] - first[0], second[1] - first[1]))))
        samples.extend((first[0] + step / count * (second[0] - first[0]), first[1] + step / count * (second[1] - first[1])) for step in range(count))
    return samples + [points[-1]]


def _context_features(grayscale, points) -> dict[str, float]:
    import numpy as np

    height, width = grayscale.shape
    samples = _sample_polyline(points)
    xs = np.clip(np.rint([point[0] for point in samples]).astype(int), 0, width - 1)
    ys = np.clip(np.rint([point[1] for point in samples]).astype(int), 0, height - 1)
    line_darkness = float(np.mean(1.0 - grayscale[ys, xs].astype(np.float32) / 255.0))
    centre_x = int(round(sum(point[0] for point in samples) / len(samples)))
    centre_y = int(round(sum(point[1] for point in samples) / len(samples)))

    def crop(radius):
        return grayscale[max(0, centre_y - radius):min(height, centre_y + radius + 1), max(0, centre_x - radius):min(width, centre_x + radius + 1)]

    near, context = crop(16), crop(48)
    return {
        "line_darkness": line_darkness,
        "dark_fraction_near": float(np.mean(near <= 96)),
        "dark_fraction_context": float(np.mean(context <= 96)),
        "mid_fraction_context": float(np.mean(context <= 176)),
        "context_std": float(np.std(context.astype(np.float32)) / 255.0),
    }


def load_model(path: Path):
    if path.suffix.lower() == ".onnx":
        try:
            return "onnx", OnnxScorer(path)
        except (OnnxScorerUnavailable, ValueError) as error:
            raise ValueError(f"ONNX scorer is unavailable or incompatible: {error}") from error
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "trained":
        raise ValueError("model report is not trained; score export is intentionally unavailable")
    return "logistic", LogisticModel.from_dict(report.get("model", {}))


def score_tile(tile: dict, vector_path: Path, model) -> dict:
    import numpy as np
    from PIL import Image

    with Image.open(_resolve(tile["raster_path"])) as image:
        grayscale = np.asarray(image.convert("L")).copy()
    collection = json.loads(vector_path.read_text(encoding="utf-8"))
    scored = []
    for feature in collection["features"]:
        properties = dict(feature["properties"])
        points = _map_to_pixel(tile, feature["geometry"]["coordinates"])
        descriptor = {**geometry_features(points), **_context_features(grayscale, points)}
        kind, scorer = model
        score = scorer.probability(source, points) if kind == "onnx" else scorer.probability(descriptor)
        properties.update(
            contour_score=round(score, 6),
            contour_score_kind="synthetic_or_review-trained logistic baseline",
            ink_support=float(properties.get("ink_support", properties.get("confidence", 0.0))),
            contour_score_review_only=True,
        )
        scored.append({"type": "Feature", "properties": properties, "geometry": feature["geometry"]})
    return {"type": "FeatureCollection", "name": f"ink_contour_scores_{tile['tile_id']}", "crs": collection.get("crs"), "features": scored}


def main():
    args = parse_args()
    index = json.loads(_resolve(args.index).read_text(encoding="utf-8"))
    ink_index = json.loads(_resolve(args.ink_index).read_text(encoding="utf-8"))
    model = load_model(_resolve(args.model))
    tiles = {tile["tile_id"]: tile for tile in index["tiles"]}
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for candidate in ink_index["tiles"]:
        tile = tiles[candidate["tile_id"]]
        if tile["split"] == "holdout_test" and not args.include_holdout:
            continue
        vector_path = _resolve(candidate["ink_vector_path"])
        collection = score_tile(tile, vector_path, model)
        output_path = output_dir / f"{tile['tile_id']}-ink-contour-scores.geojson"
        output_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append({"tile_id": tile["tile_id"], "split": tile["split"], "path": _portable(output_path), "feature_count": len(collection["features"])})
    summary = {
        "version": "1",
        "review_only": True,
        "model_kind": model[0],
        "model_feature_schema": model[1].to_dict()["feature_schema"] if model[0] == "logistic" else model[1].metadata["schema"],
        "model_path": _portable(_resolve(args.model)),
        "source_ink_index": _portable(_resolve(args.ink_index)),
        "holdout_included": bool(args.include_holdout),
        "tiles": outputs,
    }
    (output_dir / "ink_contour_score_index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_dir / "ink_contour_score_index.json")


if __name__ == "__main__":
    main()
