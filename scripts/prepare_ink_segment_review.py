"""Prepare a diverse, development-only Ink segment annotation queue.

No segment is automatically called a contour.  The output stores geometry
and raster-context features so the same human labels can train and evaluate a
classifier later without re-vectorising Ink.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.segment_review import FEATURE_NAMES, diverse_sample, geometry_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="annotation_package/index.json")
    parser.add_argument(
        "--ink-index",
        type=Path,
        default=Path("data/derived/annotation_package/ink_candidate_vectors_evidence_v2/ink_candidate_vector_index.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/annotation_package/ink_segment_review"),
    )
    parser.add_argument(
        "--score-index",
        type=Path,
        help="Optional ink_contour_score_index.json; scores stay review-only",
    )
    parser.add_argument("--per-tile", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=96)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY / path


def _portable(path: Path) -> str:
    try:
        value = path.resolve().relative_to(REPOSITORY)
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def map_to_pixel(tile: dict, coordinates) -> list[tuple[float, float]]:
    west, south, east, north = (float(value) for value in tile["bounds"])
    width, height = tile["pixel_bounds"][2:]
    return [
        (
            (float(x) - west) * width / (east - west) - 0.5,
            (north - float(y)) * height / (north - south) - 0.5,
        )
        for x, y in coordinates
    ]


def _polyline_samples(points, spacing: float = 1.0):
    samples = []
    for first, second in zip(points, points[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        count = max(1, int(math.ceil(length / spacing)))
        for step in range(count):
            fraction = step / count
            samples.append((first[0] + fraction * (second[0] - first[0]), first[1] + fraction * (second[1] - first[1])))
    samples.append(points[-1])
    return samples


def context_features(grayscale, points) -> dict[str, float]:
    import numpy as np

    height, width = grayscale.shape
    samples = _polyline_samples(points)
    xs = np.clip(np.rint([point[0] for point in samples]).astype(int), 0, width - 1)
    ys = np.clip(np.rint([point[1] for point in samples]).astype(int), 0, height - 1)
    line_darkness = float(np.mean(1.0 - grayscale[ys, xs].astype(np.float32) / 255.0))
    centre_x = int(round(sum(point[0] for point in samples) / len(samples)))
    centre_y = int(round(sum(point[1] for point in samples) / len(samples)))

    def crop(radius):
        return grayscale[max(0, centre_y - radius) : min(height, centre_y + radius + 1), max(0, centre_x - radius) : min(width, centre_x + radius + 1)]

    near, context = crop(16), crop(48)
    return {
        "line_darkness": line_darkness,
        "dark_fraction_near": float(np.mean(near <= 96)),
        "dark_fraction_context": float(np.mean(context <= 96)),
        "mid_fraction_context": float(np.mean(context <= 176)),
        "context_std": float(np.std(context.astype(np.float32)) / 255.0),
    }


def load_records(tile: dict, vector_path: Path, score_by_segment_uid: dict[str, dict] | None = None) -> tuple[list[dict], object]:
    import numpy as np
    from PIL import Image

    with Image.open(_resolve(tile["raster_path"])) as image:
        grayscale = np.asarray(image.convert("L")).copy()
    collection = json.loads(vector_path.read_text(encoding="utf-8"))
    records = []
    for feature in collection["features"]:
        properties = feature["properties"]
        segment_uid = properties.get("segment_uid", f"{tile['tile_id']}:{properties['proposal_id']}")
        score_properties = (score_by_segment_uid or {}).get(segment_uid, {})
        points = map_to_pixel(tile, feature["geometry"]["coordinates"])
        descriptor = {
            "segment_uid": segment_uid,
            "proposal_id": properties["proposal_id"],
            "segment_geometry_id": properties.get("segment_geometry_id"),
            "ink_run_id": properties.get("ink_run_id"),
            "source_raster_sha256": properties.get("source_raster_sha256"),
            "backend": properties.get("backend"),
            "upstream_commit": properties.get("upstream_commit"),
            "adapter_version": properties.get("adapter_version"),
            "ink_support": score_properties.get("ink_support", properties.get("ink_support")),
            "contour_score": score_properties.get("contour_score"),
            "tile_id": tile["tile_id"],
            "sheet_id": tile["sheet_id"],
            "split": tile["split"],
            "scene_type": tile["scene_type"],
            "geometry": feature["geometry"],
            "pixel_points": points,
            **geometry_features(points),
            **context_features(grayscale, points),
        }
        records.append(descriptor)
    return records, grayscale


def _review_feature(record: dict, sample_rank: int) -> dict:
    properties = {
        "segment_uid": record["segment_uid"],
        "proposal_id": record["proposal_id"],
        "segment_geometry_id": record["segment_geometry_id"],
        "ink_run_id": record["ink_run_id"],
        "source_raster_sha256": record["source_raster_sha256"],
        "backend": record["backend"],
        "upstream_commit": record["upstream_commit"],
        "adapter_version": record["adapter_version"],
        "ink_support": record["ink_support"],
        "contour_score": record["contour_score"],
        "tile_id": record["tile_id"],
        "sheet_id": record["sheet_id"],
        "split": record["split"],
        "scene_type": record["scene_type"],
        "cv_group": record["sheet_id"],
        "sample_rank": sample_rank,
        "review_status": "unreviewed",
        "review_note": None,
    }
    properties.update({name: round(float(record[name]), 6) for name in FEATURE_NAMES})
    return {"type": "Feature", "properties": properties, "geometry": record["geometry"]}


def _patch(grayscale, points, patch_size: int, label: str):
    from PIL import Image, ImageDraw

    half = patch_size // 2
    centre_x = int(round(sum(point[0] for point in points) / len(points)))
    centre_y = int(round(sum(point[1] for point in points) / len(points)))
    canvas = Image.new("RGB", (patch_size, patch_size), "white")
    source = Image.fromarray(grayscale).convert("RGB")
    box = (centre_x - half, centre_y - half, centre_x - half + patch_size, centre_y - half + patch_size)
    canvas.paste(source.crop(box), (0, 0))
    shifted = [(x - box[0], y - box[1]) for x, y in points]
    draw = ImageDraw.Draw(canvas)
    draw.line(shifted, fill=(225, 29, 72), width=3)
    draw.rectangle((0, 0, 31, 13), fill="white")
    draw.text((2, 1), label, fill="black")
    return canvas


def _contact_sheet(tile_id: str, selected: list[dict], grayscale, output_path: Path, patch_size: int) -> None:
    from PIL import Image, ImageDraw

    columns = 8
    cell_height = patch_size + 18
    rows = math.ceil(len(selected) / columns)
    contact = Image.new("RGB", (columns * patch_size, rows * cell_height), "white")
    draw = ImageDraw.Draw(contact)
    for index, record in enumerate(selected, 1):
        column, row = (index - 1) % columns, (index - 1) // columns
        left, top = column * patch_size, row * cell_height
        contact.paste(_patch(grayscale, record["pixel_points"], patch_size, str(index)), (left, top))
        draw.text((left + 2, top + patch_size + 2), record["proposal_id"], fill="black")
    contact.save(output_path)


def _score_lookup(score_index: dict | None) -> dict[str, dict]:
    if not score_index:
        return {}
    records = {}
    for tile in score_index.get("tiles", []):
        collection = json.loads(_resolve(tile["path"]).read_text(encoding="utf-8"))
        for feature in collection["features"]:
            properties = feature["properties"]
            if properties.get("segment_uid"):
                records[properties["segment_uid"]] = properties
    return records


def prepare(index: dict, ink_index: dict, output_dir: Path, per_tile: int, patch_size: int, *, source_ink_index: Path | None = None, score_index: dict | None = None, source_score_index: Path | None = None) -> dict:
    if per_tile < 1 or patch_size < 32:
        raise ValueError("per_tile must be positive and patch_size must be at least 32")
    if ink_index.get("holdout_included"):
        raise ValueError("refusing an Ink index that includes holdout tiles")
    tiles = {tile["tile_id"]: tile for tile in index["tiles"]}
    score_by_segment_uid = _score_lookup(score_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = []
    tile_results = []
    for ink_tile in ink_index["tiles"]:
        tile = tiles[ink_tile["tile_id"]]
        if tile["split"] == "holdout_test":
            raise ValueError(f"holdout tile reached review preparation: {tile['tile_id']}")
        vector_path = _resolve(ink_tile["ink_vector_path"])
        records, grayscale = load_records(tile, vector_path, score_by_segment_uid)
        selected = diverse_sample(records, min(per_tile, len(records)))
        selected = sorted(selected, key=lambda record: record["segment_uid"])
        contact_path = output_dir / f"{tile['tile_id']}-ink-review-contact.png"
        _contact_sheet(tile["tile_id"], selected, grayscale, contact_path, patch_size)
        features = [_review_feature(record, index) for index, record in enumerate(selected, 1)]
        selected_features.extend(features)
        tile_results.append(
            {
                "tile_id": tile["tile_id"],
                "sheet_id": tile["sheet_id"],
                "scene_type": tile["scene_type"],
                "available_count": len(records),
                "selected_count": len(selected),
                "contact_sheet_path": _portable(contact_path),
            }
        )
    queue_path = output_dir / "ink_segment_review_candidates.geojson"
    collection = {
        "type": "FeatureCollection",
        "name": "ink_segment_review_candidates",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::5132"}},
        "features": selected_features,
    }
    _atomic_json(queue_path, collection)
    result = {
        "version": "2",
        "purpose": "human labels for context-aware Ink segment classification",
        "automatic_contour_labels": False,
        "holdout_included": False,
        "cross_validation_group": "sheet_id",
        "review_statuses": ["unreviewed", "contour", "text", "road_river", "symbol", "unsure"],
        "feature_names": list(FEATURE_NAMES),
        "per_tile": per_tile,
        "patch_size": patch_size,
        "selected_count": len(selected_features),
        "source_ink_index": _portable(source_ink_index) if source_ink_index else None,
        "source_score_index": _portable(source_score_index) if source_score_index else None,
        "scored_segments_available": len(score_by_segment_uid),
        "review_candidates_path": _portable(queue_path),
        "tiles": tile_results,
    }
    _atomic_json(output_dir / "ink_segment_review_index.json", result)
    return result


def main():
    args = parse_args()
    index = json.loads(_resolve(args.index).read_text(encoding="utf-8"))
    ink_index = json.loads(_resolve(args.ink_index).read_text(encoding="utf-8"))
    score_path = _resolve(args.score_index) if args.score_index else None
    score_index = json.loads(score_path.read_text(encoding="utf-8")) if score_path else None
    result = prepare(index, ink_index, _resolve(args.output_dir), args.per_tile, args.patch_size, source_ink_index=_resolve(args.ink_index), score_index=score_index, source_score_index=score_path)
    print(result["review_candidates_path"])
    print(f"{result['selected_count']} unlabeled Ink segments; holdout excluded")


if __name__ == "__main__":
    main()
