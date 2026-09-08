"""Report exact geometry matches before migrating Ink review labels in QGIS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qgis.core import QgsApplication, QgsVectorLayer

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from histcontour_core.provenance import polyline_geometry_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="annotation_package/index.json")
    parser.add_argument("old_gpkg", type=Path)
    parser.add_argument("new_candidates", type=Path, help="new v2 review candidates GeoJSON")
    parser.add_argument("--old-layer", default="ink_segment_review")
    parser.add_argument("--output", type=Path, default=Path("data/derived/annotation_package/ink_label_reconciliation.json"))
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def map_to_pixel(tile, coordinates):
    west, south, east, north = (float(value) for value in tile["bounds"])
    width, height = tile["pixel_bounds"][2:]
    return [((float(x) - west) * width / (east - west) - 0.5, (north - float(y)) * height / (north - south) - 0.5) for x, y in coordinates]


def main():
    args = parse_args()
    index = json.loads(_resolve(args.index).read_text(encoding="utf-8"))
    tiles = {tile["tile_id"]: tile for tile in index["tiles"]}
    candidates = json.loads(_resolve(args.new_candidates).read_text(encoding="utf-8"))
    new_ids = {feature["properties"].get("segment_geometry_id") for feature in candidates["features"]}
    application = QgsApplication([], False)
    application.initQgis()
    try:
        layer = QgsVectorLayer(f"{_resolve(args.old_gpkg)}|layername={args.old_layer}", args.old_layer, "ogr")
        if not layer.isValid():
            raise SystemExit(f"Could not open {args.old_layer} from {args.old_gpkg}")
        fields = {field.name() for field in layer.fields()}
        required = {"tile_id", "review_status"}
        if not required <= fields:
            raise SystemExit(f"old layer needs fields: {', '.join(sorted(required))}")
        matched, unmatched, statuses = [], [], {}
        for feature in layer.getFeatures():
            tile_id = str(feature["tile_id"])
            tile = tiles.get(tile_id)
            if tile is None:
                unmatched.append({"feature_id": feature.id(), "reason": "unknown_tile", "tile_id": tile_id})
                continue
            geometry = feature.geometry()
            if geometry.isMultipart():
                parts = geometry.asMultiPolyline()
                if len(parts) != 1:
                    unmatched.append({"feature_id": feature.id(), "reason": "multipart_geometry", "tile_id": tile_id})
                    continue
                coordinates = [(point.x(), point.y()) for point in parts[0]]
            else:
                coordinates = [(point.x(), point.y()) for point in geometry.asPolyline()]
            try:
                geometry_id = polyline_geometry_id(map_to_pixel(tile, coordinates))
            except ValueError:
                unmatched.append({"feature_id": feature.id(), "reason": "invalid_geometry", "tile_id": tile_id})
                continue
            status = str(feature["review_status"])
            if geometry_id in new_ids:
                matched.append({"feature_id": feature.id(), "segment_geometry_id": geometry_id, "review_status": status})
                statuses[status] = statuses.get(status, 0) + 1
            else:
                unmatched.append({"feature_id": feature.id(), "reason": "geometry_changed_or_removed", "segment_geometry_id": geometry_id, "tile_id": tile_id, "review_status": status})
        report = {
            "migration_policy": "only exact normalized pixel geometry IDs are eligible for label carry-forward",
            "old_layer": args.old_layer,
            "new_candidate_count": len(new_ids),
            "exact_match_count": len(matched),
            "matched_status_counts": statuses,
            "matches": matched,
            "unmatched": unmatched,
            "automatic_write_performed": False,
        }
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output)
    finally:
        application.exitQgis()


if __name__ == "__main__":
    main()
