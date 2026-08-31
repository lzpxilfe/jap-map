import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY / "scripts" / "generate_contour_completion_candidates.py"
SPEC = importlib.util.spec_from_file_location("generate_contour_completion_candidates", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)


class ContourCompletionScriptTest(unittest.TestCase):
    def test_safe_defaults_require_reviewed_contour_anchors(self):
        with patch.object(sys, "argv", ["script", "index.json"]):
            args = SCRIPT.parse_args()
        self.assertEqual(args.anchor_status, "contour")
        self.assertFalse(args.include_holdout)

    def test_holdout_selection_requires_explicit_opt_in(self):
        index = {
            "tiles": [
                {"tile_id": "dev", "split": "development"},
                {"tile_id": "holdout", "split": "holdout_test"},
            ]
        }
        self.assertEqual([tile["tile_id"] for tile in SCRIPT.select_tiles(index, {"dev", "holdout"})], ["dev"])
        self.assertEqual(
            [tile["tile_id"] for tile in SCRIPT.select_tiles(index, {"dev", "holdout"}, True)],
            ["dev", "holdout"],
        )

    def test_pixel_and_map_coordinates_round_trip_at_pixel_centres(self):
        tile = {"bounds": [100.0, 200.0, 104.0, 202.0], "pixel_bounds": [0, 0, 4, 2]}
        for point in ((0.0, 0.0), (3.0, 1.0)):
            self.assertEqual(SCRIPT._pixel_point(tile, SCRIPT._map_point(tile, point)), point)

    def test_review_statuses_are_read_without_qgis(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "review.gpkg"
            with sqlite3.connect(package) as connection:
                connection.execute("CREATE TABLE proposal_review (proposal_uid TEXT, review_status TEXT)")
                connection.execute("INSERT INTO proposal_review VALUES ('tile:line-1', 'contour')")
            self.assertEqual(SCRIPT._review_statuses(package), {"tile:line-1": "contour"})

    def test_contour_scope_filters_unreviewed_ridge_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "ridge.geojson"
            collection = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"proposal_id": "line-1", "pixel_length": 80, "confidence": 0.8},
                        "geometry": {"type": "LineString", "coordinates": [[100.5, 201.5], [101.5, 201.5]]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"proposal_id": "line-2", "pixel_length": 80, "confidence": 0.8},
                        "geometry": {"type": "LineString", "coordinates": [[102.5, 200.5], [103.5, 200.5]]},
                    },
                ],
            }
            vector_path.write_text(json.dumps(collection), encoding="utf-8")
            tile = {
                "tile_id": "tile",
                "bounds": [100.0, 200.0, 104.0, 202.0],
                "pixel_bounds": [0, 0, 4, 2],
            }
            lines, anchors, properties = SCRIPT._load_ridge_lines(
                tile,
                vector_path,
                {"tile:line-1": "contour", "tile:line-2": "unreviewed"},
                "contour",
                SCRIPT.ContourCompletionSettings(),
            )
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(anchors), 2)
            self.assertEqual({anchor.proposal_id for anchor in anchors}, {"line-1"})
            self.assertEqual(properties["line-2"]["review_status"], "unreviewed")


if __name__ == "__main__":
    unittest.main()
