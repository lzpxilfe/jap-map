"""Synthetic contracts for the real-image comparison writer, not accuracy tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from histcontour_core.provenance import sha256_file
from scripts import compare_contour_vector_outputs as comparison
from scripts.generate_ink_centerline_candidates import _map_coordinates


class ContourVectorComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raster = self.root / "synthetic.png"
        Image.new("L", (64, 64), 255).save(self.raster)
        self.digest = sha256_file(self.raster)
        self.tiles = [{"tile_id": f"dev-{n}", "sheet_id": f"source-{n//3}", "split": "development", "scene_type": "fixture",
                       "raster_path": str(self.raster), "pixel_bounds": [0, 0, 64, 64], "bounds": [100, 200, 164, 264]}
                      for n in range(9)]
        self.index = {"tiles": self.tiles}
        self.probes = {"schema": "jap-map-contour-region-probes/1", "reference_origin": "synthetic-test-fixture",
                       "human_approved": False, "source_sha256": {"dev-0": self.digest}, "regions": [
                           {"id": f"probe-{kind}", "tile_id": "dev-0", "kind": kind, "pixel_box": [4, 4, 25, 25]}
                           for kind in ("contour", "text", "road_river")]}

    def test_holdout_unknown_duplicate_and_unbound_probes_fail(self):
        self.assertEqual(len(comparison.selected_sources(self.index, self.probes)), 9)
        for field, value in (("tile_id", "178-gongju"), ("id", "../outside"), ("pixel_box", [0, 0, 65, 5])):
            probes = copy.deepcopy(self.probes)
            probes["regions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                comparison.selected_sources(self.index, probes)
        probes = copy.deepcopy(self.probes)
        probes["regions"].append(copy.deepcopy(probes["regions"][0]))
        with self.assertRaises(ValueError):
            comparison.selected_sources(self.index, probes)
        probes = copy.deepcopy(self.probes)
        probes["source_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "frozen raster hash"):
            comparison.selected_sources(self.index, probes)

    def test_map_to_pixel_round_trip_preserves_exported_coordinates(self):
        tile = self.tiles[0]
        points = ((0., 0.), (23., 41.), (63., 63.))
        geometry = _map_coordinates(tile, 64, 64, points)
        collection = {"features": [{"properties": {"proposal_id": "line", "pixel_length": 100},
                                     "geometry": {"coordinates": geometry}}]}
        self.assertEqual(comparison.pixel_proposals(tile, collection)[0].points, points)

    def test_empty_probes_are_unscored_and_existing_outputs_are_preserved(self):
        index_path, probes_path = self.root / "index.json", self.root / "probes.json"
        index_path.write_text(json.dumps(self.index), encoding="utf-8")
        probes_path.write_text(json.dumps(self.probes), encoding="utf-8")
        rows = []
        for tile in self.tiles:
            path = self.root / f"{tile['tile_id']}.geojson"
            path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
            rows.append({"tile_id": tile["tile_id"], "source_raster_sha256": self.digest, "ink_vector_path": str(path)})
        vector_index = self.root / "vectors.json"
        vector_index.write_text(json.dumps({"tiles": rows, "holdout_included": False}), encoding="utf-8")
        output = self.root / "comparison"
        report = comparison.compare(index_path, probes_path, vector_index, vector_index, output)
        self.assertIsNone(report["probe_aggregates"]["contour"]["visible_ink_coverage"]["before_18"])
        self.assertFalse(report["promotion_passed"])
        self.assertIn("n/a", (output / "report.html").read_text())
        with self.assertRaises(FileExistsError):
            comparison.compare(index_path, probes_path, vector_index, vector_index, output)


if __name__ == "__main__":
    unittest.main()
