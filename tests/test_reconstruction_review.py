"""Contracts for the source-first development packet, no accuracy labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import prepare_contour_reconstruction_review as review


class ReconstructionReviewTests(unittest.TestCase):
    def fixture(self):
        manifest = json.loads((review.ROOT/"examples/contour_reconstruction_regions.v1.json").read_text())
        tids = sorted({r["tile_id"] for r in manifest["regions"]})
        tiles = [{"tile_id": tid, "sheet_id": tid.rsplit("-", 1)[0], "split": "development",
                  "bounds": [100, 200, 110, 210], "pixel_bounds": [0, 0, 1024, 1024],
                  "crs_authid": "EPSG:5132", "source_raster_sha256": "0"*64} for tid in tids]
        return manifest, {"schema": "jap-map-assisted-contour-drawing/1", "holdout_used": False, "tiles": tiles}

    def test_frozen_manifest_has_24_six_by_four_with_no_human_truth(self):
        manifest, report = self.fixture()
        self.assertEqual(len(review.validate_regions(manifest, report)), 9)
        self.assertFalse(manifest["human_approved"])
        self.assertFalse(manifest["training_eligible"])

    def test_bad_bounds_duplicate_and_scene_distribution_rejected(self):
        manifest, report = self.fixture()
        for change in ({"box": [0, 0, 1025, 20]}, {"box": [0, 0, 0, 20]},
                       {"box": [0, 0, 1.5, 20]}, {"region_id": "R002"},
                       {"scene_hint": "not_a_label"}, {"tile_id": "178-gongju-hydro"}):
            value = copy.deepcopy(manifest)
            value["regions"][0].update(change)
            with self.assertRaises(ValueError):
                review.validate_regions(value, report)

    def test_holdout_rejected_at_manifest_validation(self):
        manifest, report = self.fixture()
        report["tiles"][0]["split"] = "holdout"
        with self.assertRaises(ValueError):
            review.validate_regions(manifest, report)

    def test_source_panel_has_exact_nearest_pixels(self):
        pixels = np.arange(100, dtype=np.uint8).reshape(10, 10)
        source = Image.fromarray(pixels)
        actual = np.asarray(review.render_panel(source, [2, 3, 7, 8], scale=2))
        expected = np.repeat(np.repeat(pixels[3:8, 2:7], 2, axis=0), 2, axis=1)
        np.testing.assert_array_equal(actual[:, :, 0], expected)
        np.testing.assert_array_equal(np.asarray(source), pixels)

    def test_independent_lines_do_not_get_a_connector(self):
        source = Image.new("RGB", (40, 40), "white")
        panel = review.render_panel(source, [0, 0, 40, 40], lines=[[(4, 4), (8, 4)], [(30, 30), (35, 30)]])
        self.assertEqual(panel.getpixel((40, 35)), (255, 255, 255))

    def test_pixel_centre_half_shift(self):
        source = Image.new("RGB", (20, 20), "white")
        panel = np.asarray(review.render_panel(source, [5, 5, 15, 15], lines=[[(7, 5), (7, 14)]], scale=4))
        dark = 255-panel[:, :, 1].astype(float)
        xmean = np.dot(np.arange(40)+.5, dark.sum(axis=0))/dark.sum()
        self.assertAlmostEqual(xmean, (7-5+.5)*4, delta=.35)

    def test_output_exists_does_not_change_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root/"keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                review.run(root/"missing", root/"missing.json", root/"missing.geojson", root)
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
