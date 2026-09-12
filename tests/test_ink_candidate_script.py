import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY / "scripts" / "generate_ink_centerline_candidates.py"
SPEC = importlib.util.spec_from_file_location("generate_ink_centerline_candidates", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)


class InkCandidateScriptTest(unittest.TestCase):
    def test_holdout_is_excluded_unless_explicit(self):
        index = {
            "tiles": [
                {"tile_id": "dev", "split": "development"},
                {"tile_id": "holdout", "split": "holdout_test"},
            ]
        }
        self.assertEqual([tile["tile_id"] for tile in SCRIPT.select_tiles(index)], ["dev"])
        self.assertEqual(
            [tile["tile_id"] for tile in SCRIPT.select_tiles(index, include_holdout=True)],
            ["dev", "holdout"],
        )

    def test_pixel_centres_map_inside_declared_bounds(self):
        tile = {"tile_id": "tile", "bounds": [100.0, 200.0, 104.0, 202.0]}
        coordinates = SCRIPT._map_coordinates(tile, 4, 2, ((0, 0), (3, 1)))
        self.assertEqual(coordinates, [[100.5, 201.5], [103.5, 200.5]])

    def test_preview_mask_contains_only_exported_geometry(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy dependency is not installed")
        from histcontour_core.vectorization import PixelLineProposal
        mask = SCRIPT.proposal_mask([PixelLineProposal("only-this-line", ((2., 5.), (12., 5.)), 10., 1.)], (20, 20))
        self.assertEqual(int(mask.sum()), 11)
        self.assertTrue(mask[5, 2:13].all())
        self.assertFalse(mask[15].any())

    def test_process_tile_writes_separate_provenanced_outputs(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("optional Ink generation dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            output_dir = root / "outputs"
            output_dir.mkdir()
            image = np.full((96, 96), 245, dtype=np.uint8)
            image[12:84, 44:51] = 20
            Image.fromarray(image).save(source_path)
            tile = {
                "tile_id": "test-tile",
                "sheet_id": "test-sheet",
                "split": "development",
                "scene_type": "dense_mountain",
                "pixel_bounds": [128, 256, 96, 96],
                "raster_path": str(source_path),
                "bounds": [126.0, 36.0, 127.0, 37.0],
                "crs_authid": "EPSG:5132",
            }
            result = SCRIPT.process_tile((tile, str(output_dir), 18.0, 0.75))
            vector_path = Path(result["ink_vector_path"])
            preview_path = Path(result["ink_preview_path"])
            self.assertTrue(vector_path.is_file())
            self.assertTrue(preview_path.is_file())
            collection = json.loads(vector_path.read_text(encoding="utf-8"))
            self.assertGreater(len(collection["features"]), 0)
            properties = collection["features"][0]["properties"]
            self.assertEqual(properties["backend"], SCRIPT.INK_BACKEND_ID)
            self.assertEqual(properties["upstream_commit"], SCRIPT.ARCHAEOTRACE_UPSTREAM_COMMIT)
            self.assertEqual(properties["review_status"], "unreviewed")


if __name__ == "__main__":
    unittest.main()
