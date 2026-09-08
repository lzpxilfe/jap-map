import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from histcontour_core.segment_review import FEATURE_NAMES, train_logistic_baseline


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY / "scripts" / "score_ink_segments.py"
SPEC = importlib.util.spec_from_file_location("score_ink_segments", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)


def _record(value, status):
    return {**{name: float(value) for name in FEATURE_NAMES}, "review_status": status}


class ScoreInkSegmentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            import PIL
        except ImportError:
            raise unittest.SkipTest("score export needs NumPy and Pillow")

    def test_model_schema_is_checked_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waiting.json"
            path.write_text(json.dumps({"status": "waiting_for_labels"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not trained"):
                SCRIPT.load_model(path)

    def test_scored_feature_separates_ink_support_and_contour_score(self):
        import numpy as np
        from PIL import Image
        model = train_logistic_baseline([_record(2, "contour"), _record(3, "contour"), _record(-2, "text"), _record(-3, "text")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((64, 64), 242, dtype=np.uint8)
            source[30:34, 8:56] = 20
            raster = root / "source.png"
            Image.fromarray(source).save(raster)
            tile = {"tile_id": "tile", "raster_path": str(raster), "bounds": [0.0, 0.0, 64.0, 64.0], "pixel_bounds": [0, 0, 64, 64]}
            vector = root / "ink.geojson"
            vector.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"ink_support": 0.9}, "geometry": {"type": "LineString", "coordinates": [[8.5, 32.5], [55.5, 32.5]]}}]}), encoding="utf-8")
            collection = SCRIPT.score_tile(tile, vector, ("logistic", model))
        properties = collection["features"][0]["properties"]
        self.assertEqual(properties["ink_support"], 0.9)
        self.assertIn("contour_score", properties)
        self.assertTrue(properties["contour_score_review_only"])


if __name__ == "__main__":
    unittest.main()
