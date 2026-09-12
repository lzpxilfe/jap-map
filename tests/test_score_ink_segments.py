import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_threshold_subset_preserves_original_geometry_and_review_state(self):
        features = [{"properties": {"contour_score": score, "review_status": "unreviewed"},
                     "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 0]]}}
                    for score in (.05, .1, .9)]
        collection = {"type": "FeatureCollection", "features": features}
        selected = SCRIPT.filter_collection(collection, .1)
        self.assertEqual(len(selected["features"]), 2)
        self.assertEqual(len(collection["features"]), 3)
        self.assertIs(selected["features"][0], features[1])
        self.assertTrue(selected["selection"]["review_only"])
        self.assertFalse(selected["selection"]["human_approval"])
        for invalid in (True, -.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                SCRIPT.filter_collection(collection, invalid)
        features[0]["properties"]["contour_score"] = float("nan")
        with self.assertRaises(ValueError):
            SCRIPT.filter_collection(collection, .1)

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
            calls = []
            def probability(image, points):
                calls.append((image.shape, len(points)))
                return .75
            onnx_collection = SCRIPT.score_tile(tile, vector, ("onnx", SimpleNamespace(probability=probability)))
        properties = collection["features"][0]["properties"]
        self.assertEqual(properties["ink_support"], 0.9)
        self.assertIn("contour_score", properties)
        self.assertTrue(properties["contour_score_review_only"])
        self.assertEqual(calls, [((64, 64), 2)])
        self.assertEqual(onnx_collection["features"][0]["properties"]["contour_score"], .75)
        self.assertIn("onnx", onnx_collection["features"][0]["properties"]["contour_score_kind"])


if __name__ == "__main__":
    unittest.main()
