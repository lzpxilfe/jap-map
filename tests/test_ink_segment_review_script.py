import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PREPARE_PATH = REPOSITORY / "scripts" / "prepare_ink_segment_review.py"
TRAIN_PATH = REPOSITORY / "scripts" / "train_ink_segment_classifier.py"


def _load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


PREPARE = _load("prepare_ink_segment_review", PREPARE_PATH)
TRAIN = _load("train_ink_segment_classifier", TRAIN_PATH)


class InkSegmentReviewScriptTest(unittest.TestCase):
    def test_map_to_pixel_inverts_pixel_centre_coordinates(self):
        tile = {"bounds": [100.0, 200.0, 104.0, 202.0], "pixel_bounds": [20, 30, 4, 2]}
        points = PREPARE.map_to_pixel(tile, ((100.5, 201.5), (103.5, 200.5)))
        self.assertEqual(points, [(0.0, 0.0), (3.0, 1.0)])

    def test_holdout_contamination_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "includes holdout"):
                PREPARE.prepare({"tiles": []}, {"holdout_included": True, "tiles": []}, Path(directory), 2, 32)

    def test_unlabelled_queue_writes_honest_readiness_report(self):
        rows = []
        for index in range(6):
            properties = {name: float(index) for name in TRAIN.FEATURE_NAMES}
            properties.update(segment_uid=f"id-{index}", sheet_id="sheet-a", review_status="unreviewed")
            rows.append({"type": "Feature", "properties": properties, "geometry": None})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.geojson"
            path.write_text(json.dumps({"type": "FeatureCollection", "features": rows}), encoding="utf-8")
            report = TRAIN.train_report(TRAIN.read_records(path), minimum_per_class=2)
        self.assertEqual(report["status"], "waiting_for_labels")
        self.assertEqual(report["binary_training_count"], 0)

    def test_training_refuses_holdout_rows(self):
        row = {name: 0.0 for name in TRAIN.FEATURE_NAMES}
        row.update(segment_uid="holdout", tile_id="holdout", sheet_id="sheet-h", split="holdout_test", review_status="contour")
        report = TRAIN.train_report([row], minimum_per_class=2)
        self.assertEqual(report["status"], "blocked_holdout_contamination")

    def test_sheet_cross_validation_never_trains_on_held_out_sheet(self):
        rows = []
        for sheet_index, sheet in enumerate(("sheet-a", "sheet-b", "sheet-c")):
            for index in range(4):
                for status, value in (("contour", 2.0 + sheet_index / 10), ("text", -2.0 - sheet_index / 10)):
                    row = {name: value + index / 100 for name in TRAIN.FEATURE_NAMES}
                    row.update(segment_uid=f"{sheet}-{status}-{index}", sheet_id=sheet, review_status=status)
                    rows.append(row)
        report = TRAIN.train_report(rows, minimum_per_class=2)
        self.assertEqual(report["status"], "trained")
        self.assertEqual({fold["held_out_sheet"] for fold in report["folds"]}, {"sheet-a", "sheet-b", "sheet-c"})


if __name__ == "__main__":
    unittest.main()
