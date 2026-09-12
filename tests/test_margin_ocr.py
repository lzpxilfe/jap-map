"""Contract tests with synthetic rasters and a fake engine, NOT OCR accuracy."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from histcontour_core.margin_ocr import (
    MANIFEST_SCHEMA, RESULT_SCHEMA, MarginOcrError, digest_file, evaluate,
    reading_from_paddle, review_template, validate_manifest, validate_reviews,
)
from histcontour_core import paddle_margin_ocr as adapter

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("margin_pilot", ROOT / "scripts/margin_ocr_pilot.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)
VERSIONS = {"paddleocr": "3.7.0", "paddlex": "test-fixture", "paddlepaddle": "test-fixture"}


def paddle_read(text="N 36°30′00″", score=0.99):
    return {"rec_texts": [text], "rec_scores": [score], "rec_polys": [[[0, 0], [10, 0], [10, 8], [0, 8]]]}


def result_fixture(count=1):
    return {"schema": RESULT_SCHEMA, "pilot_id": "synthetic-contract-test", "corpus_kind": "historical_scan", "crops": [
        {"crop_uid": f"sheet-{index}/nw", "sheet_id": f"sheet-{index}", "region_id": "nw",
         "kind": "corner_nw", "scenario": ("clear", "old_type", "degraded")[index % 3],
         **reading_from_paddle(paddle_read(), 20, 10)} for index in range(count)
    ]}


class MarginReadingTests(unittest.TestCase):
    def test_paddle_wrapper_preserves_original_text_scores_and_polygons(self):
        reading = reading_from_paddle(SimpleNamespace(json={"res": paddle_read("  +10.4″\n舊字  ")}), 20, 10)
        self.assertEqual(reading["raw_text"], "  +10.4″\n舊字  ")
        self.assertEqual(reading["spans"][0]["confidence"], 0.99)
        self.assertEqual(reading["spans"][0]["polygon"][2], [10.0, 8.0])

    def test_empty_detection_is_distinct_from_a_read(self):
        reading = reading_from_paddle({"res": {"rec_texts": [], "rec_scores": [], "rec_polys": []}}, 20, 10)
        self.assertEqual(reading["status"], "no_text")
        self.assertEqual(reading["raw_text"], "")

    def test_malformed_or_out_of_bounds_engine_result_is_rejected(self):
        bad = [paddle_read(score=value) for value in (True, float("nan"), -0.1, 1.1)]
        bad += [{}, {**paddle_read(), "rec_scores": []}, {**paddle_read(), "rec_polys": [[[30, 0], [0, 0], [0, 1]]]}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(MarginOcrError):
                reading_from_paddle(value, 20, 10)

    def test_confidence_one_never_creates_an_approval(self):
        result = result_fixture()
        result["crops"][0].update(reading_from_paddle(paddle_read(score=1.0), 20, 10))
        item = review_template(result)["items"][0]
        self.assertEqual(item["review_status"], "unreviewed")
        self.assertIsNone(item["reviewed_text"])
        self.assertFalse(item["reference_independent"])

    def test_review_is_bound_to_exact_result(self):
        result = result_fixture()
        reviews = review_template(result)
        result["crops"][0]["raw_text"] += "x"
        with self.assertRaisesRegex(MarginOcrError, "different OCR result"):
            validate_reviews(result, reviews)

    def test_changed_text_cannot_be_approved_as_original(self):
        result = result_fixture()
        reviews = review_template(result)
        reviews["items"][0].update(review_status="approved", reviewer="researcher", reviewed_text="different")
        with self.assertRaisesRegex(MarginOcrError, "corrected"):
            validate_reviews(result, reviews)
        reviews["items"][0]["review_status"] = "corrected"
        validate_reviews(result, reviews)

    def test_unknown_duplicate_reviews_and_duplicate_results_fail(self):
        result = result_fixture()
        reviews = review_template(result)
        reviews["items"].append(copy.deepcopy(reviews["items"][0]))
        with self.assertRaises(MarginOcrError):
            validate_reviews(result, reviews)
        reviews["items"] = [{**reviews["items"][0], "crop_uid": "unknown/nw"}]
        with self.assertRaises(MarginOcrError):
            validate_reviews(result, reviews)
        result["crops"].append(copy.deepcopy(result["crops"][0]))
        with self.assertRaises(MarginOcrError):
            review_template(result)

    def test_reference_needs_explicit_independence_and_reviewer(self):
        result = result_fixture()
        reviews = review_template(result)
        reviews["items"][0]["reference_text"] = result["crops"][0]["raw_text"]
        self.assertEqual(evaluate(result, reviews)["overall"]["count"], 0)
        reviews["items"][0]["reference_independent"] = True
        with self.assertRaisesRegex(MarginOcrError, "reviewer"):
            evaluate(result, reviews)

    def test_scoring_uses_original_not_corrected_text_and_preserves_dms_symbols(self):
        result = result_fixture()
        result["crops"][0]["raw_text"] = "N 36°30′00′"
        reviews = review_template(result)
        reviews["items"][0].update(review_status="corrected", reviewer="researcher", reviewed_text="N 36°30′00″",
                                   reference_text="N 36°30′00″", reference_independent=True, review_seconds=12.5)
        report = evaluate(result, reviews)
        self.assertEqual(report["overall"]["raw_exact_match"], 0)
        self.assertEqual(report["overall"]["normalized_exact_match"], 0)
        self.assertGreater(report["overall"]["character_error_rate"], 0)
        self.assertEqual(report["review_time"]["total_seconds"], 12.5)

    def test_secondary_metric_normalizes_only_nfc_and_whitespace(self):
        result = result_fixture()
        result["crops"][0]["raw_text"] = "e\u0301   36°\n30′"
        reviews = review_template(result)
        reviews["items"][0].update(reviewer="researcher", reference_text="é 36° 30′", reference_independent=True)
        metrics = evaluate(result, reviews)["overall"]
        self.assertEqual(metrics["raw_exact_match"], 0)
        self.assertEqual(metrics["normalized_exact_match"], 1)

    def test_runtime_error_is_not_a_correct_empty_read(self):
        result = result_fixture(2)
        for row, status in zip(result["crops"], ("error", "no_text")):
            row.update(status=status, raw_text="", spans=[])
        reviews = review_template(result)
        for item in reviews["items"]:
            item.update(reviewer="researcher", reference_text="", reference_independent=True)
        report = evaluate(result, reviews)
        self.assertEqual(report["overall"]["raw_exact_match"], 0.5)
        self.assertEqual(report["runtime_errors"], 1)
        reviews["items"][0].update(review_status="approved", reviewed_text="")
        with self.assertRaisesRegex(MarginOcrError, "runtime error"):
            evaluate(result, reviews)

    def test_20_sheets_and_scenario_coverage_are_required_but_never_promote(self):
        result = result_fixture(20)
        reviews = review_template(result)
        for row, item in zip(result["crops"], reviews["items"]):
            item.update(reviewer="researcher", reference_text=row["raw_text"], reference_independent=True)
        report = evaluate(result, reviews)
        self.assertEqual(report["status"], "pilot_measured")
        self.assertFalse(report["promotion_passed"])
        reviews["items"][0]["reference_independent"] = False
        self.assertEqual(evaluate(result, reviews)["status"], "pilot_incomplete")
        for row in result["crops"]:
            row["scenario"] = "clear"
        reviews = review_template(result)
        for row, item in zip(result["crops"], reviews["items"]):
            item.update(reviewer="researcher", reference_text=row["raw_text"], reference_independent=True)
        self.assertEqual(evaluate(result, reviews)["status"], "pilot_incomplete")

    def test_unreadable_reference_is_counted_not_silently_scored(self):
        result = result_fixture()
        reviews = review_template(result)
        reviews["items"][0].update(review_status="unreadable", reviewer="researcher")
        report = evaluate(result, reviews)
        self.assertEqual(report["unreadable_reference_count"], 1)
        self.assertEqual(report["overall"]["count"], 0)
        self.assertEqual(report["status"], "pilot_incomplete")


class MarginPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic.png"
        with Image.new("RGB", (100, 80), "white") as scan:
            scan.putpixel((1, 2), (25, 50, 75))
            scan.save(self.source)
        self.manifest = {"schema": MANIFEST_SCHEMA, "pilot_id": "synthetic-contract-test", "corpus_kind": "synthetic_smoke", "sheets": [
            {"sheet_id": "sheet-a", "split": "development", "scenario": "old_type", "image_path": "synthetic.png",
             "scan_source": "test-created raster, not a historical scan", "rights": "test fixture",
             "regions": [{"region_id": "nw", "kind": "corner_nw", "pixel_box": [0, 0, 20, 10]},
                         {"region_id": "title", "kind": "title", "pixel_box": [20, 0, 40, 10]}]}
        ]}
        self.manifest_path = self.root / "manifest.json"
        PILOT.write_json(self.manifest_path, self.manifest)
        self.bundle_dir = self.root / "prepared"
        self.bundle = PILOT.prepare(self.manifest_path, self.bundle_dir)

    def make_lock(self):
        spec = {"models": {}}
        for role in ("detection", "recognition"):
            directory = self.root / role
            directory.mkdir()
            (directory / "inference.pdiparams").write_bytes(b"TEST FIXTURE NOT REAL WEIGHTS")
            spec["models"][role] = {"directory": str(directory), "model_name": f"test-{role}",
                                    "source_url": "https://example.invalid/test-fixture", "source_revision": "fixture-v1",
                                    "weight_license": "test fixture only"}
        with patch.object(adapter.metadata, "version", side_effect=VERSIONS.__getitem__):
            lock = adapter.pin_local_models(spec, self.root)
        lock_path = self.root / "model-lock.json"
        PILOT.write_json(lock_path, lock)
        return lock, lock_path

    def test_prepare_preserves_crop_pixels_source_and_bundle_identity(self):
        original_hash = digest_file(self.source)
        rows = PILOT.verify_bundle(self.bundle, self.bundle_dir)
        self.assertEqual(len(rows), 2)
        with Image.open(self.bundle_dir / rows[0]["crop_path"]) as crop:
            self.assertEqual(crop.size, (20, 10))
            self.assertEqual(crop.getpixel((1, 2)), (25, 50, 75))
        self.assertEqual(digest_file(self.source), original_hash)
        self.assertIn("OCR를 보기 전", (self.bundle_dir / "crops.html").read_text())

    def test_existing_outputs_are_never_overwritten(self):
        before = digest_file(self.bundle_dir / "bundle.json")
        with self.assertRaises(FileExistsError):
            PILOT.prepare(self.manifest_path, self.bundle_dir)
        with self.assertRaises(FileExistsError):
            PILOT.write_json(self.manifest_path, {})
        self.assertEqual(digest_file(self.bundle_dir / "bundle.json"), before)

    def test_whole_map_oversize_and_outside_boxes_are_refused(self):
        for box in ([0, 0, 100, 80], [99, 0, 101, 10], [0, 0, 100, 21], [-1, 0, 1, 2], [0, 0, True, 2]):
            with self.subTest(box=box), self.assertRaises(MarginOcrError):
                PILOT._bounds(box, 100, 80)
        with self.assertRaises(MarginOcrError):
            PILOT._bounds([0, 0, 3000, 2000], 10000, 10000)

    def test_holdout_and_path_like_identifiers_are_refused(self):
        for field, value in (("split", "holdout_test"), ("sheet_id", "178-gongju"), ("sheet_id", "../../outside")):
            manifest = copy.deepcopy(self.manifest)
            manifest["sheets"][0][field] = value
            with self.subTest(field=field), self.assertRaises(MarginOcrError):
                validate_manifest(manifest)

    def test_same_scan_cannot_inflate_the_sheet_count(self):
        manifest = copy.deepcopy(self.manifest)
        duplicate = copy.deepcopy(manifest["sheets"][0])
        duplicate["sheet_id"] = "different-name-same-scan"
        manifest["sheets"].append(duplicate)
        manifest_path = self.root / "duplicate-manifest.json"
        PILOT.write_json(manifest_path, manifest)
        output = self.root / "duplicate-prepared"
        with self.assertRaisesRegex(MarginOcrError, "same source scan"):
            PILOT.prepare(manifest_path, output)
        self.assertFalse(output.exists())

    def test_cli_returns_failure_when_crop_inference_errors(self):
        failed = {"status": "completed_with_errors", "crops": [{}], "runtime_errors": 1}
        with patch.object(PILOT, "run", return_value=failed), patch("builtins.print"):
            status = PILOT.main(["run", "bundle.json", "--model-lock", "model-lock.json", "--output", "unused"])
        self.assertEqual(status, 2)

    def test_changed_original_is_refused_before_engine_creation(self):
        _, lock_path = self.make_lock()
        self.source.write_bytes(b"source altered")
        with patch.object(PILOT, "create_engine") as create, self.assertRaisesRegex(MarginOcrError, "source scan changed"):
            PILOT.run(self.bundle_dir / "bundle.json", lock_path, self.root / "run")
        create.assert_not_called()

    def test_changed_crop_and_path_escape_are_refused(self):
        crop = self.bundle_dir / self.bundle["crops"][0]["crop_path"]
        crop.write_bytes(b"changed crop")
        with self.assertRaisesRegex(MarginOcrError, "crop changed"):
            PILOT.verify_bundle(self.bundle, self.bundle_dir)
        self.bundle["crops"][0]["crop_path"] = "../../synthetic.png"
        with self.assertRaisesRegex(MarginOcrError, "crop path"):
            PILOT.verify_bundle(self.bundle, self.bundle_dir)

    def test_model_tampering_or_additional_files_fail_inventory_check(self):
        lock, _ = self.make_lock()
        adapter.verify_models(lock, self.root, check_runtime=False)
        weights = self.root / "detection/inference.pdiparams"
        original = weights.read_bytes()
        weights.write_bytes(b"tampered")
        with self.assertRaisesRegex(MarginOcrError, "differ from source-lock"):
            adapter.verify_models(lock, self.root, check_runtime=False)
        weights.write_bytes(original)
        (weights.parent / "extra.json").write_text("{}")
        with self.assertRaises(MarginOcrError):
            adapter.verify_models(lock, self.root, check_runtime=False)

    def test_model_symlink_unknown_provenance_and_version_mismatch_fail(self):
        lock, _ = self.make_lock()
        with patch.object(adapter.metadata, "version", return_value="different-version"), self.assertRaises(MarginOcrError):
            adapter.verify_models(lock, self.root)
        changed = copy.deepcopy(lock)
        changed["models"]["detection"]["weight_license"] = "UNVERIFIED"
        with self.assertRaises(MarginOcrError):
            adapter.verify_models(changed, self.root, check_runtime=False)
        (self.root / "detection/link").symlink_to(self.source)
        with self.assertRaisesRegex(MarginOcrError, "symlink"):
            adapter.verify_models(lock, self.root, check_runtime=False)

    def test_engine_receives_only_local_models_with_optional_preprocessors_off(self):
        lock, _ = self.make_lock()
        constructor = Mock()
        with patch.object(adapter.metadata, "version", side_effect=VERSIONS.__getitem__), patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=constructor)}):
            adapter.create_engine(lock, self.root)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["text_rec_score_thresh"], 0)
        for flag in ("use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation"):
            self.assertIs(kwargs[flag], False)
        self.assertEqual(kwargs["text_detection_model_dir"], str((self.root / "detection").resolve()))

    def test_detector_override_is_bounded_and_passed_to_engine(self):
        self.assertEqual(adapter.detection_options(), {})
        for value in (True, 127, 4097, 640.0, "640"):
            with self.subTest(value=value), self.assertRaises(MarginOcrError):
                adapter.detection_options(value)
        lock, _ = self.make_lock()
        constructor = Mock()
        with patch.object(adapter.metadata, "version", side_effect=VERSIONS.__getitem__), patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=constructor)}):
            adapter.create_engine(lock, self.root, det_max_side=640)
        self.assertEqual(constructor.call_args.kwargs["text_det_limit_type"], "max")
        self.assertEqual(constructor.call_args.kwargs["text_det_limit_side_len"], 640)

    def test_predeclared_fingerprints_are_checked_at_pin_and_verify(self):
        lock, _ = self.make_lock()
        models = copy.deepcopy(lock["models"])
        for entry in models.values():
            entry["expected_files"] = {name: item["sha256"] for name, item in entry["files"].items()}
        with patch.object(adapter.metadata, "version", side_effect=VERSIONS.__getitem__):
            pinned = adapter.pin_local_models({"models": models}, self.root)
            models["detection"]["expected_files"]["inference.pdiparams"] = "0" * 64
            with self.assertRaisesRegex(MarginOcrError, "predeclared"):
                adapter.pin_local_models({"models": models}, self.root)
        pinned["models"]["recognition"]["expected_files"] = {}
        with self.assertRaisesRegex(MarginOcrError, "predeclared"):
            adapter.verify_models(pinned, self.root, check_runtime=False)

    def test_source_pixel_limit_is_finite_and_pillow_limit_is_restored(self):
        previous = Image.MAX_IMAGE_PIXELS
        with PILOT.open_scan(self.source) as image:
            self.assertEqual(image.size, (100, 80))
            self.assertEqual(Image.MAX_IMAGE_PIXELS, previous)
        with patch.object(Image, "open", side_effect=OSError("fixture metadata failure")), self.assertRaises(OSError):
            with PILOT.open_scan(self.source):
                pass
        self.assertEqual(Image.MAX_IMAGE_PIXELS, previous)
        scan = Mock(width=25001, height=10000)
        scan.__enter__ = Mock(return_value=scan)
        scan.__exit__ = Mock(return_value=False)
        with patch.object(Image, "open", return_value=scan), self.assertRaisesRegex(MarginOcrError, "250-million"):
            with PILOT.open_scan(self.source):
                pass
        self.assertEqual(Image.MAX_IMAGE_PIXELS, previous)

    def test_python_network_guard_is_scoped_and_restored(self):
        original = socket.create_connection
        previous = {key: os.environ.get(key) for key in ("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "PADDLE_PDX_CACHE_HOME")}
        with adapter.local_inference_only(cache_directory=self.root / "cache"):
            self.assertEqual(os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"], "True")
            self.assertEqual(os.environ["PADDLE_PDX_CACHE_HOME"], str((self.root / "cache").resolve()))
            with self.assertRaisesRegex(MarginOcrError, "network connection"):
                socket.create_connection(("example.invalid", 443))
            with socket.socket() as connection, self.assertRaises(MarginOcrError):
                connection.connect(("127.0.0.1", 80))
        self.assertIs(socket.create_connection, original)
        self.assertEqual({key: os.environ.get(key) for key in previous}, previous)

    def test_fake_engine_end_to_end_is_crop_only_and_stays_unreviewed(self):
        _, lock_path = self.make_lock()
        sentinel = self.root / "map.registration.json"
        sentinel.write_text('{"crs":"untouched","gcps":[]}')
        sentinel_hash = digest_file(sentinel)
        inputs = []
        output = self.root / "run"

        def predict(*, input):
            crop = Path(input)
            self.assertTrue(crop.is_relative_to(output / "crops"))
            with Image.open(crop) as image:
                self.assertEqual(image.size, (20, 10))
            inputs.append(crop)
            return [paddle_read("<script>unsafe()</script>", score=1.0)]

        engine = SimpleNamespace(predict=predict, close=Mock())
        with patch.object(adapter.metadata, "version", side_effect=VERSIONS.__getitem__), patch.object(PILOT, "create_engine", return_value=engine):
            result = PILOT.run(self.bundle_dir / "bundle.json", lock_path, output, det_max_side=640)
        self.assertEqual(len(inputs), 2)
        self.assertEqual(result["runtime_errors"], 0)
        self.assertFalse(result["gis_applied"])
        self.assertEqual(result["engine"]["detector_overrides"], adapter.detection_options(640))
        self.assertEqual(digest_file(sentinel), sentinel_hash)
        reviews = PILOT.read_json(output / "review.template.json")
        self.assertTrue(all(item["review_status"] == "unreviewed" for item in reviews["items"]))
        self.assertEqual(evaluate(result, reviews)["status"], "synthetic_smoke_only")
        report = (output / "report.html").read_text()
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)
        engine.close.assert_called_once()

    def test_fake_engine_failure_is_recorded_not_silently_empty(self):
        _, lock_path = self.make_lock()
        engine = SimpleNamespace(predict=Mock(side_effect=[RuntimeError("fixture failure"), [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]]))
        with patch.object(adapter.metadata, "version", side_effect=VERSIONS.__getitem__), patch.object(PILOT, "create_engine", return_value=engine):
            result = PILOT.run(self.bundle_dir / "bundle.json", lock_path, self.root / "run")
        self.assertEqual(result["status"], "completed_with_errors")
        self.assertEqual([row["status"] for row in result["crops"]], ["error", "no_text"])
        self.assertIn("fixture failure", result["crops"][0]["error"])

    def test_optional_import_does_not_load_paddle(self):
        completed = subprocess.run([sys.executable, "-c", "import sys; import histcontour_core.paddle_margin_ocr; assert 'paddleocr' not in sys.modules; assert 'paddle' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
