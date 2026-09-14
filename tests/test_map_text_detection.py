"""Synthetic evidence/provenance contracts, not OCR accuracy claims."""

import copy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, TiffImagePlugin

from histcontour_core import map_text_detection as detection
from histcontour_core.margin_ocr import digest_file
from histcontour_core.paddle_margin_ocr import local_inference_only, MarginOcrError


def raw_fixture():
    # A slanted quadrilateral: axis-aligned replacement would lose evidence.
    return {"dt_polys": np.array([[[2, 4], [12, 1], [14, 7], [4, 10]]]),
            "dt_scores": np.array([0.83]), "rec_scores": [0.01], "rec_texts": ["  123\n舊  "]}


def tile_fixture():
    return {"tile_id": "173-buyeo-labels", "sheet_id": "173-buyeo", "split": "development",
            "pixel_bounds": [200, 400, 20, 16], "raster_path": "source.tif",
            "bounds": [100., 168., 120., 200.], "crs_authid": "EPSG:5132", "source_raster_sha256": "0"*64}


def report_fixture():
    return {"schema": "jap-map-assisted-contour-drawing/1", "holdout_used": False,
            "human_approvals": 0, "tiles": [tile_fixture()]}


class RawEvidenceTests(unittest.TestCase):
    def test_detection_and_recognition_scores_are_distinct_and_geometry_unchanged(self):
        raw = raw_fixture()
        result = detection.raw_detection(raw, 20, 16)
        self.assertEqual(result["dt_polys"], raw["dt_polys"].tolist())
        self.assertEqual(result["polygons_on_source_pixel_center_grid"][0][0], [1.5, 3.5])
        self.assertEqual(result["dt_scores"], [0.83])
        self.assertNotIn("rec_scores", result)
        reading = detection.raw_recognition([{"rec_text": "  123\n舊  ", "rec_score": .01}], ["r-1"])
        self.assertEqual(reading["readings"][0]["rec_text"], "  123\n舊  ")
        self.assertEqual(reading["readings"][0]["rec_score"], .01)
        self.assertFalse(reading["readings"][0]["elevation_assigned"])

    def test_wrapped_json_and_empty_detection_are_supported(self):
        empty = {"dt_polys": [], "dt_scores": []}
        result = detection.raw_detection(SimpleNamespace(json={"res": empty}), 20, 16)
        self.assertEqual(result["status"], "no_regions_detected")
        self.assertEqual(detection.raw_recognition([], [])["readings"], [])

    def test_no_recognition_confidence_substitution_or_malformed_scores(self):
        bad = [{"dt_polys": [], "rec_scores": []},
               {"dt_polys": [], "dt_scores": [.9]}]
        bad += [{**raw_fixture(), "dt_scores": [v]} for v in (True, float("nan"), float("inf"), -.1, 1.1)]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(detection.MapTextDetectionError):
                detection.raw_detection(value, 20, 16)

    def test_bad_vertices_and_degenerate_geometry_fail_without_silent_clipping(self):
        for polygon in ([[[2, 4], [12, 1], [21, 7], [4, 10]]],
                        [[[2, 4], [12, 1], [14, float("nan")], [4, 10]]],
                        [[[2, 4], [12, 1], [14, True], [4, 10]]],
                        [[[0, 0], [1, 1], [2, 2], [3, 3]]]):
            with self.assertRaises(detection.MapTextDetectionError):
                detection.raw_detection({"dt_polys": polygon, "dt_scores": [.9]}, 20, 16)

    def test_recognition_count_and_text_type_are_verified(self):
        with self.assertRaises(detection.MapTextDetectionError):
            detection.raw_recognition([], ["r-1"])
        with self.assertRaises(detection.MapTextDetectionError):
            detection.raw_recognition([{"rec_text": 123, "rec_score": .8}], ["r-1"])

    def test_confidence_one_never_promotes_box_to_glyph_truth(self):
        result = detection.raw_detection({**raw_fixture(), "dt_scores": [1.]}, 20, 16)
        feature = detection.evidence_features(tile_fixture(), {"geotransform": [100, 1, 0, 200, 0, -2]}, result)[0]
        props = feature["properties"]
        for flag in ("human_approved", "training_eligible", "automatic_promotion", "glyph_pixels_identified",
                     "contour_semantics_assigned", "elevation_assigned", "erase_mask_generated"):
            self.assertFalse(props[flag])
        self.assertEqual(feature["geometry"]["coordinates"][0][0], [102, 192])
        self.assertEqual(feature["geometry"]["coordinates"][0][-1], [102, 192])
        self.assertEqual(props["geometry_role"], "text_search_region_not_glyph_mask")


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet = self.root/"packet"
        self.packet.mkdir()
        self.source = self.packet/"source.tif"
        self.report_path = self.packet/"drawing-report.json"
        self.lock_path = self.root/"model-lock.json"
        self.lock_path.write_text("{}", encoding="utf-8")
        self.report = report_fixture()
        self.write_tiff()
        self.write_report()

    def tearDown(self):
        self.temp.cleanup()

    def write_tiff(self, *, size=(20, 16), epsg=5132, split="development"):
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (1., 2., 0.)
        tags[33922] = (0., 0., 0., 100., 200., 0.)
        tags[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, epsg)
        tags[42112] = f'<GDALMetadata><Item name="split">{split}</Item></GDALMetadata>'
        Image.new("L", size, 200).save(self.source, tiffinfo=tags)
        self.report["tiles"][0]["source_raster_sha256"] = digest_file(self.source)

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")

    def test_actual_tiff_pixel_dimensions_crs_transform_and_hash(self):
        result = detection.inspect_raster(self.source, self.report["tiles"][0])
        self.assertEqual([result["source_width_px"], result["source_height_px"]], [20, 16])
        self.assertEqual(result["actual_crs_authid"], "EPSG:5132")
        self.assertEqual(result["geotransform"], [100, 1, 0, 200, 0, -2])
        self.assertEqual(result["actual_bounds"], [100, 168, 120, 200])
        self.source.write_bytes(self.source.read_bytes()+b"changed")
        with self.assertRaisesRegex(detection.MapTextDetectionError, "hash changed"):
            detection.inspect_raster(self.source, self.report["tiles"][0])

    def test_mismatched_actual_dimensions_crs_and_split_are_rejected(self):
        for kwargs, message in (({"size": (21, 16)}, "dimensions"), ({"epsg": 4326}, "CRS"),
                                ({"split": "holdout"}, "identity/split")):
            with self.subTest(kwargs=kwargs):
                self.write_tiff(**kwargs)
                with self.assertRaisesRegex(detection.MapTextDetectionError, message):
                    detection.inspect_raster(self.source, self.report["tiles"][0])

    def test_holdout_unknown_sheet_duplicate_and_missing_dimensions_rejected(self):
        for changes in ({"sheet_id": "178-gongju", "tile_id": "178-gongju-labels"},
                        {"sheet_id": "fake", "tile_id": "fake-labels"},
                        {"split": "holdout"}, {"pixel_bounds": [0, 0, 1.5, 2]}):
            report = copy.deepcopy(self.report)
            report["tiles"][0].update(changes)
            with self.assertRaises(detection.MapTextDetectionError):
                detection.validate_report(report)
        report = copy.deepcopy(self.report)
        report["tiles"].append(copy.deepcopy(report["tiles"][0]))
        with self.assertRaises(detection.MapTextDetectionError):
            detection.validate_report(report)

    def test_holdout_rejected_before_opening_any_raster(self):
        self.report["tiles"][0].update(sheet_id="178-gongju", tile_id="178-gongju-labels")
        self.write_report()
        with patch.object(detection, "inspect_raster") as inspect:
            with self.assertRaises(detection.MapTextDetectionError):
                detection.run(self.report_path, self.lock_path, self.root/"output")
            inspect.assert_not_called()

    def test_source_paths_must_remain_inside_packet(self):
        for name in ("../source.tif", str(self.source.resolve())):
            self.report["tiles"][0]["raster_path"] = name
            self.write_report()
            with self.assertRaisesRegex(detection.MapTextDetectionError, "raster path"):
                detection.run(self.report_path, self.lock_path, self.root/"output")

    def test_existing_output_or_output_inside_source_is_rejected(self):
        output = self.root/"existing"
        output.mkdir()
        sentinel = output/"keep.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            detection.run(self.report_path, self.lock_path, output)
        self.assertEqual(sentinel.read_text(), "preserve")
        with self.assertRaisesRegex(detection.MapTextDetectionError, "outside the source"):
            detection.run(self.report_path, self.lock_path, self.packet/"new")

    def test_full_fake_run_preserves_inputs_and_separate_raw_outputs(self):
        originals = {p: digest_file(p) for p in (self.source, self.report_path, self.lock_path)}
        engine = Mock()
        engine.predict.return_value = [raw_fixture()]
        rec = Mock()
        rec.predict.return_value = [{"rec_text": "  123\n舊  ", "rec_score": .01}]
        cropper = Mock(return_value=[np.zeros((6, 10, 3), dtype=np.uint8)])
        output = self.root/"output"
        with patch.object(detection, "verify_models", return_value={"detection": str(self.root/"model")}), \
             patch.object(detection, "create_engines", return_value=(engine, rec, cropper)):
            result = detection.run(self.report_path, self.lock_path, output, detection.DetectionConfig(recognize=True))
        self.assertEqual(result["counts"], {"tiles": 1, "detection_regions": 1, "recognition_readings": 1})
        self.assertEqual(result["tiles"][0]["detection"]["dt_scores"], [.83])
        self.assertEqual(result["tiles"][0]["recognition"]["readings"][0]["rec_score"], .01)
        geo = json.loads((output/"text-regions.geojson").read_text())
        self.assertEqual(geo["crs"]["properties"]["name"], "EPSG:5132")
        self.assertTrue(result["provenance"]["source_files_unchanged"])
        self.assertEqual(originals, {p: digest_file(p) for p in originals})
        self.assertFalse(any(p.suffix in (".tif", ".png") for p in output.rglob("*")))
        engine.close.assert_called_once()
        rec.close.assert_called_once()

    def test_source_changed_during_run_prevents_success_output(self):
        engine = Mock()
        def mutate(_pixels):
            self.source.write_bytes(self.source.read_bytes()+b"changed")
            return [raw_fixture()]
        engine.predict.side_effect = mutate
        output = self.root/"output"
        with patch.object(detection, "verify_models", return_value={"detection": str(self.root/"model")}), \
             patch.object(detection, "create_engines", return_value=(engine, None, None)):
            with self.assertRaisesRegex(detection.MapTextDetectionError, "input changed"):
                detection.run(self.report_path, self.lock_path, output)
        self.assertFalse((output/"text-detection.json").exists())

    def test_local_guard_rejects_network_calls(self):
        with local_inference_only(cache_directory=self.root/"cache"):
            with self.assertRaises(MarginOcrError):
                socket.create_connection(("example.com", 443))

    def test_no_arbitrary_device_or_unbounded_config(self):
        for kwargs in ({"limit_side_len": 99999}, {"cpu_threads": 0}, {"thresh": 0},
                       {"box_thresh": float("nan")}, {"unclip_ratio": 10}, {"recognize": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(detection.MapTextDetectionError):
                detection.DetectionConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
