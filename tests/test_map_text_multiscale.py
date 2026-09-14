"""Coordinate/provenance tests with fake inference, not OCR accuracy claims."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, TiffImagePlugin

from histcontour_core.map_text_detection import DetectionConfig, MapTextDetectionError
from histcontour_core.margin_ocr import digest_file
from histcontour_core.paddle_margin_ocr import MarginOcrError
from scripts import detect_map_text_multiscale as multiscale
from scripts.prepare_contour_reconstruction_review import SCENES
from tests.test_map_text_detection import raw_fixture, report_fixture


class CoordinateTest(unittest.TestCase):
    def test_resize_inverse_and_corner_to_center_shift_are_distinct(self):
        polygons = [[[2, 4], [12, 1], [14, 7], [4, 10]]]
        before = copy.deepcopy(polygons)
        result = multiscale.map_coordinates(polygons, [10, 20, 30, 40], 4, [200, 400])
        self.assertEqual(polygons, before)
        self.assertEqual(result["tile_corner_polygons"][0][0], [10.5, 21.])
        self.assertEqual(result["tile_center_grid_polygons"][0][0], [10., 20.5])
        self.assertEqual(result["source_sheet_corner_polygons"][0][0], [210.5, 421.])
        self.assertEqual(result["source_sheet_center_grid_polygons"][0][0], [210., 420.5])
        self.assertEqual(result["detector_corner_to_tile_corner_affine"], [[.25, 0., 10], [0., .25, 20], [0., 0., 1.]])
        self.assertEqual(result["padding_px"], [0, 0, 0, 0])
        self.assertEqual(result["global_rotation_degrees"], 0)


class MultiscaleRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet = self.root/"packet"
        self.packet.mkdir()
        self.source = self.packet/"source.tif"
        self.report_path = self.packet/"drawing-report.json"
        self.manifest_path = self.root/"regions.json"
        self.lock_path = self.root/"model-lock.json"
        self.output = self.root/"output"
        self.report = report_fixture()
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (1., 2., 0.)
        tags[33922] = (0., 0., 0., 100., 200., 0.)
        tags[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 5132)
        tags[42112] = '<GDALMetadata><Item name="split">development</Item></GDALMetadata>'
        Image.new("RGB", (20, 16), (180, 190, 200)).save(self.source, tiffinfo=tags)
        self.report["tiles"][0]["source_raster_sha256"] = digest_file(self.source)
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        self.lock_path.write_text("{}", encoding="utf-8")
        self.manifest = {
            "schema": "jap-map-contour-reconstruction-regions/1",
            "dataset_role": "development_diagnostics_not_independent_evaluation",
            "human_approved": False, "training_eligible": False, "holdout_used": False,
            "drawing_report_sha256": digest_file(self.report_path),
            "regions": [{"region_id": f"R{i+1:03d}", "tile_id": "173-buyeo-labels", "scene_hint": scene, "box": [2, 3, 18, 15]}
                        for i, scene in enumerate(scene for scene in sorted(SCENES) for _ in range(4))],
        }
        self.write_manifest()
        self.detector = Mock()
        self.detector.predict.return_value = [raw_fixture()]
        self.recognizer = Mock()
        self.recognizer.predict.return_value = [{"rec_text": "  505\n舊  ", "rec_score": .01}]
        self.cropper = Mock(return_value=[np.zeros((6, 10, 3), dtype=np.uint8)])

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def run_fake(self, **kwargs):
        with patch.object(multiscale, "verify_models", return_value={"detection": str(self.root/"det-model"), "recognition": str(self.root/"rec-model")}) as verify, \
             patch.object(multiscale, "create_engines", return_value=(self.detector, self.recognizer, self.cropper)):
            result = multiscale.run(self.report_path, self.manifest_path, self.lock_path, self.output, **kwargs)
            self.assertEqual(verify.call_count, 2)
            return result

    def test_three_regions_two_scales_keep_raw_outputs_and_exact_source_mapping(self):
        paths = (self.source, self.report_path, self.manifest_path, self.lock_path)
        before = {path: digest_file(path) for path in paths}
        result = self.run_fake()
        self.assertEqual(result["counts"], {"regions": 3, "passes": 6, "raw_detection_regions": 6, "recognition_readings": 6})
        self.assertEqual([row["region_id"] for row in result["regions"]], ["R004", "R015", "R009"])
        self.assertEqual([call.args[0].shape for call in self.detector.predict.call_args_list], [(24, 32, 3), (48, 64, 3)] * 3)
        first, second = result["regions"][0]["passes"]
        self.assertEqual(first["raw_detector"]["dt_polys"], raw_fixture()["dt_polys"].tolist())
        self.assertEqual(second["raw_detector"]["dt_scores"], [.83])
        self.assertEqual(first["mapped_source_coordinates"]["tile_corner_polygons"][0][0], [3., 5.])
        self.assertEqual(second["mapped_source_coordinates"]["tile_corner_polygons"][0][0], [2.5, 4.])
        self.assertEqual(first["mapped_source_coordinates"]["tile_center_grid_polygons"][0][0], [2.5, 4.5])
        self.assertEqual(result["regions"][0]["source_sheet_crop_origin_px"], [202, 403])
        self.assertEqual(first["recognition"]["readings"][0]["rec_text"], "  505\n舊  ")
        self.assertEqual(first["recognition"]["readings"][0]["rec_score"], .01)
        self.assertFalse(first["recognition"]["readings"][0]["elevation_assigned"])
        self.assertFalse(first["erase_mask_generated"])
        self.assertEqual(before, {path: digest_file(path) for path in paths})
        collection = json.loads((self.output/"text-regions.geojson").read_text())
        self.assertEqual(collection["features"][0]["geometry"]["coordinates"][0][0], [103., 190.])
        self.assertEqual(len({feature["properties"]["region_id"] for feature in collection["features"]}), 6)
        self.assertTrue((self.output/"R004/2x.json").exists())
        self.assertTrue((self.output/"R004/comparison.png").exists())
        self.detector.close.assert_called_once()
        self.recognizer.close.assert_called_once()

    def test_empty_detection_is_preserved_for_every_pass(self):
        self.detector.predict.return_value = [{"dt_polys": [], "dt_scores": []}]
        result = self.run_fake()
        self.assertEqual(result["counts"]["passes"], 6)
        self.assertEqual(result["counts"]["raw_detection_regions"], 0)
        self.assertEqual(result["regions"][0]["passes"][0]["raw_detector"]["status"], "no_regions_detected")
        self.recognizer.predict.assert_not_called()

    def test_holdout_is_rejected_before_any_raster_is_opened(self):
        self.report["tiles"][0].update(sheet_id="178-gongju", tile_id="178-gongju-labels", split="holdout")
        self.report_path.write_text(json.dumps(self.report))
        with patch.object(multiscale, "inspect_raster") as inspect, self.assertRaises(ValueError):
            self.run_fake()
        inspect.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_report_pin_and_fixed_manifest_membership_are_enforced(self):
        self.manifest["drawing_report_sha256"] = "0"*64
        self.write_manifest()
        with self.assertRaisesRegex(MapTextDetectionError, "fixed manifest hash"):
            self.run_fake()
        self.manifest["drawing_report_sha256"] = digest_file(self.report_path)
        self.write_manifest()
        with self.assertRaisesRegex(MapTextDetectionError, "distinct region"):
            self.run_fake(region_ids=("R004", "R999"))
        with self.assertRaisesRegex(MapTextDetectionError, "distinct region"):
            self.run_fake(region_ids=("R004", "R004"))
        with self.assertRaisesRegex(MapTextDetectionError, "scales"):
            self.run_fake(scales=(3,))

    def test_existing_output_and_output_inside_source_are_rejected(self):
        self.output.mkdir()
        sentinel = self.output/"keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_fake()
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaisesRegex(MapTextDetectionError, "outside the source"):
            multiscale.run(self.report_path, self.manifest_path, self.lock_path, self.packet/"new")

    def test_changed_source_or_input_payload_prevents_success_report(self):
        def mutate_file(_pixels):
            self.source.write_bytes(self.source.read_bytes()+b"change")
            return [raw_fixture()]
        self.detector.predict.side_effect = mutate_file
        with self.assertRaisesRegex(MapTextDetectionError, "input changed"):
            self.run_fake()
        self.assertFalse((self.output/"multiscale-text-detection.json").exists())

    def test_model_cannot_mutate_supplied_inference_pixels(self):
        def mutate_pixels(pixels):
            pixels[0, 0] = 0
            return [raw_fixture()]
        self.detector.predict.side_effect = mutate_pixels
        with self.assertRaisesRegex(MapTextDetectionError, "mutated"):
            self.run_fake()
        self.assertFalse((self.output/"multiscale-text-detection.json").exists())

    def test_recognition_cropper_cannot_change_saved_raw_polygons(self):
        def mutate_polygons(_pixels, polygons):
            polygons[0][0][0] = 9
            return [np.zeros((6, 10, 3), dtype=np.uint8)]
        self.cropper.side_effect = mutate_polygons
        result = self.run_fake()
        self.assertEqual(result["regions"][0]["passes"][0]["raw_detector"]["dt_polys"], raw_fixture()["dt_polys"].tolist())

    def test_failed_post_inference_model_verification_prevents_success_report(self):
        with patch.object(multiscale, "verify_models", side_effect=[{"detection": str(self.root/"model")}, MarginOcrError("model changed")]), \
             patch.object(multiscale, "create_engines", return_value=(self.detector, self.recognizer, self.cropper)):
            with self.assertRaisesRegex(MarginOcrError, "model changed"):
                multiscale.run(self.report_path, self.manifest_path, self.lock_path, self.output)
        self.assertFalse((self.output/"multiscale-text-detection.json").exists())
        self.detector.close.assert_called_once()
        self.recognizer.close.assert_called_once()

    def test_inference_stays_under_socket_guard(self):
        def connect(_pixels):
            socket.create_connection(("example.com", 443))
        self.detector.predict.side_effect = connect
        with self.assertRaises(MarginOcrError):
            self.run_fake()
        self.assertFalse((self.output/"multiscale-text-detection.json").exists())


if __name__ == "__main__":
    unittest.main()
