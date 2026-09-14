"""Direct-recognition coordinate and provenance contracts; no accuracy claims."""

import copy
import json
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from histcontour_core.map_text_detection import MapTextDetectionError
from histcontour_core.margin_ocr import digest_file
from histcontour_core.paddle_margin_ocr import MarginOcrError
from scripts import recognize_component_text_candidates as recognition
from tests import test_map_text_multiscale as fixtures


def candidate_fixture(shape=(12, 16), origin=(2, 3)):
    groups = []
    for rank, offset in ((3, 6), (1, 0), (2, 1)):
        quad = [[2.+offset, 2.], [10.+offset, 1.], [11.+offset, 6.], [3.+offset, 7.]]
        source = [[x+origin[0], y+origin[1]] for x, y in quad]
        groups.append({"group_id": f"fixture-{rank}", "rank": rank, "group_component_ids": [1, 2],
                       "quad_crop_pixel_centers": quad, "quad_source_pixel_centers": source,
                       "quad_source_image_corners": [[x+.5, y+.5] for x, y in source],
                       "ranking_score_not_probability": 1.-rank*.1, "ambiguous": True,
                       "human_approved": False, "training_eligible": False,
                       "evidence": {"quad_within_crop_bounds": rank != 3, "height_ratio": 1.1}})
    labels = np.zeros(shape, dtype=np.int32)
    labels[2, 2], labels[3, 4], labels[3, 5] = 1, 2, 3
    return SimpleNamespace(source_ink=labels > 0, component_labels=labels,
                           components=tuple({"component_id": cid, "fixture": True} for cid in (1, 2, 3)), groups=tuple(groups),
                           provenance={"counts": {"group_candidates_before_output_cap": 4, "returned_groups": 3},
                                       "omitted_groups": [{"group_id": "fixture-4", "rank": 4, "group_component_ids": [2, 3],
                                                           "ranking_score_not_probability": .5, "reason": "output_group_cap"}],
                                       "truncation_flags": {"output_group_cap": True}})


class CoordinateAndEngineTest(unittest.TestCase):
    def test_half_pixel_shift_precedes_enlargement_and_source_mapping_round_trips(self):
        group = candidate_fixture().groups[1]
        self.assertEqual(recognition.group_input_quad(group, 2)[0], [5., 5.])
        self.assertEqual(recognition.group_input_quad(group, 4)[0], [10., 10.])
        mapped, within = recognition.validate_group(group, [2, 3, 18, 15], [200, 400])
        self.assertTrue(within)
        self.assertEqual(mapped["tile_corner_polygons"][0][0], [4.5, 5.5])
        self.assertEqual(mapped["tile_center_grid_polygons"][0][0], [4., 5.])
        self.assertEqual(mapped["source_sheet_corner_polygons"][0][0], [204.5, 405.5])
        self.assertEqual(mapped["source_sheet_center_grid_polygons"][0][0], [204., 405.])

    def test_inconsistent_source_transform_is_rejected_without_clipping(self):
        group = candidate_fixture().groups[1]
        group["quad_source_pixel_centers"][0][0] += .5
        with self.assertRaisesRegex(MapTextDetectionError, "transforms disagree"):
            recognition.validate_group(group, [2, 3, 18, 15], [200, 400])
        outside = candidate_fixture().groups[0]
        before = copy.deepcopy(outside)
        _mapped, within = recognition.validate_group(outside, [2, 3, 18, 15], [200, 400])
        self.assertFalse(within)
        self.assertEqual(outside, before)

    def test_recognizer_factory_never_constructs_a_text_detector(self):
        detector, recognizer, cropper = Mock(), Mock(), Mock()
        lock = {"models": {"recognition": {"model_name": "pinned-recognizer"}}}
        with patch.object(recognition, "verify_models", return_value={"recognition": "/local/recognizer"}), \
             patch.dict(sys.modules, {"paddleocr": SimpleNamespace(TextRecognition=recognizer, TextDetection=detector),
                                      "paddlex.inference.pipelines.components": SimpleNamespace(CropByPolys=cropper)}):
            recognition.create_recognizer(lock, "/local", 1)
        detector.assert_not_called()
        self.assertEqual(recognizer.call_args.kwargs["model_dir"], "/local/recognizer")
        self.assertEqual(recognizer.call_args.kwargs["device"], "cpu")
        cropper.assert_called_once_with(det_box_type="quad")


class ComponentRecognitionTest(unittest.TestCase):
    def setUp(self):
        # Reuse the synthetic GeoTIFF/fixed 24-region manifest fixture only.
        fixtures.MultiscaleRunTest.setUp(self)
        self.recognizer.predict.side_effect = lambda crops: [
            {"rec_text": "" if index == 0 else " 100\n", "rec_score": 0. if index == 0 else .99}
            for index in range(len(crops))]
        self.generator = Mock(side_effect=lambda gray, origin: (candidate_fixture(gray.shape, origin), {"synthetic": True}))

    def write_manifest(self):
        fixtures.MultiscaleRunTest.write_manifest(self)

    def tearDown(self):
        fixtures.MultiscaleRunTest.tearDown(self)

    def run_fake(self, **kwargs):
        with patch.object(recognition, "verify_models", return_value={"recognition": str(self.root/"rec-model")}) as verify, \
             patch.object(recognition, "create_recognizer", return_value=(self.recognizer, self.cropper)), \
             patch.object(recognition, "create_candidates", self.generator):
            result = recognition.run(self.report_path, self.manifest_path, self.lock_path, self.output, **kwargs)
        self.assertEqual(verify.call_count, 2)
        return result

    def test_all_ranks_boundary_skips_and_raw_readings_are_preserved(self):
        paths = (self.source, self.report_path, self.manifest_path, self.lock_path)
        before = {path: digest_file(path) for path in paths}
        result = self.run_fake()
        self.assertEqual(result["counts"], {"regions": 3, "retained_groups": 9, "recognition_attempts": 12, "recognition_skips": 6})
        self.assertEqual([row["region_id"] for row in result["regions"]], ["R004", "R015", "R009"])
        self.assertEqual([group["selection_rank"] for group in result["regions"][0]["groups"]], [1, 2, 3])
        first, second, outside = result["regions"][0]["groups"]
        self.assertEqual([p["scale"] for p in first["passes"]], [2, 4])
        self.assertEqual(first["passes"][0]["recognition"]["rec_text"], "")
        self.assertEqual(first["passes"][0]["recognition"]["rec_score"], 0.)
        self.assertEqual(second["passes"][0]["recognition"]["rec_text"], " 100\n")
        self.assertFalse(second["passes"][0]["recognition"]["elevation_assigned"])
        self.assertEqual(outside["selection_reason"], "quad_outside_crop_no_clipping")
        self.assertTrue(all(p["status"] == "not_run" for p in outside["passes"]))
        self.assertEqual(result["regions"][0]["component_generation_provenance"]["omitted_groups"][0]["rank"], 4)
        self.assertFalse(result["provenance"]["text_detector_used"])
        self.assertNotIn("dt_scores", json.dumps(result))
        self.assertNotIn("detection_score", json.dumps(result))
        self.assertEqual(before, {path: digest_file(path) for path in paths})
        self.assertTrue((self.output/first["passes"][0]["recognition_crop_image"]).exists())
        with np.load(self.output/"R004/component-pixels.npz") as arrays:
            np.testing.assert_array_equal(arrays["component_labels"], candidate_fixture().component_labels)
            np.testing.assert_array_equal(arrays["source_ink"], arrays["component_labels"] > 0)
        geo = json.loads((self.output/"component-search-regions.geojson").read_text())
        self.assertEqual(geo["features"][0]["geometry"]["coordinates"][0][0], [104.5, 189.])
        self.assertEqual(len(geo["features"]), 9)
        self.recognizer.close.assert_called_once()
        self.detector.predict.assert_not_called()

    def test_explicit_partial_single_component_search_is_allowed(self):
        def single(gray, origin):
            result = candidate_fixture(gray.shape, origin)
            for group in result.groups:
                group.update(group_component_ids=[1], candidate_kind="intra_component_hole_axis",
                             partial_component_search=True, full_component_containment=False)
            return result, {"synthetic": True}
        self.generator.side_effect = single
        result = self.run_fake()
        self.assertEqual(result["counts"]["recognition_attempts"], 12)
        self.assertTrue(all(g["candidate"]["partial_component_search"] for r in result["regions"] for g in r["groups"]))

    def test_unqualified_single_component_search_is_rejected(self):
        def single(gray, origin):
            result = candidate_fixture(gray.shape, origin)
            for group in result.groups:
                group.update(group_component_ids=[1], candidate_kind="intra_component_hole_axis",
                             partial_component_search=True, full_component_containment=True)
            return result, {"synthetic": True}
        self.generator.side_effect = single
        with self.assertRaisesRegex(MapTextDetectionError, "explicit partial"):
            self.run_fake()

    def test_recognition_rank_cap_does_not_drop_candidate_or_fabricate_score(self):
        result = self.run_fake(max_recognition_groups_per_region=1)
        self.assertEqual(result["counts"]["recognition_attempts"], 6)
        second = result["regions"][0]["groups"][1]
        self.assertEqual(second["selection_rank"], 2)
        self.assertEqual(second["selection_reason"], "recognition_group_cap")
        self.assertEqual(second["passes"][0]["status"], "not_run")
        self.assertNotIn("recognition", second["passes"][0])
        self.assertEqual(second["candidate"]["ranking_score_not_probability"], .8)

    def test_empty_candidate_result_has_no_fabricated_reading(self):
        def empty(gray, origin):
            result = candidate_fixture(gray.shape, origin)
            result.groups = ()
            return result, {"synthetic": True}
        self.generator.side_effect = empty
        result = self.run_fake()
        self.assertEqual(result["counts"]["recognition_attempts"], 0)
        self.assertEqual(result["counts"]["retained_groups"], 0)
        self.recognizer.predict.assert_not_called()
        self.assertTrue((self.output/"component-text-recognition.json").exists())

    def test_numeric_worker_uses_existing_interpreter_and_only_data_packets(self):
        result = self.run_fake(candidate_python=sys.executable)
        # The genuine core sees constant synthetic TIFF pixels and returns no
        # groups. Fake candidate metadata cannot leak into the subprocess.
        self.generator.assert_not_called()
        self.recognizer.predict.assert_not_called()
        self.assertEqual(result["counts"]["retained_groups"], 0)
        execution = result["regions"][0]["component_generation_execution"]
        self.assertEqual(execution["mode"], "isolated_numeric_subprocess")
        self.assertTrue(execution["runtime_verified_before_and_after"])
        self.assertIn("scipy", execution["runtime"]["packages"])
        self.assertTrue((self.output/execution["packet_path"]).exists())
        self.assertIn("candidate_python_executable", result["provenance"]["input_files"])

    def test_changed_model_after_inference_blocks_success_report(self):
        with patch.object(recognition, "verify_models", side_effect=[{"recognition": str(self.root/"model")}, MarginOcrError("model changed")]), \
             patch.object(recognition, "create_recognizer", return_value=(self.recognizer, self.cropper)), \
             patch.object(recognition, "create_candidates", self.generator):
            with self.assertRaisesRegex(MarginOcrError, "model changed"):
                recognition.run(self.report_path, self.manifest_path, self.lock_path, self.output)
        self.assertFalse((self.output/"component-text-recognition.json").exists())
        self.recognizer.close.assert_called_once()

    def test_pixel_budget_omits_by_rank_and_records_all_skips(self):
        with patch.object(recognition, "MAX_CROPPED_PIXELS_PER_PASS", 60):
            result = self.run_fake()
        self.assertEqual(result["counts"]["recognition_attempts"], 6)
        second = result["regions"][0]["groups"][1]
        self.assertTrue(all(p["reason"] == "recognition_pixel_budget" for p in second["passes"]))

    def test_holdout_and_mismatched_report_hash_fail_before_raster_or_model_use(self):
        self.manifest["drawing_report_sha256"] = "0"*64
        self.write_manifest()
        with patch.object(recognition, "inspect_raster") as inspect, self.assertRaisesRegex(MapTextDetectionError, "fixed region manifest"):
            self.run_fake()
        inspect.assert_not_called()
        self.report["tiles"][0].update(sheet_id="178-gongju", tile_id="178-gongju-labels", split="holdout")
        self.report_path.write_text(json.dumps(self.report))
        with patch.object(recognition, "inspect_raster") as inspect, self.assertRaises(ValueError):
            self.run_fake()
        inspect.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_source_blocks_success_report(self):
        def change(crops):
            self.source.write_bytes(self.source.read_bytes()+b"changed")
            return [{"rec_text": "100", "rec_score": 1.} for _ in crops]
        self.recognizer.predict.side_effect = change
        with self.assertRaisesRegex(MapTextDetectionError, "input changed"):
            self.run_fake()
        self.assertFalse((self.output/"component-text-recognition.json").exists())

    def test_local_socket_guard_still_applies_to_direct_recognition(self):
        self.recognizer.predict.side_effect = lambda _crops: socket.create_connection(("example.com", 443))
        with self.assertRaises(MarginOcrError):
            self.run_fake()
        self.assertFalse((self.output/"component-text-recognition.json").exists())

    def test_existing_or_source_nested_output_and_invalid_caps_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_fake()
        with self.assertRaisesRegex(MapTextDetectionError, "outside the source"):
            recognition.run(self.report_path, self.manifest_path, self.lock_path, self.packet/"new")
        for limit in (0, 129, True):
            with self.assertRaisesRegex(MapTextDetectionError, "group cap"):
                self.run_fake(max_recognition_groups_per_region=limit)


if __name__ == "__main__":
    unittest.main()
