"""Source/ROI/publication contracts for experimental reconstruction runs.

Most run tests use a small deterministic extraction substitute while retaining
real TIFF inspection, source hashes, ownership, export and approval-copy paths.
The ROI tests also exercise the actual bounded curve refiner.
"""

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, TiffImagePlugin

from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.observed_linework import ObservedLineworkConfig
from histcontour_core.provenance import sha256_file
from scripts import reconstruct_observed_contours as reconstruction
from scripts.prepare_contour_reconstruction_review import intersects


TILE_ID = "173-buyeo-mountain"


def path_record(points, *, role="observed_linework_review", identity="fixture-001"):
    return {"path_id": identity, "points": copy.deepcopy(points),
            "length_px": sum(math.dist(a, b) for a, b in zip(points, points[1:])),
            "role": role, "strong_seed_connected": True, "mean_continuous_support": .8,
            "text_avoidance_fraction": 0., "human_approved": False,
            "contour_semantics_assigned": False, "training_eligible": False, "inferred_gap": False}


def fake_extraction(paths, shape=(64, 64)):
    skeleton = np.zeros(shape, bool)
    for path in paths:
        points = np.rint(path["points"]).astype(int)
        skeleton[points[:, 1], points[:, 0]] = True
    return {"paths": copy.deepcopy(paths), "masks": {"skeleton": skeleton},
            "stage_counts": {"skeleton": int(skeleton.sum())}, "config": asdict(ObservedLineworkConfig())}


class ObservedReconstructionRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.packet = self.root/"packet"
        self.packet.mkdir()
        self.source = self.packet/"source.tif"
        self.report_path = self.packet/"drawing-report.json"
        self.manifest_path = self.root/"regions.json"
        self.detection_path = self.root/"text-detection.json"
        self.approved_path = self.root/"approved.geojson"
        self.output = self.root/"new-output"
        self.tile = {"tile_id": TILE_ID, "sheet_id": "173-buyeo", "split": "development",
                     "pixel_bounds": [200, 400, 64, 64], "raster_path": "source.tif",
                     "bounds": [100., 184., 108., 200.], "crs_authid": "EPSG:5132"}
        self.write_tiff()
        self.report = {"schema": "jap-map-assisted-contour-drawing/1", "holdout_used": False,
                       "human_approvals": 0, "tiles": [self.tile]}
        self.manifest = json.loads((reconstruction.ROOT/"examples/contour_reconstruction_regions.v1.json").read_text())
        for row in self.manifest["regions"]:
            row.update(tile_id=TILE_ID, box=[20, 20, 40, 40])
        self.detections = {"schema": "jap-map-map-text-detection/1", "holdout_used": False,
                           "tiles": [{**self.tile, "detection": {"polygons_on_source_pixel_center_grid": [], "dt_scores": []}}]}
        self.approved = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:5132"}},
                         "features": [{"type": "Feature", "properties": {"proposal_id": "human-fixture",
                             "human_approved": True, "reviewer_note": "原文 그대로"},
                             "geometry": {"type": "LineString", "coordinates": [[100.125, 198.25], [101.5, 197.75]]}}]}
        self.paths = [path_record([[5., 5.], [10., 5.]])]
        self.save_inputs()

    def tearDown(self):
        self.temporary.cleanup()

    def write_tiff(self, *, size=(64, 64), epsg=5132, split="development"):
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (.125, .25, 0.)
        tags[33922] = (0., 0., 0., 100., 200., 0.)
        tags[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, epsg)
        tags[42112] = f'<GDALMetadata><Item name="split">{split}</Item></GDALMetadata>'
        pixels = np.full(tuple(reversed(size)), 255, np.uint8)
        pixels[5, 5:11] = 35
        Image.fromarray(pixels).save(self.source, tiffinfo=tags)
        self.tile["source_raster_sha256"] = sha256_file(self.source)

    def save_inputs(self, *, sync_detection=False):
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        self.manifest["drawing_report_sha256"] = sha256_file(self.report_path)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        if sync_detection:
            self.detections["tiles"][0].update(self.tile)
        self.detection_path.write_text(json.dumps(self.detections), encoding="utf-8")
        # Deliberately noncanonical spacing/newlines checks byte identity.
        self.approved_path.write_text(json.dumps(self.approved, ensure_ascii=False, indent=3)+"\n\n", encoding="utf-8")

    def run_fixture(self, region_ids=("R001",), *, extraction=None, output=None):
        value = fake_extraction(self.paths) if extraction is None else extraction
        with patch.object(reconstruction, "extract_observed_linework", return_value=value) as extract:
            result = reconstruction.run(self.packet, self.manifest_path, self.detection_path,
                                        self.approved_path, output or self.output, region_ids)
        return result, extract

    def test_real_tiff_source_contract_native_coordinates_and_approved_bytes(self):
        inputs = [self.source, self.report_path, self.manifest_path, self.detection_path, self.approved_path]
        before = {p: p.read_bytes() for p in inputs}
        result, extract = self.run_fixture()
        self.assertEqual(before, {p: p.read_bytes() for p in inputs})
        self.assertEqual((self.output/"approved-connections-unchanged.geojson").read_bytes(), before[self.approved_path])
        self.assertEqual(extract.call_args.kwargs["tile_origin"], (200, 400))
        np.testing.assert_array_equal(extract.call_args.args[0], np.asarray(Image.open(self.source)))
        self.assertEqual(extract.call_args.kwargs["text_avoidance_score"].shape, (64, 64))
        raw = json.loads((self.output/"raw-observed.geojson").read_text())
        draft = json.loads((self.output/"candidate-observed.geojson").read_text())
        self.assertEqual(raw["crs"]["properties"]["name"], "EPSG:5132")
        self.assertEqual(raw["features"], draft["features"])
        feature = raw["features"][0]
        self.assertEqual(feature["geometry"]["coordinates"][0], [100.+5.5*.125, 200.-5.5*.25])
        np.testing.assert_allclose(map_to_pixel(self.tile, feature["geometry"]["coordinates"]), self.paths[0]["points"], atol=1e-12)
        for flag in ("human_approved", "whole_line_semantics_approved", "contour_semantics_assigned", "training_eligible"):
            self.assertFalse(feature["properties"][flag])
        self.assertEqual(feature["properties"]["dataset_role"], "review_only_not_training")
        self.assertEqual(result["status"], "experimental_not_applied")
        self.assertEqual(result["human_approvals"], 0)
        self.assertTrue(result["provenance"]["inputs_unchanged"])
        self.assertTrue(result["provenance"]["implementation_unchanged"])
        for path, digest in result["provenance"]["implementation_sha256"].items():
            self.assertEqual(sha256_file(path), digest)
        names = {Path(p).name for p in result["provenance"]["implementation_sha256"]}
        self.assertTrue({"ink.py", "vectorization.py", "contours.py", "text_ownership.py", "observed_curve.py",
                         "prepare_contour_reconstruction_review.py"}.issubset(names))
        self.assertEqual(set(result["provenance"]["runtime_versions"]), {"numpy", "scipy", "Pillow", "scikit-image"})

    def test_suspected_text_and_short_context_are_exported_without_semantic_promotion(self):
        self.paths += [path_record([[24., 24.], [29., 24.]], role="suspected_text_review", identity="text"),
                       path_record([[44., 44.], [45., 44.]], role="short_context", identity="short")]
        self.run_fixture()
        for name, identity in (("suspected-text", "text"), ("short-context", "short")):
            collection = json.loads((self.output/(name+".geojson")).read_text())
            self.assertEqual(len(collection["features"]), 1)
            properties = collection["features"][0]["properties"]
            self.assertEqual(properties["path_id"], identity)
            self.assertFalse(properties["human_approved"])
            self.assertFalse(properties["training_eligible"])
        raw = json.loads((self.output/"raw-observed.geojson").read_text())
        self.assertEqual([f["properties"]["path_id"] for f in raw["features"]], ["fixture-001"])

    def test_smoothed_export_remeasures_length_and_names_raw_support_scope(self):
        self.paths = [path_record([[22., 25.], [26., 28.], [32., 25.], [38., 25.]])]
        with patch.object(reconstruction, "smooth_path", return_value=(
                [[22., 25.], [38., 25.]], {"adopted_in_draft": True})):
            self.run_fixture()
        feature = json.loads((self.output/"candidate-observed.geojson").read_text())["features"][0]
        props = feature["properties"]
        self.assertEqual(props["length_px"], 16.)
        self.assertGreater(props["source_trace_length_px"], 16.)
        self.assertEqual(props["source_trace_mean_continuous_support"], .8)
        self.assertNotIn("mean_continuous_support", props)
        self.assertIn("original raster graph", props["evidence_measurement_scope"])

    def test_empty_duplicate_unknown_region_ids_fail_before_extraction(self):
        for ids in ([], ["R001", "R001"], ["R999"]):
            with self.subTest(ids=ids), patch.object(reconstruction, "extract_observed_linework") as extract:
                with self.assertRaises(ValueError):
                    reconstruction.run(self.packet, self.manifest_path, self.detection_path,
                                       self.approved_path, self.output, ids)
                extract.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_overwrite_and_output_inside_source_snapshot_are_rejected(self):
        self.output.mkdir()
        sentinel = self.output/"keep.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(sentinel.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "outside source packet"):
            self.run_fixture(output=self.packet/"new-output")

    def test_source_path_escape_absolute_and_external_symlink_fail(self):
        outside = self.root/"outside.tif"
        outside.write_bytes(self.source.read_bytes())
        link = self.packet/"outside-link.tif"
        link.symlink_to(outside)
        for name in ("../outside.tif", str(outside.resolve()), "outside-link.tif"):
            with self.subTest(name=name):
                self.tile["raster_path"] = name
                self.save_inputs(sync_detection=True)
                with self.assertRaisesRegex(ValueError, "remain inside packet"):
                    self.run_fixture()
                self.assertFalse(self.output.exists())

    def test_report_manifest_detection_and_actual_raster_holdout_fail(self):
        cases = ("report", "tile", "manifest", "detection", "actual_raster")
        for case in cases:
            with self.subTest(case=case):
                if case == "report": self.report["holdout_used"] = True
                elif case == "tile": self.tile["split"] = "holdout"
                elif case == "manifest": self.manifest["holdout_used"] = True
                elif case == "detection": self.detections["holdout_used"] = True
                else: self.write_tiff(split="holdout")
                self.save_inputs(sync_detection=True)
                with self.assertRaises(ValueError): self.run_fixture()
                self.assertFalse(self.output.exists())
                self.report["holdout_used"] = self.manifest["holdout_used"] = self.detections["holdout_used"] = False
                self.tile["split"] = "development"
                self.write_tiff()
                self.save_inputs(sync_detection=True)

    def test_actual_raster_dimensions_crs_and_digest_are_verified(self):
        for change in ("dimensions", "crs", "digest"):
            with self.subTest(change=change):
                if change == "dimensions": self.write_tiff(size=(63, 64))
                elif change == "crs": self.write_tiff(epsg=4326)
                else: self.source.write_bytes(self.source.read_bytes()+b"changed")
                self.save_inputs(sync_detection=True)
                with self.assertRaises(ValueError): self.run_fixture()
                self.assertFalse(self.output.exists())
                self.write_tiff()
                self.save_inputs(sync_detection=True)

    def test_stale_report_manifest_hash_fails(self):
        self.manifest["drawing_report_sha256"] = "0"*64
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "source report differs"):
            self.run_fixture()

    def test_text_source_metadata_mismatch_fails(self):
        for field, value in (("crs_authid", "EPSG:4326"), ("source_raster_sha256", "0"*64),
                             ("pixel_bounds", [0, 0, 64, 64]), ("bounds", [100, 184, 109, 200])):
            with self.subTest(field=field):
                self.detections["tiles"][0].update(self.tile)
                self.detections["tiles"][0][field] = value
                self.save_inputs()
                with self.assertRaisesRegex(ValueError, "text evidence source mismatch"):
                    self.run_fixture()
                self.assertFalse(self.output.exists())

    def test_approved_native_crs_or_geometry_type_mismatch_fails(self):
        self.approved["crs"]["properties"]["name"] = "EPSG:4326"
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "approved geometry"):
            self.run_fixture()
        self.approved["crs"]["properties"]["name"] = "EPSG:5132"
        self.approved["features"][0]["geometry"] = {"type": "Point", "coordinates": [100, 200]}
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "approved geometry"):
            self.run_fixture()

    def test_enclosing_loop_is_not_fitted_just_because_its_bbox_overlaps(self):
        loop = [[10., 10.], [50., 10.], [50., 50.], [10., 50.], [10., 10.]]
        self.paths = [path_record(loop)]
        with patch.object(reconstruction, "smooth_path") as smooth:
            result, _ = self.run_fixture()
        smooth.assert_not_called()
        self.assertEqual(result["curve_audits"], [])
        self.assertFalse(intersects(loop, [20, 20, 40, 40]))
        raw = json.loads((self.output/"raw-observed.geojson").read_text())
        draft = json.loads((self.output/"candidate-observed.geojson").read_text())
        self.assertEqual(raw["features"], draft["features"])
        self.assertEqual(draft["features"][0]["geometry"]["coordinates"][0], draft["features"][0]["geometry"]["coordinates"][-1])

    def test_actual_crossing_passes_only_requested_boxes_to_curve_refiner(self):
        self.paths = [path_record([[5., 30.], [25., 30.], [45., 30.], [58., 30.]])]
        self.manifest["regions"][1]["box"] = [40, 10, 60, 20]
        self.save_inputs()
        with patch.object(reconstruction, "smooth_path", return_value=(self.paths[0]["points"], {"adopted_in_draft": False})) as smooth:
            self.run_fixture(["R001"])
        smooth.assert_called_once()
        self.assertEqual(smooth.call_args.args[3], [[20, 20, 40, 40]])

    def test_source_mutation_during_extraction_prevents_success_report(self):
        def changed(*_args, **_kwargs):
            self.source.write_bytes(self.source.read_bytes()+b"test-only mutation")
            return fake_extraction(self.paths)
        with patch.object(reconstruction, "extract_observed_linework", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "input changed"):
                reconstruction.run(self.packet, self.manifest_path, self.detection_path,
                                   self.approved_path, self.output, ["R001"])
        self.assertFalse((self.output/"reconstruction-report.json").exists())

    def test_implementation_hash_change_during_extraction_prevents_success_report(self):
        # Simulate a changed implementation fingerprint; never edit real code.
        selected = (reconstruction.ROOT/"histcontour_core"/"ink.py").resolve()
        state = {"changed": False}
        def hash_checked(path):
            if state["changed"] and Path(path).resolve() == selected:
                return "0"*64
            return sha256_file(path)
        def changed(*_args, **_kwargs):
            state["changed"] = True
            return fake_extraction(self.paths)
        original_digest = sha256_file(selected)
        with patch.object(reconstruction, "sha256_file", side_effect=hash_checked), \
             patch.object(reconstruction, "extract_observed_linework", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                reconstruction.run(self.packet, self.manifest_path, self.detection_path,
                                   self.approved_path, self.output, ["R001"])
        self.assertEqual(sha256_file(selected), original_digest)
        self.assertFalse((self.output/"reconstruction-report.json").exists())


class ObservedReconstructionRoiTests(unittest.TestCase):
    def test_segment_intersection_uses_centre_grid_boundaries(self):
        box = [20, 20, 40, 40]
        self.assertTrue(intersects([[0, 30], [60, 30]], box))
        self.assertTrue(intersects([[19.5, 0], [19.5, 60]], box))
        self.assertFalse(intersects([[19.49, 0], [19.49, 60]], box))
        self.assertFalse(intersects([[0, 0], [0, 10]], box))
        self.assertFalse(intersects([[30, 30]], box))

    def test_outside_vertices_and_partially_crossing_edges_are_locked(self):
        points = [[10., 30.], [20., 30.], [30., 30.], [40., 30.], [50., 30.]]
        gray = np.full((64, 64), 255, np.uint8)
        skeleton = np.zeros_like(gray, dtype=bool)
        fake = {"points": [], "audit": {"changed": False, "before": {"total_turn_degrees": 0}, "after": {"total_turn_degrees": 0}}}
        with patch.object(reconstruction, "refine_observed_curve", return_value=fake) as refine:
            output, audit = reconstruction.smooth_path(gray, skeleton, points, [[20, 20, 40, 40]])
        self.assertEqual(output, points)
        self.assertFalse(audit["adopted_in_draft"])
        np.testing.assert_array_equal(refine.call_args.kwargs["locked_indices"], [0, 3, 4])
        self.assertEqual(refine.call_args.kwargs["locked_spans"], [(0, 1), (2, 3), (3, 4)])

    def test_edge_between_distinct_allowed_boxes_remains_locked(self):
        points = [[20., 30.], [40., 30.]]
        gray = np.full((64, 64), 255, np.uint8)
        fake = {"points": [], "audit": {"changed": False, "before": {"total_turn_degrees": 0}, "after": {"total_turn_degrees": 0}}}
        with patch.object(reconstruction, "refine_observed_curve", return_value=fake) as refine:
            reconstruction.smooth_path(gray, np.zeros_like(gray, dtype=bool), points, [[18, 28, 24, 33], [38, 28, 44, 33]])
        self.assertEqual(refine.call_args.kwargs["locked_spans"], [(0, 1)])

    def test_moved_sample_leaving_allowed_union_rejects_entire_fit(self):
        points = [[10., 30.], [25., 30.], [35., 30.], [50., 30.]]
        gray = np.full((64, 64), 255, np.uint8)
        def escaping(_gray, local, **_kwargs):
            moved = np.array(local, copy=True)
            moved[1, 1] += 20
            distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
            return {"points": moved.tolist(), "point_source_distances": distances.tolist(), "spans": [],
                    "audit": {"changed": True, "before": {"total_turn_degrees": 100}, "after": {"total_turn_degrees": 10}}}
        with patch.object(reconstruction, "refine_observed_curve", side_effect=escaping):
            output, audit = reconstruction.smooth_path(gray, np.zeros_like(gray, dtype=bool), points, [[20, 20, 40, 40]])
        self.assertEqual(output, points)
        self.assertFalse(audit["adopted_in_draft"])
        self.assertEqual(audit["draft_rejection"], "moved_outside_requested_region")

    def test_real_curve_refinement_preserves_outside_geometry_and_source_arrays(self):
        yy, _xx = np.indices((64, 64))
        gray = (255.-180.*np.exp(-.5*((yy-32.)/.85)**2)).astype(np.float32)
        points = [[float(x), 31.+float(x % 2)] for x in range(5, 59)]
        skeleton = np.zeros(gray.shape, bool)
        indices = np.asarray(points, int)
        skeleton[indices[:, 1], indices[:, 0]] = True
        before_gray, before_skeleton, before_points = gray.copy(), skeleton.copy(), copy.deepcopy(points)
        output, audit = reconstruction.smooth_path(gray, skeleton, points, [[20, 20, 44, 44]])
        self.assertTrue(audit["adopted_in_draft"])
        np.testing.assert_array_equal(gray, before_gray)
        np.testing.assert_array_equal(skeleton, before_skeleton)
        self.assertEqual(points, before_points)
        self.assertEqual(output[0], points[0])
        self.assertEqual(output[-1], points[-1])
        for vertex in points:
            if vertex[0] < 19.5 or vertex[0] > 43.5:
                self.assertIn(vertex, output)
        # Every output sample outside the allowed ROI remains on the exact
        # original polyline, including resampled points on locked segments.
        original = np.asarray(points)
        for point in np.asarray(output):
            if 19.5 <= point[0] <= 43.5 and 19.5 <= point[1] <= 43.5:
                continue
            starts, vectors = original[:-1], np.diff(original, axis=0)
            fraction = np.clip(np.sum((point-starts)*vectors, axis=1)/np.sum(vectors*vectors, axis=1), 0, 1)
            distance = np.min(np.linalg.norm(point-(starts+fraction[:, None]*vectors), axis=1))
            self.assertLess(distance, 1e-9)


if __name__ == "__main__":
    unittest.main()
