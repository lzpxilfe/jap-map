import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from histcontour_core.contour_context import (
    BASE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, CONTEXT_SCHEMA, NEIGHBOR_FEATURE_NAMES,
    NeighborhoodFeatures, fit_classifier, probability, recall_first_threshold, validate_classifier,
)
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_semantic_samples import sample_indices
from scripts.run_contour_context_experiment import metrics, nested_validation, validate_sample_labels
from scripts.score_contour_context_candidates import load_experiments, score, select_flags


def training_rows():
    rows = []
    for sheet_index, sheet in enumerate(("173-buyeo", "174-cheongyang", "177-nonsan")):
        for index in range(12):
            positive = index < 8
            rows.append({"sample_id": f"{sheet}-{index}", "sheet_id": sheet,
                         "class": "contour" if positive else ("text" if index < 10 else "road_river"),
                         **{name: (1 if positive else -1) + 0.03*index + 0.01*sheet_index + j*0.001 for j, name in enumerate(CONTEXT_FEATURE_NAMES)}})
    return rows


class NeighborhoodFeatureTests(unittest.TestCase):
    def test_parallel_family_and_immutable_input(self):
        single = np.full((192, 192), 255, np.uint8)
        single[94:97, :] = 0
        family = single.copy()
        for y in (48, 64, 80, 112, 128, 144):
            family[y:y+2, :] = 0
        points = [(30.0, 95.0), (160.0, 95.0)]
        before = family.copy()
        one = NeighborhoodFeatures(single).describe(points)
        many = NeighborhoodFeatures(family).describe(points)
        np.testing.assert_array_equal(before, family)
        self.assertEqual(set(many), set(NEIGHBOR_FEATURE_NAMES))
        self.assertGreater(many["profile_peaks"], one["profile_peaks"] + 2)
        self.assertGreater(many["normal_alignment_r36"], 0.9)
        self.assertGreater(many["profile_bilateral"], 0.9)
        self.assertTrue(all(math.isfinite(value) for value in many.values()))

    def test_direction_and_reversal(self):
        image = np.full((192, 192), 255, np.uint8)
        image[::12, :] = 0
        cache = NeighborhoodFeatures(image)
        horizontal = cache.describe([(36.0, 96.0), (156.0, 96.0)])
        reversed_line = cache.describe([(156.0, 96.0), (36.0, 96.0)])
        vertical = cache.describe([(96.0, 36.0), (96.0, 156.0)])
        self.assertGreater(horizontal["normal_alignment_r36"], 0.9)
        self.assertLess(vertical["normal_alignment_r36"], -0.9)
        for name in NEIGHBOR_FEATURE_NAMES:
            self.assertAlmostEqual(horizontal[name], reversed_line[name], places=5, msg=name)

    def test_blank_edge_and_closed_loop_finite(self):
        cache = NeighborhoodFeatures(np.full((32, 32), 255, np.uint8))
        for points in ([(0, 0), (31, 31)], [(8, 8), (10, 8), (10, 10), (8, 10), (8, 8)]):
            values = cache.describe(points)
            self.assertTrue(all(math.isfinite(value) for value in values.values()))
            self.assertEqual(values["profile_peaks"], 0)
            self.assertLess(values["profile_valid_fraction"], 1)

    def test_invalid_inputs(self):
        for array in (np.empty((0, 5)), np.zeros((3, 3, 3)), np.full((3, 3), np.nan), np.full((3, 3), 256)):
            with self.assertRaises(ValueError):
                NeighborhoodFeatures(array)
        cache = NeighborhoodFeatures(np.zeros((32, 32)))
        for points in ([(0, 0)], [(1, 1), (1, 1)], [(0, 0), (32, 2)], [(0, 0), (float("nan"), 2)]):
            with self.assertRaises(ValueError):
                cache.describe(points)


class ContextClassifierTests(unittest.TestCase):
    def test_model_fit_and_validation(self):
        rows = training_rows()
        model = validate_classifier(fit_classifier(rows))
        self.assertGreater(probability(model, rows[0]), probability(model, rows[11]))
        self.assertAlmostEqual(model["means"][0], np.mean([row[CONTEXT_FEATURE_NAMES[0]] for row in rows]))
        self.assertEqual(model["training_count"], len(rows))
        for mutation in ({"feature_schema": "wrong"}, {"feature_names": ["sheet_id"]}, {"scales": [0]*len(CONTEXT_FEATURE_NAMES)}, {"intercept": float("nan")}, {"coefficients": []}):
            with self.assertRaises(ValueError):
                validate_classifier({**model, **mutation})
        broken = {**rows[0], CONTEXT_FEATURE_NAMES[0]: float("nan")}
        with self.assertRaises(ValueError):
            probability(model, broken)

    def test_no_identifier_features_and_exclude_ambiguous(self):
        rows = training_rows()
        altered = [{**row, "sample_id": "different", "sheet_id": "arbitrary", "tile_id": "no target leakage"} for row in rows]
        self.assertEqual(fit_classifier(rows), fit_classifier(altered))
        with self.assertRaises(ValueError):
            fit_classifier([{**row, "class": "unsure"} for row in rows])
        with self.assertRaises(ValueError):
            fit_classifier(rows, ("sheet_id",))
        with self.assertRaises(ValueError):
            fit_classifier(rows, l2=float("nan"))

    def test_recall_threshold_counts_and_ties(self):
        rows = [{"class": "contour", "score": value} for value in (0.1, 0.3, 0.3, 0.8)]
        rows.append({"class": "text", "score": 0.999})
        self.assertEqual(recall_first_threshold(rows, target_recall=1), 0.1)
        self.assertEqual(recall_first_threshold(rows, target_recall=0.75), 0.3)
        for invalid in (0, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                recall_first_threshold(rows, target_recall=invalid)
        with self.assertRaises(ValueError):
            recall_first_threshold([{"class": "text", "score": 0.8}])
        with self.assertRaises(ValueError):
            recall_first_threshold([{"class": "contour", "score": float("nan")}])

    def test_nested_folds_test_sheet_never_calibrates_itself(self):
        rows = training_rows()
        result = nested_validation(rows, BASE_FEATURE_NAMES, target_recall=1)
        for fold in result["folds"]:
            train, test = set(fold["training_sample_ids"]), set(fold["test_sample_ids"])
            self.assertFalse(train & test)
            calibrated = {row["sample_id"] for row in fold["inner_predictions"]}
            self.assertEqual(calibrated, train)
            for inner in fold["inner_folds"]:
                self.assertFalse(set(inner["training_sample_ids"]) & set(inner["validation_sample_ids"]))
                self.assertFalse(test & set(inner["training_sample_ids"]))
                self.assertFalse(test & set(inner["validation_sample_ids"]))
        altered = copy.deepcopy(rows)
        for row in altered:
            if row["sheet_id"] == "173-buyeo":
                for name in CONTEXT_FEATURE_NAMES:
                    row[name] += 5
        other = nested_validation(altered, BASE_FEATURE_NAMES, target_recall=1)
        self.assertEqual(result["folds"][0]["threshold"], other["folds"][0]["threshold"])
        self.assertEqual(result["folds"][0]["model"], other["folds"][0]["model"])


class SemanticSamplingTests(unittest.TestCase):
    def test_score_blind_length_bins(self):
        features = [{"properties": {"pixel_length": length+i, "segment_uid": f"{length}-{i}", "contour_score": 0}} for length in (18, 30, 100) for i in range(8)]
        original = sample_indices(features, seed=42, per_bin=6)
        for feature in features:
            feature["properties"]["contour_score"] = 1
        self.assertEqual(original, sample_indices(features, seed=42, per_bin=6))
        self.assertEqual(len(original), 18)
        with self.assertRaises(ValueError):
            sample_indices(features, seed=42, per_bin=9)
        for mutated in (features+[features[0]], [{"properties": {"pixel_length": float("nan"), "segment_uid": "a"}}], [{"properties": {"pixel_length": 17, "segment_uid": "a"}}]):
            with self.assertRaises(ValueError):
                sample_indices(mutated, seed=42, per_bin=1)

    def test_labels_fingerprint_unique_ids_and_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = {"holdout_used": False, "samples": [
                {"sample_id": f"S{i}", "segment_uid": f"UID{i}", "sheet_id": sheet, "tile_id": f"{sheet}-{kind}"}
                for i, (sheet, kind) in enumerate((s, k) for s in ("173-buyeo", "174-cheongyang", "177-nonsan") for k in ("mountain", "labels", "hydro"))]}
            source, labels_path = root/"manifest.json", root/"labels.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            labels = {"sample_manifest_sha256": sha256_file(source), "human_approved": False, "reference_origin": "ai_visual_provisional",
                      "items": [{"sample_id": row["sample_id"], "class": "contour"} for row in manifest["samples"]]}
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            self.assertEqual(len(validate_sample_labels(source, labels_path)[1]), 9)
            for changed in ({**labels, "human_approved": True}, {**labels, "sample_manifest_sha256": "0"*64}, {**labels, "items": labels["items"]+[labels["items"][0]]}):
                labels_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_sample_labels(source, labels_path)
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            manifest["holdout_used"] = True
            source.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_sample_labels(source, labels_path)


class ContextExportContracts(unittest.TestCase):
    def test_full_export_keeps_geometry_uncertainty_and_escapes_notes(self):
        from PIL import Image
        from histcontour_core.segment_review import LogisticModel
        from scripts.generate_ink_centerline_candidates import _map_coordinates
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def save(name, value):
                path = root/name
                path.write_text(json.dumps(value), encoding="utf-8")
                return path

            raster = root/"synthetic.png"
            Image.new("L", (64, 64), 255).save(raster)
            digest = sha256_file(raster)
            tiles, raw_entries, samples, original_geometry = [], [], [], {}
            for i, (sheet, kind) in enumerate((s, k) for s in ("173-buyeo", "174-cheongyang", "177-nonsan") for k in ("mountain", "labels", "hydro")):
                tile_id, sid = f"{sheet}-{kind}", f"S{i+1:03}"
                tile = {"tile_id": tile_id, "sheet_id": sheet, "split": "development", "scene_type": "synthetic_contract", "raster_path": str(raster), "pixel_bounds": [0, 0, 64, 64], "bounds": [0, 0, 64, 64]}
                tiles.append(tile)
                points = [(10, 32), (50, 32)]
                geometry = {"type": "LineString", "coordinates": _map_coordinates(tile, 64, 64, points)}
                original_geometry[tile_id] = geometry
                vector = save(f"{tile_id}.geojson", {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"proposal_id": sid, "segment_uid": sid, "pixel_length": 40}, "geometry": geometry}]})
                raw_entries.append({"tile_id": tile_id, "ink_vector_path": str(vector), "source_raster_sha256": digest})
                samples.append({"sample_id": sid, "segment_uid": sid, "tile_id": tile_id, "sheet_id": sheet, "source_raster_sha256": digest,
                                "source_vector_sha256": sha256_file(vector), "pixel_box": [0, 0, 64, 64], "pixel_points": points})
            index_path = save("index.json", {"tiles": tiles})
            vector_path = save("vectors.json", {"tiles": raw_entries, "holdout_included": False})
            samples_path = save("samples.json", {"schema": "jap-map-contour-semantic-samples/1", "holdout_used": False, "samples": samples, "vector_index_sha256": sha256_file(vector_path)})
            labels_path = save("labels.json", {"human_approved": False, "reference_origin": "ai_visual_provisional", "items": [{"sample_id": row["sample_id"], "class": "contour", "note": "<script>bad()</script>"} for row in samples]})
            legacy = LogisticModel(BASE_FEATURE_NAMES, (0.,)*11, (1.,)*11, (0.,)*11, 0.)
            baseline_path = save("legacy.json", {"status": "trained", "model": legacy.to_dict()})
            model = {"feature_schema": CONTEXT_SCHEMA, "feature_names": list(CONTEXT_FEATURE_NAMES), "means": [0.]*34, "scales": [1.]*34, "coefficients": [0.]*34, "intercept": 0.}
            predictions = [{"sample_id": row["sample_id"], "sheet_id": row["sheet_id"], "class": "contour", "score": 0.5, "retained": False} for row in samples]
            experiment_paths = []
            for name, target, threshold in (("balanced", 0.95, 0.6), ("conservative", 1., 0.4)):
                folds = [{"test_sheet": sheet, "threshold": threshold, "model": model,
                          "training_sample_ids": [row["sample_id"] for row in samples if row["sheet_id"] != sheet],
                          "test_sample_ids": [row["sample_id"] for row in samples if row["sheet_id"] == sheet]}
                         for sheet in ("173-buyeo", "174-cheongyang", "177-nonsan")]
                report = {"schema": "jap-map-contour-context-experiment/1", "status": "completed_research_only", "human_approved": False, "holdout_used": False,
                          "reference_origin": "ai_visual_provisional", "target_inner_contour_retention": target, "class_counts": {"contour": 9},
                          "source_index_sha256": sha256_file(index_path), "sample_manifest_sha256": sha256_file(samples_path),
                          "labels_sha256": sha256_file(labels_path), "legacy_model_sha256": sha256_file(baseline_path), "feature_schema": CONTEXT_SCHEMA, "l2_fixed_for_both_models": 0.1,
                          "legacy_fixed_010": {"metrics": metrics([{**row, "retained": True} for row in predictions])},
                          "alternatives": {"real_weak_neighborhood": {"folds": folds, "outer_predictions": predictions, "metrics": metrics([{**row, "retained": threshold <= .5} for row in predictions])}}}
                experiment_paths.append(save(f"{name}.json", report))
            probes_path = save("probes.json", {"schema": "jap-map-contour-region-probes/1", "human_approved": False, "regions": [], "source_sha256": {}})
            output = root/"result"
            result = score(index_path, vector_path, baseline_path, *experiment_paths, samples_path, labels_path, probes_path, output, illustration_ids=["S001"])
            self.assertEqual(result["counts"], {"legacy": 9, "balanced": 0, "conservative": 9, "uncertain": 9})
            self.assertFalse(result["automatic_promotion"])
            self.assertIsNone(result["probe_aggregates"]["contour"]["coverage"]["balanced"])
            for row in result["tiles"]:
                exported = json.loads(Path(row["all_scores_path"]).read_text())["features"]
                self.assertEqual(exported[0]["geometry"], original_geometry[row["tile_id"]])
                self.assertFalse(exported[0]["properties"]["human_approved"])
            document = (output/"report.html").read_text()
            self.assertNotIn("<script>", document)
            self.assertIn("&lt;script&gt;", document)
            self.assertEqual(document.count("<details id="), 9)
            self.assertTrue((output/"context-tradeoff-examples.png").exists())
            with self.assertRaises(FileExistsError):
                score(index_path, vector_path, baseline_path, *experiment_paths, samples_path, labels_path, probes_path, output)

    def test_uncertain_union_and_thresholds(self):
        # A plausible old candidate is never silently discarded from the review queue.
        self.assertEqual(select_flags(0.9, 0.05, 0.3, 0.1), {"legacy": True, "balanced": False, "conservative": False, "uncertain": True})
        self.assertTrue(select_flags(0.01, 0.2, 0.3, 0.1)["uncertain"])
        self.assertFalse(select_flags(0.9, 0.9, 0.3, 0.1)["uncertain"])
        for parameters in ((0.2, 0.3, 0.2, 0.4), (0.2, float("nan"), 0.3, 0.2), (True, 0.2, 0.3, 0.1)):
            with self.assertRaises(ValueError):
                select_flags(*parameters)

    def test_two_threshold_policies_cannot_change_models(self):
        result = nested_validation(training_rows(), CONTEXT_FEATURE_NAMES)
        first = {"schema": "jap-map-contour-context-experiment/1", "status": "completed_research_only",
                 "target_inner_contour_retention": 0.95, "human_approved": False, "holdout_used": False,
                 **{key: "a"*64 for key in ("source_index_sha256", "sample_manifest_sha256", "labels_sha256", "legacy_model_sha256")},
                 "reference_origin": "ai_visual_provisional", "alternatives": {"real_weak_neighborhood": result}}
        second = {**copy.deepcopy(first), "target_inner_contour_retention": 1.0}
        for row in second["alternatives"]["real_weak_neighborhood"]["folds"]:
            row["threshold"] = 0.0
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp)/name for name in ("balanced.json", "conservative.json")]
            for path, report in zip(paths, (first, second)):
                path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(len(load_experiments(*paths)[1][0]), 3)
            second["alternatives"]["real_weak_neighborhood"]["folds"][0]["model"]["intercept"] += 1
            paths[1].write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiments(*paths)


if __name__ == "__main__":
    unittest.main()
