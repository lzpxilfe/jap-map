import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from histcontour_core.human_feedback import collect_feedback, geometry_digest, rasterize_sparse_training_feedback, source_geometry_matches
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_feedback_training import prepare, validate_training_labels
from scripts.evaluate_contour_forward import comparison_gate


def packet_fixture():
    tile = {"tile_id": "fixture", "sheet_id": "173-buyeo", "split": "development", "bounds": [0., 0., 64., 64.],
            "pixel_bounds": [0, 0, 64, 64], "source_raster_sha256": "a"*64}
    cases = []
    for index, (sample, role, y) in enumerate((("S001", "training", 10), ("S002", "training", 20), ("E001", "evaluation_only", 30), ("S003", "training", 40)), 1):
        points = [[10., float(y)], [30., float(y)]]
        geometry = {"type": "LineString", "coordinates": [[x+.5, 63.5-y] for x, y in points]}
        cases.append({"case_id": f"H{index:03}", "sample_id": sample, "dataset_role": role, "tile_id": "fixture", "sheet_id": tile["sheet_id"],
                      "source_raster_sha256": tile["source_raster_sha256"], "segment_uid": sample, "original_geometry": geometry,
                      "original_geometry_sha256": geometry_digest(geometry), "pixel_points": points, "pixel_box": [0, 0, 64, 64],
                      "review_status": "unreviewed", "geometry_decision": "unreviewed", "annotator": "", "human_approved": False, "review_note": ""})
    return {"schema": "jap-map-contour-human-packet/1", "holdout_used": False, "tiles": [tile], "cases": cases}


def approved(row, status="contour", action="accept_original"):
    return {**row, "human_approved": True, "annotator": "Synthetic test reviewer", "review_status": status, "geometry_decision": action}


class HumanFeedbackTests(unittest.TestCase):
    def test_nothing_is_approved_from_initial_packet(self):
        packet = packet_fixture()
        result = collect_feedback(packet, copy.deepcopy(packet["cases"]))
        self.assertEqual(result["human_approval_count"], 0)
        self.assertEqual(result["training_labels"], [])
        self.assertEqual(result["evaluation_labels"], [])
        self.assertEqual(result["training_readiness"]["status"], "waiting_for_human_labels")

    def test_roles_and_sparse_unknown_background(self):
        packet = packet_fixture()
        decisions = copy.deepcopy(packet["cases"])
        decisions[0] = approved(decisions[0])
        decisions[1] = approved(decisions[1], "road_river", "reject")
        decisions[2] = approved(decisions[2])
        feedback = collect_feedback(packet, decisions)
        self.assertEqual(len(feedback["training_labels"]), 2)
        self.assertEqual(len(feedback["evaluation_labels"]), 1)
        with tempfile.TemporaryDirectory() as temp:
            result = rasterize_sparse_training_feedback(packet, feedback, Path(temp)/"labels")
            image = np.asarray(Image.open(Path(temp)/"labels"/result["tiles"][0]["path"]))
            self.assertTrue((image[10, 10:31] == 1).all())
            self.assertTrue((image[20, 10:31] == 0).all())
            self.assertTrue((image[30, 10:31] == 255).all())  # E-series never trains.
            self.assertEqual(int((image == 255).sum()), 64*64-42)

    def test_corrected_path_and_inferred_gap_are_distinct(self):
        packet = packet_fixture()
        decisions = copy.deepcopy(packet["cases"])
        decisions[0] = approved(decisions[0], action="replace_with_trace")
        traces = [{"case_id": "H001", "record_id": "t1", "trace_kind": "observed_contour", "pixel_points": [[10., 11.], [30., 11.]], "note": "shifted actual contour"},
                  {"case_id": "H001", "record_id": "t2", "trace_kind": "inferred_gap", "pixel_points": [[10., 12.], [30., 12.]], "note": "not observed ink"}]
        feedback = collect_feedback(packet, decisions, traces)
        self.assertEqual(feedback["training_labels"], [])
        self.assertEqual(feedback["geometry_references"][0]["original_pixel_points"], [[10., 10.], [30., 10.]])
        self.assertEqual(feedback["geometry_references"][0]["human_pixel_points"], [[10., 11.], [30., 11.]])
        with tempfile.TemporaryDirectory() as temp:
            result = rasterize_sparse_training_feedback(packet, feedback, Path(temp)/"labels")
            image = np.asarray(Image.open(Path(temp)/"labels"/result["tiles"][0]["path"]))
            self.assertEqual(image[11, 20], 1)
            self.assertEqual(image[12, 20], 255)
            self.assertEqual(image[10, 20], 255)

    def test_explicit_approval_names_and_consistent_actions_required(self):
        packet = packet_fixture()
        mutations = [dict(human_approved=True), dict(human_approved=True, annotator="NULL", review_status="contour", geometry_decision="accept_original"),
                     dict(human_approved="true"), dict(human_approved=True, annotator="Tester", review_status="road_river", geometry_decision="accept_original"),
                     dict(human_approved=True, annotator="Tester", review_status="mixed", geometry_decision="unsure", review_note="")]
        for mutation in mutations:
            decisions = copy.deepcopy(packet["cases"])
            decisions[0].update(mutation)
            with self.assertRaises(ValueError):
                collect_feedback(packet, decisions)
        decisions = copy.deepcopy(packet["cases"])
        decisions[0] = approved(decisions[0], action="replace_with_trace")
        with self.assertRaisesRegex(ValueError, "separate drawn trace"):
            collect_feedback(packet, decisions)

    def test_metadata_and_eval_role_cannot_be_rewritten(self):
        packet = packet_fixture()
        decisions = copy.deepcopy(packet["cases"])
        decisions[2] = approved(decisions[2])
        decisions[2]["dataset_role"] = "training"
        with self.assertRaises(ValueError):
            collect_feedback(packet, decisions)
        packet["cases"][2]["dataset_role"] = "training"
        with self.assertRaisesRegex(ValueError, "E-series"):
            collect_feedback(packet, copy.deepcopy(packet["cases"]))

    def test_trace_bounds_unknown_ids_and_duplicate_record_ids(self):
        packet = packet_fixture()
        decisions = copy.deepcopy(packet["cases"])
        decisions[0] = approved(decisions[0], action="replace_with_trace")
        trace = {"case_id": "H001", "record_id": "one", "trace_kind": "observed_contour", "pixel_points": [[10., 11.], [30., 11.]]}
        for mutation in (dict(case_id="missing"), dict(pixel_points=[[0., 0.], [70., 0.]]), dict(pixel_points=[[0., 0.], [float("nan"), 1.]]), dict(trace_kind="unreviewed")):
            with self.assertRaises(ValueError):
                collect_feedback(packet, decisions, [{**trace, **mutation}])
        with self.assertRaises(ValueError):
            collect_feedback(packet, decisions, [trace, trace])

    def test_ignore_and_conflicting_pixels_do_not_become_labels(self):
        packet = packet_fixture()
        decisions = copy.deepcopy(packet["cases"])
        decisions[0] = approved(decisions[0])
        traces = [{"case_id": "H001", "record_id": "negative", "trace_kind": "hard_negative", "pixel_points": [[15., 10.], [20., 10.]]}]
        masks = [{"case_id": "H001", "record_id": "mask", "reason": "unreadable overlap", "pixel_rings": [[[21., 9.], [26., 9.], [26., 11.], [21., 11.], [21., 9.]]]}]
        feedback = collect_feedback(packet, decisions, traces, masks)
        with tempfile.TemporaryDirectory() as temp:
            result = rasterize_sparse_training_feedback(packet, feedback, Path(temp)/"labels")
            image = np.asarray(Image.open(Path(temp)/"labels"/result["tiles"][0]["path"]))
            self.assertTrue((image[10, 15:27] == 255).all())
            self.assertEqual(result["tiles"][0]["conflicting_pixels_ignored"], 6)

    def test_geometry_role_forgery_is_rejected_before_mask_export(self):
        packet = packet_fixture()
        decisions = copy.deepcopy(packet["cases"])
        decisions[2] = approved(decisions[2])
        feedback = collect_feedback(packet, decisions)
        feedback["geometry_references"][0]["dataset_role"] = "training"
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            rasterize_sparse_training_feedback(packet, feedback, Path(temp)/"labels")

    def test_subpixel_serialization_roundoff_only(self):
        packet = packet_fixture()
        case, tile = packet["cases"][0], packet["tiles"][0]
        geometry = copy.deepcopy(case["original_geometry"])
        geometry["coordinates"][0][0] += 1e-14
        self.assertTrue(source_geometry_matches(case, tile, geometry))
        geometry["coordinates"][0][0] += .001
        self.assertFalse(source_geometry_matches(case, tile, geometry))

    def test_model_readiness_cannot_fit_unapproved_or_eval_examples(self):
        packet = packet_fixture()
        feedback = collect_feedback(packet, copy.deepcopy(packet["cases"]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packet_path = root/"packet.json"
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            feedback["packet_sha256"] = sha256_file(packet_path)
            feedback_path = root/"feedback.json"
            feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
            result = prepare(packet_path, feedback_path, root/"training", fit_if_ready=True)
            self.assertFalse(result["model_fitted"])
            self.assertEqual(result["status"], "waiting_for_human_labels")
        decisions = copy.deepcopy(packet["cases"])
        decisions[2] = approved(decisions[2])
        feedback = collect_feedback(packet, decisions)
        feedback["training_labels"] = copy.deepcopy(feedback["evaluation_labels"])
        with self.assertRaises(ValueError):
            validate_training_labels(packet, feedback)

    def test_ready_synthetic_human_fixture_exercises_fit_without_eval_leakage(self):
        from histcontour_core.segment_review import FEATURE_NAMES, LogisticModel
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            legacy_path = root/"legacy.json"
            legacy = LogisticModel(FEATURE_NAMES, (0.,)*11, (1.,)*11, (0.,)*11, 0.)
            legacy_path.write_text(json.dumps({"status": "trained", "model": legacy.to_dict()}), encoding="utf-8")
            tiles, cases, decisions = [], [], []
            for group, sheet in enumerate(("173-buyeo", "174-cheongyang", "177-nonsan")):
                tile_id = f"fixture-{group}"
                raster = root/f"{tile_id}.png"
                image = Image.new("L", (64, 64), 255)
                draw = ImageDraw.Draw(image)
                for i in range(8):
                    draw.line([(10, 5+5*i), (30, 5+5*i)], fill=0, width=1)
                image.save(raster)
                tile = {"tile_id": tile_id, "sheet_id": sheet, "split": "development", "bounds": [0., 0., 64., 64.], "pixel_bounds": [0, 0, 64, 64],
                        "raster_path": raster.name, "source_raster_sha256": sha256_file(raster), "reference_vector_path": f"{tile_id}.geojson"}
                features = []
                for i in range(8):
                    number = len(cases)+1
                    case_id, sid, y = f"H{number:03}", f"S{number:03}", 5.+5*i
                    points = [[10., y], [30., y]]
                    geometry = {"type": "LineString", "coordinates": [[x+.5, 63.5-y] for x, y in points]}
                    case = {"case_id": case_id, "sample_id": sid, "dataset_role": "training", "tile_id": tile_id, "sheet_id": sheet,
                            "segment_uid": sid, "source_raster_sha256": tile["source_raster_sha256"], "original_geometry": geometry,
                            "original_geometry_sha256": geometry_digest(geometry), "pixel_points": points, "pixel_box": [0, 0, 64, 64]}
                    cases.append(case)
                    decisions.append(approved(case, "contour" if i < 4 else "road_river", "accept_original" if i < 4 else "reject"))
                    features.append({"type": "Feature", "properties": {"segment_uid": sid, "proposal_id": sid, "pixel_length": 20}, "geometry": geometry})
                vector = root/tile["reference_vector_path"]
                vector.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
                tile["source_vector_sha256"] = sha256_file(vector)
                tiles.append(tile)
            evaluation = {**copy.deepcopy(cases[0]), "case_id": "H025", "sample_id": "E001", "segment_uid": "E001", "dataset_role": "evaluation_only"}
            cases.append(evaluation)
            decisions.append(approved(evaluation))
            packet = {"schema": "jap-map-contour-human-packet/1", "holdout_used": False, "tiles": tiles, "cases": cases,
                      "legacy_model_path": legacy_path.name, "legacy_model_sha256": sha256_file(legacy_path)}
            feedback = collect_feedback(packet, decisions, minimum_per_class=2)
            packet_path, feedback_path = root/"packet.json", root/"feedback.json"
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            feedback["packet_sha256"] = sha256_file(packet_path)
            feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
            result = prepare(packet_path, feedback_path, root/"fitted-synthetic-fixture", fit_if_ready=True, minimum_per_class=2)
            self.assertTrue(result["model_fitted"])
            self.assertEqual(result["training_label_count"], 24)
            self.assertEqual(result["evaluation_labels_not_used"], 1)
            for fold in result["validation"]["folds"]:
                self.assertNotIn("H025", fold["training_sample_ids"])
                self.assertNotIn("H025", fold["test_sample_ids"])

    def test_forward_gate_uses_identity_not_just_positive_totals(self):
        rows = [{"sample_id": "E1", "class": "contour", "legacy_retained": True, "candidate_retained": False},
                {"sample_id": "E2", "class": "contour", "legacy_retained": False, "candidate_retained": True},
                {"sample_id": "E3", "class": "road_river", "legacy_retained": True, "candidate_retained": False}]
        result = comparison_gate(rows)
        self.assertEqual(result["additional_contour_misses"], ["E1"])
        self.assertFalse(result["criteria_met_on_provisional_forward_samples"])
        rows[0]["candidate_retained"] = True
        self.assertTrue(comparison_gate(rows)["criteria_met_on_provisional_forward_samples"])
        self.assertFalse(comparison_gate(rows)["automatic_promotion"])


if __name__ == "__main__":
    unittest.main()
