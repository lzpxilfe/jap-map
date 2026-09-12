import copy
from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from histcontour_core.contour_chains import (
    CHAIN_FEATURE_NAMES, PRIOR_CHAIN_FEATURE_NAMES, PRIOR_CONTEXT_FEATURE_NAMES,
    chain_features, fit_chain_classifier, legacy_log_odds, validate_chain_classifier, visible_chains,
)
from histcontour_core.contour_context import NeighborhoodFeatures, probability
from histcontour_core.provenance import sha256_file
from histcontour_core.vectorization import PixelLineProposal
from scripts.prepare_contour_semantic_samples import sample_indices
from scripts.run_contour_chain_development import select_development_candidate, validate_separation
from scripts.run_contour_context_experiment import metrics, validate_sample_labels


def proposals(*lines):
    return [PixelLineProposal(str(i), tuple(tuple(point) for point in line), sum(math.dist(a, b) for a, b in zip(line, line[1:])), 1.0) for i, line in enumerate(lines)]


def edges(lines):
    return Counter(tuple(sorted((a, b))) for line in lines for a, b in zip(line, line[1:]))


class TouchingChainTests(unittest.TestCase):
    def test_touching_straight_arms_and_side_branch(self):
        source = proposals([(0, 16), (16, 16)], [(16, 16), (32, 16)], [(16, 16), (16, 28)])
        original = copy.deepcopy(source)
        chains, audit = visible_chains(source)
        self.assertEqual(sorted(len(chain.member_indices) for chain in chains), [1, 2])
        self.assertEqual(audit["paired_junctions"], 1)
        self.assertEqual(edges([line.points for line in source]), edges([chain.points for chain in chains]))
        self.assertEqual(original, source)
        self.assertEqual(audit["gap_connections_created"], 0)

    def test_gap_four_way_and_ambiguous_junction_are_never_paired(self):
        fixtures = [
            proposals([(0, 0), (16, 0)], [(17, 0), (32, 0)]),
            proposals([(0, 16), (16, 16)], [(16, 16), (32, 16)], [(16, 16), (16, 0)], [(16, 16), (16, 32)]),
            proposals([(16, 16), (32, 16)], [(16, 16), (0, 14)], [(16, 16), (0, 18)]),
        ]
        for source in fixtures:
            chains, audit = visible_chains(source)
            self.assertEqual(len(chains), len(source))
            self.assertEqual(audit["paired_junctions"], 0)

    def test_short_side_evidence_but_not_short_through_arm(self):
        source = proposals([(0, 16), (16, 16)], [(16, 16), (32, 16)], [(16, 16), (16, 18)])
        self.assertEqual(visible_chains(source)[1]["paired_junctions"], 1)
        too_short = proposals([(0, 16), (16, 16)], [(16, 16), (20, 16)], [(16, 16), (16, 28)])
        self.assertEqual(visible_chains(too_short)[1]["paired_junctions"], 0)

    def test_closed_input_and_cycle_partition_preserve_all_edges(self):
        ring = [(32+24*math.cos(i*math.pi/16), 32+24*math.sin(i*math.pi/16)) for i in range(33)]
        ring[-1] = ring[0]
        source = proposals(ring[:17], ring[16:])
        chains, _ = visible_chains(source, maximum_deviation=30)
        self.assertEqual(sorted(index for chain in chains for index in chain.member_indices), [0, 1])
        self.assertEqual(edges([line.points for line in source]), edges([chain.points for chain in chains]))
        single = proposals([(10, 10), (20, 10), (20, 20), (10, 10)])
        chains, _ = visible_chains(single)
        self.assertEqual(chains[0].points, single[0].points)

    def test_isolated_delta_features_are_exactly_zero(self):
        image = np.full((96, 96), 255, np.uint8)
        image[48, 10:86] = 0
        source = proposals([(10, 48), (85, 48)])
        cache = NeighborhoodFeatures(image)
        descriptors, chains, audit = chain_features(source, cache)
        self.assertEqual(len(chains), 1)
        self.assertTrue(all(value == 0.0 for value in descriptors[0].values()))
        self.assertEqual(audit["members_in_multi_member_chains"], 0)

    def test_invalid_input_and_model_contracts(self):
        for source in (proposals([(1, 1), (1, 1)]), proposals([(0, 0), (float("nan"), 1)])):
            with self.assertRaises(ValueError):
                visible_chains(source)
        with self.assertRaises(ValueError):
            visible_chains([], tangent_distance=0)
        self.assertEqual(legacy_log_odds(0), -6)
        self.assertEqual(legacy_log_odds(1), 6)
        self.assertAlmostEqual(legacy_log_odds(.5), 0)
        for value in (float("nan"), -1, True):
            with self.assertRaises(ValueError):
                legacy_log_odds(value)
        rows = [{"class": "contour" if i < 4 else "road_river", **{name: float(i) for name in PRIOR_CHAIN_FEATURE_NAMES}} for i in range(8)]
        model = validate_chain_classifier(fit_chain_classifier(rows, feature_names=PRIOR_CHAIN_FEATURE_NAMES))
        self.assertGreater(probability(model, rows[0]), probability(model, rows[-1]))
        for mutation in ({"feature_schema": "wrong"}, {"feature_names": ["sheet_id"]}, {"intercept": float("nan")}, {"scales": [0]*len(PRIOR_CHAIN_FEATURE_NAMES)}):
            with self.assertRaises(ValueError):
                validate_chain_classifier({**model, **mutation})


class ForwardSeparationTests(unittest.TestCase):
    def test_exclusion_does_not_select_old_identities(self):
        features = [{"properties": {"pixel_length": length+i, "segment_uid": f"{length}-{i}"}} for length in (18, 30, 100) for i in range(8)]
        old = sample_indices(features, seed=7, per_bin=2)
        excluded = {features[i]["properties"]["segment_uid"] for _, i in old}
        new = sample_indices(features, seed=8, per_bin=2, excluded_ids=excluded)
        self.assertFalse(excluded & {features[i]["properties"]["segment_uid"] for _, i in new})

    def test_equal_retained_count_cannot_hide_additional_missed_contours(self):
        baseline = [{"sample_id": "A", "class": "contour", "retained": True}, {"sample_id": "B", "class": "contour", "retained": False},
                    {"sample_id": "N", "class": "road_river", "retained": True}]
        wrong = [{**baseline[0], "retained": False}, {**baseline[1], "retained": True}, {**baseline[2], "retained": False}]
        good = [{**row, "retained": row["class"] == "contour"} for row in baseline]
        results = {"wrong": {"outer_predictions": wrong, "metrics": metrics(wrong), "final_model": {"feature_names": list(PRIOR_CONTEXT_FEATURE_NAMES)}},
                   "good": {"outer_predictions": good, "metrics": metrics(good), "final_model": {"feature_names": list(CHAIN_FEATURE_NAMES)}}}
        self.assertEqual(select_development_candidate(results, baseline), "good")
        self.assertEqual(results["wrong"]["additional_development_contour_misses_vs_legacy"], ["A"])

    def test_forward_manifest_is_bound_disjoint_and_rejected_for_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vector_path = root/"vectors.json"
            vector_path.write_text("{}", encoding="utf-8")
            training_path, forward_path, labels_path = root/"train.json", root/"forward.json", root/"labels.json"
            training = {"vector_index_sha256": sha256_file(vector_path), "source_index_sha256": "x", "samples": [{"segment_uid": "old"}]}
            training_path.write_text(json.dumps(training), encoding="utf-8")
            forward = {"role": "forward_development_evaluation", "holdout_used": False, "excluded_manifest_sha256": sha256_file(training_path),
                       "source_index_sha256": "x", "vector_index_sha256": sha256_file(vector_path), "samples": [{"segment_uid": "new"}]}
            forward_path.write_text(json.dumps(forward), encoding="utf-8")
            labels_path.write_text("{}", encoding="utf-8")
            validate_separation(training_path, forward_path, vector_path)
            with self.assertRaisesRegex(ValueError, "forward-evaluation-only"):
                validate_sample_labels(forward_path, labels_path)
            forward["samples"][0]["segment_uid"] = "old"
            forward_path.write_text(json.dumps(forward), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                validate_separation(training_path, forward_path, vector_path)


if __name__ == "__main__":
    unittest.main()
