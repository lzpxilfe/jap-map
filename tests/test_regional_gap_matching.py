"""Synthetic matching contracts, not historical contour performance."""
import copy
from dataclasses import replace
import itertools
import math
import unittest

from histcontour_core.regional_gap_matching import RegionalGapMatchingConfig, match_endpoint_banks


def endpoint(source, x, y, tangent=(1., 0.), support=None):
    result = {"source_path_id": source, "point": [x, y], "outward_tangent": list(tangent)}
    if support is not None:
        result["support"] = support
    return result


def banks(ys_a, ys_b, distance=20):
    return ([endpoint(f"left-{i}", 0, y) for i, y in enumerate(ys_a)],
            [endpoint(f"right-{j}", distance, y, (-1, 0)) for j, y in enumerate(ys_b)])


def matches(hypothesis):
    return [(m["a_index"], m["b_index"]) for m in hypothesis["matches"]]


class RegionalGapMatchingTests(unittest.TestCase):
    def test_joint_alignment_avoids_a_nearest_pair_order_swap(self):
        a = [endpoint("a0", 0, 0), endpoint("a1", 20, 10)]
        b = [endpoint("b0", 30, 0, (-1, 0)), endpoint("b1", 21, 10, (-1, 0))]
        self.assertLess(math.dist(a[0]["point"], b[1]["point"]), math.dist(a[0]["point"], b[0]["point"]))
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        self.assertEqual(matches(result["best"]), [(0, 0), (1, 1)])
        self.assertFalse(result["human_approved"])

    def test_missing_middle_line_and_unequal_banks_skip_without_shifting_others(self):
        a, b = banks([0, 10, 20, 30], [0, 20, 30])
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1),
                                     config=replace(RegionalGapMatchingConfig(), maximum_lateral_displacement_px=4))
        self.assertEqual(matches(result["best"]), [(0, 0), (2, 1), (3, 2)])
        self.assertEqual(result["best"]["skipped_a_indices"], [1])
        self.assertEqual(result["best"]["skipped_b_indices"], [])

    def test_input_order_is_not_bank_order_and_endpoint_coordinates_are_exact(self):
        a, b = banks([20.125, .25, 10.5], [10.5, 20.125, .25])
        result = match_endpoint_banks(a, b, ordering_axis=(0, 8))
        self.assertEqual(matches(result["best"]), [(1, 2), (2, 0), (0, 1)])
        for match in result["best"]["matches"]:
            self.assertEqual(match["start"], a[match["a_index"]]["point"])
            self.assertEqual(match["end"], b[match["b_index"]]["point"])
            self.assertTrue(match["inferred_gap"])
            self.assertFalse(match["human_approved"])
            self.assertFalse(match["routing_performed"])

    def test_near_tie_and_competing_endpoint_are_explicit(self):
        a, b = banks([0], [-1, 1])
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        self.assertIsNotNone(result["runner_up"])
        self.assertAlmostEqual(result["cost_margin"], 0)
        self.assertNotEqual(matches(result["best"]), matches(result["runner_up"]))
        self.assertIn("near_tied_hypotheses", result["ambiguity_reasons"])
        self.assertIn("competing_endpoint_pair", result["ambiguity_reasons"])
        self.assertFalse(result["best"]["eligible_for_routing"])

    def test_runner_up_is_a_distinct_matching_not_another_skip_order(self):
        a, b = banks([0, 100], [0, 100])
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1),
                                     config=replace(RegionalGapMatchingConfig(), maximum_lateral_displacement_px=2))
        self.assertEqual(len(result["best"]["matches"]), 2)
        self.assertEqual(len(result["runner_up"]["matches"]), 1)
        self.assertAlmostEqual(result["cost_margin"], 2*RegionalGapMatchingConfig().skip_cost-20/96)

    def test_best_and_runner_up_match_exhaustive_small_alignments(self):
        a, b = banks([0, 13, 25], [1, 11, 28, 44])
        config = RegionalGapMatchingConfig()
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1), config=config)
        costs = {}
        # One-pair calls expose exactly the same public pair costs/gates.
        for i, aa in enumerate(a):
            for j, bb in enumerate(b):
                single = match_endpoint_banks([aa], [bb], ordering_axis=(0, 1), config=config)
                for hypothesis in (single["best"], single["runner_up"]):
                    if hypothesis and hypothesis["matches"]:
                        costs[i, j] = hypothesis["matches"][0]["cost"]
        all_alignments = []
        for k in range(min(len(a), len(b))+1):
            for ii in itertools.combinations(range(len(a)), k):
                for jj in itertools.combinations(range(len(b)), k):
                    selected = tuple(zip(ii, jj))
                    if all(pair in costs for pair in selected):
                        total = sum(costs[pair] for pair in selected)+(len(a)+len(b)-2*k)*config.skip_cost
                        all_alignments.append((total, selected))
        expected = sorted(all_alignments)
        for actual, wanted in zip((result["best"], result["runner_up"]), expected[:2]):
            self.assertAlmostEqual(actual["cost"], wanted[0])
            self.assertEqual(tuple(matches(actual)), wanted[1])

    def test_monotone_order_does_not_hide_geometric_chord_crossing(self):
        a = [endpoint("a0", -100, .5), endpoint("a1", 0, 1)]
        b = [endpoint("b0", 100, 2, (-1, 0)), endpoint("b1", 10, 3, (-1, 0))]
        config = replace(RegionalGapMatchingConfig(), maximum_pair_distance_px=256, skip_cost=2)
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1), config=config)
        self.assertEqual(matches(result["best"]), [(0, 0), (1, 1)])
        self.assertIn("straight_chords_cross_or_touch", result["ambiguity_reasons"])
        self.assertFalse(result["best"]["eligible_for_routing"])

    def test_duplicate_endpoint_reuse_and_same_source_loop_are_not_allowed(self):
        a, b = banks([0, 10], [0, 10])
        with self.assertRaisesRegex(ValueError, "same source endpoint"):
            match_endpoint_banks(a+[copy.deepcopy(a[0])], b, ordering_axis=(0, 1))
        with self.assertRaisesRegex(ValueError, "same source endpoint"):
            match_endpoint_banks(a, b+[copy.deepcopy(a[0])], ordering_axis=(0, 1))
        b[0]["source_path_id"] = a[0]["source_path_id"]
        result = match_endpoint_banks(a[:1], b[:1], ordering_axis=(0, 1))
        self.assertEqual(result["best"]["matches"], [])
        self.assertEqual(result["diagnostics"]["rejected_pairs"], {"same_source_path": 1})

    def test_every_endpoint_is_used_at_most_once_in_each_hypothesis(self):
        a, b = banks(range(8), range(6))
        result = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        for hypothesis in (result["best"], result["runner_up"]):
            ids = [identity for match in hypothesis["matches"] for identity in match["endpoint_ids"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_rotation_translation_and_axis_reversal_preserve_solution_costs(self):
        a, b = banks([0, 12, 26], [1, 13, 24])
        original = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        angle = .731
        def rotate(v):
            return [v[0]*math.cos(angle)-v[1]*math.sin(angle), v[0]*math.sin(angle)+v[1]*math.cos(angle)]
        aa, bb = copy.deepcopy(a), copy.deepcopy(b)
        for value in aa+bb:
            value["point"] = [v+t for v, t in zip(rotate(value["point"]), (70, -35))]
            value["outward_tangent"] = rotate(value["outward_tangent"])
        rotated = match_endpoint_banks(aa, bb, ordering_axis=rotate((0, 1)))
        reversed_axis = match_endpoint_banks(a, b, ordering_axis=(0, -1))
        self.assertEqual(matches(original["best"]), matches(rotated["best"]))
        self.assertEqual(matches(original["best"]), matches(reversed_axis["best"])[::-1])
        self.assertAlmostEqual(original["best"]["cost"], rotated["best"]["cost"])
        self.assertAlmostEqual(original["cost_margin"], rotated["cost_margin"])

    def test_direction_distance_order_and_boundary_cases_remain_explicit(self):
        a, b = banks([0], [0])
        b[0]["outward_tangent"] = [1, 0]
        rejected = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        self.assertEqual(rejected["diagnostics"]["rejected_pairs"], {"tangent_bound": 1})
        a, b = banks([0], [0], distance=95)
        boundary = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        self.assertIn("pair_near_distance_boundary", boundary["ambiguity_reasons"])
        a, b = banks([0, .1], [0, 10])
        tied = match_endpoint_banks(a, b, ordering_axis=(0, 1))
        self.assertIn("bank_order_tie", tied["ambiguity_reasons"])
        empty = match_endpoint_banks(a, [], ordering_axis=(0, 1))
        self.assertIsNone(empty["runner_up"])
        self.assertEqual(empty["best"]["matches"], [])
        self.assertIn("empty_endpoint_bank", empty["ambiguity_reasons"])

    def test_inputs_and_hypotheses_do_not_alias(self):
        a, b = banks([0, 12], [0, 12])
        a[0]["support"] = .8
        before = copy.deepcopy((a, b))
        result = match_endpoint_banks(a, b, ordering_axis=[0, 1])
        self.assertEqual((a, b), before)
        if result["runner_up"]["matches"]:
            old_runner = copy.deepcopy(result["runner_up"])
            result["best"]["matches"][0]["start"][0] = 999
            result["best"]["matches"][0]["cost_terms"]["distance"] = 999
            self.assertEqual(result["runner_up"], old_runner)
        self.assertEqual((a, b), before)

    def test_input_validation_and_caps_bound_allocation(self):
        a, b = banks([0], [0])
        for axis in ((0, 0), (math.nan, 1), (True, 1), (1,), "x"):
            with self.assertRaises(ValueError):
                match_endpoint_banks(a, b, ordering_axis=axis)
        for field, value in (("point", [math.inf, 0]), ("point", [1e10, 0]),
                             ("outward_tangent", [0, 0]), ("support", -1),
                             ("support", True), ("source_path_id", "")):
            invalid = copy.deepcopy(a)
            invalid[0][field] = value
            with self.assertRaises(ValueError):
                match_endpoint_banks(invalid, b, ordering_axis=(0, 1))
        for config in (replace(RegionalGapMatchingConfig(), maximum_endpoints_per_bank=True),
                       replace(RegionalGapMatchingConfig(), maximum_dp_cells=1),
                       replace(RegionalGapMatchingConfig(), maximum_pair_distance_px=1000),
                       replace(RegionalGapMatchingConfig(), skip_cost=1e308),
                       replace(RegionalGapMatchingConfig(), skip_cost=math.nan)):
            with self.assertRaises(ValueError):
                match_endpoint_banks(a, b, ordering_axis=(0, 1), config=config)
        a, b = banks(range(65), [0])
        with self.assertRaisesRegex(ValueError, "work cap"):
            match_endpoint_banks(a, b, ordering_axis=(0, 1))


if __name__ == "__main__":
    unittest.main()
