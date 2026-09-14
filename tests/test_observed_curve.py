import copy
from dataclasses import replace
import unittest

import numpy as np

from histcontour_core.observed_curve import ObservedCurveConfig, refine_observed_curve


def graph_ink(function, *, shape=(120, 140), contrast=210., sigma=.65):
    yy, xx = np.indices(shape, dtype=float)
    return 245. - contrast * np.exp(-.5 * ((yy - function(xx)) / sigma) ** 2)


def sampled_original(result):
    points = np.asarray(result["original_points"])
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.column_stack([np.interp(result["point_source_distances"], distance, points[:, axis]) for axis in (0, 1)])


class ObservedCurveTests(unittest.TestCase):
    def test_straight_jitter_fits_source_centres_on_a_full_length_route(self):
        image = graph_ink(lambda x: 50.)
        x = np.arange(10., 121.)
        points = np.c_[x, 50 + .35 * np.sin(2.1 * x)]
        result = refine_observed_curve(image, points)
        candidate = np.asarray(result["points"])
        self.assertTrue(result["audit"]["changed"])
        self.assertGreater(result["audit"]["before"]["length_pixels"], 100)
        self.assertLess(np.std(candidate[10:-10, 1] - 50), .02)
        self.assertLess(result["audit"]["after"]["total_turn_degrees"], .1 * result["audit"]["before"]["total_turn_degrees"])
        np.testing.assert_array_equal(candidate[[0, -1]], points[[0, -1]])
        self.assertEqual(result["audit"]["source_support_fraction"], 1.)

    def test_real_s_bends_are_retained_and_jitter_removed(self):
        curve = lambda x: 55 + 12 * np.sin((x - 10) * 2 * np.pi / 110)
        image = graph_ink(curve)
        x = np.arange(10., 121.)
        points = np.c_[x, curve(x) + .3 * np.sin(x * 2.2)]
        result = refine_observed_curve(image, points)
        candidate = np.asarray(result["points"])
        interior = candidate[10:-10]
        self.assertLess(np.sqrt(np.mean((interior[:, 1] - curve(interior[:, 0])) ** 2)), .12)
        self.assertGreater(candidate[:, 1].max(), 66.5)
        self.assertLess(candidate[:, 1].min(), 43.5)
        self.assertLess(result["audit"]["after"]["total_turn_degrees"], result["audit"]["before"]["total_turn_degrees"])

    def test_sparse_route_gains_source_supported_curvature_not_just_more_vertices(self):
        curve = lambda x: 50 + 1.1 * np.sin((x - 10) * 2 * np.pi / 110)
        image = graph_ink(curve)
        points = [[10., 50.], [65., 50.], [120., 50.]]
        result = refine_observed_curve(image, points)
        candidate = np.asarray(result["points"])
        self.assertGreater(candidate[:, 1].max(), 50.85)
        self.assertLess(candidate[:, 1].min(), 49.15)
        self.assertLess(np.sqrt(np.mean((candidate[:, 1] - curve(candidate[:, 0])) ** 2)), .12)

    def test_sharp_supported_valley_survives_with_its_depth_and_flanks(self):
        valley = lambda x: 35 + .85 * abs(x - 65)
        image = graph_ink(valley, sigma=.8)
        x = np.arange(15., 116.)
        points = np.c_[x, valley(x) + .22 * np.sin(2.1 * x)]
        result = refine_observed_curve(image, points)
        candidate = np.asarray(result["points"])
        self.assertLess(candidate[:, 1].min(), 35.5)
        self.assertLess(np.mean(abs(candidate[10:-10, 1] - valley(candidate[10:-10, 0]))), .18)
        self.assertGreater(candidate[0, 1] - candidate[:, 1].min(), 41)
        self.assertGreater(candidate[-1, 1] - candidate[:, 1].min(), 41)

    def test_faint_single_stroke_uses_local_contrast(self):
        x = np.arange(10., 121.)
        points = np.c_[x, 50 + .3 * np.sin(x * 2.1)]
        dark = refine_observed_curve(graph_ink(lambda x: 50.), points)
        faint = refine_observed_curve(graph_ink(lambda x: 50., contrast=14), points)
        self.assertTrue(faint["audit"]["changed"])
        np.testing.assert_allclose(faint["points"], dark["points"], atol=1e-7)
        too_faint = refine_observed_curve(graph_ink(lambda x: 50., contrast=3), points)
        self.assertFalse(too_faint["audit"]["changed"])
        self.assertEqual(too_faint["audit"]["source_support_fraction"], 0)

    def test_parallel_double_peaks_never_attract_a_route_to_their_midpoint(self):
        image = np.minimum(graph_ink(lambda x: 49.), graph_ink(lambda x: 52.))
        for y in (49., 50.5):
            points = [[10., y], [65., y + .2], [120., y]]
            result = refine_observed_curve(image, points)
            self.assertFalse(result["audit"]["changed"])
            self.assertEqual(result["audit"]["source_support_fraction"], 0.)
            self.assertGreater(result["audit"]["abstentions"].get("ambiguous_multiple_peaks", 0), 0)
            np.testing.assert_allclose(result["points"], sampled_original(result), atol=1e-12)

    def test_darker_stroke_truncated_at_profile_boundary_cannot_crash_half_width(self):
        # The central peak is 0.45--0.5 after the darker boundary peak sets
        # normalization. Before the fix, its absolute half-height set was
        # empty and _profiles raised IndexError instead of abstaining.
        for boundary_y in (46., 54.):
            image = np.minimum(graph_ink(lambda x: 50., contrast=105., sigma=.6),
                               graph_ink(lambda x: boundary_y, sigma=.6))
            result = refine_observed_curve(image, [[10., 50.], [120., 50.]])
            self.assertEqual(result["audit"]["source_support_fraction"], 0.)
            self.assertFalse(result["audit"]["changed"])
            self.assertGreater(result["audit"]["abstentions"].get("ambiguous_truncated_boundary_ink", 0), 0)
            np.testing.assert_array_equal(result["points"], sampled_original(result))

    def test_truncated_neighbour_is_not_false_single_lobe_support(self):
        # find_peaks omits endpoint peaks. A central peak above half-height
        # previously avoided the crash but falsely appeared unambiguous.
        for boundary_y in (46., 54.):
            image = np.minimum(graph_ink(lambda x: 50., contrast=150., sigma=.6),
                               graph_ink(lambda x: boundary_y, sigma=.6))
            result = refine_observed_curve(image, [[10., 50.], [120., 50.]])
            self.assertEqual(result["audit"]["source_support_fraction"], 0.)
            self.assertTrue(all(not span["observed_source_support"] for span in result["spans"]))
            self.assertFalse(result["audit"]["changed"])

    def test_adjacent_glyph_stroke_splits_the_supported_fit(self):
        image = graph_ink(lambda x: 50.)
        yy, xx = np.indices(image.shape, dtype=float)
        glyph = 245 - 210 * np.exp(-.5 * ((yy - 53.) / .6) ** 2)
        image[:, 54:69] = np.minimum(image[:, 54:69], glyph[:, 54:69])
        x = np.arange(10., 121.)
        points = np.c_[x, 50 + .28 * np.sin(x * 2.1)]
        result = refine_observed_curve(image, points)
        candidate = np.asarray(result["points"])
        base = sampled_original(result)
        contaminated = (base[:, 0] >= 55) & (base[:, 0] <= 68)
        np.testing.assert_allclose(candidate[contaminated], base[contaminated], atol=1e-12)
        self.assertTrue(result["audit"]["changed"])
        self.assertGreater(result["audit"]["abstentions"].get("ambiguous_multiple_peaks", 0), 0)

    def test_long_white_gap_is_unchanged_and_explicitly_not_observed(self):
        image = graph_ink(lambda x: 50.)
        image[:, 45:90] = 245
        x = np.arange(10., 121.)
        points = np.c_[x, 50 + .3 * np.sin(x * 2.1)]
        result = refine_observed_curve(image, points)
        candidate = np.asarray(result["points"]); base = sampled_original(result)
        blank = (base[:, 0] >= 46) & (base[:, 0] <= 88)
        np.testing.assert_allclose(candidate[blank], base[blank], atol=1e-12)
        unsupported_length = sum(span["source_distance_end"] - span["source_distance_start"]
                                 for span in result["spans"] if not span["observed_source_support"])
        self.assertGreater(unsupported_length, 40)
        self.assertTrue(result["audit"]["changed"])
        for span in result["spans"]:
            if not span["observed_source_support"]:
                self.assertFalse(span["changed"])

    def test_anchors_and_approved_spans_preserve_exact_original_coordinates(self):
        image = graph_ink(lambda x: 50.)
        x = np.arange(10., 121.)
        points = np.c_[x, 50 + .3 * np.sin(x * 2.1)]
        result = refine_observed_curve(image, points, anchor_indices=[20], locked_indices=[75], locked_spans=[(40, 60)])
        candidate = np.asarray(result["points"])
        indices = result["original_vertex_indices"]
        for index in [0, 20, 75, len(points) - 1, *range(40, 61)]:
            np.testing.assert_array_equal(candidate[indices[index]], points[index])
        s = np.asarray(result["point_source_distances"])
        interval = (s >= s[indices[40]]) & (s <= s[indices[60]])
        np.testing.assert_allclose(candidate[interval], sampled_original(result)[interval], atol=1e-12)
        self.assertTrue(result["audit"]["changed"])

    def test_fully_approved_path_is_exactly_unchanged_and_not_relabelled(self):
        image = graph_ink(lambda x: 50.)
        points = [[10., 50.1], [10., 50.1], [60., 49.8], [120., 50.2]]
        result = refine_observed_curve(image, points, fully_approved=True)
        self.assertEqual(result["points"], points)
        self.assertFalse(result["audit"]["changed"])
        self.assertNotIn("human_approved", result)
        self.assertNotIn("label", result)
        self.assertEqual(result["audit"]["abstentions"], {"fully_locked_input": 1})

    def test_stroke_width_and_nearby_line_clearance_bound_movement(self):
        image = graph_ink(lambda x: 50.)
        points = [[10., 50.6], [60., 50.6], [120., 50.6]]
        normal = refine_observed_curve(image, points)
        self.assertTrue(normal["audit"]["changed"])
        restricted = refine_observed_curve(image, points, clearance=1.)
        self.assertFalse(restricted["audit"]["changed"])
        self.assertGreater(restricted["audit"]["abstentions"].get("centre_outside_movement_bound", 0), 0)
        field = np.full_like(image, 1.5)
        field[:, :40] = 8.
        varying = refine_observed_curve(image, points, clearance=field)
        self.assertLessEqual(varying["audit"]["maximum_displacement_pixels"], ObservedCurveConfig().maximum_normal_shift)
        candidate = np.asarray(varying["points"]); base = sampled_original(varying)
        np.testing.assert_allclose(candidate[base[:, 0] > 42], base[base[:, 0] > 42], atol=1e-12)
        far = refine_observed_curve(image, [[10., 53.], [120., 53.]])
        self.assertFalse(far["audit"]["changed"])

    def test_border_profiles_abstain_without_out_of_bounds_sampling(self):
        image = graph_ink(lambda x: 1.5)
        result = refine_observed_curve(image, [[10., 1.5], [120., 1.5]])
        self.assertFalse(result["audit"]["changed"])
        self.assertGreater(result["audit"]["abstentions"].get("source_border", 0), 0)
        self.assertTrue(np.isfinite(result["points"]).all())

    def test_right_angle_rotation_and_direction_reversal_are_equivariant(self):
        image = graph_ink(lambda x: 50., shape=(140, 140))
        x = np.arange(10., 121.)
        points = np.c_[x, 50 + .3 * np.sin(x * 2.1)]
        result = refine_observed_curve(image, points)
        rotated_points = np.c_[points[:, 1], 139 - points[:, 0]]
        rotated = refine_observed_curve(np.rot90(image), rotated_points)
        expected = np.c_[np.array(result["points"])[:, 1], 139 - np.array(result["points"])[:, 0]]
        np.testing.assert_allclose(rotated["points"], expected, atol=1e-8)
        reverse = refine_observed_curve(image, points[::-1])
        # Arc-distance grids can differ on reversal. Compare the same locus.
        reverse_points = np.array(reverse["points"])[::-1]
        candidate = np.array(result["points"])
        np.testing.assert_allclose(np.interp(candidate[5:-5, 0], reverse_points[:, 0], reverse_points[:, 1]), candidate[5:-5, 1], atol=.015)

    def test_oblique_straight_stroke_is_fitted_in_local_normals(self):
        yy, xx = np.indices((140, 140), dtype=float)
        angle = .61; tangent = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        origin = np.array([18., 30.])
        distance = (xx - origin[0]) * normal[0] + (yy - origin[1]) * normal[1]
        image = 245 - 210 * np.exp(-.5 * (distance / .8) ** 2)
        s = np.arange(0., 111.)
        points = origin + s[:, None] * tangent + (.25 * np.sin(2.1 * s))[:, None] * normal
        result = refine_observed_curve(image, points)
        candidate = np.array(result["points"])
        self.assertLess(np.std((candidate[10:-10] - origin) @ normal), .06)

    def test_arrays_and_lists_are_not_mutated_and_results_are_independent(self):
        image = graph_ink(lambda x: 50.)
        points = [[10., 50.2], [60., 49.9], [120., 50.2]]
        original_image = image.copy(); original_points = copy.deepcopy(points)
        clearance = np.full_like(image, 10.); original_clearance = clearance.copy()
        result = refine_observed_curve(image, points, clearance=clearance)
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(clearance, original_clearance)
        self.assertEqual(points, original_points)
        result["original_points"][0][0] = 999
        result["points"][0][0] = 888
        self.assertEqual(points, original_points)

    def test_closed_routes_and_duplicate_vertices_remain_well_defined(self):
        yy, xx = np.indices((100, 100), dtype=float)
        image = 245 - 210 * np.exp(-.5 * ((np.hypot(xx - 50, yy - 50) - 25) / .8) ** 2)
        t = np.linspace(0, 2 * np.pi, 101)
        points = np.c_[50 + 25 * np.cos(t), 50 + 25 * np.sin(t)]
        points[-1] = points[0]
        points = np.insert(points, 21, points[20], axis=0)
        result = refine_observed_curve(image, points)
        self.assertEqual(result["points"][0], result["points"][-1])
        self.assertTrue(np.isfinite(result["points"]).all())
        self.assertLess(result["audit"]["maximum_displacement_pixels"], .4)

    def test_invalid_inputs_and_bounded_work_budget(self):
        image = graph_ink(lambda x: 50.)
        points = [[10., 50.], [120., 50.]]
        for kwargs in ({"anchor_indices": [-1]}, {"locked_indices": [True]}, {"locked_spans": [(1, 0)]},
                       {"locked_spans": [(0, 2)]}, {"fully_approved": 1}, {"clearance": -1},
                       {"clearance": np.zeros((2, 2))}, {"clearance": float("nan")}):
            with self.assertRaises(ValueError):
                refine_observed_curve(image, points, **kwargs)
        for invalid in ([], [[1., 1.], [1., 1.]], [[-1., 50.], [20., 50.]], [[1., 1.], [np.nan, 4.]]):
            with self.assertRaises(ValueError):
                refine_observed_curve(image, invalid)
        for invalid_image in (np.zeros((0, 0)), np.zeros((5, 5, 3)), image * np.nan, image + 300):
            with self.assertRaises(ValueError):
                refine_observed_curve(invalid_image, points)
        for config in (replace(ObservedCurveConfig(), sample_spacing=2),
                       replace(ObservedCurveConfig(), clearance_fraction=.6),
                       replace(ObservedCurveConfig(), maximum_normal_shift=5),
                       replace(ObservedCurveConfig(), maximum_samples=True)):
            with self.assertRaises(ValueError):
                refine_observed_curve(image, points, config=config)
        budget = refine_observed_curve(image, points, config=replace(ObservedCurveConfig(), maximum_samples=16))
        self.assertEqual(budget["points"], points)
        self.assertEqual(budget["audit"]["abstentions"], {"sample_budget_exceeded": 1})


if __name__ == "__main__":
    unittest.main()
