import copy
from dataclasses import replace
import unittest

import numpy as np

from histcontour_core.gap_refinement import GapRefinementConfig, propose_gap_refinement, regularize_gap_path, shape_audit
from histcontour_core.manual_gap_bridge import ManualGapBridgeConfig, build_manual_gap_bridge


def synthetic_ink(contrast=210, *, parallel=False):
    yy, xx = np.indices((90, 100), dtype=float)
    ink = np.exp(-.5*((yy-40)/.65)**2)
    if parallel:
        ink = np.maximum(ink, np.exp(-.5*((yy-43)/.65)**2))
    gray = 245-contrast*ink
    gray[:, 31:40] = 245  # label blank is never ink evidence
    source = [[8., 40.], [28., 40.], [30., 41.2]]
    target = [[40., 38.8], [42., 40.], [64., 40.]]
    return gray, source, target, [source[-1], target[0]]


class GapRefinementTests(unittest.TestCase):
    def test_tiny_s_bend_is_removed_with_exact_endpoints_and_bound(self):
        t = np.linspace(0, 1, 41)
        original = np.c_[10+10*t, 40+.15*np.sin(2*np.pi*t)].tolist()
        before = copy.deepcopy(original)
        new, audit = regularize_gap_path(original)
        self.assertTrue(audit["changed"])
        self.assertEqual(new[0], original[0]); self.assertEqual(new[-1], original[-1])
        self.assertEqual(audit["after"]["inflection_count"], 0)
        self.assertLessEqual(audit["maximum_displacement_pixels"], .35)
        self.assertEqual(original, before)
        # Audit the saved polyline, not only its ideal quadratic curve.
        x = np.linspace(10,20,10001)
        difference = abs(np.interp(x,np.array(new)[:,0],np.array(new)[:,1])-np.interp(x,np.array(original)[:,0],np.array(original)[:,1]))
        self.assertLessEqual(difference.max(),audit["maximum_displacement_pixels"]+1e-10)

    def test_real_single_bend_and_large_s_and_long_label_are_not_flattened(self):
        t = np.linspace(0, 1, 51)
        for x, y in ((10*t, 4*t*(1-t)), (10*t, 2*np.sin(2*np.pi*t)), (40*t, .1*np.sin(2*np.pi*t))):
            points = np.c_[x, y].tolist()
            new, audit = regularize_gap_path(points)
            self.assertFalse(audit["changed"])
            self.assertEqual(new, points)

    def test_smoothing_is_rotation_and_direction_independent(self):
        t = np.linspace(0, 1, 35); p = np.c_[10*t, .1*np.sin(2*np.pi*t)]
        rotation = np.array([[.6, -.8], [.8, .6]])
        result, _ = regularize_gap_path(p)
        transformed, _ = regularize_gap_path((p@rotation+[20,30])[::-1])
        np.testing.assert_allclose(np.array(transformed)[::-1], np.array(result)@rotation+[20,30], atol=1e-10)

    def test_duplicate_vertices_do_not_crash_and_closed_or_nonfinite_is_rejected(self):
        self.assertEqual(shape_audit([[1,1],[1,1],[2,2]])["detour_ratio"], 1)
        for points in ([], [[0,0],[0,0]], [[0,0],[1,1],[0,0]], [[0,0],[float('nan'),1]]):
            with self.assertRaises(ValueError):
                regularize_gap_path(points)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"maximum_gap_pixels": 17}, {"maximum_tail_trim": 20}, {"minimum_samples_per_side": 2},
                       {"minimum_contrast": 0}, {"profile_radius": float('nan')}, {"maximum_fixed_displacement": True}):
            with self.assertRaises(ValueError):
                replace(GapRefinementConfig(), **kwargs).validate()

    def test_connected_ink_reanchors_tips_but_never_approves(self):
        gray, source, target, points = synthetic_ink()
        before = copy.deepcopy((source, target, points)); pixels = gray.copy()
        result = propose_gap_refinement(gray, source, target, points)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["kind"], "local_tail_replacement")
        self.assertTrue(result["requires_source_tail_replacement"])
        self.assertFalse(result["append_only_safe"])
        self.assertFalse(result["human_approved"]); self.assertFalse(result["model_fitted"])
        middle = [p for p in result["points"] if 31 <= p[0] <= 39]
        self.assertLess(max(abs(p[1]-40) for p in middle), .3)
        self.assertEqual(result["support"]["gap_pixels_sampled_as_ink"], 0)
        self.assertEqual(before, (source, target, points)); np.testing.assert_array_equal(gray, pixels)

    def test_faint_stroke_uses_local_contrast_not_absolute_darkness(self):
        dark = propose_gap_refinement(*synthetic_ink())
        faint = propose_gap_refinement(*synthetic_ink(contrast=14))
        self.assertEqual(faint["status"], "proposed")
        np.testing.assert_allclose(faint["points"], dark["points"], atol=.001)

    def test_ambiguous_parallel_strokes_abstain(self):
        result = propose_gap_refinement(*synthetic_ink(parallel=True))
        self.assertEqual(result["status"], "context_review_required")
        self.assertIn("insufficient_or_ambiguous_connected_ink", result["reasons"])

    def test_blank_and_low_contrast_and_border_support_abstain(self):
        gray, source, target, points = synthetic_ink()
        for image in (np.full_like(gray, 245), synthetic_ink(contrast=3)[0]):
            result = propose_gap_refinement(image, source, target, points)
            self.assertEqual(result["status"], "context_review_required")
            self.assertEqual(result["points"], points)
        shifted = [np.asarray(p)-[0,38] for p in (source, target, points)]
        result = propose_gap_refinement(gray, *shifted)
        self.assertEqual(result["status"], "context_review_required")

    def test_printed_ink_inside_blank_cannot_attract_fit(self):
        gray, source, target, points = synthetic_ink()
        first = propose_gap_refinement(gray, source, target, points)
        gray[25:55, 32:38] = 0
        second = propose_gap_refinement(gray, source, target, points)
        self.assertEqual(first, second)

    def test_relocating_to_opposite_side_of_existing_bend_is_not_cleanup(self):
        gray, source, target, _ = synthetic_ink()
        source[-1][1] = 41.2; target[0][1] = 41.2
        t = np.linspace(0,1,41)
        points = np.c_[30+10*t,41.2+4*t*(1-t)*.6]
        result = propose_gap_refinement(gray,source,target,points)
        self.assertEqual(result["status"],"context_review_required")
        self.assertIn("local_fit_reverses_existing_bend",result["reasons"])

    def test_long_gap_requires_regional_review(self):
        gray, source, target, points = synthetic_ink()
        target = (np.asarray(target)+[20,0]).tolist(); points = [source[-1],target[0]]
        result = propose_gap_refinement(gray, source, target, points)
        self.assertEqual(result["reasons"], ["long_gap_requires_regional_anchor_review"])
        self.assertEqual(result["points"], points)

    def test_wrong_endpoint_outside_raster_or_empty_image_is_rejected(self):
        gray, source, target, points = synthetic_ink()
        for image, s, p in ((gray, source, [[29,41.2],points[-1]]),
                            (gray, source, [[-1,41.2],points[-1]]),
                            (np.zeros((0,0)), source, points)):
            with self.assertRaises(ValueError):
                propose_gap_refinement(image, s, target, p)

    def test_qgis_builder_uses_short_gap_cleanup_without_reanchoring(self):
        old = build_manual_gap_bridge((20,40),(30,40),(1,.2),(1,.2), config=ManualGapBridgeConfig(regularize_short_gaps=False))
        new = build_manual_gap_bridge((20,40),(30,40),(1,.2),(1,.2))
        self.assertIsNone(old.geometry_refinement)
        self.assertTrue(new.geometry_refinement["changed"])
        self.assertEqual(new.points[0], old.points[0]); self.assertEqual(new.points[-1],old.points[-1])
        self.assertEqual(shape_audit(new.points)["inflection_count"], 0)
        long = build_manual_gap_bridge((20,40),(60,40),(1,.2),(1,.2))
        self.assertIsNone(long.geometry_refinement)


if __name__ == "__main__":
    unittest.main()
