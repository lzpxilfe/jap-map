import copy
import unittest

import numpy as np

from scripts.refine_assisted_contour_curve import curved_revision, subtle_curve_offset


def cubic(controls, count):
    p = np.asarray(controls, dtype=float)
    t = np.linspace(0, 1, count)[:, None]
    return ((1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t*t*p[2]+t**3*p[3]).tolist()


class AssistedCurveRefinementTests(unittest.TestCase):
    def test_subtle_policy_caps_magnitude_but_never_guesses_bend_direction(self):
        self.assertEqual(subtle_curve_offset([[0, 0], [10, 0]], "positive"), .5)
        self.assertEqual(subtle_curve_offset([[0, 0], [4, 0]], "negative"), -.2)
        with self.assertRaises(ValueError):
            subtle_curve_offset([[0, 0], [10, 0]], None)

    def test_local_bulge_preserves_endpoints_tangents_and_original_samples(self):
        controls = [[0., 0.], [3., 3.], [7., 3.], [10., 0.]]
        original = cubic(controls, 12)
        before = copy.deepcopy(original)
        points, audit = curved_revision(original, 1.25)
        self.assertEqual(original, before)
        self.assertEqual(points[0], original[0])
        self.assertEqual(points[-1], original[-1])
        np.testing.assert_allclose(audit["endpoint_derivatives_xy"], [[9., 9.], [9., -9.]], atol=1e-10)
        baseline = np.asarray(cubic(controls, 81))
        np.testing.assert_allclose(np.asarray(points)[40]-baseline[40], [0., 1.25], atol=1e-10)
        self.assertAlmostEqual(audit["maximum_displacement_pixels"], 1.25)
        self.assertLess(audit["original_cubic_fit_max_error_pixels"], 1e-10)
        self.assertTrue(audit["forward_progress_monotone"])

    def test_zero_offset_and_opposite_offsets_have_expected_geometry(self):
        original = cubic([[0, 10], [1, 7], [1, 3], [0, 0]], 14)
        zero, _ = curved_revision(original, 0.)
        left, _ = curved_revision(original, -1.)
        right, _ = curved_revision(original, 1.)
        np.testing.assert_allclose((np.asarray(left)+right)/2, zero, atol=1e-10)
        self.assertGreater(right[40][0], zero[40][0])
        np.testing.assert_allclose(zero, cubic([[0, 10], [1, 7], [1, 3], [0, 0]], 81), atol=1e-10)

    def test_invalid_large_noncubic_or_looping_edits_are_rejected(self):
        original = cubic([[0, 0], [3, 3], [7, 3], [10, 0]], 12)
        for offset in (3., float("nan")):
            with self.assertRaises(ValueError):
                curved_revision(original, offset)
        with self.assertRaises(ValueError):
            curved_revision(original, 1., samples=80)
        with self.assertRaises(ValueError):
            curved_revision(original[:3], 1.)
        noncubic = copy.deepcopy(original)
        noncubic[5][0] += 2.
        with self.assertRaisesRegex(ValueError, "not an evenly sampled cubic"):
            curved_revision(noncubic, 1.)
        with self.assertRaisesRegex(ValueError, "doubles back"):
            curved_revision(cubic([[0, 0], [30, 0], [-20, 0], [10, 0]], 12), 1.)


if __name__ == "__main__":
    unittest.main()
