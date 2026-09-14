"""Synthetic evidence contracts, not semantic contour accuracy measurements."""

import dataclasses
import math
from types import SimpleNamespace
import unittest

import numpy as np

from histcontour_core.directional_trace import DirectionalTraceConfig, DirectionalTraceError, trace_observed_ink
from histcontour_core.ink import InkCenterlineResult, ink_centerline_candidates


def evidence_from_support(support, tx=None, ty=None, coherence=None):
    support = np.asarray(support, dtype=np.float32)
    shape = support.shape
    centre = support > 0
    return InkCenterlineResult(
        centerline=centre, center_score=np.ones(shape, dtype=np.float32), support_score=support,
        scale_px=np.ones(shape, dtype=np.float32), tangent_x=np.ones(shape) if tx is None else tx,
        tangent_y=np.zeros(shape) if ty is None else ty, coherence=np.ones(shape) if coherence is None else coherence,
        centerline_fraction=float(centre.mean()),
    )


def painted_path(shape, points, value=.7):
    support, tx, ty = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    for first, second in zip(points, points[1:]):
        length = math.dist(first, second)
        for fraction in np.linspace(0., 1., max(2, int(math.ceil(length * 4.)) + 1)):
            x, y = (int(round(a + fraction * (b - a))) for a, b in zip(first, second))
            support[y, x] = value
            if length:
                tx[y, x], ty[y, x] = (second[0] - first[0]) / length, (second[1] - first[1]) / length
    return evidence_from_support(support, tx, ty)


class DirectionalTraceTest(unittest.TestCase):
    def test_white_image_abstains_even_when_center_score_is_saturated(self):
        evidence = evidence_from_support(np.zeros((24, 40)))
        result = trace_observed_ink(evidence, (3, 12), (36, 12))
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.audit.reason, "no_ink_support")
        self.assertEqual(result.points, ())

    def test_single_white_vertex_is_reported_not_lost_between_samples(self):
        support = np.zeros((24, 40))
        support[12, 3:37] = 1.
        support[12, 20] = 0.
        result = trace_observed_ink(evidence_from_support(support), (3, 12), (36, 12))
        self.assertEqual(result.status, "supported_with_gaps")
        self.assertEqual(len(result.audit.unsupported_spans), 1)
        self.assertAlmostEqual(result.audit.maximum_unsupported_run_px, .16, places=6)
        self.assertAlmostEqual(result.audit.unsupported_spans[0].start_distance_px, 16.92, places=6)
        self.assertLess(result.audit.supported_length_fraction, 1.)
        strict = trace_observed_ink(evidence_from_support(support), (3, 12), (36, 12),
                                    config=DirectionalTraceConfig(max_unsupported_run_px=0.))
        self.assertEqual(strict.status, "abstained")

    def test_long_white_label_gap_abstains(self):
        support = np.zeros((24, 40))
        support[12, 3:16] = support[12, 24:37] = 1.
        result = trace_observed_ink(evidence_from_support(support), (3, 12), (36, 12))
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.audit.reason, "no_supported_route")

    def test_diagonal_gap_budget_counts_only_actual_unsupported_distance(self):
        support = np.zeros((12, 12))
        support[4, 4] = support[6, 6] = 1.
        result = trace_observed_ink(evidence_from_support(support), (4, 4), (6, 6),
                                    config=DirectionalTraceConfig(max_unsupported_run_px=1., minimum_supported_fraction=.5))
        self.assertEqual(result.status, "supported_with_gaps")
        self.assertAlmostEqual(result.audit.maximum_unsupported_run_px, .8, places=6)

    def test_faint_path_does_not_need_a_skeleton_or_coherence_gate(self):
        support = np.zeros((24, 40))
        support[12, 3:37] = .09
        evidence = dataclasses.replace(evidence_from_support(support), centerline=np.zeros(support.shape, dtype=bool),
                                       center_score=np.zeros(support.shape), coherence=np.zeros(support.shape), centerline_fraction=0.)
        result = trace_observed_ink(evidence, (3, 12), (36, 12))
        self.assertEqual(result.status, "observed")
        self.assertAlmostEqual(result.audit.mean_support, .09, places=6)
        self.assertTrue(all(y == 12. for _, y in result.points))

    def test_subpixel_quadratic_support_dip_cannot_be_called_observed(self):
        support = np.zeros((12, 12))
        support[4, 4] = support[5, 5] = 1.
        result = trace_observed_ink(evidence_from_support(support), (4, 4), (5, 5),
                                    config=DirectionalTraceConfig(minimum_support=.6, minimum_supported_fraction=.5))
        self.assertEqual(result.status, "supported_with_gaps")
        self.assertAlmostEqual(result.audit.maximum_unsupported_run_px, math.sqrt(.4), places=6)

    def test_subpixel_endpoint_below_threshold_is_audited(self):
        support = np.zeros((12, 12))
        support[4, 4] = support[4, 5] = .1
        result = trace_observed_ink(evidence_from_support(support), (4., 4.49), (4.2, 4.),
                                    config=DirectionalTraceConfig(minimum_supported_fraction=.5))
        self.assertNotEqual(result.status, "observed")
        self.assertTrue(result.audit.unsupported_spans)

    def test_incoming_heading_retains_more_expensive_arrival_at_junction(self):
        # At (10, 7) the upper route costs 20; the straight arrival costs 20.6.
        # Only the straight arrival avoids the subsequent 2-unit right turn.
        support = np.zeros((12, 22))
        support[7, 2:19] = 1.
        support[7, 3:10] = .1
        support[3:8, 2] = support[3:8, 10] = support[3, 2:11] = 1.
        result = trace_observed_ink(evidence_from_support(support), (2, 7), (18, 7), exclusion_mask=support == 0.,
                                    config=DirectionalTraceConfig(turn_weight=2., tangent_weight=0., max_unsupported_run_px=0.))
        self.assertEqual(result.status, "observed")
        self.assertTrue(all(y == 7. for _, y in result.points))
        self.assertAlmostEqual(result.audit.cost, 28.6, places=5)

    def test_turn_cost_resists_right_angle_dark_detour(self):
        support = np.zeros((48, 68))
        support[28, 8:57] = .65
        support[20:29, 8] = support[20:29, 56] = support[20, 8:57] = 1.
        evidence = evidence_from_support(support, coherence=np.zeros(support.shape))
        options = dict(exclusion_mask=support == 0.)
        unpenalized = trace_observed_ink(evidence, (8, 28), (56, 28), **options,
                                       config=DirectionalTraceConfig(turn_weight=0., max_unsupported_run_px=0.))
        penalized = trace_observed_ink(evidence, (8, 28), (56, 28), **options,
                                     config=DirectionalTraceConfig(turn_weight=12., max_unsupported_run_px=0.))
        self.assertEqual(unpenalized.status, "observed")
        self.assertLess(min(y for _, y in unpenalized.points), 28.)
        self.assertEqual(penalized.status, "observed")
        self.assertTrue(all(y == 28. for _, y in penalized.points))

    def test_text_avoidance_keeps_faint_continuation_past_darker_glyph(self):
        support = np.zeros((44, 72))
        support[24, 6:66] = .3
        support[18:25, 16] = support[18:25, 54] = support[18, 16:55] = 1.
        support[10:32, 34] = 1.  # A dark crossing glyph stroke.
        text = np.zeros(support.shape)
        text[10:32, 15:56] = 1.
        text[24, :] = 0.  # Caller-owned crossing pixels stay available as ink.
        result = trace_observed_ink(evidence_from_support(support), (6, 24), (65, 24), text_avoidance_score=text)
        self.assertEqual(result.status, "observed")
        self.assertTrue(all(y == 24. for _, y in result.points))
        self.assertTrue(result.audit.used_text_avoidance)

    def test_reference_corridor_excludes_tempting_parallel_line(self):
        support = np.zeros((44, 72))
        support[24, 6:66] = .25
        support[18:25, 6] = support[18:25, 65] = support[18, 6:66] = 1.
        reference = [(6., 24.), (65., 24.)]
        result = trace_observed_ink(evidence_from_support(support), *reference, reference_path=reference,
                                    config=DirectionalTraceConfig(corridor_radius_px=2.))
        self.assertEqual(result.status, "observed")
        self.assertTrue(all(y == 24. for _, y in result.points))
        self.assertLessEqual(result.audit.maximum_corridor_distance_px, 2.)

    def test_corridor_applies_to_exact_endpoint_not_just_rounded_pixel(self):
        result = trace_observed_ink(evidence_from_support(np.ones((8, 8))), (2, 4.49), (6, 4),
                                    reference_path=((2, 4), (6, 4)), config=DirectionalTraceConfig(corridor_radius_px=.25))
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.audit.reason, "exact_route_leaves_corridor")

    def test_text_score_is_soft_when_no_other_observed_line_exists(self):
        support = np.zeros((16, 24))
        support[8, 3:21] = .6
        result = trace_observed_ink(evidence_from_support(support), (3, 8), (20, 8),
                                    text_avoidance_score=np.ones(support.shape), config=DirectionalTraceConfig(max_unsupported_run_px=0.))
        self.assertEqual(result.status, "observed")
        self.assertTrue(all(y == 8. for _, y in result.points))

    def test_real_valley_and_s_bends_remain_in_the_trace(self):
        curves = (
            ((8, 12), (16, 28), (28, 40), (40, 28), (48, 12)),
            tuple((x, 32. + 12. * math.sin((x - 8) * math.pi / 28.)) for x in range(8, 65)),
        )
        for reference in curves:
            with self.subTest(reference=reference[:2]):
                result = trace_observed_ink(painted_path((64, 80), reference), reference[0], reference[-1], reference_path=reference,
                                            config=DirectionalTraceConfig(corridor_radius_px=3., max_unsupported_run_px=0.))
                self.assertEqual(result.status, "observed", result.audit)
                self.assertGreater(max(y for _, y in result.points), 38.)
                self.assertEqual(len(set(result.points)), len(result.points))
                self.assertLessEqual(result.audit.maximum_corridor_distance_px, 3.)
        self.assertLess(min(y for _, y in result.points), 23.)

    def test_hard_exclusion_blocks_crossing_and_diagonal_corner_cut(self):
        support = np.ones((12, 16))
        excluded = np.zeros(support.shape, dtype=bool)
        excluded[:, 8] = True
        result = trace_observed_ink(evidence_from_support(support), (2, 6), (13, 6), exclusion_mask=excluded)
        self.assertEqual(result.status, "abstained")
        excluded = np.ones(support.shape, dtype=bool)
        excluded[4, 4] = excluded[5, 5] = False
        result = trace_observed_ink(evidence_from_support(support), (4, 4), (5, 5), exclusion_mask=excluded)
        self.assertEqual(result.status, "abstained")

    def test_exact_endpoints_and_inputs_are_preserved(self):
        support = np.ones((20, 40))
        evidence = evidence_from_support(support)
        start, end = [3.2, 10.1], [35.3, 10.2]
        reference = [start.copy(), end.copy()]
        text, excluded = np.zeros(support.shape), np.zeros(support.shape, dtype=bool)
        snapshots = [array.copy() for array in (evidence.support_score, evidence.tangent_x, evidence.tangent_y, evidence.coherence, text, excluded)]
        result = trace_observed_ink(evidence, start, end, reference_path=reference, text_avoidance_score=text, exclusion_mask=excluded)
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.points[0], tuple(start))
        self.assertEqual(result.points[-1], tuple(end))
        self.assertEqual(reference, [start, end])
        for before, after in zip(snapshots, (evidence.support_score, evidence.tangent_x, evidence.tangent_y, evidence.coherence, text, excluded)):
            np.testing.assert_array_equal(before, after)
        start[0] = end[0] = reference[0][0] = 100.
        self.assertEqual(result.points[0], (3.2, 10.1))
        self.assertEqual(result.points[-1], (35.3, 10.2))

    def test_exclusion_checks_exact_subpixel_boundary_touch(self):
        excluded = np.zeros((8, 8), dtype=bool)
        excluded[4, 5] = True
        result = trace_observed_ink(evidence_from_support(np.ones((8, 8))), (4.5, 4.2), (4.4, 4.3), exclusion_mask=excluded)
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.audit.reason, "exact_endpoint_segment_enters_exclusion")

    def test_different_endpoints_in_one_pixel_are_not_rounded_away(self):
        result = trace_observed_ink(evidence_from_support(np.ones((8, 8))), (3.1, 3.1), (3.2, 3.3))
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.points, ((3.1, 3.1), (3.2, 3.3)))

    def test_real_detector_does_not_mutate_image_and_blank_abstains(self):
        image = np.full((40, 72), 255, dtype=np.uint8)
        blank = trace_observed_ink(ink_centerline_candidates(image), (6, 20), (65, 20))
        self.assertEqual(blank.status, "abstained")
        image[19:22, 5:67] = 170
        snapshot = image.copy()
        result = trace_observed_ink(ink_centerline_candidates(image), (6, 20), (65, 20))
        np.testing.assert_array_equal(image, snapshot)
        self.assertEqual(result.status, "observed")

    def test_resource_caps_are_finite_and_abstain_without_geometry(self):
        evidence = evidence_from_support(np.ones((48, 80)))
        for config, reason in (
            (DirectionalTraceConfig(max_states=1), "state_budget_exhausted"),
            (DirectionalTraceConfig(max_queue_entries=1), "queue_budget_exhausted"),
        ):
            with self.subTest(reason=reason):
                result = trace_observed_ink(evidence, (6, 20), (65, 20), config=config)
                self.assertEqual(result.status, "abstained")
                self.assertEqual(result.points, ())
                self.assertEqual(result.audit.reason, reason)
                self.assertLessEqual(result.audit.discovered_states, config.max_states)
                self.assertLessEqual(result.audit.queue_peak, config.max_queue_entries)
        with self.assertRaisesRegex(DirectionalTraceError, "max_window"):
            trace_observed_ink(evidence, (6, 20), (65, 20), config=DirectionalTraceConfig(max_window_size_px=32))
        with self.assertRaisesRegex(DirectionalTraceError, "max_corridor"):
            trace_observed_ink(evidence, (6, 20), (65, 20), reference_path=((6, 20), (65, 20)),
                               config=DirectionalTraceConfig(max_corridor_evaluations=10))
        with self.assertRaisesRegex(DirectionalTraceError, "point budget"):
            trace_observed_ink(evidence, (6, 20), (65, 20), reference_path=((6, 20), (30, 20), (65, 20)),
                               config=DirectionalTraceConfig(max_reference_points=2))

    def test_invalid_configs_shapes_values_and_endpoints_are_rejected(self):
        for kwargs in ({"minimum_support": 0.}, {"support_weight": float("nan")}, {"turn_weight": float("inf")},
                       {"max_states": True}, {"max_window_size_px": 2048}, {"max_unsupported_run_px": -1.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(DirectionalTraceError):
                DirectionalTraceConfig(**kwargs)
        evidence = evidence_from_support(np.ones((8, 8)))
        for start in ((-0.1, 4), (7.1, 4), (float("nan"), 4), "bad", (True, 4)):
            with self.subTest(start=start), self.assertRaises(DirectionalTraceError):
                trace_observed_ink(evidence, start, (6, 4))
        with self.assertRaises(DirectionalTraceError):
            trace_observed_ink(evidence, (2, 4), (6, 4), text_avoidance_score=np.ones((2, 2)))
        with self.assertRaises(DirectionalTraceError):
            trace_observed_ink(evidence, (2, 4), (6, 4), exclusion_mask=np.ones((8, 8)))
        broken = SimpleNamespace(**{name: getattr(evidence, name).copy() for name in ("support_score", "tangent_x", "tangent_y", "coherence")})
        broken.support_score[4, 4] = np.nan
        with self.assertRaises(DirectionalTraceError):
            trace_observed_ink(broken, (2, 4), (6, 4))


if __name__ == "__main__":
    unittest.main()
