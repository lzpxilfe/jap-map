import unittest

from histcontour_core.completion import (
    ContourCompletionSettings,
    anchors_from_polyline,
    propose_contour_completions,
    rasterize_polylines,
)


class ContourCompletionScenarioTest(unittest.TestCase):
    """Synthetic failure cases seen around historical-map contour breaks."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("contour completion requires NumPy")
        cls.np = np
        cls.shape = (100, 120)

    def anchors(self, *, left=((8, 50), (34, 50)), right=((76, 50), (108, 50)), length=80):
        first = anchors_from_polyline("left", left, source_length_px=length)
        second = anchors_from_polyline("right", right, source_length_px=length)
        return (next(anchor for anchor in first if anchor.endpoint_role == "end"), next(anchor for anchor in second if anchor.endpoint_role == "start"))

    def arrays(self, polylines=()):
        centerline = rasterize_polylines(self.shape, polylines)
        score = self.np.zeros(self.shape, dtype=self.np.float32)
        score[centerline] = 0.2
        return centerline, score

    def existing(self, left=((8, 50), (34, 50)), right=((76, 50), (108, 50))):
        return rasterize_polylines(self.shape, (left, right))

    def test_dark_to_faint_line_uses_supported_ink_path(self):
        right = ((72, 50), (108, 50))
        centerline, score = self.arrays((((34, 50), (72, 50)),))
        score[50, 35:55] = 1.0
        score[50, 55:72] = 0.05
        result = propose_contour_completions(
            self.anchors(right=right), centerline, score, existing_linework=self.existing(right=right)
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.mode, "ink_path")
        self.assertEqual(candidate.points[0], (34.0, 50.0))
        self.assertEqual(candidate.points[-1], (72.0, 50.0))

    def test_text_strokes_crossing_line_trigger_amodal_interpolation(self):
        centerline, score = self.arrays(
            (
                ((34, 50), (76, 50)),
                ((50, 42), (50, 58)),
                ((56, 42), (56, 58)),
            )
        )
        result = propose_contour_completions(
            self.anchors(), centerline, score, existing_linework=self.existing()
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].mode, "hermite_occlusion")
        self.assertGreater(result.candidates[0].junction_pixels, 0)

    def test_long_label_gap_uses_lower_confidence_hermite_gap(self):
        left = ((8, 50), (28, 50))
        right = ((88, 50), (112, 50))
        centerline, score = self.arrays()
        result = propose_contour_completions(
            self.anchors(left=left, right=right, length=100),
            centerline,
            score,
            existing_linework=self.existing(left, right),
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.mode, "hermite_gap")
        self.assertGreater(candidate.direct_gap_px, 50)
        self.assertLess(candidate.score, 0.7)

    def test_symbol_on_line_is_crossed_geometrically_not_traced(self):
        # A box symbol has no reliable through-route: the bridge should preserve
        # the contour tangent instead of following the symbol perimeter.
        centerline, score = self.arrays(
            (
                ((34, 50), (44, 50)),
                ((44, 42), (64, 42), (64, 58), (44, 58), (44, 42)),
                ((64, 50), (76, 50)),
            )
        )
        result = propose_contour_completions(
            self.anchors(), centerline, score, existing_linework=self.existing()
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.mode, "hermite_occlusion")
        self.assertTrue(all(abs(y - 50.0) < 0.01 for _x, y in candidate.points))

    def test_short_text_strokes_cannot_be_anchors(self):
        centerline, score = self.arrays((((34, 50), (76, 50)),))
        result = propose_contour_completions(
            self.anchors(length=22), centerline, score, existing_linework=self.existing()
        )
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.rejection_counts["short_anchor"], 2)

    def test_perpendicular_symbol_or_road_branch_is_not_connected(self):
        left = anchors_from_polyline("left", ((8, 50), (34, 50)), source_length_px=80)
        vertical = anchors_from_polyline("vertical", ((76, 20), (76, 50)), source_length_px=80)
        selected = (
            next(anchor for anchor in left if anchor.endpoint_role == "end"),
            next(anchor for anchor in vertical if anchor.endpoint_role == "end"),
        )
        centerline, score = self.arrays((((34, 50), (76, 50)),))
        result = propose_contour_completions(selected, centerline, score)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.rejection_counts["tangent_mismatch"], 1)

    def test_nearby_parallel_contours_do_not_cross_connect(self):
        first = anchors_from_polyline("upper-left", ((8, 44), (34, 44)), source_length_px=80)
        second = anchors_from_polyline("lower-right", ((76, 56), (108, 56)), source_length_px=80)
        # The displacement angle is too far from both endpoint tangents under
        # a deliberately tighter setting, representing adjacent contour bands.
        selected = (
            next(anchor for anchor in first if anchor.endpoint_role == "end"),
            next(anchor for anchor in second if anchor.endpoint_role == "start"),
        )
        centerline, score = self.arrays((((34, 44), (76, 56)),))
        settings = ContourCompletionSettings(maximum_ink_tangent_error_degrees=12, maximum_interpolation_tangent_error_degrees=8)
        result = propose_contour_completions(selected, centerline, score, settings=settings)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.rejection_counts["tangent_mismatch"], 1)

    def test_ambiguous_endpoint_rejects_both_equally_plausible_targets(self):
        left = anchors_from_polyline("left", ((8, 50), (34, 50)), source_length_px=90)
        upper = anchors_from_polyline("upper", ((76, 48), (108, 48)), source_length_px=90)
        lower = anchors_from_polyline("lower", ((76, 52), (108, 52)), source_length_px=90)
        selected = (
            next(anchor for anchor in left if anchor.endpoint_role == "end"),
            next(anchor for anchor in upper if anchor.endpoint_role == "start"),
            next(anchor for anchor in lower if anchor.endpoint_role == "start"),
        )
        centerline, score = self.arrays(
            (
                ((34, 50), (76, 48)),
                ((34, 50), (76, 52)),
            )
        )
        result = propose_contour_completions(selected, centerline, score, existing_linework=self.existing())
        self.assertEqual(result.candidates, ())
        self.assertGreaterEqual(result.rejection_counts["ambiguous_endpoint"], 2)

    def test_existing_line_is_not_emitted_as_a_completion(self):
        centerline, score = self.arrays((((34, 50), (76, 50)),))
        already_complete = rasterize_polylines(self.shape, (((8, 50), (108, 50)),))
        result = propose_contour_completions(
            self.anchors(), centerline, score, existing_linework=already_complete
        )
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.rejection_counts["already_covered"], 1)

    def test_tile_border_end_is_left_for_cross_tile_processing(self):
        line = ((1, 50), (20, 50))
        border_anchor = next(
            anchor
            for anchor in anchors_from_polyline("border", line, source_length_px=80)
            if anchor.endpoint_role == "start"
        )
        centerline, score = self.arrays()
        result = propose_contour_completions(
            (border_anchor,),
            centerline,
            score,
            existing_linework=rasterize_polylines(self.shape, (line,)),
        )
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.rejection_counts["border_anchor"], 1)

    def test_closed_symbol_loop_does_not_create_endpoint_anchors(self):
        anchors = anchors_from_polyline(
            "closed-symbol",
            ((30, 30), (50, 30), (50, 50), (30, 50), (30, 30)),
            source_length_px=80,
        )
        self.assertEqual(anchors, ())

    def test_gap_beyond_interpolation_limit_is_not_considered(self):
        shape = (100, 180)
        left = ((8, 50), (34, 50))
        right = ((130, 50), (170, 50))
        anchors = (
            next(anchor for anchor in anchors_from_polyline("left", left, source_length_px=120) if anchor.endpoint_role == "end"),
            next(anchor for anchor in anchors_from_polyline("right", right, source_length_px=120) if anchor.endpoint_role == "start"),
        )
        centerline = rasterize_polylines(shape, (((34, 50), (130, 50)),))
        score = self.np.zeros(shape, dtype=self.np.float32)
        score[centerline] = 1.0
        result = propose_contour_completions(anchors, centerline, score)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.evaluated_pair_count, 0)

    def test_same_source_line_cannot_shortcut_between_its_own_ends(self):
        line = ((25, 50), (45, 35), (65, 50))
        anchors = anchors_from_polyline("one-line", line, source_length_px=90)
        centerline, score = self.arrays((((25, 50), (65, 50)),))
        result = propose_contour_completions(anchors, centerline, score)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.evaluated_pair_count, 0)

    def test_gently_curved_faint_line_can_use_ink_path(self):
        left = ((8, 50), (30, 50))
        right = ((68, 58), (108, 58))
        anchors = self.anchors(left=left, right=right, length=90)
        curve = ((30, 50), (40, 50), (58, 58), (68, 58))
        centerline, score = self.arrays((curve,))
        score[centerline] = 0.06
        result = propose_contour_completions(
            anchors,
            centerline,
            score,
            existing_linework=self.existing(left, right),
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].mode, "ink_path")

    def test_long_ink_detour_is_not_followed_around_symbol(self):
        left = ((8, 50), (34, 50))
        right = ((76, 50), (108, 50))
        detour = ((34, 50), (34, 25), (76, 25), (76, 50))
        centerline, score = self.arrays((detour,))
        result = propose_contour_completions(
            self.anchors(left=left, right=right, length=100),
            centerline,
            score,
            existing_linework=self.existing(left, right),
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertNotEqual(result.candidates[0].mode, "ink_path")
        self.assertLess(result.candidates[0].path_length_px, 50)


if __name__ == "__main__":
    unittest.main()
