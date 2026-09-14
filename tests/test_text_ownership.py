"""Synthetic safety contracts for soft ownership, not text-recognition accuracy."""

from dataclasses import replace
import unittest

import numpy as np
from PIL import Image, ImageDraw

from histcontour_core.text_ownership import TextOwnershipConfig, derive_text_ownership


BOX = [[20., 20.], [76., 20.], [76., 76.], [20., 76.]]


def canvas():
    image = Image.new("L", (96, 96), 255)
    return image, ImageDraw.Draw(image)


def digit_two(draw, *, offset=(0, 0), fill=25):
    points = [(34, 33), (43, 33), (46, 36), (44, 41), (34, 51), (46, 51)]
    draw.line([(x+offset[0], y+offset[1]) for x, y in points], fill=fill, width=1)


def digit_three(draw, *, fill=25):
    draw.line([(56, 33), (66, 33), (66, 40), (59, 43), (66, 46), (66, 53), (56, 53)], fill=fill, width=1)


class TextOwnershipTests(unittest.TestCase):
    def assert_partition(self, result):
        soft = result.soft_text_avoidance > 0
        expected = result.source_ink & result.text_region
        self.assertTrue(np.array_equal(soft | result.ambiguous_ink | result.throughgoing_ink, expected))
        self.assertFalse(np.any(soft & result.ambiguous_ink))
        self.assertFalse(np.any(soft & result.throughgoing_ink))
        self.assertFalse(np.any(result.ambiguous_ink & result.throughgoing_ink))
        self.assertFalse(np.any(result.soft_text_avoidance[~result.source_ink]))
        self.assertLessEqual(float(result.soft_text_avoidance.max()), .65)

    def test_compact_dark_digits_produce_only_soft_original_ink_evidence(self):
        image, draw = canvas()
        digit_two(draw)
        digit_three(draw)
        gray = np.asarray(image)
        result = derive_text_ownership(gray, [BOX], [.95])
        self.assertGreater(np.count_nonzero(result.soft_text_avoidance), 30)
        self.assertTrue(np.all(result.soft_text_avoidance[gray == 255] == 0))
        self.assertFalse(result.throughgoing_ink.any())
        self.assert_partition(result)
        self.assertFalse(result.provenance["human_approved"])
        self.assertFalse(result.provenance["training_eligible"])
        self.assertFalse(result.provenance["glyph_truth_assigned"])

    def test_crossing_contour_is_retained_with_two_distinct_supported_boundaries(self):
        image, draw = canvas()
        draw.line([(3, 47), (91, 47)], fill=90, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertTrue(result.throughgoing_ink[47, 20:77].all())
        self.assertFalse(result.soft_text_avoidance.any())
        record = result.region_evidence[0]["components"][0]
        self.assertGreater(record["boundary_separation_px"], 40)
        self.assertGreater(record["minimum_direction_alignment"], .99)
        self.assert_partition(result)

    def test_nearby_faint_contour_is_preserved_beside_compact_digit(self):
        image, draw = canvas()
        digit_two(draw)
        draw.line([(3, 57), (91, 57)], fill=229, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX], [.99])
        self.assertGreater(np.count_nonzero(result.soft_text_avoidance), 20)
        self.assertTrue(result.throughgoing_ink[57, 20:77].all())
        self.assertTrue(np.all(result.soft_text_avoidance[57] == 0))
        self.assert_partition(result)

    def test_connected_glyph_contour_junction_is_ambiguous(self):
        image, draw = canvas()
        digit_two(draw)
        draw.line([(3, 42), (91, 42)], fill=70, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertFalse(result.throughgoing_ink.any())
        self.assertTrue(result.ambiguous_ink.any())
        self.assertIn("connected_junction_ownership_unresolved", result.region_evidence[0]["decision_counts"])
        self.assert_partition(result)

    def test_crossed_lines_are_ambiguous_at_the_connected_junction(self):
        image, draw = canvas()
        draw.line([(3, 48), (91, 48)], fill=90, width=1)
        draw.line([(48, 3), (48, 91)], fill=100, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertTrue(result.ambiguous_ink[48, 48])
        self.assertFalse(result.soft_text_avoidance.any())
        self.assert_partition(result)

    def test_one_sided_external_connection_is_not_throughgoing(self):
        image, draw = canvas()
        draw.line([(3, 48), (47, 48)], fill=70, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertFalse(result.throughgoing_ink.any())
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertTrue(result.ambiguous_ink[48, 20:48].all())

    def test_short_exterior_extensions_do_not_count_as_throughgoing(self):
        image, draw = canvas()
        draw.line([(18, 48), (78, 48)], fill=70, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertFalse(result.throughgoing_ink.any())
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertIn("external_extension_too_short", result.region_evidence[0]["decision_counts"])

    def test_directionally_incompatible_two_exits_remain_ambiguous(self):
        image, draw = canvas()
        draw.line([(5, 30), (30, 30), (30, 66), (5, 66)], fill=70, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertFalse(result.throughgoing_ink.any())
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertIn("boundary_directions_do_not_support_throughgoing_ink", result.region_evidence[0]["decision_counts"])

    def test_zero_closed_contour_and_one_short_line_are_deferred(self):
        image, draw = canvas()
        draw.ellipse((30, 31, 44, 52), outline=25, width=1)
        draw.line([(60, 31), (60, 52)], fill=25, width=1)
        result = derive_text_ownership(np.asarray(image), [BOX], [1.])
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertFalse(result.throughgoing_ink.any())
        self.assertIn("closed_loop_zero_or_contour_unresolved", result.region_evidence[0]["decision_counts"])
        self.assertIn("near_straight_one_or_short_line_unresolved", result.region_evidence[0]["decision_counts"])
        self.assert_partition(result)

    def test_faint_compact_bent_ink_is_not_owned_as_text(self):
        image, draw = canvas()
        digit_two(draw, fill=229)
        result = derive_text_ownership(np.asarray(image), [BOX])
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertIn("faint_compact_ink_not_owned_as_text", result.region_evidence[0]["decision_counts"])

    def test_rotated_polygon_does_not_claim_its_axis_aligned_box(self):
        image, draw = canvas()
        digit_two(draw)
        draw.line([(19, 20), (24, 20), (24, 24)], fill=25, width=1)
        diamond = [[46., 15.], [78., 46.], [46., 78.], [15., 46.]]
        result = derive_text_ownership(np.asarray(image), [diamond])
        self.assertGreater(np.count_nonzero(result.soft_text_avoidance), 20)
        self.assertFalse(result.text_region[20, 20])
        self.assertFalse(result.soft_text_avoidance[20, 20])
        self.assertFalse(result.ambiguous_ink[20, 20])
        self.assert_partition(result)

    def test_rotated_throughgoing_line_uses_actual_boundary_directions(self):
        image, draw = canvas()
        draw.line([(4, 10), (89, 80)], fill=70, width=1)
        diamond = [[46., 15.], [78., 46.], [46., 78.], [15., 46.]]
        result = derive_text_ownership(np.asarray(image), [diamond])
        self.assertGreater(np.count_nonzero(result.throughgoing_ink), 20)
        self.assertFalse(result.soft_text_avoidance.any())
        self.assert_partition(result)

    def test_detector_scores_are_recorded_without_promoting_or_rescaling(self):
        image, draw = canvas()
        digit_two(draw)
        first = derive_text_ownership(np.asarray(image), [BOX], [.1])
        second = derive_text_ownership(np.asarray(image), [BOX], [1.])
        self.assertTrue(np.array_equal(first.soft_text_avoidance, second.soft_text_avoidance))
        self.assertEqual(first.region_evidence[0]["detector_score_not_probability"], .1)
        self.assertEqual(second.region_evidence[0]["detector_score_not_probability"], 1.)

    def test_blank_and_no_regions_create_no_text_ownership(self):
        blank = np.full((96, 96), 255, np.uint8)
        for polygons in ([], [BOX]):
            result = derive_text_ownership(blank, polygons)
            self.assertFalse(result.source_ink.any())
            self.assertFalse(result.soft_text_avoidance.any())
            self.assert_partition(result)
        image, draw = canvas()
        digit_two(draw)
        result = derive_text_ownership(np.asarray(image), [])
        self.assertTrue(result.source_ink.any())
        self.assertFalse(result.soft_text_avoidance.any())

    def test_no_input_mutation_and_result_arrays_are_read_only(self):
        image, draw = canvas()
        digit_two(draw)
        gray = np.asarray(image).copy()
        polygons = np.asarray([BOX])
        scores = np.asarray([.9])
        before = gray.copy(), polygons.copy(), scores.copy()
        result = derive_text_ownership(gray, polygons, scores)
        for original, snapshot in zip((gray, polygons, scores), before):
            self.assertTrue(np.array_equal(original, snapshot))
        for name in ("source_ink", "text_region", "soft_text_avoidance", "ambiguous_ink", "throughgoing_ink"):
            self.assertFalse(getattr(result, name).flags.writeable)
            self.assertFalse(np.shares_memory(getattr(result, name), gray))

    def test_overlapping_regions_defer_conflicting_component_ownership(self):
        image, draw = canvas()
        digit_two(draw)
        narrow = [[39., 28.], [50., 28.], [50., 56.], [39., 56.]]
        result = derive_text_ownership(np.asarray(image), [BOX, narrow])
        overlap_ink = result.text_region & result.source_ink
        self.assertTrue(result.ambiguous_ink.any())
        self.assertFalse(np.any(result.soft_text_avoidance[result.ambiguous_ink]))
        self.assertTrue(overlap_ink.any())
        self.assert_partition(result)

    def test_bounded_region_defers_and_total_work_limit_rejects(self):
        image, draw = canvas()
        digit_two(draw)
        result = derive_text_ownership(np.asarray(image), [BOX], config=replace(TextOwnershipConfig(), max_region_work_pixels=32))
        self.assertFalse(result.soft_text_avoidance.any())
        self.assertTrue(result.ambiguous_ink.any())
        self.assertEqual(result.region_evidence[0]["status"], "deferred_work_bound")
        with self.assertRaisesRegex(ValueError, "work exceeds"):
            derive_text_ownership(np.asarray(image), [BOX], config=replace(TextOwnershipConfig(), max_total_work_pixels=32))

    def test_invalid_gray_polygons_scores_and_config_rejected(self):
        gray = np.full((96, 96), 255, np.uint8)
        for invalid in (np.ones((3, 3, 3)), np.ones((0, 3)), np.full((3, 3), np.nan),
                        np.full((3, 3), -1), np.full((3, 3), 256), np.ones((3, 3), bool)):
            with self.assertRaises(ValueError):
                derive_text_ownership(invalid, [BOX])
        for polygons, scores in (([BOX], []), ([BOX], [True]), ([BOX], [float("nan")]),
                                 ([BOX], [1.1]), ([[[0, 0], [1, 1], [2, 2]]], None),
                                 ([[[-1, 1], [3, 1], [3, 3], [-1, 3]]], None)):
            with self.assertRaises(ValueError):
                derive_text_ownership(gray, polygons, scores)
        with self.assertRaises(ValueError):
            derive_text_ownership(gray, [BOX], config=replace(TextOwnershipConfig(), soft_avoidance_weight=1.))
        with self.assertRaises(ValueError):
            derive_text_ownership(gray, [BOX], config=replace(TextOwnershipConfig(), max_total_work_pixels=10**12))

    def test_float_normalized_gray_matches_uint8_source(self):
        image, draw = canvas()
        digit_two(draw)
        integer = np.asarray(image)
        first = derive_text_ownership(integer, [BOX])
        second = derive_text_ownership(integer.astype(np.float32)/255, [BOX])
        self.assertTrue(np.array_equal(first.source_ink, second.source_ink))
        self.assertTrue(np.array_equal(first.soft_text_avoidance, second.soft_text_avoidance))


if __name__ == "__main__":
    unittest.main()
