"""Synthetic search-group recall and safety, never semantic OCR accuracy."""

from dataclasses import replace
import unittest

import numpy as np
from PIL import Image, ImageDraw

from histcontour_core.component_text_candidates import ComponentTextCandidateConfig, find_component_text_candidates


def digits_fixture(text="100", *, angle=0, scale=1):
    size = (160*scale, 96*scale)
    masks = []
    for index, digit in enumerate(text):
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        x0, y0 = (55+14*index)*scale, 42*scale
        xy = lambda x, y: (x0+x*scale, y0+y*scale)
        if digit == "0":
            draw.ellipse((*xy(0, 0), *xy(8, 12)), outline=255, width=scale)
        elif digit == "1":
            draw.line([xy(4, 0), xy(4, 12)], fill=255, width=scale)
        elif digit == "2":
            draw.line([xy(0, 2), xy(2, 0), xy(6, 0), xy(8, 3), xy(7, 5), xy(0, 12), xy(8, 12)], fill=255, width=scale)
        else:
            raise ValueError(digit)
        if angle:
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, center=(80*scale, 48*scale))
        masks.append(np.asarray(mask) > 0)
    gray = np.full(tuple(reversed(size)), 255, np.uint8)
    gray[np.logical_or.reduce(masks)] = 25
    return gray, masks


def component_ids_for_masks(result, masks):
    ids = []
    for mask in masks:
        values, counts = np.unique(result.component_labels[mask], return_counts=True)
        pairs = [(int(cid), int(count)) for cid, count in zip(values, counts) if cid > 0]
        ids.append(max(pairs, key=lambda row: row[1])[0])
    return ids


def group_for_ids(result, ids):
    return next((group for group in result.groups if set(group["group_component_ids"]) == set(ids)), None)


def linked_lobes_fixture(*, angle=0, tail=True):
    """Three joined closed lobes plus an attached curve; no character label."""
    mask = Image.new("L", (150, 100), 0)
    draw = ImageDraw.Draw(mask)
    for x in (50, 65, 80):
        draw.ellipse((x-4, 44, x+4, 56), outline=255, width=2)
    draw.line([(50, 56), (80, 56)], fill=255, width=2)
    if tail:
        draw.line([(80, 44), (89, 33), (96, 25)], fill=255, width=2)
    if angle:
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, center=(75, 50))
    gray = np.full((100, 150), 255, np.uint8)
    gray[np.asarray(mask) > 0] = 25
    return gray


class ComponentTextCandidateTests(unittest.TestCase):
    def test_attached_curve_does_not_set_a_linked_lobes_search_axis(self):
        for angle in (-35, 0, 35, 70):
            with self.subTest(angle=angle):
                gray = linked_lobes_fixture(angle=angle)
                before = gray.copy()
                result = find_component_text_candidates(gray, source_origin_xy=(430, 598))
                groups = [g for g in result.groups if g["candidate_kind"] == "intra_component_hole_axis"]
                self.assertTrue(groups, result.provenance["hole_orientation_attempts"])
                candidate = next(g for g in groups if g["orientation_hypothesis"] == "enclosed_hole_centroids")
                self.assertAlmostEqual(candidate["evidence"]["baseline_angle_degrees"], -angle, delta=4.)
                self.assertEqual(len(candidate["group_component_ids"]), 1)
                self.assertTrue(candidate["partial_component_search"])
                self.assertFalse(candidate["full_component_containment"])
                self.assertLess(candidate["evidence"]["component_ink_inside_quad_fraction"], 1.)
                self.assertFalse(candidate["evidence"]["whole_component_ownership_supported"])
                self.assertTrue(candidate["evidence"]["hole_count_is_not_character_count"])
                np.testing.assert_array_equal(gray, before)
                np.testing.assert_array_equal(result.component_labels > 0, gray < 160)
                quad = np.asarray(candidate["quad_crop_pixel_centers"])
                np.testing.assert_allclose(candidate["quad_source_pixel_centers"], quad+[430, 598])
                np.testing.assert_allclose(candidate["quad_source_image_corners"], quad+[430.5, 598.5])
                self.assertTrue(any(a["orientation_hypothesis"] == "whole_component_ink_pca_control"
                                    and a["status"] == "rejected" for a in result.provenance["hole_orientation_attempts"]))

    def test_hole_axis_hypotheses_never_label_lobes_as_glyphs(self):
        result = find_component_text_candidates(linked_lobes_fixture())
        self.assertEqual(len(result.components), 1)
        self.assertEqual(result.components[0]["character_type"], "unknown")
        for candidate in result.groups:
            self.assertFalse(candidate["glyph_truth_assigned"])
            self.assertFalse(candidate["elevation_assigned"])
            self.assertFalse(candidate["erase_mask_generated"])
            self.assertFalse(candidate["human_approved"])
            self.assertTrue(candidate["ambiguous"])
        np.testing.assert_array_equal(result.component_labels > 0, result.source_ink)

    def test_boundary_context_is_bounded_and_keeps_original_hypothesis(self):
        gray = linked_lobes_fixture()
        result = find_component_text_candidates(gray)
        expansions = [g for g in result.groups if g["orientation_hypothesis"].endswith("_bounded_normal_context")]
        self.assertTrue(expansions)
        for group in expansions:
            evidence = group["evidence"]
            base = next(g for g in result.groups if g["quad_crop_pixel_centers"] == evidence["base_quad_crop_pixel_centers"])
            self.assertGreater(evidence["component_ink_inside_quad_pixels"], base["evidence"]["component_ink_inside_quad_pixels"])
            self.assertLessEqual(evidence["bounded_normal_context_px"], 32)
            self.assertTrue(group["partial_component_search"])
            self.assertFalse(group["full_component_containment"])
            self.assertFalse(evidence["whole_component_ownership_supported"])
            self.assertTrue(evidence["context_may_include_attached_contour"])
        np.testing.assert_array_equal(result.source_ink, gray < 160)

    def test_closed_lobes_without_external_ink_do_not_claim_a_partial_search(self):
        result = find_component_text_candidates(linked_lobes_fixture(tail=False))
        self.assertEqual(result.groups, ())
        self.assertTrue(any(a.get("reason") == "no_partial_component_context" for a in result.provenance["hole_orientation_attempts"]))

    def test_broken_open_lobes_are_not_closed_to_manufacture_an_axis(self):
        gray = linked_lobes_fixture()
        for x in (50, 65, 80):
            gray[43:48, x-1:x+2] = 255
        before = gray.copy()
        result = find_component_text_candidates(gray)
        self.assertEqual(result.provenance["counts"]["intra_component_candidates"], 0)
        self.assertTrue(all(g["candidate_kind"] != "intra_component_hole_axis" for g in result.groups))
        np.testing.assert_array_equal(gray, before)
        np.testing.assert_array_equal(result.source_ink, gray < 160)

    def test_hole_axis_budget_and_output_omissions_keep_alternatives(self):
        gray = linked_lobes_fixture()
        limited = find_component_text_candidates(gray, config=replace(ComponentTextCandidateConfig(), max_hole_axis_evaluations=1))
        self.assertTrue(limited.provenance["truncation_flags"]["hole_orientation_evaluation_cap"])
        self.assertEqual(limited.provenance["counts"]["hole_orientation_evaluations"], 1)
        self.assertTrue(any(a.get("reason") == "hole_orientation_evaluation_cap" for a in limited.provenance["hole_orientation_attempts"]))
        doubled = np.concatenate((gray, gray), axis=1)
        capped = find_component_text_candidates(doubled, config=replace(ComponentTextCandidateConfig(), max_groups=1))
        self.assertTrue(capped.provenance["truncation_flags"]["output_group_cap"])
        omitted = capped.provenance["omitted_groups"]
        self.assertTrue(omitted)
        self.assertTrue(all("quad_source_pixel_centers" in g and "evidence" in g for g in omitted))
        self.assertTrue(any(a["status"] == "output_group_cap" for a in capped.provenance["hole_orientation_attempts"]))
        hole_capped = find_component_text_candidates(gray, config=replace(ComponentTextCandidateConfig(), max_enclosed_holes_per_component=2))
        holes = hole_capped.components[0]["enclosed_holes"]
        self.assertEqual(len(holes), 3)
        self.assertEqual(sum(hole["used_for_axis"] for hole in holes), 2)
        self.assertTrue(any(hole["reason"] == "enclosed_hole_cap" for hole in holes))

    def test_noncollinear_internal_holes_have_explicit_failed_axes(self):
        image = Image.new("L", (140, 100), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 25, 94, 80), fill=25)
        for x, y in ((50, 38), (84, 38), (67, 68)):
            draw.ellipse((x-3, y-3, x+3, y+3), fill=255)
        result = find_component_text_candidates(np.asarray(image), config=replace(ComponentTextCandidateConfig(), max_component_pixels=4096))
        self.assertEqual(result.groups, ())
        self.assertTrue(result.provenance["hole_orientation_attempts"])
        self.assertTrue(all(a["status"] == "rejected" for a in result.provenance["hole_orientation_attempts"]))
        self.assertTrue(any(a.get("reason") == "hole_centroids_not_collinear" for a in result.provenance["hole_orientation_attempts"]))

    def test_two_component_pca_degeneracy_is_not_reported_as_word_evidence(self):
        gray, _ = digits_fixture("00")
        result = find_component_text_candidates(gray)
        self.assertTrue(all(g["evidence"]["centroid_pca_two_point_degeneracy"] for g in result.groups))
        self.assertTrue(all(len(g["evidence"]["member_ink_axes_degrees"]) == 2 for g in result.groups))

    def test_tilted_100_and_200_have_complete_three_component_search_groups(self):
        for text in ("100", "200"):
            for angle in (-35, 0, 35, 70):
                with self.subTest(text=text, angle=angle):
                    gray, masks = digits_fixture(text, angle=angle, scale=2)
                    result = find_component_text_candidates(gray)
                    ids = component_ids_for_masks(result, masks)
                    self.assertEqual(len(set(ids)), 3)
                    group = group_for_ids(result, ids)
                    self.assertIsNotNone(group, result.provenance)
                    self.assertEqual(group["component_count"], 3)
                    self.assertTrue(group["evidence"]["quad_within_crop_bounds"])
                    # PIL positive rotation is counter-clockwise; image y is down.
                    self.assertAlmostEqual(group["evidence"]["baseline_angle_degrees"], -angle, delta=5)

    def test_source_pixel_scale_variants_remain_detectable_within_declared_bounds(self):
        for scale in (1, 2, 4):
            with self.subTest(scale=scale):
                gray, masks = digits_fixture("200", scale=scale)
                result = find_component_text_candidates(gray)
                group = group_for_ids(result, component_ids_for_masks(result, masks))
                self.assertIsNotNone(group)
                self.assertGreater(min(group["evidence"]["projected_heights_px"]), 10*scale)

    def test_single_zero_or_one_is_not_classified_and_requires_a_group(self):
        for text in ("0", "1"):
            gray, _ = digits_fixture(text)
            result = find_component_text_candidates(gray)
            self.assertEqual(result.groups, ())
            eligible = [c for c in result.components if c["eligible_for_grouping"]]
            self.assertEqual(len(eligible), 1)
            self.assertEqual(eligible[0]["character_type"], "unknown")
            self.assertFalse(eligible[0]["glyph_truth_assigned"])

    def test_aligned_small_circles_can_only_be_ambiguous_search_evidence(self):
        gray, _ = digits_fixture("00")
        result = find_component_text_candidates(gray)
        self.assertGreater(len(result.groups), 0)
        for group in result.groups:
            self.assertTrue(group["ambiguous"])
            self.assertFalse(group["glyph_truth_assigned"])
            self.assertFalse(group["contour_truth_assigned"])
            self.assertFalse(group["elevation_assigned"])
            self.assertFalse(group["erase_mask_generated"])
            self.assertIn("components_may_be_short_contours_or_symbols", group["ambiguous_reasons"])

    def test_large_closed_contours_are_excluded_without_component_splitting(self):
        image = Image.new("L", (240, 130), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 15, 100, 110), outline=25, width=2)
        draw.ellipse((130, 15, 220, 110), outline=25, width=2)
        result = find_component_text_candidates(np.asarray(image))
        self.assertEqual(result.groups, ())
        self.assertEqual(len(result.components), 2)
        self.assertTrue(all("long_or_large_component_not_split" in c["exclusion_reasons"] for c in result.components))
        self.assertEqual(set(np.unique(result.component_labels)), {0, 1, 2})

    def test_glyph_attached_to_a_large_contour_is_never_split_into_a_group(self):
        gray, masks = digits_fixture("200", scale=2)
        # Attach the first character to an existing long source line.
        image = Image.fromarray(gray)
        draw = ImageDraw.Draw(image)
        first_y, first_x = np.column_stack(np.nonzero(masks[0]))[0]
        draw.line([(10, int(first_y)), (int(first_x), int(first_y))], fill=25, width=2)
        gray = np.asarray(image)
        before = gray.copy()
        result = find_component_text_candidates(gray)
        cid = int(result.component_labels[first_y, first_x])
        record = next(c for c in result.components if c["component_id"] == cid)
        self.assertFalse(record["eligible_for_grouping"])
        self.assertIn("long_or_large_component_not_split", record["exclusion_reasons"])
        self.assertTrue(all(cid not in g["group_component_ids"] for g in result.groups))
        self.assertEqual(int(result.component_labels[first_y, 10]), cid)
        np.testing.assert_array_equal(gray, before)

    def test_touching_multiple_glyphs_in_one_component_do_not_imply_character_count(self):
        gray, masks = digits_fixture("200", scale=2)
        image = Image.fromarray(gray)
        draw = ImageDraw.Draw(image)
        # Join the first two characters using existing dark source ink.
        draw.line([(126, 108), (146, 108)], fill=25, width=2)
        result = find_component_text_candidates(np.asarray(image))
        self.assertEqual(len(result.components), 2)
        self.assertGreater(len(result.groups), 0)
        self.assertEqual(set(result.groups[0]["group_component_ids"]), {c["component_id"] for c in result.components})
        self.assertTrue(all(g["evidence"]["component_count_is_not_character_count"] for g in result.groups))
        self.assertTrue(all(c["character_type"] == "unknown" for c in result.components))

    def test_isolated_noise_and_blank_do_not_create_groups(self):
        gray = np.full((100, 140), 255, np.uint8)
        gray[10:90:7, 10:130:7] = 20
        result = find_component_text_candidates(gray)
        self.assertEqual(result.groups, ())
        self.assertTrue(all("too_few_ink_pixels" in c["exclusion_reasons"] for c in result.components))
        empty = find_component_text_candidates(np.full((32, 40), 255, np.uint8))
        self.assertEqual(empty.groups, ())
        self.assertFalse(empty.source_ink.any())

    def test_faint_components_are_not_used_as_dark_group_candidates(self):
        gray, _ = digits_fixture("100")
        gray[gray < 255] = 225
        result = find_component_text_candidates(gray)
        self.assertEqual(result.groups, ())
        self.assertFalse(result.source_ink.any())

    def test_inconsistent_heights_or_excessive_spacing_prevent_pairing(self):
        image = Image.new("L", (180, 100), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((30, 30, 36, 36), outline=20, width=1)
        draw.ellipse((44, 22, 58, 46), outline=20, width=1)
        self.assertEqual(find_component_text_candidates(np.asarray(image)).groups, ())
        # Build two same-size components much farther apart than a text gap.
        image = Image.new("L", (180, 100), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((15, 30, 23, 42), outline=20, width=1)
        draw.ellipse((145, 30, 153, 42), outline=20, width=1)
        self.assertEqual(find_component_text_candidates(np.asarray(image)).groups, ())

    def test_rotated_quads_contain_all_member_ink_and_have_explicit_coordinate_transforms(self):
        gray, masks = digits_fixture("100", angle=35, scale=2)
        result = find_component_text_candidates(gray, source_origin_xy=(430, 598))
        group = group_for_ids(result, component_ids_for_masks(result, masks))
        quad = np.asarray(group["quad_crop_pixel_centers"])
        np.testing.assert_allclose(group["quad_crop_image_corners"], quad+.5, atol=1e-12)
        np.testing.assert_allclose(group["quad_source_pixel_centers"], quad+[430, 598], atol=1e-12)
        np.testing.assert_allclose(group["quad_source_image_corners"], quad+[430.5, 598.5], atol=1e-12)
        rows, cols = np.nonzero(np.isin(result.component_labels, group["group_component_ids"]))
        points = np.column_stack((cols, rows))
        cross = []
        for a, b in zip(quad, np.roll(quad, -1, axis=0)):
            vector = b-a
            cross.append(vector[0]*(points[:, 1]-a[1])-vector[1]*(points[:, 0]-a[0]))
        cross = np.asarray(cross)
        self.assertTrue(np.all(cross >= -1e-9) or np.all(cross <= 1e-9))
        self.assertGreater(abs(quad[1, 1]-quad[0, 1]), 5)

    def test_quad_padding_outside_crop_is_explicit_not_silently_clipped(self):
        image = Image.new("L", (60, 30), 255)
        draw = ImageDraw.Draw(image)
        draw.ellipse((4, 1, 12, 13), outline=20)
        draw.ellipse((18, 1, 26, 13), outline=20)
        result = find_component_text_candidates(np.asarray(image))
        self.assertGreater(len(result.groups), 0)
        group = result.groups[0]
        self.assertFalse(group["evidence"]["quad_within_crop_bounds"])
        self.assertIn("crop_boundary_overrun", group["ambiguous_reasons"])
        self.assertLess(min(p[1] for p in group["quad_crop_pixel_centers"]), -.5)

    def test_output_pair_component_and_evaluation_caps_are_audited(self):
        gray, _ = digits_fixture("100")
        result = find_component_text_candidates(gray, config=replace(ComponentTextCandidateConfig(), max_groups=1))
        self.assertEqual(len(result.groups), 1)
        self.assertTrue(result.provenance["truncation_flags"]["output_group_cap"])
        self.assertEqual(len(result.provenance["omitted_groups"]), result.provenance["counts"]["group_candidates_before_output_cap"]-1)
        self.assertTrue(all(row["reason"] == "output_group_cap" for row in result.provenance["omitted_groups"]))
        pair_capped = find_component_text_candidates(gray, config=replace(ComponentTextCandidateConfig(), max_pairs=1))
        self.assertTrue(pair_capped.provenance["truncation_flags"]["pair_cap"])
        self.assertLessEqual(pair_capped.provenance["counts"]["neighbor_pairs_after_cap"], 1)
        eval_capped = find_component_text_candidates(gray, config=replace(ComponentTextCandidateConfig(), max_group_evaluations=1))
        self.assertTrue(eval_capped.provenance["truncation_flags"]["group_evaluation_cap"])
        self.assertLessEqual(eval_capped.provenance["counts"]["group_evaluations"], 1)
        component_capped = find_component_text_candidates(gray, config=replace(ComponentTextCandidateConfig(), max_components_for_grouping=2))
        self.assertTrue(component_capped.provenance["truncation_flags"]["component_grouping_cap"])
        self.assertEqual(component_capped.provenance["counts"]["components_used_for_grouping"], 2)
        self.assertTrue(any("component_grouping_cap" in c["exclusion_reasons"] for c in component_capped.components))

    def test_ranking_and_group_identity_are_deterministic_without_semantic_approval(self):
        gray, _ = digits_fixture("200", angle=20, scale=2)
        first = find_component_text_candidates(gray)
        second = find_component_text_candidates(gray)
        self.assertEqual(first.groups, second.groups)
        self.assertEqual(first.provenance, second.provenance)
        scores = [g["ranking_score_not_probability"] for g in first.groups]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([g["rank"] for g in first.groups], list(range(1, len(first.groups)+1)))
        for group in first.groups:
            self.assertFalse(group["human_approved"])
            self.assertFalse(group["training_eligible"])
            self.assertEqual(group["geometry_role"], "recognizer_search_quad_not_glyph_mask")

    def test_input_pixels_are_unchanged_and_arrays_are_read_only(self):
        gray, _ = digits_fixture("100", scale=2)
        before = gray.copy()
        result = find_component_text_candidates(gray)
        np.testing.assert_array_equal(gray, before)
        np.testing.assert_array_equal(result.component_labels > 0, result.source_ink)
        self.assertFalse(result.source_ink.flags.writeable)
        self.assertFalse(result.component_labels.flags.writeable)
        self.assertFalse(np.shares_memory(result.source_ink, gray))
        self.assertFalse(np.shares_memory(result.component_labels, gray))
        self.assertTrue(result.provenance["source_pixels_unchanged"])
        self.assertFalse(result.provenance["erase_mask_generated"])

    def test_normalized_float_input_uses_the_same_source_ink_geometry(self):
        gray, _ = digits_fixture("200")
        first = find_component_text_candidates(gray)
        second = find_component_text_candidates(gray.astype(np.float32)/255)
        np.testing.assert_array_equal(first.source_ink, second.source_ink)
        np.testing.assert_array_equal(first.component_labels, second.component_labels)

    def test_invalid_inputs_and_unbounded_config_fail(self):
        for gray in (np.ones((2, 3, 4)), np.ones((0, 5)), np.ones((10, 10), bool),
                     np.full((10, 10), np.nan), np.full((10, 10), -1), np.full((10, 10), 256)):
            with self.assertRaises(ValueError): find_component_text_candidates(gray)
        gray, _ = digits_fixture("100")
        for origin in ((-.5, 0), (True, 0), (1,), (0., 0.)):
            with self.assertRaises(ValueError): find_component_text_candidates(gray, source_origin_xy=origin)
        for config in (replace(ComponentTextCandidateConfig(), max_group_members=6),
                       replace(ComponentTextCandidateConfig(), dark_gray_ceiling=255),
                       replace(ComponentTextCandidateConfig(), max_group_evaluations=10**10),
                       replace(ComponentTextCandidateConfig(), max_image_pixels=64),
                       replace(ComponentTextCandidateConfig(), max_total_components=2),
                       replace(ComponentTextCandidateConfig(), max_enclosed_holes_per_component=1),
                       replace(ComponentTextCandidateConfig(), max_hole_axis_evaluations=100_000),
                       replace(ComponentTextCandidateConfig(), hole_band_margin_height_ratio=float("nan"))):
            with self.assertRaises(ValueError): find_component_text_candidates(gray, config=config)


if __name__ == "__main__":
    unittest.main()
