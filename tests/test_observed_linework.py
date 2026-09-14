"""Synthetic observed-ink contracts; not historical contour accuracy."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from histcontour_core.observed_linework import extract_observed_linework, ObservedLineworkConfig


class ObservedLineworkTests(unittest.TestCase):
    def fake_evidence(self, points, strong_points=()):
        image = np.ones((64, 64), dtype=np.float32)
        support, tx, ty = [np.zeros_like(image) for _ in range(3)]
        for x, y, dx, dy in points:
            image[y, x] = .3
            support[y, x], tx[y, x], ty[y, x] = .08, dx, dy
        for x, y in strong_points:
            support[y, x] = .5
        return image, SimpleNamespace(support_score=support, tangent_x=tx, tangent_y=ty,
            coherence=support*.9, centerline=support > 0)

    def line_image(self, faint=False):
        yy, xx = np.indices((64, 96))
        strength = np.where(xx < 40, .8, .08) if faint else .6
        return (1-strength*np.exp(-.5*((yy-30)/.9)**2)).astype(np.float32)

    def test_blank_never_creates_observed_or_inferred_ink(self):
        r = extract_observed_linework(np.ones((32, 40), np.float32))
        self.assertEqual(r["paths"], [])
        self.assertEqual(r["stage_counts"]["retained_observed"], 0)

    def test_faint_extension_survives_without_mutating_source(self):
        image = self.line_image(True)
        original = image.copy()
        r = extract_observed_linework(image)
        np.testing.assert_array_equal(image, original)
        self.assertTrue(r["masks"]["skeleton"][28:33, 60:85].any(axis=0).all())
        self.assertTrue(any(p["role"] == "observed_linework_review" for p in r["paths"]))
        self.assertFalse(r["contour_semantics_assigned"])
        self.assertTrue(all(not p["inferred_gap"] for p in r["paths"]))

    def test_parallel_lines_remain_disconnected(self):
        yy, xx = np.indices((64, 96))
        image = 1-.6*np.exp(-.5*((yy-20)/.8)**2)-.3*np.exp(-.5*((yy-28)/.8)**2)
        r = extract_observed_linework(image)
        self.assertFalse(r["masks"]["skeleton"][24, 10:85].any())
        for p in r["paths"]:
            ys = [q[1] for q in p["points"]]
            self.assertLess(max(ys)-min(ys), 5)

    def test_white_gap_is_not_filled(self):
        image = self.line_image()
        image[:, 43:54] = 1
        r = extract_observed_linework(image)
        self.assertFalse(r["masks"]["skeleton"][:, 46:51].any())
        for p in r["paths"]:
            xs = [q[0] for q in p["points"]]
            self.assertFalse(min(xs) < 43 and max(xs) > 54)

    def test_full_component_text_evidence_is_separate_not_deleted(self):
        image = self.line_image()
        avoidance = np.zeros_like(image)
        avoidance[26:35] = .6
        r = extract_observed_linework(image, text_avoidance_score=avoidance)
        self.assertTrue(any(p["role"] == "suspected_text_review" for p in r["paths"]))
        self.assertTrue(r["masks"]["suspected_text_skeleton"].any())
        self.assertTrue(r["paths"])

    def test_small_text_box_cannot_delete_throughgoing_line(self):
        image = self.line_image()
        avoidance = np.zeros_like(image)
        avoidance[26:35, 40:50] = .6
        r = extract_observed_linework(image, text_avoidance_score=avoidance)
        self.assertFalse(any(p["role"] == "suspected_text_review" for p in r["paths"]))

    def test_masks_readonly_paths_keep_raw_vertices(self):
        r = extract_observed_linework(self.line_image())
        self.assertGreater(max(len(p["points"]) for p in r["paths"]), 60)
        with self.assertRaises(ValueError):
            r["masks"]["skeleton"][0, 0] = True

    def test_invalid_inputs_and_resource_limit(self):
        for a in (np.zeros((2, 30)), np.full((32, 32), np.nan), np.full((32, 32), -1)):
            with self.assertRaises(ValueError):
                extract_observed_linework(a)
        with self.assertRaises(ValueError):
            extract_observed_linework(self.line_image(), config=ObservedLineworkConfig(max_pixels=64))
        with self.assertRaises(ValueError):
            extract_observed_linework(self.line_image(), text_avoidance_score=np.zeros((3, 3)))

    def test_long_weak_branch_cannot_bypass_directional_rejection(self):
        horizontal = [(x, 32, 1., 0.) for x in range(8, 56)]
        branch = [(32, y, 0., 1.) for y in range(33, 57)]
        image, evidence = self.fake_evidence(horizontal+branch, [(x, 32) for x in range(8, 56)])
        with patch('histcontour_core.observed_linework.ink_centerline_candidates', return_value=evidence):
            result = extract_observed_linework(image)
        self.assertFalse(result['masks']['directionally_grown'][34:57, 32].any())
        self.assertFalse(result['masks']['retained_observed'][34:57, 32].any())

    def test_isolated_faint_length_is_not_a_rotation_biased_vertex_count(self):
        fixtures = [[(x, 20, 1., 0.) for x in range(10, 29)],
                    [(10+i, 10+i, 2**-.5, 2**-.5) for i in range(14)]]
        for points in fixtures:
            image, evidence = self.fake_evidence(points)
            with patch('histcontour_core.observed_linework.ink_centerline_candidates', return_value=evidence):
                result = extract_observed_linework(image)
            long = [p for p in result['paths'] if p['role'] == 'observed_linework_review']
            self.assertEqual(len(long), 1)
            self.assertGreaterEqual(long[0]['length_px'], 18)
            self.assertFalse(long[0]['strong_seed_connected'])


if __name__ == "__main__":
    unittest.main()
