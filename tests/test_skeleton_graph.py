"""Connectivity contracts for vectorization; not semantic contour accuracy."""

import unittest

import numpy as np

from histcontour_core.vectorization import _join_tangent_pairs, _trace_skeleton_array, skeleton_to_pixel_line_proposals


class SkeletonGraphTest(unittest.TestCase):
    def test_connected_staircase_is_not_lost_to_false_junctions(self):
        mask = np.zeros((64, 64), dtype=bool)
        for n in range(5, 55):
            mask[n, n:n + 2] = True
        legacy = skeleton_to_pixel_line_proposals(mask, minimum_length_px=18, diagonal_policy="full8")
        self.assertEqual(legacy, [])
        proposals = skeleton_to_pixel_line_proposals(mask, minimum_length_px=18, simplify_tolerance_px=0)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].pixel_length, 99)
        self.assertEqual(set(proposals[0].points), {(float(x), float(y)) for y, x in zip(*np.nonzero(mask))})

    def test_true_orthogonal_and_diagonal_junctions_remain_split(self):
        for diagonal in (False, True):
            mask = np.zeros((41, 41), dtype=bool)
            if diagonal:
                for n in range(5, 36):
                    mask[n, n] = mask[n, 40 - n] = True
            else:
                mask[20, 5:36] = True
                mask[5:36, 20] = True
            lines = skeleton_to_pixel_line_proposals(mask, minimum_length_px=3, simplify_tolerance_px=0)
            self.assertEqual(len(lines), 4)
            for line in lines:
                self.assertIn((20.0, 20.0), (line.points[0], line.points[-1]))

    def test_closed_loop_remains_closed(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[5, 5:26] = mask[25, 5:26] = True
        mask[5:26, 5] = mask[5:26, 25] = True
        lines = skeleton_to_pixel_line_proposals(mask, minimum_length_px=5, simplify_tolerance_px=0)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].points[0], lines[0].points[-1])
        self.assertEqual(lines[0].pixel_length, 80)

    def test_close_parallel_lines_and_real_gaps_are_not_joined(self):
        mask = np.zeros((32, 64), dtype=bool)
        mask[12, 4:24] = mask[12, 27:58] = mask[14, 4:58] = True
        lines = skeleton_to_pixel_line_proposals(mask, minimum_length_px=5, simplify_tolerance_px=0)
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertEqual(len({point[1] for point in line.points}), 1)
            self.assertTrue(all(mask[int(y), int(x)] for x, y in line.points))

    def test_random_masks_preserve_all_nonisolated_foreground_without_new_edges(self):
        rng = np.random.default_rng(918)
        for _ in range(30):
            mask = rng.random((12, 15)) < .25
            old = _trace_skeleton_array(mask, np, diagonal_policy="full8")
            new = _trace_skeleton_array(mask, np, diagonal_policy="corner_safe")
            self.assertEqual({point for line in old for point in line}, {point for line in new for point in line})
            edges = []
            for line in new:
                edges.extend(tuple(sorted((a, b))) for a, b in zip(line, line[1:]))
            self.assertEqual(len(edges), len(set(edges)))
            self.assertTrue(all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) == 1 for a, b in edges))

    def test_invalid_policy_fails(self):
        with self.assertRaisesRegex(ValueError, "diagonal_policy"):
            skeleton_to_pixel_line_proposals(np.zeros((10, 10), dtype=bool), diagonal_policy="guess_crossings")

    def test_optional_pairing_keeps_true_crossings_and_gaps_unjoined(self):
        mask = np.zeros((41, 41), dtype=bool)
        mask[20, 5:36] = True
        mask[5:36, 20] = True
        lines = skeleton_to_pixel_line_proposals(mask, minimum_length_px=3, junction_policy="tangent_pairs")
        self.assertEqual(len(lines), 4)
        mask[20, 20] = False
        lines = skeleton_to_pixel_line_proposals(mask, minimum_length_px=3, junction_policy="tangent_pairs")
        self.assertTrue(all((20.,20.) not in line.points for line in lines))

    def test_optional_pairing_joins_opposing_arms_but_preserves_side_branch(self):
        source = [((0.,0.), (10.,0.)), ((10.,0.), (20.,0.)), ((10.,0.), (10.,5.))]
        joined = _join_tangent_pairs(source)
        self.assertEqual(len(joined), 2)
        self.assertIn(((0.,0.), (10.,0.), (20.,0.)), joined)
        self.assertIn(((10.,0.), (10.,5.)), joined)
        self.assertEqual({point for line in source for point in line}, {point for line in joined for point in line})

    def test_optional_pairing_rejects_ambiguous_y_and_short_arms(self):
        source = [((0.,0.), (10.,0.)), ((10.,0.), (20.,3.)), ((10.,0.), (20.,-3.))]
        self.assertEqual(len(_join_tangent_pairs(source)), 3)
        source = [((0.,0.), (4.,0.)), ((4.,0.), (8.,0.)), ((4.,0.), (4.,5.))]
        self.assertEqual(len(_join_tangent_pairs(source)), 3)


if __name__ == "__main__":
    unittest.main()
