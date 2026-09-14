import unittest
import numpy as np

from histcontour_core.numeric_line_partition import partition_pixel_line, native_part_coordinates, rasterize_line_cells


class NumericLinePartitionTests(unittest.TestCase):
    def test_exact_boundary_splits_preserve_every_geometric_part(self):
        mask = np.zeros((10, 12), bool)
        mask[4, 4:7] = True
        result = partition_pixel_line([[1., 4.], [10., 4.]], mask)
        self.assertEqual([p['kind'] for p in result['parts']], ['retained', 'numeric_candidate', 'retained'])
        self.assertEqual(result['parts'][1]['points'][0], [3.5, 4.])
        self.assertEqual(result['parts'][1]['points'][-1], [6.5, 4.])
        self.assertAlmostEqual(result['numeric_candidate_length_px'], 3.)
        self.assertAlmostEqual(result['partitioned_length_px'], 9.)

    def test_protected_and_ambiguous_ink_override_numeric_hypothesis(self):
        glyph = np.ones((10, 12), bool)
        protected = np.zeros_like(glyph); protected[:, 4:6] = True
        uncertain = np.zeros_like(glyph); uncertain[:, 6:8] = True
        result = partition_pixel_line([[1., 4.], [10., 4.]], glyph, protected=protected, ambiguous=uncertain)
        retained = [p for p in result['parts'] if p['kind'] == 'retained']
        self.assertEqual(len(retained), 1)
        self.assertAlmostEqual(retained[0]['length_px'], 4.)
        self.assertAlmostEqual(result['protected_overlap_length_px'], 2.)
        self.assertAlmostEqual(result['ambiguous_overlap_length_px'], 2.)

    def test_no_numeric_overlap_returns_exact_original_vertices_and_native_values(self):
        mask = np.zeros((10, 12), bool)
        points = [[1., 2.], [2., 3.], [2., 3.], [7., 8.]]
        result = partition_pixel_line(points, mask)
        self.assertEqual(result['parts'][0]['points'], points)
        native = [[127.+x*.000123, 36.-y*.000117] for x, y in points]
        self.assertEqual(native_part_coordinates(native, result['parts'][0]['source_edge_locations']), native)

    def test_native_interpolation_only_changes_new_cut_positions(self):
        glyph = np.zeros((10, 12), bool); glyph[4, 5] = True
        result = partition_pixel_line([[1., 4.], [10., 4.]], glyph)
        original = [[127.0003, 36.001], [127.0009, 36.001]]
        parts = [native_part_coordinates(original, p['source_edge_locations']) for p in result['parts']]
        self.assertEqual(parts[0][0], original[0]); self.assertEqual(parts[-1][-1], original[-1])
        self.assertEqual(parts[0][-1], parts[1][0]); self.assertEqual(parts[1][-1], parts[2][0])

    def test_diagonal_and_reversal_use_true_length_not_vertex_count(self):
        glyph = np.zeros((12, 12), bool); glyph[5, 5] = True
        forward = partition_pixel_line([[1., 1.], [10., 10.]], glyph)
        reverse = partition_pixel_line([[10., 10.], [1., 1.]], glyph)
        self.assertAlmostEqual(forward['numeric_candidate_length_px'], 2**.5)
        self.assertAlmostEqual(forward['numeric_candidate_length_px'], reverse['numeric_candidate_length_px'])
        self.assertAlmostEqual(forward['partitioned_length_px'], 9*2**.5)

    def test_closed_loop_and_inputs_are_preserved(self):
        glyph = np.zeros((12, 12), bool); glyph[1, 3:6] = True
        before = glyph.copy()
        points = [[1., 1.], [9., 1.], [9., 9.], [1., 9.], [1., 1.]]
        result = partition_pixel_line(points, glyph)
        np.testing.assert_array_equal(glyph, before)
        self.assertEqual(result['parts'][0]['points'][0], points[0])
        self.assertEqual(result['parts'][-1]['points'][-1], points[-1])
        self.assertAlmostEqual(result['partitioned_length_px'], 32.)
        self.assertTrue(all(not p['human_approved'] and not p['training_eligible'] for p in result['parts']))

    def test_invalid_grid_geometry_and_work_budget(self):
        glyph = np.zeros((12, 12), bool)
        for points in ([[1., 1.]], [[0., 0.], [13., 0.]], [[1., 1.], [float('nan'), 4.]], [[1., 1.], [1., 1.]]):
            with self.assertRaises(ValueError): partition_pixel_line(points, glyph)
        with self.assertRaises(ValueError): partition_pixel_line([[1., 1.], [10., 10.]], glyph, max_intervals=2)
        with self.assertRaises(ValueError): partition_pixel_line([[1., 1.], [2., 2.]], glyph.astype(float))
        with self.assertRaises(ValueError): partition_pixel_line([[1., 1.], [2., 2.]], glyph, protected=np.zeros((4, 4), bool))

    def test_exact_interval_budget_excludes_endpoints_and_duplicate_corner_cuts(self):
        mask = np.zeros((4, 4), bool)
        for points, budget in (([[.5, 1.], [1.5, 1.]], 1),
                               ([[.5, .5], [1.5, 1.5]], 1),
                               ([[0., 1.], [.5, 1.], [1.5, 1.]], 2),
                               ([[.5-1e-13, 1.], [1.5+1e-13, 1.]], 1)):
            result = partition_pixel_line(points, mask, max_intervals=budget)
            self.assertEqual(result['intervals_examined'], budget)

    def test_approved_protection_uses_same_tiny_crossed_cell_as_partitioner(self):
        points = [[.49, .4], [1.51, 1.5]]
        glyph = np.zeros((4, 4), bool); glyph[0, 1] = True
        unprotected = partition_pixel_line(points, glyph)
        self.assertGreater(unprotected['numeric_candidate_length_px'], .1)
        protected = rasterize_line_cells((4, 4), [points])
        self.assertTrue(protected[0, 1])
        result = partition_pixel_line(points, glyph, protected=protected)
        self.assertEqual(result['numeric_candidate_length_px'], 0.)
        self.assertEqual(result['parts'][0]['points'], points)


if __name__ == '__main__':
    unittest.main()
