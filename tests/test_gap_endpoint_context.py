import unittest
from histcontour_core.gap_endpoint_context import terminal_context, endpoint_banks
from histcontour_core.regional_gap_matching import match_endpoint_banks


def line(name, a, b, y):
    return {'source_path_id': name, 'points': [[x, y] for x in range(a, b+1)]}


class GapEndpointContextTests(unittest.TestCase):
    def test_two_ordered_lines_and_missing_middle_do_not_shift_neighbors(self):
        paths = [line('a'+str(y), 2, 15, y) for y in (12, 20, 28)]
        paths += [line('b'+str(y), 35, 48, y) for y in (12, 28)]
        result = terminal_context(paths, [18, 8, 32, 32], image_shape=(64, 64))
        a, b = endpoint_banks(result['endpoints'], [18, 8, 32, 32], [1, 0])
        match = match_endpoint_banks(a, b, ordering_axis=[0, 1])
        self.assertEqual([(m['start'][1], m['end'][1]) for m in match['best']['matches']], [(12, 12), (28, 28)])
        self.assertFalse(match['human_approved'])

    def test_terminal_touching_path_interior_is_not_a_gap(self):
        paths = [line('horizontal', 2, 30, 20), {'source_path_id': 'branch', 'points': [[15,y] for y in range(10,21)]}]
        result = terminal_context(paths, [18, 8, 32, 32], image_shape=(64,64))
        self.assertNotIn([15,20], [e['point'] for e in result['endpoints']])

    def test_short_tail_and_context_interior_are_excluded(self):
        result = terminal_context([line('short', 12, 15, 20), line('inside', 20, 30, 20)], [18,8,32,32], image_shape=(64,64))
        self.assertEqual(result['endpoints'], [])

    def test_smoothed_or_decimated_paths_are_rejected(self):
        for points in ([[[1.1,2],[2,2]]], [[[1,2],[3,2]]]):
            with self.assertRaises(ValueError):
                terminal_context([{'source_path_id':'x','points':points[0]}], [18,8,32,32], image_shape=(64,64))

    def test_ray_must_actually_enter_box(self):
        row = {'source_path_id':'x','point':[10,40],'outward_tangent':[1,0]}
        self.assertEqual(endpoint_banks([row], [18,8,32,32], [1,0]), [[],[]])


if __name__ == '__main__': unittest.main()
