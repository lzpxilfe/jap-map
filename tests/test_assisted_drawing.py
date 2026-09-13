import copy
import math
import unittest

import numpy as np

from histcontour_core.assisted_drawing import DrawingConfig, assess_path, endpoint, endpoint_pairs, native_feature_collection
from scripts.generate_assisted_contour_drawing import world
from histcontour_core.human_feedback import map_to_pixel
from scripts.screen_assisted_contour_drawing import screening_reasons


def line(uid, points, retained=True):
    return {"uid": uid, "points": points, "retained": retained, "score": .8}


class AssistedDrawingTests(unittest.TestCase):
    def test_native_geojson_has_explicit_crs_and_does_not_mutate_inputs(self):
        features = [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[1., 2.], [3., 4.]]}}]
        before = copy.deepcopy(features)
        result = native_feature_collection(features, [{"crs_authid": "EPSG:5132"}])
        self.assertEqual(result["crs"]["properties"]["name"], "EPSG:5132")
        result["features"][0]["properties"]["test"] = True
        self.assertEqual(features, before)
        for tiles in ([], [{}], [{"crs_authid": "EPSG:5132"}, {"crs_authid": "EPSG:4326"}]):
            with self.assertRaises(ValueError):
                native_feature_collection(features, tiles)
        with self.assertRaises(ValueError):
            native_feature_collection(features, [{"crs_authid": "EPSG:5132"}],
                                      source_crs={"type": "name", "properties": {"name": "EPSG:4326"}})

    def test_simple_gap_is_proposal_and_originals_are_unchanged(self):
        lines = [line("left", [(0, 30), (30, 30)]), line("right", [(40, 30), (70, 30)])]
        before = copy.deepcopy(lines)
        pairs, audit = endpoint_pairs(lines)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["gap_pixels"], 10)
        self.assertEqual(lines, before)
        self.assertEqual(audit["eligible_endpoints"], 4)
        self.assertNotIn("human_approved", pairs[0])

    def test_rejected_branch_still_counts_as_a_junction(self):
        lines = [line("left", [(0, 30), (30, 30)]), line("right", [(40, 30), (70, 30)]),
                 line("side", [(30, 30), (30, 50)], False)]
        self.assertEqual(endpoint_pairs(lines)[0], [])

    def test_parallel_or_backward_endpoints_are_not_bridged(self):
        for target in ([(0, 40), (30, 40)], [(40, 30), (40, 60)]):
            self.assertEqual(endpoint_pairs([line("a", [(0, 30), (30, 30)]), line("b", target)])[0], [])

    def test_competing_pairs_are_marked_not_merged(self):
        lines = [line("a", [(0, 30), (30, 30)]), line("b", [(45, 28), (75, 28)]),
                 line("c", [(45, 32), (75, 32)])]
        pairs, _ = endpoint_pairs(lines)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["competing_endpoint_pair"] for pair in pairs))

    def test_long_gaps_closed_inputs_duplicates_and_invalids(self):
        self.assertEqual(endpoint_pairs([line("a", [(0, 30), (30, 30)]), line("b", [(140, 30), (180, 30)])])[0], [])
        self.assertEqual(endpoint_pairs([line("ring", [(0, 0), (30, 0), (30, 30), (0, 0)])])[0], [])
        with self.assertRaises(ValueError):
            endpoint_pairs([line("x", [(0, 0), (30, 0)]), line("x", [(40, 0), (70, 0)])])
        with self.assertRaises(ValueError):
            endpoint_pairs([line("x", [(math.nan, 0), (30, 0)])])
        with self.assertRaises(ValueError):
            DrawingConfig(maximum_gap=129).validate()

    def test_endpoint_uses_distance_not_vertex_count(self):
        point, inner, direction = endpoint([(0, 0), (100, 0)], 1)
        self.assertEqual(point, (100, 0))
        self.assertEqual(inner, (88, 0))
        self.assertEqual(direction, (1, 0))

    def test_route_quality_distinguishes_new_geometry_and_bad_endpoints(self):
        score = np.ones((40, 80), np.float32)
        original = np.zeros_like(score, dtype=bool)
        original[20, :21] = True
        original[20, 50:] = True
        quality = assess_path([(20, 20), (50, 20)], (20, 20), (50, 20), score, original)
        self.assertEqual(quality["supported_fraction"], 1)
        self.assertEqual(quality["detour_ratio"], 1)
        self.assertGreater(quality["new_fraction_outside_original_1_5px"], .8)
        shifted = assess_path([(20, 20), (48, 20)], (20, 20), (50, 20), score, original)
        self.assertEqual(shifted["endpoint_error_pixels"], 2)
        with self.assertRaises(ValueError):
            assess_path([(20, 20), (80, 20)], (20, 20), (80, 20), score, original)

    def test_world_coordinates_preserve_pixel_centres_non_square_grid(self):
        tile = {"bounds": [100., 200., 140., 300.], "pixel_bounds": [50, 60, 20, 25]}
        points = [[0., 0.], [19., 24.], [3.25, 8.5]]
        mapped = world(tile, points)
        self.assertEqual(mapped[0], [101., 298.])
        np.testing.assert_allclose(map_to_pixel(tile, mapped), points, atol=1e-12)

    def test_screen_rejects_crossed_third_line_but_not_own_anchor(self):
        row = {"mode": "ink_livewire", "gap_pixels": 20., "source_uid": "a", "target_uid": "b",
               "pixel_points": [(10, 20), (30, 20)], "start": (10, 20), "end": (30, 20)}
        owner = np.zeros((40, 50), np.int32)
        owner[20, 10:15] = 1
        owner[20, 28:31] = 2
        self.assertEqual(screening_reasons(row, owner, {"a": 1, "b": 2}), [])
        owner[18:23, 20] = 3
        self.assertIn("crosses_or_touches_third_retained_source_line", screening_reasons(row, owner, {"a": 1, "b": 2}))

    def test_automatic_long_inferred_gaps_are_not_a_safe_draft(self):
        row = {"mode": "contextual_gap", "gap_pixels": 20., "source_uid": "a", "target_uid": "b",
               "pixel_points": [(10, 20), (30, 20)], "start": (10, 20), "end": (30, 20)}
        reasons = screening_reasons(row, np.zeros((40, 50), np.int32), {"a": 1, "b": 2})
        self.assertEqual(reasons, ["automatic_inferred_gap_over_16px_requires_explicit_anchor_review"])


if __name__ == "__main__":
    unittest.main()
