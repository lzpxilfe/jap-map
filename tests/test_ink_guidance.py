import unittest

from histcontour_core.manual_gap_bridge import ManualGapBridgeError, build_manual_gap_bridge, sample_evidence_tangent
from histcontour_core.provenance import execution_id, polyline_geometry_id


class InkGuidanceTest(unittest.TestCase):
    def test_geometry_identity_ignores_trace_direction(self):
        forward = polyline_geometry_id(((1, 2), (3, 4), (5, 6)))
        backward = polyline_geometry_id(((5, 6), (3, 4), (1, 2)))
        self.assertEqual(forward, backward)

    def test_execution_identity_changes_when_settings_change(self):
        common = {"raster_sha256": "a" * 64, "backend": "ink", "upstream_commit": "b" * 40, "vectorization": {"minimum": 18}}
        first = execution_id(**common, settings={"scale": 9})
        second = execution_id(**common, settings={"scale": 15})
        self.assertNotEqual(first, second)

    def test_manual_gap_bridge_preserves_explicit_endpoints(self):
        bridge = build_manual_gap_bridge((10, 10), (50, 10), (1, 0), (1, 0))
        self.assertEqual(bridge.points[0], (10.0, 10.0))
        self.assertEqual(bridge.points[-1], (50.0, 10.0))
        self.assertLessEqual(bridge.detour_ratio, 1.25)

    def test_manual_gap_rejects_perpendicular_tangent(self):
        with self.assertRaises(ManualGapBridgeError):
            build_manual_gap_bridge((10, 10), (50, 10), (0, 1), (0, 1))

    def test_run_comparison_reports_both_pixel_and_segment_change(self):
        from scripts.compare_ink_runs import compare
        report = compare(
            {"version": "1", "backend": "old", "tiles": [{"tile_id": "a", "centerline_pixels": 10, "proposal_count": 2}]},
            {"version": "2", "backend": "new", "tiles": [{"tile_id": "a", "centerline_pixels": 12, "proposal_count": 4}]},
        )
        row = report["tiles"][0]
        self.assertEqual(row["centerline_pixel_delta"], 2)
        self.assertEqual(row["proposal_delta"], 2)


class InkDirectionAndLiveWireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("Ink guidance requires NumPy")
        from histcontour_core.ink import ink_centerline_candidates
        cls.np = np
        image = np.full((80, 112), 245, dtype=np.uint8)
        image[36:43, 10:102] = 15
        cls.evidence = ink_centerline_candidates(image)

    def test_ink_provides_immutable_direction_fields(self):
        evidence = self.evidence
        self.assertEqual(evidence.tangent_x.shape, evidence.centerline.shape)
        self.assertEqual(evidence.coherence.shape, evidence.centerline.shape)
        self.assertFalse(evidence.tangent_x.flags.writeable)
        self.assertGreater(float(evidence.coherence[evidence.centerline].max()), 0.1)

    def test_livewire_follows_stroke_and_guidance_remains_soft(self):
        from histcontour_core.livewire import trace_ink_path
        from histcontour_core.trace_guidance import guidance_from_boxes
        guidance = guidance_from_boxes(self.evidence.centerline.shape, ((48, 32, 62, 48),))
        path = trace_ink_path(self.evidence, (12, 39), (100, 39), guidance=guidance)
        self.assertEqual(path.points[0], (12.0, 39.0))
        self.assertEqual(path.points[-1], (100.0, 39.0))
        self.assertTrue(path.used_guidance)
        self.assertGreater(len(path.points), 30)

    def test_sampled_tangent_supports_manual_bridge(self):
        tangent = sample_evidence_tangent(self.evidence, (15, 39))
        self.assertIsNotNone(tangent)
        self.assertGreater(abs(tangent[0]), 0.75)

    def test_explicit_label_gap_bridge_uses_local_ink_not_parallel_line(self):
        np = self.np
        from histcontour_core.ink import ink_centerline_candidates
        image = np.full((88, 120), 245, dtype=np.uint8)
        image[37:43, 10:48] = 20
        image[37:43, 72:110] = 20
        image[51:57, 10:110] = 20  # tempting remote parallel contour
        image[29:49, 52:68] = 25   # printed label inside the actual blank
        evidence = ink_centerline_candidates(image)
        first, second = sample_evidence_tangent(evidence, (44, 40)), sample_evidence_tangent(evidence, (76, 40))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        bridge = build_manual_gap_bridge((44, 40), (76, 40), first, second)
        self.assertLess(max(abs(y - 40) for _x, y in bridge.points), 4.0)


if __name__ == "__main__":
    unittest.main()
