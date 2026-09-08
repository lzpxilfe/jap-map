import unittest

from histcontour_core.ink import (
    ARCHAEOTRACE_UPSTREAM_COMMIT,
    INK_BACKEND_ID,
    InkCenterlineSettings,
    ink_centerline_candidates,
)
from histcontour_core.vectorization import skeleton_to_pixel_line_proposals


class InkSettingsTest(unittest.TestCase):
    def test_defaults_are_pinned_to_archaeotrace_ink_v2(self):
        settings = InkCenterlineSettings()
        self.assertEqual(settings.scales_px, (9, 15, 31))
        self.assertEqual(settings.tile_size_px, 128)
        self.assertEqual(INK_BACKEND_ID, "archaeotrace_ink_v2_evidence")
        self.assertEqual(ARCHAEOTRACE_UPSTREAM_COMMIT, "f55d45da6228bd0c60e02618a2bb5031a55c54b4")

    def test_invalid_scale_and_halo_are_rejected_without_numpy(self):
        with self.assertRaises(ValueError):
            InkCenterlineSettings(scales_px=(8, 15, 31))
        with self.assertRaises(ValueError):
            InkCenterlineSettings(tile_halo_px=4)


class InkCenterlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError:
            raise unittest.SkipTest("optional Ink backend dependency NumPy is not installed")
        cls.np = np

    def test_dark_stroke_produces_read_only_single_pixel_centerline(self):
        np = self.np
        image = np.full((96, 96), 245, dtype=np.uint8)
        image[12:84, 44:51] = 20
        result = ink_centerline_candidates(image, tile_origin=(128, 256))
        self.assertGreater(int(result.centerline.sum()), 60)
        self.assertTrue(result.centerline[16:80, 45:50].any())
        self.assertEqual(result.centerline.shape, image.shape)
        self.assertEqual(result.center_score.shape, image.shape)
        self.assertEqual(result.support_score.shape, image.shape)
        self.assertEqual(result.scale_px.shape, image.shape)
        self.assertEqual(result.tangent_x.shape, image.shape)
        self.assertEqual(result.tangent_y.shape, image.shape)
        self.assertEqual(result.coherence.shape, image.shape)
        self.assertFalse(result.centerline.flags.writeable)
        self.assertFalse(result.center_score.flags.writeable)
        self.assertFalse(result.support_score.flags.writeable)
        self.assertAlmostEqual(result.centerline_fraction, float(result.centerline.mean()))

    def test_rgb_channels_preserve_coloured_ink(self):
        np = self.np
        image = np.full((96, 96, 3), 240, dtype=np.uint8)
        image[46:51, 12:84] = (210, 35, 35)
        result = ink_centerline_candidates(image)
        self.assertGreater(int(result.centerline.sum()), 50)
        self.assertTrue(result.centerline[44:53, 16:80].any())

    def test_invalid_tile_origin_is_rejected(self):
        np = self.np
        image = np.full((32, 32), 255, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "integer"):
            ink_centerline_candidates(image, tile_origin=(0.5, 0))

    def test_existing_centerline_is_vectorized_without_rethinning(self):
        np = self.np
        skeleton = np.zeros((64, 96), dtype=bool)
        skeleton[32, 10:86] = True
        score = np.zeros(skeleton.shape, dtype=np.float32)
        score[skeleton] = 0.8
        proposals = skeleton_to_pixel_line_proposals(
            skeleton,
            score,
            minimum_length_px=20,
            simplify_tolerance_px=0.5,
            proposal_prefix="ink-line",
        )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].proposal_id, "ink-line-1")
        self.assertGreater(proposals[0].pixel_length, 70)
        self.assertAlmostEqual(proposals[0].confidence, 0.8, places=5)
        self.assertEqual(len(proposals[0].points), 2)

    def test_numpy_direction_fallback_produces_finite_axial_field(self):
        from histcontour_core.ink import _ink_evidence_direction
        np = self.np
        score = np.zeros((24, 48), dtype=np.float32)
        score[10:14, 5:43] = 0.8
        tangent_x, tangent_y, coherence = _ink_evidence_direction(np, None, score)
        self.assertTrue(np.isfinite(tangent_x).all())
        self.assertTrue(np.isfinite(tangent_y).all())
        self.assertGreater(float(coherence.max()), 0.1)

    def test_junctions_are_split_instead_of_guessed_through(self):
        np = self.np
        skeleton = np.zeros((48, 48), dtype=bool)
        skeleton[24, 5:43] = True
        skeleton[5:25, 24] = True
        proposals = skeleton_to_pixel_line_proposals(
            skeleton,
            minimum_length_px=8,
            simplify_tolerance_px=0,
        )
        self.assertGreaterEqual(len(proposals), 3)
        self.assertTrue(all(proposal.pixel_length >= 8 for proposal in proposals))


if __name__ == "__main__":
    unittest.main()
