import unittest

from histcontour_core.synthetic_ink import SCENES_PER_TERRAIN, build_synthetic_case, terrain_split


class SyntheticInkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            import PIL
        except ImportError:
            raise unittest.SkipTest("synthetic Ink fixtures need NumPy and Pillow")

    def test_fixed_terrain_split_has_no_variant_leakage(self):
        self.assertEqual(terrain_split(0), "train")
        self.assertEqual(terrain_split(79), "train")
        self.assertEqual(terrain_split(80), "validation")
        self.assertEqual(terrain_split(100), "test")
        self.assertEqual(SCENES_PER_TERRAIN, 8)

    def test_label_gap_truth_stays_separate_from_visible_truth(self):
        case = build_synthetic_case(83, 3, size=128)
        self.assertEqual(case.image.shape, (128, 128, 3))
        self.assertGreater(int(case.complete_contour.sum()), int(case.visible_contour.sum()))
        self.assertGreater(int(case.label_gap.sum()), 0)
        self.assertTrue((case.label_gap & case.complete_contour & ~case.visible_contour).any())

    def test_non_contour_truth_is_present_in_crossing_scene(self):
        case = build_synthetic_case(12, 7, size=128)
        self.assertGreater(int(case.non_contour.sum()), 0)

    def test_segment_patch_carries_two_scales_and_three_positions(self):
        from histcontour_core.patch_context import patch_metadata, segment_context_tensor
        case = build_synthetic_case(12, 5, size=128)
        tensor = segment_context_tensor(case.image, ((12.0, 35.0), (64.0, 40.0), (116.0, 36.0)))
        metadata = patch_metadata()
        self.assertEqual(tensor.shape, (metadata["channels"], metadata["patch_size"], metadata["patch_size"]))
        self.assertGreater(float(tensor[1::2].sum()), 0.0)

    def test_case_writer_keeps_all_four_truth_surfaces(self):
        from scripts.run_synthetic_ink_experiment import write_cases
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_cases(output, terrain_count=1, variants=1, size=96)
            case_dir = output / "cases"
            self.assertTrue((case_dir / "terrain-000-variant-0.png").is_file())
            self.assertTrue((case_dir / "terrain-000-variant-0-visible_contour.png").is_file())
            self.assertTrue((case_dir / "terrain-000-variant-0-complete_contour.png").is_file())
            self.assertTrue((case_dir / "terrain-000-variant-0-non_contour.png").is_file())
            self.assertTrue((case_dir / "terrain-000-variant-0-label_gap.png").is_file())


if __name__ == "__main__":
    unittest.main()
