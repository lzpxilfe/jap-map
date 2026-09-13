import unittest

import numpy as np
from PIL import Image, ImageDraw

from scripts.register_assisted_review_sketch import blue_centres, colour_masks, geotransform, register, register_board_panel, render_registered_preview, split_red_annotations


class AssistedReviewSketchTests(unittest.TestCase):
    def fixture(self):
        source = Image.new("RGB", (121, 121), "white")
        draw = ImageDraw.Draw(source)
        for i in range(11):
            points = [(int(8+i*10+5*np.sin(y*.1+i)), y) for y in range(121)]
            draw.line(points, fill=(35, 35, 35), width=2)
        draw.text((10, 8), "140", fill="black")
        endpoints = [[35, 43], [46, 80]]
        for x, y in endpoints:
            draw.ellipse((x-3, y-3, x+3, y+3), outline=(0, 100, 235), width=1)
        screenshot = Image.new("RGB", (660, 660), "white")
        screenshot.paste(source.resize((605, 605), Image.Resampling.BICUBIC), (19, 27))
        draw = ImageDraw.Draw(screenshot)
        draw.line([(90, 30), (120, 130), (70, 300), (180, 620)], fill=(185, 30, 5), width=5)
        return source, screenshot, endpoints

    def test_red_pen_is_not_orange_proposal_or_grey_ink(self):
        rgb = np.array([[[185, 30, 5], [230, 115, 0], [60, 60, 60], [0, 100, 235]]], dtype=np.uint8)
        blue, red, _ = colour_masks(rgb)
        self.assertEqual(red.tolist(), [[True, False, False, False]])
        self.assertEqual(blue.tolist(), [[False, False, False, True]])

    def test_registration_recovers_scale_translation_without_changing_images(self):
        source, screenshot, endpoints = self.fixture()
        before = [source.tobytes(), screenshot.tobytes()]
        result = register(source, screenshot, endpoints)
        fit = result["screenshot_from_crop"]
        self.assertAlmostEqual(fit["scale_x"], 5, delta=.03)
        self.assertAlmostEqual(fit["scale_y"], 5, delta=.03)
        # Resized pixel centres add (scale-1)/2 to the paste offset.
        self.assertAlmostEqual(fit["offset_x"], 21, delta=.6)
        self.assertAlmostEqual(fit["offset_y"], 29, delta=.6)
        self.assertGreater(result["background_alignment_ncc"], .95)
        self.assertEqual(before, [source.tobytes(), screenshot.tobytes()])

    def test_missing_blue_anchors_or_red_sketch_fails(self):
        with self.assertRaises(ValueError):
            blue_centres(np.full((100, 100, 3), 255, np.uint8))
        source, _, endpoints = self.fixture()
        with self.assertRaisesRegex(ValueError, "no substantial red"):
            register(source, source.resize((605, 605)), endpoints)

    def test_geotransform_preserves_pixel_centres_and_north_up(self):
        tile = {"bounds": [100., 200., 140., 300.], "pixel_bounds": [0, 0, 20, 25]}
        fit = {"scale_x": 5., "scale_y": 4., "offset_x": 2., "offset_y": -3.}
        gt = geotransform(tile, [3, 7, 10, 20], fit)
        for u, v in [(0., 0.), (20., 30.), (100.25, 77.5)]:
            crop_x, crop_y = (u-2)/5, (v+3)/4
            expected = [100+(3+crop_x+.5)*2, 300-(7+crop_y+.5)*4]
            actual = [gt[0]+(u+.5)*gt[1], gt[3]+(v+.5)*gt[5]]
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertGreater(gt[1], 0)
        self.assertLess(gt[5], 0)

    def board_fixture(self):
        source, _, _ = self.fixture()
        original = Image.new("RGB", (512, 300), "white")
        original.paste(source.resize((240, 240), Image.Resampling.NEAREST), (8, 41))
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        draw.line([(23, 43), (45, 83), (38, 180), (80, 280)], fill=(230, 20, 0), width=2)
        return source, original, annotated

    def test_board_registration_uses_verified_rendered_pixel_centres(self):
        source, original, annotated = self.board_fixture()
        panel, result = register_board_panel(source, annotated, original, 0)
        self.assertEqual(panel.size, (256, 300))
        self.assertEqual(result["board_panel_pixel_box"], [0, 0, 256, 300])
        self.assertEqual(result["board_unchanged_background_fraction"], 1.)
        fit = result["screenshot_from_crop"]
        self.assertEqual(fit["scale_x"], 240/121)
        self.assertEqual(fit["offset_x"], 8+(240/121-1)/2)
        self.assertEqual(fit["offset_y"], 41+(240/121-1)/2)

    def test_board_rejects_wrong_source_row_dimensions_or_changed_background(self):
        source, original, annotated = self.board_fixture()
        with self.assertRaisesRegex(ValueError, "source panel"):
            register_board_panel(source.transpose(Image.Transpose.FLIP_LEFT_RIGHT), annotated, original, 0)
        for row in (-1, 1, True):
            with self.assertRaises(ValueError):
                register_board_panel(source, annotated, original, row)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            register_board_panel(source, annotated.resize((1024, 600)), original, 0)
        changed = annotated.copy()
        ImageDraw.Draw(changed).rectangle((50, 100, 150, 200), fill=(120, 120, 120))
        with self.assertRaisesRegex(ValueError, "background changed"):
            register_board_panel(source, changed, original, 0)

    def test_board_does_not_treat_original_colour_markers_as_user_sketch(self):
        source, original, _ = self.board_fixture()
        with self.assertRaisesRegex(ValueError, "no substantial red"):
            register_board_panel(source, original, original, 0)

    def test_low_resolution_pen_pixel_keeps_its_area_in_preview(self):
        red = np.zeros((20, 20), dtype=bool)
        red[10, 10] = True
        reference = Image.new("RGB", (10, 10), "white")
        fit = {"scale_x": 2., "scale_y": 2., "offset_x": 0., "offset_y": 0.}
        preview = render_registered_preview(reference, red, fit, "fixture")
        self.assertGreaterEqual(int(np.all(np.asarray(preview) == [190, 35, 10], axis=2).sum()), 4)

    def test_annotation_split_preserves_every_red_pixel_without_approving_paths(self):
        red = np.zeros((20, 30), dtype=bool)
        red[4:12, 3] = True
        red[5:8, 23:26] = True
        before = red.copy()
        sketch, symbols = split_red_annotations(red, [[22, 4, 28, 10]])
        np.testing.assert_array_equal(sketch | symbols, red)
        self.assertFalse((sketch & symbols).any())
        np.testing.assert_array_equal(red, before)
        self.assertEqual(int(symbols.sum()), 9)
        self.assertEqual(int(sketch.sum()), 8)
        for box in ([0, 0, 1, 1], [-1, 0, 4, 4], [22., 4, 28, 10]):
            with self.assertRaises(ValueError):
                split_red_annotations(red, [box])


if __name__ == "__main__":
    unittest.main()
