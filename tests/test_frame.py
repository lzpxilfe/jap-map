import unittest

from jap_map.core.frame import Corner, CornerRole, FrameValidationError, SheetFrame


def normal_frame():
    return {
        CornerRole.NW: Corner(CornerRole.NW, 127, 38),
        CornerRole.NE: Corner(CornerRole.NE, 128, 38),
        CornerRole.SE: Corner(CornerRole.SE, 128, 37),
        CornerRole.SW: Corner(CornerRole.SW, 127, 37),
    }


class FrameValidationTest(unittest.TestCase):
    def test_preserves_exact_quadrilateral(self):
        corners = normal_frame()
        corners[CornerRole.NE] = Corner(CornerRole.NE, 128.2, 38.1)
        frame = SheetFrame.create("sample", "EPSG:5132", corners)
        self.assertEqual(frame.ring_xy(), ((127, 38), (128.2, 38.1), (128, 37), (127, 37)))

    def test_rejects_duplicate_corner(self):
        corners = normal_frame()
        corners[CornerRole.SE] = corners[CornerRole.NE]
        with self.assertRaises(FrameValidationError):
            SheetFrame.create("sample", "EPSG:5132", corners)

    def test_rejects_crossing_or_wrong_role_order(self):
        corners = normal_frame()
        corners[CornerRole.SE] = Corner(CornerRole.SE, 127, 37)
        corners[CornerRole.SW] = Corner(CornerRole.SW, 128, 37)
        with self.assertRaises(FrameValidationError):
            SheetFrame.create("sample", "EPSG:5132", corners)

    def test_rejects_crossing_coordinates_in_any_crs(self):
        corners = normal_frame()
        corners[CornerRole.SE] = Corner(CornerRole.SE, 127, 37)
        corners[CornerRole.SW] = Corner(CornerRole.SW, 128, 37)
        with self.assertRaises(FrameValidationError):
            SheetFrame.create("sample", "EPSG:9999", corners)


if __name__ == "__main__":
    unittest.main()
