import unittest

from jap_map.core.frame import Corner, CornerRole, FrameValidationError, SheetFrame


def normal_frame():
    return {
        CornerRole.NW: Corner(CornerRole.NW, 38, 127),
        CornerRole.NE: Corner(CornerRole.NE, 38, 128),
        CornerRole.SE: Corner(CornerRole.SE, 37, 128),
        CornerRole.SW: Corner(CornerRole.SW, 37, 127),
    }


class FrameValidationTest(unittest.TestCase):
    def test_preserves_exact_quadrilateral(self):
        corners = normal_frame()
        corners[CornerRole.NE] = Corner(CornerRole.NE, 38.1, 128.2)
        frame = SheetFrame.create("sample", "EPSG:5132", corners)
        self.assertEqual(frame.ring_xy(), ((127, 38), (128.2, 38.1), (128, 37), (127, 37)))

    def test_rejects_duplicate_corner(self):
        corners = normal_frame()
        corners[CornerRole.SE] = corners[CornerRole.NE]
        with self.assertRaises(FrameValidationError):
            SheetFrame.create("sample", "EPSG:5132", corners)

    def test_rejects_crossing_or_wrong_role_order(self):
        corners = normal_frame()
        corners[CornerRole.SE] = Corner(CornerRole.SE, 39, 128)
        with self.assertRaises(FrameValidationError):
            SheetFrame.create("sample", "EPSG:5132", corners)

    def test_rejects_antimeridian_span(self):
        corners = {
            CornerRole.NW: Corner(CornerRole.NW, 10, 179),
            CornerRole.NE: Corner(CornerRole.NE, 10, -179),
            CornerRole.SE: Corner(CornerRole.SE, 9, -179),
            CornerRole.SW: Corner(CornerRole.SW, 9, 179),
        }
        with self.assertRaises(FrameValidationError):
            SheetFrame.create("sample", "EPSG:4326", corners)


if __name__ == "__main__":
    unittest.main()
