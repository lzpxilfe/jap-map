import unittest

from jap_map.core.coordinates import CoordinateParseError, parse_angle


class CoordinateParserTest(unittest.TestCase):
    def test_decimal(self):
        self.assertAlmostEqual(parse_angle("37.5", "lat"), 37.5)
        self.assertAlmostEqual(parse_angle("127.25", "lon"), 127.25)

    def test_degree_minute_second_and_cardinal(self):
        self.assertAlmostEqual(parse_angle("37°30′00″N", "lat"), 37.5)
        self.assertAlmostEqual(parse_angle("127 15 00 E", "lon"), 127.25)
        self.assertAlmostEqual(parse_angle("37:30:00S", "lat"), -37.5)

    def test_negative_decimal(self):
        self.assertAlmostEqual(parse_angle("-37.5", "lat"), -37.5)
        self.assertAlmostEqual(parse_angle("-127.25", "lon"), -127.25)

    def test_axis_rejects_wrong_cardinal(self):
        with self.assertRaises(CoordinateParseError):
            parse_angle("127E", "lat")
        with self.assertRaises(CoordinateParseError):
            parse_angle("37N", "lon")

    def test_rejects_sign_and_cardinal(self):
        with self.assertRaises(CoordinateParseError):
            parse_angle("-37N", "lat")

    def test_rejects_invalid_minutes_and_range(self):
        with self.assertRaises(CoordinateParseError):
            parse_angle("37°60′", "lat")
        with self.assertRaises(CoordinateParseError):
            parse_angle("91", "lat")
        with self.assertRaises(CoordinateParseError):
            parse_angle("181", "lon")


if __name__ == "__main__":
    unittest.main()
