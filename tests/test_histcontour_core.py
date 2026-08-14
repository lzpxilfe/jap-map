import json
import tempfile
import unittest
from pathlib import Path

from histcontour_core.contours import ContourLine, extract_visible_contours, generate_link_candidates
from histcontour_core.models import ControlPoint, MapProfile, MapSheet, MetadataError
from histcontour_core.pilot import BaselineMetrics, PILOT_SCENARIOS, PilotManifest, PilotManifestError, PilotSheet, make_baseline_report
from histcontour_core.registration import GroundControlPoint, RegistrationError, SheetRegistration, apply_projective, fit_projective


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.gcps = (
            GroundControlPoint(0, 0, 100, 200, "NW"),
            GroundControlPoint(99, 0, 200, 200, "NE"),
            GroundControlPoint(99, 99, 200, 100, "SE"),
            GroundControlPoint(0, 99, 100, 100, "SW"),
        )

    def test_projective_corners_and_serialization(self):
        registration = SheetRegistration.create("sheet-1", "scan.tif", 100, 100, "EPSG:9999", self.gcps)
        self.assertAlmostEqual(registration.transform(99, 99).x, 200)
        self.assertAlmostEqual(registration.transform(99, 99).y, 100)
        self.assertAlmostEqual(registration.rmse, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registration.json"
            registration.write_json(path)
            self.assertEqual(SheetRegistration.read_json(path), registration)

    def test_extra_gcp_and_invalid_control_points(self):
        extra = GroundControlPoint(50, 50, 150.505050505, 149.494949495, "centre")
        coefficients = fit_projective((*self.gcps, extra))
        self.assertAlmostEqual(apply_projective(coefficients, 50, 50).x, extra.map_x, places=6)
        with self.assertRaises(RegistrationError):
            fit_projective(self.gcps[:3])


class MetadataTest(unittest.TestCase):
    def test_profile_and_required_sheet_provenance(self):
        profile = MapProfile("series-a", "Series A", (100, 120, 140), crs_presets=("EPSG:4326",))
        self.assertEqual(MapProfile.from_dict(profile.to_dict()), profile)
        values = dict(sheet_id="a-1", source_title="原題", display_title="Display", series="A", edition="1", producer="Survey", survey_purpose="Topographic", survey_year="1930", publication_year="1932", scale="1:50000", contour_interval_m=20, source_language="ja", source_script="Jpan", horizontal_crs="EPSG:4301", vertical_datum="Unknown", scan_source="Archive", rights="Public domain")
        sheet = MapSheet(**values)
        self.assertEqual(sheet.sheet_id, "a-1")
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "sheet.json"
            sheet.write_json(metadata_path)
            self.assertEqual(MapSheet.read_json(metadata_path), sheet)
        values["producer"] = ""
        with self.assertRaises(MetadataError): MapSheet(**values)


class ContourTest(unittest.TestCase):
    def test_colour_extraction_and_non_crossing_candidate(self):
        background, contour = (255, 255, 255), (120, 80, 60)
        image = [[background for _ in range(12)] for _ in range(12)]
        for x in range(2, 10): image[5][x] = contour
        lines = extract_visible_contours(image, contour, 1, 3)
        self.assertTrue(lines)
        proposals = generate_link_candidates((ContourLine("a", ((0, 0), (1, 0)), 1), ContourLine("b", ((3, 0), (4, 0)), 1)), 4)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].status, "proposed")


class PilotTest(unittest.TestCase):
    def setUp(self):
        self.sheets = tuple(
            PilotSheet(f"sheet-{index}", scenario, f"data/raw/{scenario}.tif", "profile.json", f"data/derived/{scenario}.registration.json")
            for index, scenario in enumerate(PILOT_SCENARIOS, start=1)
        )

    def test_manifest_requires_the_three_initial_scenarios_and_round_trips(self):
        manifest = PilotManifest("korea-initial", self.sheets)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "pilot.json"
            manifest.write_json(manifest_path)
            self.assertEqual(PilotManifest.read_json(manifest_path), manifest)
        with self.assertRaises(PilotManifestError):
            PilotManifest("missing-scenario", self.sheets[:2])

    def test_manifest_can_include_an_extra_connected_sheet(self):
        extra = PilotSheet("sheet-4", "mountain_clear", "data/raw/extra.tif", "profile.json", "extra.registration.json")
        self.assertEqual(len(PilotManifest("four-sheet-block", (*self.sheets, extra)).sheets), 4)

    def test_source_controlled_four_sheet_pilot_is_complete(self):
        repository = Path(__file__).resolve().parents[1]
        manifest = PilotManifest.read_json(repository / "examples" / "korea_four_sheet_pilot" / "manifest.json")
        self.assertEqual(len(manifest.sheets), 4)
        profile = MapProfile.from_dict(json.loads((repository / manifest.sheets[0].profile_path).read_text(encoding="utf-8")))
        self.assertIsNone(profile.contour_rgb)
        self.assertEqual(profile.contour_rules["segmentation"]["backend"], "grayscale_ridge_v1")
        registrations = {}
        for pilot_sheet in manifest.sheets:
            metadata = MapSheet.read_json(repository / pilot_sheet.metadata_path)
            registration = SheetRegistration.read_json(repository / pilot_sheet.registration_path)
            self.assertEqual(metadata.sheet_id, pilot_sheet.sheet_id)
            self.assertEqual(registration.sheet_id, pilot_sheet.sheet_id)
            self.assertEqual(registration.crs_authid, "EPSG:5132")
            registrations[pilot_sheet.sheet_id] = registration
        self.assertAlmostEqual(registrations["174-cheongyang"].gcps[2].map_y, registrations["173-buyeo"].gcps[0].map_y)
        self.assertAlmostEqual(registrations["174-cheongyang"].gcps[1].map_x, registrations["178-gongju"].gcps[0].map_x)

    def test_baseline_report_is_diagnostic_not_an_accuracy_claim(self):
        lines = (ContourLine("a", ((0, 0), (3, 4)), 1),)
        metrics = BaselineMetrics.from_results(100, 100, lines, ())
        report = make_baseline_report("sheet-1", "profile-1", metrics, 1.0)
        self.assertEqual(report["sheet_id"], "sheet-1")
        self.assertEqual(metrics.endpoint_count, 2)
        self.assertAlmostEqual(metrics.total_visible_length_px, 5)
        self.assertIn("not precision/recall", report["interpretation"]["limitations"])


if __name__ == "__main__":
    unittest.main()
