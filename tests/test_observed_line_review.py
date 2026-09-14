"""Exact-geometry question packets, distinct from human answers or training."""
import json
import unittest

from histcontour_core.human_feedback import geometry_digest
from scripts.prepare_observed_line_review import run
from tests import test_observed_reconstruction_script as fixtures


class ObservedLineReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ObservedReconstructionRunTests()
        self.fixture.setUp()
        self.fixture.paths = [fixtures.path_record([[22., 25.], [28., 30.], [35., 26.]])]
        self.fixture.run_fixture()
        self.output = self.fixture.root/"review"

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self, cases=(("N001", "R001", "fixture-001"),)):
        f = self.fixture
        return run(f.packet, f.manifest_path, f.output, self.output, cases)

    def test_question_preserves_exact_geometry_unknown_decisions_and_nontraining_role(self):
        before = (self.fixture.output/"candidate-observed.geojson").read_bytes()
        result = self.prepare()
        self.assertEqual(before, (self.fixture.output/"candidate-observed.geojson").read_bytes())
        original = json.loads(before)["features"][0]
        row = result["questions"][0]
        selected = json.loads((self.output/"question-geometries.geojson").read_text())["features"][0]
        self.assertEqual(selected["geometry"], original["geometry"])
        self.assertEqual(row["candidate_geometry_sha256"], geometry_digest(original["geometry"]))
        self.assertEqual(row["human_semantics"], "unknown")
        self.assertEqual(row["human_shape"], "unknown")
        self.assertFalse(row["human_approved"])
        self.assertFalse(row["training_eligible"])
        self.assertEqual(result["human_approvals"], 0)
        self.assertTrue((self.output/"N001.png").is_file())

    def test_empty_duplicate_invalid_ids_and_more_than_three_questions_rejected(self):
        for cases in ([], [("N001", "R001", "fixture-001")]*2,
                      [("bad", "R001", "fixture-001")], [("N001", "unknown", "fixture-001")],
                      [(f"N{i:03d}", "R001", "fixture-001") for i in range(4)]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.prepare(cases)
            self.assertFalse(self.output.exists())

    def test_incorrect_native_crs_is_rejected_even_when_both_collections_agree(self):
        for name in ("raw-observed.geojson", "candidate-observed.geojson"):
            path = self.fixture.output/name
            value = json.loads(path.read_text())
            value["crs"]["properties"]["name"] = "EPSG:4326"
            path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "native source CRS"):
            self.prepare()

    def test_source_mutation_is_detected_before_question_creation(self):
        self.fixture.source.write_bytes(self.fixture.source.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_review_is_not_overwritten(self):
        self.prepare()
        original = (self.output/"questions.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.output/"questions.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
