import json
import unittest
from scripts.inspect_numeric_gap_context import run
from tests import test_numeric_separation_script as fixtures


class NumericGapContextScriptTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.NumericSeparationScriptTests();self.f.setUp();self.f.run_case()
        # The upstream fixture is deliberately decimated. Supply a genuine
        # adjacent-pixel path for this raw-only endpoint contract.
        vectors=json.loads(self.f.vectors.read_text())
        vectors['features'][0]['geometry']['coordinates']=[[100+(x+.5)*.125,200-5.5*.25] for x in range(5,11)]
        self.f.vectors.write_text(json.dumps(vectors))
        self.ownership=self.f.output/'numeric-separation-report.json'
        data=json.loads(self.ownership.read_text())
        data['regions'][0]['hypotheses']=[{'hypothesis_id':'synthetic','glyph_candidate_component_ids':[1],
            'reference_quad_crop_pixel_centers':[[7,4],[9,4],[9,6],[7,6]]}]
        self.ownership.write_text(json.dumps(data))
        self.output=self.f.f.root/'gap-context'

    def tearDown(self):self.f.tearDown()

    def run_case(self):
        return run(self.f.f.packet,self.f.f.manifest_path,self.ownership,self.f.vectors,self.output)

    def test_source_preserved_and_only_unapproved_alternatives_exported(self):
        paths=[self.ownership,self.f.vectors,self.f.f.source]
        before=[p.read_bytes() for p in paths]
        result=self.run_case()
        self.assertEqual(before,[p.read_bytes() for p in paths])
        self.assertEqual(result['connections_applied'],0)
        self.assertEqual(result['human_approvals'],0)
        self.assertEqual(len(result['cases'][0]['orientation_alternatives']),4)
        self.assertTrue((self.output/result['cases'][0]['preview']).exists())

    def test_inferred_input_cannot_supply_observed_endpoints(self):
        data=json.loads(self.f.vectors.read_text());data['features'][0]['properties']['inferred_gap']=True
        self.f.vectors.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'unsplit observed'):self.run_case()

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir(); sentinel=self.output/'keep';sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):self.run_case()
        self.assertEqual(sentinel.read_text(),'keep')


if __name__=='__main__':unittest.main()
