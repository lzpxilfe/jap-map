import unittest
from histcontour_core.context_gap_preview import build_context_gap_previews,resolve_straight_feedback


class ContextGapPreviewTests(unittest.TestCase):
    def test_local_straight_feedback_is_pinned_and_not_global_semantic_approval(self):
        r,e,p=self.fixture();curve=build_context_gap_previews(r,e,p)['previews'][0]
        report={'cases':[{'case_id':'case','tile_id':'tile','curve_previews':{'previews':[curve]},'contextual_ranking':r}]}
        feedback={'schema':'jap-map-context-gap-feedback/1','source_report_sha256':'digest','decisions':[{
            'case_id':'case','candidate_id':'x','tile_id':'tile','source_path_ids':['a','b'],
            'expected_endpoints':[[10,20],[30,20]],'scope':'this_connection_only','connection_decision':'approved',
            'geometry_decision':'straight_acceptable','semantic_decision':'not_separately_confirmed',
            'training_eligible':False,'user_quote':'직선으로 이어도 무방'}]}
        revision=resolve_straight_feedback(report,feedback,'digest')[0]
        self.assertEqual(revision['points'],[[10,20],[30,20]])
        self.assertTrue(revision['connection_human_approved'])
        self.assertFalse(revision['contour_semantics_assigned'])
        self.assertFalse(curve['human_approved'])
        with self.assertRaises(ValueError):resolve_straight_feedback(report,feedback,'changed')
        feedback['decisions'][0]['expected_endpoints'][0][0]+=1
        with self.assertRaises(ValueError):resolve_straight_feedback(report,feedback,'digest')

    def fixture(self):
        ranking={'candidates':[{'candidate_id':'x','start':[10,20],'end':[30,20],
                               'source_path_ids':['a','b'],'ranking_cost':.5}]}
        ends=[{'source_path_id':'a','point':[10,20],'outward_tangent':[1,0]},
              {'source_path_id':'b','point':[30,20],'outward_tangent':[-1,0]}]
        paths=[{'source_path_id':'a','points':[[x,20] for x in range(2,11)]},
               {'source_path_id':'b','points':[[x,20] for x in range(30,39)]}]
        return ranking,ends,paths

    def test_collinear_preview_preserves_endpoints_and_parents(self):
        r,e,p=self.fixture();result=build_context_gap_previews(r,e,p);preview=result['previews'][0]
        self.assertTrue(preview['geometry_checks_passed'])
        self.assertEqual(preview['points'][0],[10,20]);self.assertEqual(preview['points'][-1],[30,20])
        self.assertAlmostEqual(preview['detour_ratio'],1)
        self.assertFalse(preview['source_tails_modified'])
        self.assertEqual(result['connections_applied'],0)

    def test_observed_crossing_blocks_geometry_check(self):
        r,e,p=self.fixture();p.append({'source_path_id':'cross','points':[[20,10],[20,30]]})
        preview=build_context_gap_previews(r,e,p)['previews'][0]
        self.assertFalse(preview['geometry_checks_passed'])
        self.assertIn('cross',preview['observed_collision_path_ids'])

    def test_wrong_signed_direction_cannot_be_flipped_to_fake_continuity(self):
        r,e,p=self.fixture();e[0]['outward_tangent']=[-1,0]
        preview=build_context_gap_previews(r,e,p)['previews'][0]
        self.assertEqual(preview['points'],[])

    def test_context_abstention_does_not_make_curve(self):
        r,e,p=self.fixture();r['candidates'][0]['ranking_cost']=None
        self.assertEqual(build_context_gap_previews(r,e,p)['previews'][0]['points'],[])

    def test_slope_clamping_is_not_reported_as_matching_source_direction(self):
        r,e,p=self.fixture();e[0]['outward_tangent']=[1,.9];e[1]['outward_tangent']=[-1,.9]
        preview=build_context_gap_previews(r,e,p)['previews'][0]
        self.assertTrue(preview['points'])
        self.assertIn('bounded_curve_does_not_match_source_tangent',preview['reasons'])
        self.assertFalse(preview['geometry_checks_passed'])

    def test_other_path_touch_at_endpoint_is_not_exempted(self):
        r,e,p=self.fixture();p.append({'source_path_id':'foreign','points':[[10,20],[10,25]]})
        preview=build_context_gap_previews(r,e,p)['previews'][0]
        self.assertIn('foreign',preview['observed_collision_path_ids'])


if __name__=='__main__':unittest.main()
