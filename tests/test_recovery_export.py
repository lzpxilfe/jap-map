import copy
import unittest
from histcontour_core.recovery_export import export_recovery_additions


class RecoveryExportTests(unittest.TestCase):
    def fixture(self):
        tiles={'tile':{'bounds':[0,0,10,10],'pixel_bounds':[0,0,10,10],'crs_authid':'EPSG:5132','source_raster_sha256':'source'}}
        candidate={'candidate_id':'x','points':[[1,1],[2,1],[3,1]],'source_path_ids':['a','b'],
                   'status':'observed','human_approved':False,'training_eligible':False,
                   'topology_audit':{'status':'passed'},'numeric_ink_audit':{'status':'no_numeric_candidate_overlap'},
                   'edge_partition':{'requires_topology_resolution':False,'parts':[
                       {'kind':'existing_observed_edge','points':[[1,1],[2,1]]},
                       {'kind':'new_recovery_edge','points':[[2,1],[3,1]]}]}}
        report={'schema':'jap-map-short-recovery-run/1','holdout_used':False,'human_approvals':0,'numeric_candidate_masks_checked':True,
                'regions':[{'tile_id':'tile','weak_sensitivity':{'candidates':[candidate]}}]}
        return report,tiles,candidate

    def test_only_new_edges_exported_with_half_pixel_mapping(self):
        r,t,c=self.fixture();before=copy.deepcopy(r);out=export_recovery_additions(r,t)
        self.assertEqual(out['features'][0]['geometry']['coordinates'],[[2.5,8.5],[3.5,8.5]])
        self.assertEqual(r,before);self.assertFalse(out['features'][0]['properties']['human_approved'])

    def test_reversed_duplicate_preserves_both_ids_without_double_geometry(self):
        r,t,c=self.fixture();other=copy.deepcopy(c);other['candidate_id']='y';other['edge_partition']['parts'][-1]['points'].reverse()
        r['regions'][0]['weak_sensitivity']['candidates'].append(other)
        out=export_recovery_additions(r,t)
        self.assertEqual(len(out['features']),1);self.assertEqual(out['features'][0]['properties']['candidate_ids'],['x','y'])

    def test_numeric_conflict_is_omitted_not_exported_as_valid(self):
        r,t,c=self.fixture();c['numeric_ink_audit']['status']='held_for_numeric_review'
        out=export_recovery_additions(r,t);self.assertEqual(out['features'],[]);self.assertEqual(len(out['omitted_candidates']),1)


if __name__=='__main__':unittest.main()
