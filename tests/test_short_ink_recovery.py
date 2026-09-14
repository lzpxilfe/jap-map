import unittest
import numpy as np
from histcontour_core.short_ink_recovery import recover_short_ink,contrast_guarded_evidence,audit_short_recovery_topology,audit_recovery_numeric_ink
from histcontour_core.directional_trace import DirectionalTraceConfig
from histcontour_core.short_ink_recovery import recover_short_ink_batched
from tests.test_directional_trace import evidence_from_support


class ShortInkRecoveryTests(unittest.TestCase):
    def test_batches_match_single_pass_and_cover_last_partial_batch(self):
        ends=[];support=np.zeros((150,30))
        for i in range(6):
            y=10+i*20;support[y,3:11]=.8
            ends.extend([{'source_path_id':f'a{i}','point':[3,y],'outward_tangent':[1,0]},
                         {'source_path_id':f'b{i}','point':[10,y],'outward_tangent':[-1,0]}])
        evidence=evidence_from_support(support)
        batched=recover_short_ink_batched(evidence,ends,maximum_pairs=4)
        single=recover_short_ink(evidence,ends,maximum_pairs=64)
        self.assertEqual(batched['candidates'],single['candidates'])
        self.assertEqual(batched['completed_batches'],[{'first_pair':0,'stop_pair_exclusive':4},{'first_pair':4,'stop_pair_exclusive':6}])
        with self.assertRaises(ValueError):recover_short_ink_batched(evidence,ends,maximum_pairs=4,maximum_batches=1)

    def test_batch_boundary_does_not_hide_competing_endpoint(self):
        ends=self.ends()+[{'source_path_id':'c','point':[18,11],'outward_tangent':[-1,0]}]
        result=recover_short_ink_batched(evidence_from_support(np.ones((30,30))*.8),ends,maximum_pairs=1)
        self.assertTrue(all(c['status']=='competing_endpoints_abstained' for c in result['candidates']))

    def test_numeric_overlap_between_vertices_is_flagged_without_erasing_route(self):
        r={'candidates':[{'points':[[2,5],[8,5]],'status':'observed'}]}
        mask=np.zeros((12,12),bool);mask[5,5]=True
        result=audit_recovery_numeric_ink(r,mask)['candidates'][0]
        self.assertEqual(result['numeric_ink_audit']['status'],'held_for_numeric_review')
        self.assertEqual(result['points'],r['candidates'][0]['points'])
        self.assertEqual(result['status'],'observed')
        self.assertNotIn('numeric_ink_audit',r['candidates'][0])

    def test_no_numeric_overlap_is_not_a_contour_approval(self):
        r={'candidates':[{'points':[[2,5],[8,5]]}]}
        result=audit_recovery_numeric_ink(r,np.zeros((12,12),bool))['candidates'][0]
        self.assertEqual(result['numeric_ink_audit']['overlapping_pixel_cells'],0)
        self.assertFalse(result['numeric_ink_audit']['contour_semantics_proven'])

    def test_sparse_dense_region_endpoints_use_local_neighbor_search(self):
        ends=[{'source_path_id':str(i),'point':[i*20,10],'outward_tangent':[1,0]} for i in range(300)]
        r=recover_short_ink(evidence_from_support(np.zeros((30,30))),ends)
        self.assertEqual(r['endpoint_count'],300)
        self.assertEqual(r['nearby_pairs_examined'],0)
        self.assertEqual(r['candidates'],[])

    def test_topology_checks_crossing_between_samples_and_preserves_original(self):
        result={'candidates':[{'points':[[10,10],[18,10]],'start':[10,10],'end':[18,10],
                              'source_path_ids':['a','b'],'status':'observed'}]}
        paths=[{'source_path_id':'foreign','points':[[14,8],[14,12]]}]
        audited=audit_short_recovery_topology(result,paths)
        self.assertEqual(audited['candidates'][0]['topology_audit']['status'],'held_for_review')
        self.assertEqual(audited['candidates'][0]['status'],'observed')
        self.assertNotIn('topology_audit',result['candidates'][0])

    def test_parallel_line_clearance_and_legitimate_parent_touch(self):
        result={'candidates':[{'points':[[10,10],[18,10]],'start':[10,10],'end':[18,10],
                              'source_path_ids':['a','b'],'status':'observed'}]}
        parents=[{'source_path_id':'a','points':[[5,10],[10,10]]},{'source_path_id':'b','points':[[18,10],[22,10]]}]
        self.assertEqual(audit_short_recovery_topology(result,parents)['candidates'][0]['topology_audit']['status'],'passed')
        parents.append({'source_path_id':'near','points':[[12,10.5],[16,10.5]]})
        audit=audit_short_recovery_topology(result,parents)['candidates'][0]['topology_audit']
        self.assertEqual(audit['intersected_source_path_ids'],[])
        self.assertEqual(audit['insufficient_clearance_path_ids'],['near'])

    def test_contrast_guard_blocks_flat_paper_even_with_high_support(self):
        evidence=evidence_from_support(np.ones((30,30))*.8)
        guarded,audit=contrast_guarded_evidence(np.full((30,30),255,np.uint8),evidence)
        self.assertEqual(audit['supported_pixels_after'],0)
        self.assertTrue(np.all(evidence.support_score==np.float32(.8)))
        self.assertEqual(recover_short_ink(guarded,self.ends())['candidates'][0]['status'],'abstained')

    def test_fixed_weak_sensitivity_recovers_faint_source_not_default(self):
        gray=np.full((30,30),255,np.uint8);gray[10,8:21]=225
        support=np.zeros((30,30));support[10,8:21]=.04
        guarded,_=contrast_guarded_evidence(gray,evidence_from_support(support))
        baseline=recover_short_ink(guarded,self.ends())
        weak=recover_short_ink(guarded,self.ends(),trace_config=DirectionalTraceConfig(
            corridor_radius_px=2.,max_length_ratio=1.6,minimum_support=.025))
        self.assertEqual(baseline['candidates'][0]['status'],'abstained')
        self.assertEqual(weak['candidates'][0]['status'],'observed')
        self.assertEqual(weak['connections_applied'],0)

    def ends(self):return [{'source_path_id':'a','point':[10,10],'outward_tangent':[1,0]},
                          {'source_path_id':'b','point':[18,10],'outward_tangent':[-1,0]}]

    def test_existing_ink_is_recovered_without_merging(self):
        support=np.zeros((30,30));support[10,8:21]=.8
        r=recover_short_ink(evidence_from_support(support),self.ends())
        self.assertEqual(r['candidates'][0]['status'],'observed')
        self.assertFalse(r['candidates'][0]['inferred_gap'])
        self.assertEqual(r['connections_applied'],0)

    def test_white_gap_cannot_be_reported_as_observed(self):
        r=recover_short_ink(evidence_from_support(np.zeros((30,30))),self.ends())
        self.assertEqual(r['candidates'][0]['status'],'abstained')
        self.assertEqual(r['candidates'][0]['points'],[])

    def test_competing_endpoint_abstains(self):
        ends=self.ends()+[{'source_path_id':'c','point':[18,11],'outward_tangent':[-1,0]}]
        r=recover_short_ink(evidence_from_support(np.ones((30,30))*.8),ends)
        self.assertTrue(all(c['status']=='competing_endpoints_abstained' for c in r['candidates']))

    def test_opposing_direction_and_same_path_do_not_pair(self):
        ends=self.ends();ends[1]['outward_tangent']=[1,0]
        self.assertEqual(recover_short_ink(evidence_from_support(np.ones((30,30))),ends)['candidates'],[])


if __name__=='__main__':unittest.main()
