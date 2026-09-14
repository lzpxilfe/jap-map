import unittest
import numpy as np
from histcontour_core.fragment_chain_recovery import recover_via_source_fragment
from histcontour_core.recovery_edge_partition import partition_recovery_edges
from tests.test_directional_trace import evidence_from_support


class FragmentChainRecoveryTests(unittest.TestCase):
    def fixture(self):
        paths=[{'source_path_id':pid,'points':[[x,10] for x in range(a,b+1)]} for pid,a,b in [('a',0,3),('mid',7,10),('b',14,17)]]
        r=partition_recovery_edges({'candidates':[{'start':[3,10],'end':[14,10],
            'source_path_ids':['a','b'],'points':[[x,10] for x in range(3,15)]}]},paths)
        r['candidates'][0]['edge_partition']['requires_topology_resolution']=True
        support=np.zeros((25,25));support[10,:18]=.8
        return r,paths,evidence_from_support(support)

    def test_complete_middle_fragment_is_reused_without_branch(self):
        r,p,e=self.fixture();result=recover_via_source_fragment(r,p,e);a=result['alternatives'][0]
        self.assertEqual(a['status'],'source_fragment_preserving_draft')
        self.assertEqual(a['preserved_source_fragment_points'],p[1]['points'])
        self.assertFalse(a['edge_partition']['requires_topology_resolution'])
        self.assertEqual(a['edge_partition']['reused_edge_count'],3)
        self.assertEqual(result['connections_applied'],0)

    def test_white_subgap_is_not_invented(self):
        r,p,e=self.fixture();e=evidence_from_support(np.zeros((25,25)))
        a=recover_via_source_fragment(r,p,e)['alternatives'][0]
        self.assertEqual(a['status'],'abstained');self.assertEqual(a['points'],[])

    def test_reverse_source_order_does_not_reverse_connection(self):
        r,p,e=self.fixture();p[1]['points'].reverse()
        a=recover_via_source_fragment(r,p,e)['alternatives'][0]
        self.assertEqual(a['points'][0],[3,10]);self.assertEqual(a['points'][-1],[14,10])
        self.assertEqual(a['preserved_source_fragment_points'],list(reversed(p[1]['points'])))

    def test_one_supported_subgap_survives_without_fabricating_full_chain(self):
        r,p,e=self.fixture();support=np.zeros((25,25));support[10,:11]=.8
        a=recover_via_source_fragment(r,p,evidence_from_support(support))['alternatives'][0]
        self.assertEqual(a['status'],'partial_subgap_draft_not_complete_chain')
        self.assertEqual(a['points'],[])
        self.assertEqual(len(a['supported_subgap_previews']),1)
        self.assertEqual(a['supported_subgap_previews'][0]['end'],[7,10])


if __name__=='__main__':unittest.main()
