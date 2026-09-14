import unittest
from histcontour_core.recovery_edge_partition import partition_recovery_edges


class RecoveryEdgePartitionTests(unittest.TestCase):
    def test_middle_fragment_is_not_emitted_as_new_ink(self):
        original={'candidates':[{'points':[[x,10] for x in range(1,8)]}]}
        sources=[{'source_path_id':'middle','points':[[3,10],[4,10],[5,10]]}]
        result=partition_recovery_edges(original,sources)['candidates'][0]['edge_partition']
        self.assertEqual([p['kind'] for p in result['parts']],['new_recovery_edge','existing_observed_edge','new_recovery_edge'])
        self.assertEqual(result['new_edge_count'],4);self.assertEqual(result['reused_edge_count'],2)
        self.assertFalse(result['requires_topology_resolution'])
        self.assertNotIn('edge_partition',original['candidates'][0])

    def test_entering_fragment_interior_does_not_silently_create_branch(self):
        r={'candidates':[{'points':[[3,8],[3,9],[3,10],[4,10],[5,10]]}]}
        paths=[{'source_path_id':'middle','points':[[2,10],[3,10],[4,10],[5,10]]}]
        audit=partition_recovery_edges(r,paths)['candidates'][0]['edge_partition']
        self.assertTrue(audit['requires_topology_resolution'])
        self.assertEqual(audit['attachments'][0]['combined_degree'],3)

    def test_reversed_existing_edges_are_still_reused(self):
        r={'candidates':[{'points':[[3,10],[2,10],[1,10]]}]}
        p=[{'source_path_id':'source','points':[[1,10],[2,10],[3,10]]}]
        self.assertEqual(partition_recovery_edges(r,p)['candidates'][0]['edge_partition']['new_edge_count'],0)

    def test_smoothed_coordinates_cannot_be_rounded_into_fake_ownership(self):
        with self.assertRaises(ValueError):partition_recovery_edges({'candidates':[{'points':[[1.2,2],[2,2]]}]},[])


if __name__=='__main__':unittest.main()
