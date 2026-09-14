import unittest
from histcontour_core.contextual_gap_ranking import rank_gap_context


def match(start,end,ids=('a','b')):
    return {'start':start,'end':end,'source_path_ids':list(ids),'cost':.4}


class ContextualGapRankingTests(unittest.TestCase):
    def paths(self):return [{'source_path_id':str(y),'points':[[x,y] for x in range(5,56)]} for y in (8,12,16,44,48,52)]

    def test_parallel_neighbor_flow_prefers_parallel_gap(self):
        r=rank_gap_context([match([18,30],[42,30]),match([30,18],[30,42],('c','d'))],self.paths(),[20,20,40,40])
        self.assertEqual(r['candidates'][0]['source_path_ids'],['a','b'])
        self.assertAlmostEqual(r['candidates'][0]['context_direction_penalty'],0)
        self.assertGreater(r['candidates'][1]['context_direction_penalty'],.99)
        self.assertEqual(r['conflicts'][0]['reason'],'chords_cross_or_touch')

    def test_parents_cannot_vote_for_their_own_gap(self):
        paths=self.paths()[:2]
        r=rank_gap_context([match([18,30],[42,30],('8','12'))],paths,[20,20,40,40])
        self.assertIsNone(r['candidates'][0]['ranking_cost'])
        self.assertTrue(all(not s['neighbor_path_ids'] for s in r['candidates'][0]['context_samples']))

    def test_one_long_path_cannot_supply_multiple_independent_votes(self):
        r=rank_gap_context([match([18,30],[42,30])],self.paths()[:1],[20,20,40,40])
        self.assertFalse(r['candidates'][0]['context_evidence_sufficient'])
        self.assertIsNone(r['candidates'][0]['context_direction_penalty'])

    def test_reversed_duplicate_pair_is_not_counted_twice(self):
        r=rank_gap_context([match([18,30],[42,30]),match([42,30],[18,30],('b','a'))],self.paths(),[20,20,40,40])
        self.assertEqual(len(r['candidates']),1)
        self.assertFalse(r['candidates'][0]['eligible_for_automatic_connection'])

    def test_unknown_does_not_outrank_supported_context(self):
        r=rank_gap_context([match([18,30],[42,30]),match([150,150],[170,150],('c','d'))],self.paths(),[20,20,40,40])
        self.assertIsNone(r['candidates'][-1]['ranking_cost'])
        self.assertEqual(r['connections_applied'],0)

    def test_repeated_orientation_proposals_use_lowest_cost_independent_of_order(self):
        a=match([18,30],[42,30]);b=dict(a,cost=.8)
        first=rank_gap_context([a,b],self.paths(),[20,20,40,40])
        second=rank_gap_context([b,a],self.paths(),[20,20,40,40])
        self.assertEqual(first,second)
        self.assertEqual(first['candidates'][0]['original_endpoint_cost'],.4)

    def test_reversing_neighbor_paths_preserves_axial_direction(self):
        paths=self.paths()
        forward=rank_gap_context([match([18,30],[42,30])],paths,[20,20,40,40])
        for path in paths:path['points'].reverse()
        reverse=rank_gap_context([match([18,30],[42,30])],paths,[20,20,40,40])
        self.assertAlmostEqual(forward['candidates'][0]['context_direction_penalty'],reverse['candidates'][0]['context_direction_penalty'])

    def test_endpoint_reuse_is_explicit_conflict(self):
        r=rank_gap_context([match([18,30],[42,30]),match([18,30],[42,31],('a','c'))],self.paths(),[20,20,40,40])
        self.assertEqual(r['conflicts'][0]['reason'],'endpoint_reuse')

    def test_parallel_chord_outside_label_cannot_be_ranked_as_label_gap(self):
        r=rank_gap_context([match([18,18],[42,18])],self.paths(),[20,20,40,40])
        candidate=r['candidates'][0]
        self.assertTrue(candidate['context_evidence_sufficient'])
        self.assertIsNone(candidate['ranking_cost'])
        self.assertIn('chord_does_not_traverse_label_context',candidate['ranking_abstention_reasons'])


if __name__=='__main__':unittest.main()
