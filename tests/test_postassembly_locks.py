import unittest
from scripts.smooth_assembled_reconstruction import source_locks


class PostassemblyLockTests(unittest.TestCase):
    def test_approval_and_inferred_spans_remain_locked_after_join(self):
        parts={
            'a':{'properties':{'part_kind':'retained_source'},'geometry':{'coordinates':[[0,0],[1,0]]}},
            'b':{'properties':{'part_kind':'approved_connection'},'geometry':{'coordinates':[[1,0],[2,0]]}},
            'c':{'properties':{'part_kind':'automatic_connection','inferred_gap':True},'geometry':{'coordinates':[[3,0],[2,0]]}}}
        membership={'members':[{'part_id':'a','reversed':False},{'part_id':'b','reversed':False},{'part_id':'c','reversed':True}]}
        self.assertEqual(source_locks(membership,parts,[[0,0],[1,0],[2,0],[3,0]]),[(1,2),(2,3)])

    def test_changed_membership_cannot_silently_reanchor_approval(self):
        parts={'a':{'properties':{'part_kind':'approved_connection'},'geometry':{'coordinates':[[0,0],[1,0]]}}}
        with self.assertRaises(ValueError):source_locks({'members':[{'part_id':'a','reversed':False}]},parts,[[0,0],[2,0]])


if __name__=='__main__':unittest.main()
