import unittest
from histcontour_core.semantic_reference_overlap import reference_overlap


class SemanticReferenceOverlapTests(unittest.TestCase):
    def test_local_reference_does_not_label_rest_of_long_line(self):
        r=reference_overlap([[0,0],[100,0]],[[40,0],[50,0]],tolerance_px=.1)
        self.assertAlmostEqual(r['aligned_overlap_length_px'],10)
        self.assertEqual(r['sampled_spans'],[[40.,50.]])
        self.assertFalse(r['whole_line_semantics_assigned'])

    def test_crossing_line_is_not_aligned_semantic_overlap(self):
        r=reference_overlap([[45,-10],[45,10]],[[40,0],[50,0]])
        self.assertEqual(r['aligned_overlap_length_px'],0)

    def test_reversed_reference_retains_axial_match(self):
        r=reference_overlap([[0,0],[100,0]],[[50,0],[40,0]],tolerance_px=.1)
        self.assertAlmostEqual(r['aligned_overlap_length_px'],10)


if __name__=='__main__':unittest.main()
