"""Source-coupled numeric partition I/O; the classifier is deliberately mocked."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts import separate_numeric_ink as separation
from scripts.detect_map_text_multiscale import _pixel_digest
from tests import test_observed_reconstruction_script as fixtures


class NumericSeparationScriptTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.ObservedReconstructionRunTests(); self.f.setUp()
        for r in self.f.manifest['regions']: r['box'] = [0, 0, 64, 64]
        self.f.save_inputs(); self.f.run_fixture()
        self.rec_dir = self.f.root/'recognition'; self.rec_dir.mkdir()
        (self.rec_dir/'R001').mkdir()
        self.output = self.f.root/'numeric-output'
        self.gray = np.asarray(Image.open(self.f.source))
        self.ink = self.gray <= 160
        self.labels, _ = ndimage.label(self.ink, structure=np.ones((3, 3), bool))
        self.array_path = self.rec_dir/'R001/component-pixels.npz'
        np.savez_compressed(self.array_path, source_ink=self.ink, component_labels=self.labels)
        self.row = {'region_id': 'R001', 'tile_id': fixtures.TILE_ID, 'sheet_id': '173-buyeo',
            'box': [0, 0, 64, 64], 'split': 'development', 'crs_authid': 'EPSG:5132',
            'source_raster_sha256': self.f.tile['source_raster_sha256'],
            'source_crop_gray_payload_sha256': _pixel_digest(self.gray),
            'component_pixel_arrays': 'R001/component-pixels.npz',
            'component_generation_configuration': {'background_window_px': 31, 'ink_contrast_floor': 40., 'dark_gray_ceiling': 160.},
            'components': [], 'groups': []}
        self.recognition = {'schema': 'jap-map-component-text-recognition/1', 'holdout_used': False,
                            'human_approvals': 0, 'regions': [self.row]}
        self.rec_path = self.rec_dir/'component-text-recognition.json'; self.save_recognition()
        self.vectors = self.f.output/'candidate-observed.geojson'
        glyph = np.zeros_like(self.ink); glyph[5, 7:9] = True
        self.model = SimpleNamespace(source_ink=self.ink, glyph_candidate=glyph,
            ambiguous_ink=self.ink & ~glyph, considered_ink=glyph,
            protected_throughgoing=np.zeros_like(glyph), hypotheses=(), component_evidence=(),
            provenance={'counts': {'glyph_candidate_pixels': 2}})

    def tearDown(self): self.f.tearDown()

    def save_recognition(self): self.rec_path.write_text(json.dumps(self.recognition))

    def run_case(self):
        with patch.object(separation, 'infer_numeric_ink_ownership', return_value=self.model):
            return separation.run(self.f.packet, self.f.manifest_path, self.rec_path,
                                  self.vectors, self.f.approved_path, self.output)

    def test_source_and_approved_bytes_preserved_and_all_numeric_parts_kept(self):
        originals = {p: p.read_bytes() for p in (self.f.source, self.vectors, self.f.approved_path, self.rec_path, self.array_path)}
        result = self.run_case()
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        self.assertEqual((self.output/'before.geojson').read_bytes(), originals[self.vectors])
        self.assertEqual((self.output/'approved-connections-unchanged.geojson').read_bytes(), originals[self.f.approved_path])
        counts = result['counts']
        self.assertEqual(counts['parent_lines_partitioned'], 1)
        self.assertAlmostEqual(counts['numeric_hypothesis_length_px'], 2.)
        self.assertEqual(counts['all-parts'], counts['retained']+counts['numeric-candidates'])
        parts = json.loads((self.output/'all-parts.geojson').read_text())['features']
        self.assertEqual(parts[0]['geometry']['coordinates'][0], json.loads(originals[self.vectors])['features'][0]['geometry']['coordinates'][0])
        self.assertEqual(parts[-1]['geometry']['coordinates'][-1], json.loads(originals[self.vectors])['features'][0]['geometry']['coordinates'][-1])
        self.assertTrue(all(not f['properties']['human_approved'] and not f['properties']['training_eligible'] for f in parts))
        self.assertAlmostEqual(sum(p['properties']['length_px'] for p in parts), 5.)
        self.assertTrue((self.output/'R001/comparison.png').exists())
        self.assertEqual(result['human_approvals'], 0)
        self.assertTrue(result['provenance']['inputs_and_implementation_unchanged'])

    def test_approved_cells_override_numeric_and_keep_original_feature(self):
        source = json.loads(self.vectors.read_text())
        approved = {'type': 'FeatureCollection', 'crs': source['crs'], 'features': source['features']}
        self.f.approved_path.write_text(json.dumps(approved))
        result = self.run_case()
        self.assertEqual(result['counts']['parent_lines_partitioned'], 0)
        self.assertEqual(json.loads((self.output/'retained.geojson').read_text())['features'], source['features'])
        with np.load(self.output/result['merged_masks'][0]['path'], allow_pickle=False) as m:
            self.assertFalse(m['glyph'].any())

    def test_native_precision_collapse_preserves_whole_parent(self):
        source = json.loads(self.vectors.read_text())
        with patch.object(separation, 'native_part_coordinates', return_value=[[127., 36.], [127., 36.]]):
            result = self.run_case()
        self.assertEqual(result['counts']['parent_lines_partitioned'], 0)
        self.assertEqual(result['counts']['numeric_hypothesis_length_px'], 0)
        self.assertEqual(json.loads((self.output/'retained.geojson').read_text())['features'], source['features'])

    def test_human_or_semantic_approved_input_is_never_partitioned(self):
        source = json.loads(self.vectors.read_text())
        source['features'][0]['properties']['whole_line_semantics_approved'] = True
        self.vectors.write_text(json.dumps(source))
        result = self.run_case()
        self.assertEqual(result['counts']['parent_lines_partitioned'], 0)
        self.assertEqual(json.loads((self.output/'retained.geojson').read_text())['features'], source['features'])

    def test_unexamined_unknown_does_not_veto_another_crop(self):
        shape = (4, 4)
        target = {k: np.zeros(shape, bool) for k in ('glyph', 'protected', 'ambiguous', 'considered')}
        first = SimpleNamespace(glyph_candidate=np.ones(shape, bool), protected_throughgoing=np.zeros(shape, bool),
                                ambiguous_ink=np.zeros(shape, bool), considered_ink=np.ones(shape, bool))
        unexamined = SimpleNamespace(glyph_candidate=np.zeros(shape, bool), protected_throughgoing=np.zeros(shape, bool),
                                     ambiguous_ink=np.ones(shape, bool), considered_ink=np.zeros(shape, bool))
        separation.merge_evidence(target, [0, 0, 4, 4], first)
        separation.merge_evidence(target, [0, 0, 4, 4], unexamined)
        self.assertTrue(target['glyph'].all()); self.assertFalse(target['ambiguous'].any())

    def test_labels_must_reconstruct_from_exact_source_threshold(self):
        altered = self.labels.copy(); altered[5, 7] = 2
        np.savez_compressed(self.array_path, source_ink=self.ink, component_labels=altered)
        with self.assertRaisesRegex(ValueError, 'declared original ink'): self.run_case()
        self.assertFalse(self.output.exists())

    def test_array_path_escape_and_crop_digest_fail(self):
        self.row['component_pixel_arrays'] = '../outside.npz'; self.save_recognition()
        with self.assertRaisesRegex(ValueError, 'escape'): self.run_case()
        self.row['component_pixel_arrays'] = 'R001/component-pixels.npz'
        self.row['source_crop_gray_payload_sha256'] = '0'*64; self.save_recognition()
        with self.assertRaisesRegex(ValueError, 'original source pixels'): self.run_case()

    def test_holdout_crs_duplicate_region_and_vector_identity_fail(self):
        self.recognition['holdout_used'] = True; self.save_recognition()
        with self.assertRaises(ValueError): self.run_case()
        self.recognition['holdout_used'] = False
        self.row['crs_authid'] = 'EPSG:4326'; self.save_recognition()
        with self.assertRaises(ValueError): self.run_case()
        self.row['crs_authid'] = 'EPSG:5132'
        self.recognition['regions'].append(copy.deepcopy(self.row)); self.save_recognition()
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.run_case()
        self.recognition['regions'].pop(); self.save_recognition()
        vector = json.loads(self.vectors.read_text()); vector['features'].append(copy.deepcopy(vector['features'][0]))
        self.vectors.write_text(json.dumps(vector))
        with self.assertRaisesRegex(ValueError, 'distinct'): self.run_case()

    def test_existing_output_and_output_inside_input_are_rejected(self):
        self.output.mkdir(); sentinel = self.output/'keep'; sentinel.write_text('keep')
        with self.assertRaises(FileExistsError): self.run_case()
        self.assertEqual(sentinel.read_text(), 'keep')
        self.output = self.rec_dir/'new'
        with self.assertRaisesRegex(ValueError, 'outside'): self.run_case()

    def test_implementation_mutation_prevents_success_report(self):
        original = separation.sha256_file
        seen = 0
        def modified(path):
            nonlocal seen
            if str(path).endswith('numeric_line_partition.py'):
                seen += 1
                if seen > 1: return '0'*64
            return original(path)
        with patch.object(separation, 'sha256_file', side_effect=modified):
            with self.assertRaisesRegex(ValueError, 'implementation changed'): self.run_case()
        self.assertFalse((self.output/'numeric-separation-report.json').exists())


if __name__ == '__main__': unittest.main()
