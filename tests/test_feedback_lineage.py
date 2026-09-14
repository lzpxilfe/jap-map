import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.build_reconstruction_review_project import verify_feedback_lineage
from histcontour_core.provenance import sha256_file


class FeedbackLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.drawing=self.root/'drawing-report.json';self.drawing.write_text('{}')
        self.preview=self.root/'gap-context-report.json';self.preview.write_text('{}')
        self.decision=self.root/'decision.json';self.decision.write_text(json.dumps({'schema':'jap-map-context-gap-feedback/1'}))
        self.application={'input_sha256':{str(p):sha256_file(p) for p in (self.drawing,self.preview,self.decision)},'revisions':[{'original':'scope'}]}

    def tearDown(self):self.tmp.cleanup()

    def test_original_snapshot_verified_without_current_rerun_dependency(self):
        with patch('scripts.build_reconstruction_review_project.resolve_straight_feedback',return_value=[{'original':'scope'}]):
            self.assertEqual(verify_feedback_lineage(self.application,self.drawing),[self.preview,self.decision])

    def test_changed_original_preview_rejected(self):
        self.preview.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'original approved preview'):verify_feedback_lineage(self.application,self.drawing)

    def test_modified_approval_scope_rejected(self):
        changed=copy.deepcopy(self.application);changed['revisions'][0]['original']='expanded scope'
        with patch('scripts.build_reconstruction_review_project.resolve_straight_feedback',return_value=[{'original':'scope'}]):
            with self.assertRaisesRegex(ValueError,'saved local revisions'):verify_feedback_lineage(changed,self.drawing)


if __name__=='__main__':unittest.main()
