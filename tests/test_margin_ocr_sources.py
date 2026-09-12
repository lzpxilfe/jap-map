"""Offline archive-acquisition safeguards. No requests or real OCR in tests."""

import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.request import Request

from histcontour_core.margin_ocr import MarginOcrError, digest_value, review_template
from tests.test_margin_ocr import PILOT, result_fixture

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"margin_ocr_pilot": PILOT}):
        spec.loader.exec_module(module)
    return module


FETCH = load_script("fetch_margin_ocr_corpus")
PROPOSE = load_script("propose_margin_crops")
SUMMARY = load_script("summarize_margin_ocr_pilot")


def index_fixture():
    return {"features": [
        {"properties": {"available": True, "call_num": f"SHEET {group}-1", "label": f"Fixture-{number}",
                        "recid": f"aa{number:03}aa0000", "websiteUrl": f"https://purl.stanford.edu/aa{number:03}aa0000",
                        "iiifUrl": f"https://purl.stanford.edu/aa{number:03}aa0000/iiif/manifest"}, "geometry": None}
        for number, group in enumerate(FETCH.GROUPS)
    ]}


def response(body=b"fixture", *, status=200, length=None):
    result = io.BytesIO(body)
    result.status = status
    result.url = "https://stacks.stanford.edu/fixture"
    result.headers = {} if length is None else {"Content-Length": str(length)}
    return result


class MarginSourceTests(unittest.TestCase):
    def test_selection_is_20_unique_regions_and_excludes_gongju_group(self):
        index = index_fixture()
        held_out = copy.deepcopy(index["features"][0])
        held_out["properties"]["call_num"] = "SHEET D11-1"
        index["features"].append(held_out)
        selected = FETCH.select_sources(index)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({row["record_id"] for row in selected}), 20)
        self.assertNotIn("D11", {row["regional_group"] for row in selected})

    def test_duplicate_source_ids_or_mismatched_manifest_urls_are_refused(self):
        index = index_fixture()
        index["features"][1]["properties"]["recid"] = "aa000aa0000"
        index["features"][1]["properties"]["iiifUrl"] = "https://purl.stanford.edu/aa000aa0000/iiif/manifest"
        with self.assertRaisesRegex(MarginOcrError, "duplicate"):
            FETCH.select_sources(index)
        index = index_fixture()
        index["features"][0]["properties"]["iiifUrl"] = "https://example.invalid/manifest"
        with self.assertRaisesRegex(MarginOcrError, "manifest URL"):
            FETCH.select_sources(index)

    def test_non_archive_urls_are_rejected_before_opening_a_connection(self):
        for url in ("http://purl.stanford.edu/x", "https://example.invalid/x", "https://purl.stanford.edu:444/x", "https://user@purl.stanford.edu/x"):
            with self.subTest(url=url), patch.object(FETCH._OPENER, "open") as open_request, self.assertRaises(MarginOcrError):
                FETCH.fetch_bytes(url, 100)
            open_request.assert_not_called()

    def test_redirect_is_validated_before_following(self):
        handler = FETCH._ArchiveRedirectHandler()
        request = Request("https://purl.stanford.edu/x")
        with self.assertRaises(MarginOcrError):
            handler.redirect_request(request, None, 302, "Found", {}, "https://example.invalid/x")
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://stacks.stanford.edu/x")
        self.assertEqual(redirected.full_url, "https://stacks.stanford.edu/x")

    def test_http_200_nonempty_bounded_complete_payload_is_required(self):
        cases = [(b"fixture", 202, None), (b"", 200, None), (b"fixture", 200, 200),
                 (b"fixture", 200, 8), (b"fixture", 200, 0), (b"x" * 101, 200, None)]
        for body, status, length in cases:
            with self.subTest(status=status, length=length), patch.object(FETCH._OPENER, "open", return_value=response(body, status=status, length=length)), self.assertRaises(MarginOcrError):
                FETCH.fetch_bytes("https://purl.stanford.edu/x", 100)
        with patch.object(FETCH._OPENER, "open", return_value=response(length=7)):
            self.assertEqual(FETCH.fetch_bytes("https://purl.stanford.edu/x", 100), b"fixture")

    def test_each_original_needs_pdm_and_matching_single_canvas_jp2(self):
        canvas = {"rendering": [{"format": "image/jp2", "@id": "https://stacks.stanford.edu/file/aa000aa0000/map.jp2"}],
                  "images": [{"resource": {"service": {"@id": "https://stacks.stanford.edu/image/iiif/map"}}}]}
        manifest = {"license": FETCH.PUBLIC_DOMAIN, "sequences": [{"canvases": [canvas]}]}
        self.assertEqual(FETCH.image_source(manifest, "aa000aa0000")[0], canvas)
        changed = copy.deepcopy(manifest)
        changed["license"] = "unspecified"
        with self.assertRaisesRegex(MarginOcrError, "Public Domain"):
            FETCH.image_source(changed, "aa000aa0000")
        with self.assertRaisesRegex(MarginOcrError, "matching original"):
            FETCH.image_source(manifest, "bb111bb1111")
        manifest["sequences"][0]["canvases"].append(canvas)
        with self.assertRaisesRegex(MarginOcrError, "single-sheet"):
            FETCH.image_source(manifest, "aa000aa0000")

    def test_proposals_remain_bounded_native_pixel_regions(self):
        regions = PROPOSE.proposed_regions([[1500, 1200], [9000, 1200], [9000, 9000], [1500, 9000]], 10000, 10000)
        self.assertEqual(len(regions), 6)
        for row in regions:
            PILOT._bounds(row["pixel_box"], 10000, 10000)
            self.assertNotIn("gcp", row)

    def test_initial_run_summary_keeps_ai_probe_out_of_formal_evaluation(self):
        result = result_fixture(3)
        notes_rows, source_rows = [], []
        for index, row in enumerate(result["crops"]):
            rid = f"aa{index:03}aa0000"
            sid = f"stanford-{rid}"
            row.update(sheet_id=sid, region_id="title", crop_uid=f"{sid}/title", kind="title", scenario="old_type",
                       pixel_box=[0, 0, 20, 10], crop_size=[20, 10], crop_sha256="fixture", inference_seconds=0.1)
            notes_rows.append({"record_id": rid, "title_visual_ltr": None if index == 2 else row["raw_text"],
                               "ocr_seen_before_reference": index == 1})
            source_rows.append({"sheet_id": sid, "record_id": rid, "regional_group": f"fixture-{index}", "index_label": "fixture",
                                "source_page": "fixture", "source_url": "fixture", "iiif_manifest_url": "fixture", "image_size": [100, 80],
                                "image_bytes": 1, "image_sha256": "fixture", "source_manifest_sha256": "fixture", "license": "synthetic fixture"})
        bundle = {"manifest": {"visual_notes_sha256": "notes", "source_catalog_sha256": "sources",
                               "sheets": [{"scenario": "old_type"}] * 3}}
        lock = {"models": {}}
        result.update(bundle_sha256=digest_value(bundle), model_lock_sha256=digest_value(lock),
                      created_utc="synthetic fixture", status="completed", engine={})
        reviews = review_template(result)
        documents = {"result.json": result, "bundle.json": bundle, "model-lock.json": lock,
                     "review.template.json": reviews,
                     "notes": {"reference_origin": "ai_visual_provisional", "human_approved": False, "sheets": notes_rows},
                     "sources": {"sources": source_rows}}
        with patch.object(SUMMARY, "read_json", side_effect=lambda path: documents[path.name]), patch.object(SUMMARY, "verify_models"), patch.object(SUMMARY, "digest_file", side_effect=lambda path: path.name):
            report = SUMMARY.summarize(Path("fixture-run"), Path("notes"), Path("sources"))
            self.assertEqual(report["provisional_title_probe"]["count"], 1)
            self.assertEqual(report["provisional_title_probe"]["raw_exact_count"], 1)
            self.assertEqual(report["formal_evaluation"]["overall"]["count"], 0)
            self.assertEqual(report["human_approvals"], 0)
            self.assertFalse(report["promotion_passed"])
            reviews["items"][0].update(review_status="approved", reviewer="fixture", reviewed_text=result["crops"][0]["raw_text"])
            with self.assertRaisesRegex(MarginOcrError, "untouched"):
                SUMMARY.summarize(Path("fixture-run"), Path("notes"), Path("sources"))
            bundle["manifest"]["visual_notes_sha256"] = "changed after run"
            with self.assertRaisesRegex(MarginOcrError, "frozen run"):
                SUMMARY.summarize(Path("fixture-run"), Path("notes"), Path("sources"))


if __name__ == "__main__":
    unittest.main()
