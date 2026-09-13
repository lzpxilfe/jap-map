import copy
import json
from pathlib import Path
import tempfile
import unittest

from histcontour_core.assisted_review import build_review_outputs
from histcontour_core.contour_network import merge_parts
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import read, world
from scripts.refine_assisted_drawing_from_feedback import refine
from scripts.vectorize_assisted_contours import assemble, run
from tests.test_feedback_refinement_script import fixture


TILE = {"tile_id":"t","bounds":[0,0,100,100],"pixel_bounds":[0,0,100,100]}


def edge(identity,points,*,kind="retained_source",tile="t"):
    return {"type":"Feature","geometry":{"type":"LineString","coordinates":world(TILE,points)},
            "properties":{"part_id":identity,"part_kind":kind,"tile_id":tile}}


class ContourNetworkTests(unittest.TestCase):
    def test_merge_keeps_vertices_and_does_not_propagate_connector_approval(self):
        parts = [edge("a",[[5,20],[10,21],[15,20]]),edge("b",[[25,20],[30,20]]),
                 edge("h",[[15,20],[20,19],[25,20]],kind="approved_connection")]
        before = copy.deepcopy(parts); result = merge_parts(parts,{"t":TILE})
        self.assertEqual(result["line_count"],1); self.assertEqual(result["open_endpoint_nodes"],2)
        feature = result["features"][0]
        self.assertFalse(feature["properties"]["human_approved"])
        self.assertEqual(feature["properties"]["approved_connection_count"],1)
        self.assertEqual(feature["properties"]["source_part_count"],2)
        actual = feature["geometry"]["coordinates"]
        for part in parts:
            for point in part["geometry"]["coordinates"]:self.assertIn(point,actual)
        self.assertEqual(before,parts)

    def test_branches_are_not_arbitrarily_threaded(self):
        parts = [edge("a",[[10,20],[20,20]]),edge("b",[[20,20],[30,20]]),edge("c",[[20,20],[20,30]])]
        result = merge_parts(parts,{"t":TILE})
        self.assertEqual(result["line_count"],3)
        self.assertEqual(result["tiles"][0]["branch_nodes"],1)

    def test_cycles_and_reversed_edges_are_merged_once(self):
        parts = [edge("a",[[10,10],[20,10]]),edge("b",[[20,20],[20,10]]),edge("c",[[20,20],[10,10]])]
        result = merge_parts(parts,{"t":TILE})
        self.assertEqual(result["line_count"],1)
        self.assertTrue(result["features"][0]["properties"]["closed"])
        members = result["memberships"][0]["members"]
        self.assertEqual(sorted(m["part_id"] for m in members),["a","b","c"])
        self.assertTrue(any(m["reversed"] for m in members))

    def test_roundoff_does_not_erase_exact_vertices_or_snap_real_gaps(self):
        parts = [edge("a",[[10,20],[20,20]]),edge("b",[[20+1e-6,20],[30,20]])]
        result = merge_parts(parts,{"t":TILE})
        self.assertEqual(result["line_count"],1)
        coordinates = result["features"][0]["geometry"]["coordinates"]
        self.assertIn(parts[0]["geometry"]["coordinates"][-1],coordinates)
        self.assertIn(parts[1]["geometry"]["coordinates"][0],coordinates)
        parts[1] = edge("b",[[20+.001,20],[30,20]])
        self.assertEqual(merge_parts(parts,{"t":TILE})["line_count"],2)

    def test_tiles_are_not_merged_even_at_identical_map_coordinates(self):
        parts = [edge("a",[[10,20],[20,20]]),edge("b",[[20,20],[30,20]],tile="other")]
        self.assertEqual(merge_parts(parts,{"t":TILE,"other":TILE})["line_count"],2)

    def test_invalid_identity_geometry_and_snapping_tolerance_are_rejected(self):
        part = edge("a",[[10,20],[20,20]])
        with self.assertRaises(ValueError):merge_parts([part,part],{"t":TILE})
        with self.assertRaises(ValueError):merge_parts([part],{})
        with self.assertRaises(ValueError):merge_parts([part],{"t":TILE},endpoint_tolerance_pixels=.5)
        with self.assertRaises(ValueError):merge_parts([edge("a",[[10,20],[10,20]])],{"t":TILE})
        self.assertEqual(merge_parts([],{"t":TILE})["line_count"],0)


class VectorizationAssemblyTests(unittest.TestCase):
    def prepare(self,root):
        packet,ledger_path,report,originals,base = fixture(root)
        revisions = {"type":"FeatureCollection","crs":base["crs"],"features":[]}
        path = root/"approved-revisions.geojson"; path.write_text(json.dumps(revisions))
        ledger = read(ledger_path); ledger["revision_collection_sha256"] = sha256_file(path)
        ledger_path.write_text(json.dumps(ledger))
        drafts = refine(packet,ledger_path,root/"drafts",revisions_path=path,previews=0)
        candidates = read(root/"drafts/proposed-revisions.geojson")
        reviewed = build_review_outputs(report,originals,ledger,revisions,base_lines=base)
        return packet,ledger_path,path,(report,originals,base,reviewed,drafts,candidates)

    def test_auto_tail_replacement_is_applied_only_to_separate_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,_,args = self.prepare(Path(directory)); before = copy.deepcopy(args)
            result = assemble(*args)
            self.assertEqual(args,before)
            self.assertEqual(result["automatic_connection_count"],1)
            self.assertEqual(result["automatic_source_tail_replacements"],2)
            self.assertEqual(result["baseline"]["line_count"],6)
            self.assertEqual(result["reviewed_network"]["line_count"],5)
            self.assertEqual(result["enhanced_network"]["line_count"],4)
            self.assertEqual(result["approved_connections"],args[3]["collections"]["approved_contour"]["features"])
            self.assertTrue(all(not f["properties"]["human_approved"] for f in result["enhanced_network"]["features"]))
            self.assertEqual(result["enhanced_network"]["open_endpoint_nodes"],8)

    def test_weak_anchor_candidate_is_not_assembled(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,_,args = self.prepare(Path(directory))
            args[-1]["features"][0]["properties"]["anchor_review_required"] = True
            result = assemble(*args)
            self.assertEqual(result["automatic_connection_count"],0)
            self.assertEqual(result["automatic_source_tail_replacements"],0)
            self.assertEqual(result["omitted"][0]["reason"],"weak_anchor_evidence_not_assembled")

    def test_geometry_identity_approval_and_lock_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _,_,_,args = self.prepare(Path(directory))
            for mutation in ("geometry","approval","lock","contract"):
                a = copy.deepcopy(args)
                if mutation == "geometry":a[-1]["features"][0]["geometry"]["coordinates"][0][0] += 1
                elif mutation == "approval":a[-1]["features"][0]["properties"]["human_approved"] = True
                elif mutation == "lock":a[-2]["decisions"][0]["status"] = "unchanged"
                else:a[-2]["revisions"][0]["tail_replacement_contract"]["source_tail_trim_pixels"] += 1
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):assemble(*a)

    def test_file_pipeline_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);packet,ledger,revisions,_ = self.prepare(root)
            paths = [p for p in root.rglob('*') if p.is_file()]
            before = {p:sha256_file(p) for p in paths}
            result = run(packet,ledger,revisions,root/"drafts",root/"vectorized")
            self.assertEqual(result["counts"]["enhanced_candidate_lines"],4)
            self.assertEqual(result["counts"]["unapplied_review_connections"],1)
            self.assertEqual(before,{p:sha256_file(p) for p in paths})
            self.assertEqual(read(root/"vectorized/contour-candidates.geojson")["crs"]["properties"]["name"],"EPSG:3857")
            with self.assertRaises(FileExistsError):run(packet,ledger,revisions,root/"drafts",root/"vectorized")
            with self.assertRaises(ValueError):run(packet,ledger,revisions,root/"drafts",packet/"unsafe")


if __name__ == "__main__":unittest.main()
