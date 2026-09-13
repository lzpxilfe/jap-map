import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from histcontour_core.assisted_review import _tail_replacements
from histcontour_core.gap_refinement import GapRefinementConfig
from histcontour_core.human_feedback import geometry_digest
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import world
from scripts.refine_assisted_drawing_from_feedback import balanced_review_queue, geometry_draft, owner_mask, refine, touches_other
from scripts.render_assisted_refinement import render_panel
from tests.test_gap_refinement import synthetic_ink


def fixture(directory):
    """Synthetic user attestations for contract tests, NOT real judgments."""
    packet = directory/"packet"; packet.mkdir(); (packet/"sources").mkdir()
    yy, xx = np.indices((90,260), dtype=float)
    gray = (245-210*np.exp(-.5*((yy-40)/.65)**2)).astype(np.uint8)
    Image.fromarray(gray).save(packet/"sources/synthetic.tif")
    tile = {"tile_id": "synthetic", "sheet_id": "synthetic", "split": "development", "crs_authid": "EPSG:3857",
            "bounds": [1000., 2000., 1520., 2180.], "pixel_bounds": [0,0,260,90],
            "raster_path": "sources/synthetic.tif", "source_raster_sha256": sha256_file(packet/"sources/synthetic.tif")}
    crs = {"type": "name", "properties": {"name": "EPSG:3857"}}
    rows, features, bases = [], [], []
    for i in range(3):
        _, source, target, points = synthetic_ink()
        source, target, points = [(np.asarray(p)+[i*80,0]).tolist() for p in (source,target,points)]
        sid = f"T{i:04}"
        props = {"proposal_id": sid, "tile_id": "synthetic", "source_uid": sid+"-a", "target_uid": sid+"-b",
                 "dataset_role": "review_only_not_training", "human_approved": False, "mode": "contextual_gap"}
        rows.append({**props, "pixel_points": points, "start": points[0], "end": points[-1], "pixel_box": [i*80,10,i*80+70,70]})
        features.append({"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": world(tile,points)}})
        for suffix, path in (("a",source),("b",target)):
            bases.append({"type": "Feature", "properties": {"segment_uid":sid+"-"+suffix,"tile_id":"synthetic","human_approved":False},
                          "geometry": {"type":"LineString","coordinates":world(tile,path)}})
    report = {"schema":"jap-map-assisted-contour-drawing/1","holdout_used":False,"human_approvals":0,
              "tiles":[tile],"proposals":rows,"proposal_count":len(rows)}
    collection = lambda f: {"type":"FeatureCollection","crs":crs,"features":f}
    for name, value in (("drawing-report.json",report),("ai-proposals.geojson",collection(features)),("base-lines.geojson",collection(bases))):
        (packet/name).write_text(json.dumps(value),encoding="utf-8")
    decision = {k: rows[0][k] for k in ("proposal_id","tile_id","source_uid","target_uid")}
    decision.update(source_raster_sha256=tile["source_raster_sha256"],original_geometry_sha256=geometry_digest(features[0]["geometry"]),
                    geometry_decision="accept_original",semantic_decision="contour",human_judgment_received=True,evidence_event_ids=["synthetic-event"])
    ledger = {"schema":"jap-map-assisted-human-review/1","review_origin":"explicit_user_chat",
              "dataset_role":"review_only_not_training","holdout_used":False,"native_crs":"EPSG:3857",
              "source_report_sha256":sha256_file(packet/"drawing-report.json"),"source_proposals_sha256":sha256_file(packet/"ai-proposals.geojson"),
              "source_base_lines_sha256":sha256_file(packet/"base-lines.geojson"),"revisions":[],"decisions":[decision],
              "deferred_original_ids":["T0002"],"events":[{"event_id":"synthetic-event","proposal_id":"T0000","origin":"explicit_user_chat",
                "case_actions":{"T0000":"accept_original"},"case_semantics":{"T0000":"contour"}}]}
    path = directory/"ledger.json"; path.write_text(json.dumps(ledger),encoding="utf-8")
    return packet, path, report, collection(features), collection(bases)


class FeedbackRefinementScriptTests(unittest.TestCase):
    def test_queue_alternates_tiles_and_keeps_weak_evidence_last(self):
        items = [{"proposal_id":sid,"tile_id":tile,"kind":"local_tail_replacement","anchor_review_required":weak}
                 for sid,tile,weak in (("1","a",False),("2","a",False),("3","b",False),("0","b",True))]
        self.assertEqual([r["proposal_id"] for r in balanced_review_queue(items)],["1","3","2","0"])

    def test_batch_locks_approved_and_deferred_keeps_crs_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); packet, ledger, report, originals, base = fixture(root)
            before = {p:sha256_file(p) for p in root.rglob('*') if p.is_file()}
            result = refine(packet,ledger,root/"out",previews=1)
            self.assertEqual(result["counts"], {"locked_human_review":1,"proposed":1,"deferred_context":1})
            self.assertEqual(result["approved_geometries_modified"],0)
            self.assertFalse(result["model_fitted"]); self.assertFalse(result["holdout_used"])
            features = json.loads((root/"out/proposed-revisions.geojson").read_text())
            self.assertEqual(features["crs"],originals["crs"])
            candidate = features["features"][0]
            self.assertEqual(candidate["properties"]["parent_proposal_id"],"T0001")
            self.assertFalse(candidate["properties"]["human_approved"])
            self.assertFalse(candidate["properties"]["append_only_safe"])
            self.assertEqual({p:sha256_file(p) for p in before},before)
            spec = result["revisions"][0]
            tails = _tail_replacements(spec,candidate,originals["features"][1]["geometry"]["coordinates"],report["proposals"][1],report["tiles"][0],base,base["crs"])
            self.assertEqual(len(tails),2)
            self.assertEqual(len(list((root/"out").glob('*.png'))),1)
            with self.assertRaises(FileExistsError):
                refine(packet,ledger,root/"out",previews=0)

    def test_mutated_files_and_held_out_sources_are_rejected_before_output(self):
        for mutation in ("raster","ledger_hash","crs","holdout"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); packet, ledger, report, originals, base = fixture(root)
                value = json.loads(ledger.read_text())
                if mutation == "raster":
                    Image.new('L',(260,90),245).save(packet/"sources/synthetic.tif")
                elif mutation == "ledger_hash":
                    value["source_base_lines_sha256"] = "f"*64
                else:
                    report["tiles"][0]["split" if mutation == "holdout" else "crs_authid"] = "holdout" if mutation == "holdout" else "EPSG:4326"
                    (packet/"drawing-report.json").write_text(json.dumps(report))
                    value["source_report_sha256"] = sha256_file(packet/"drawing-report.json")
                ledger.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    refine(packet,ledger,root/"out",previews=0)
                self.assertFalse((root/"out").exists())

    def test_never_writes_inside_original_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); packet, ledger, *_ = fixture(root)
            with self.assertRaises(ValueError):
                refine(packet,ledger,packet/"new-refinements")
            self.assertFalse((packet/"new-refinements").exists())

    def test_other_end_of_reviewed_source_needs_combined_tail_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); packet, ledger_path, report, originals, base = fixture(root)
            tile = report["tiles"][0]
            common_uid = "T0000-b"
            common = next(f for f in base["features"] if f["properties"]["segment_uid"] == common_uid)
            common["geometry"]["coordinates"] = world(tile,[[40,38.8],[42,40],[108,40],[110,41.2]])
            base["features"] = [f for f in base["features"] if f["properties"]["segment_uid"] != "T0001-a"]
            report["proposals"][1]["source_uid"] = common_uid
            originals["features"][1]["properties"]["source_uid"] = common_uid
            ledger = json.loads(ledger_path.read_text())
            for name,value,key in (("drawing-report.json",report,"source_report_sha256"),
                                   ("ai-proposals.geojson",originals,"source_proposals_sha256"),
                                   ("base-lines.geojson",base,"source_base_lines_sha256")):
                (packet/name).write_text(json.dumps(value))
                ledger[key] = sha256_file(packet/name)
            ledger_path.write_text(json.dumps(ledger))
            result = refine(packet,ledger_path,root/"out",previews=0)
            item = next(d for d in result["decisions"] if d["proposal_id"] == "T0001")
            self.assertEqual(item["status"],"context_review_required")
            self.assertIn("shared_source_tail_needs_combined_review_contract",item["reasons"])

    def test_entire_join_and_shared_pixel_collisions_are_screened(self):
        owner = owner_mask((50,50),[(1,[[5,20],[20,20]]),(2,[[30,20],[45,20]]),(3,[[21,19],[21,15]])])
        self.assertTrue(touches_other([[20,20],[30,20]],owner,[1,2]))
        shared = owner_mask((50,50),[(1,[[5,20],[20,20]]),(2,[[20,20],[30,20]])])
        self.assertEqual(shared[20,20],-1)
        self.assertTrue(touches_other([[20,20],[30,20]],shared,[1,2]))
        self.assertTrue(touches_other([[-1,10],[10,10]],owner,[1,2]))
        self.assertFalse(touches_other([[20,35],[30,35]],owner,[1,2]))

    def test_weak_anchor_evidence_stays_visible_on_smoothing_draft(self):
        t = np.linspace(0,1,35); points = np.c_[30+10*t,40+.1*np.sin(2*np.pi*t)]
        result = geometry_draft(np.full((90,100),245.),[[10,40],points[0]], [points[-1],[70,40]],points,GapRefinementConfig())
        self.assertEqual(result["status"],"proposed")
        self.assertEqual(result["kind"],"fixed_endpoints")
        self.assertTrue(result["anchor_review_required"])
        self.assertFalse(result["human_approved"])
        self.assertIn("insufficient_or_ambiguous_connected_ink",result["reasons"])

    def test_competing_endpoint_hypotheses_are_not_reanchored(self):
        args = synthetic_ink()
        result = geometry_draft(*args,GapRefinementConfig(),competing_pair=True)
        self.assertEqual(result["status"],"context_review_required")
        self.assertEqual(result["points"],args[-1])
        self.assertEqual(result["reasons"],["competing_endpoint_pair_needs_context"])

    def test_common_span_error_does_not_penalize_extended_tails(self):
        from scripts.evaluate_feedback_refinement import gap_span_error
        original = [[10,20],[20,20]]; reference = [[5,21],[25,21]]
        self.assertEqual(gap_span_error(original,reference,reference)["mean_normal_error_pixels"],0)
        self.assertEqual(gap_span_error(original,original,reference)["mean_normal_error_pixels"],1)
        with self.assertRaises(ValueError):
            gap_span_error(original,[[10,20],[16,20],[15,20],[20,20]],reference)

    def test_panel_preserves_pixel_centres_raster_values_and_aspect(self):
        pixels = np.arange(32,dtype=np.uint8).reshape(4,8)
        source = Image.fromarray(pixels)
        rendered = np.asarray(render_panel(source,(0,0,8,4),size=(80,80)))
        self.assertTrue(np.all(rendered[:20] == 255))
        self.assertTrue(np.all(rendered[60:] == 255))
        np.testing.assert_array_equal(rendered[20:60,:,0],np.repeat(np.repeat(pixels,10,axis=0),10,axis=1))
        overlay = np.asarray(render_panel(source,(0,0,8,4),size=(80,80),endpoints=[(2,1)]))
        # The circle centre is at native centre (2.5,1.5), with top letterbox.
        blue = (overlay[:,:,2].astype(int)-overlay[:,:,0].astype(int)) > 80
        ys,xs = np.nonzero(blue)
        self.assertLess(abs(xs.mean()-25),1); self.assertLess(abs(ys.mean()-35),1)


if __name__ == "__main__":
    unittest.main()
