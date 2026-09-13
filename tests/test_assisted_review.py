import copy
import json
from pathlib import Path
import tempfile
import unittest

from histcontour_core.assisted_review import build_review_outputs
from histcontour_core.human_feedback import geometry_digest
from histcontour_core.provenance import sha256_file
from scripts.export_assisted_human_review import export
from scripts.publish_assisted_review_ledger import assert_no_private_paths, public_event


def review_fixture():
    """Synthetic attestations exercise contracts; these are not user judgments."""
    tile = {"tile_id": "synthetic", "sheet_id": "synthetic-sheet", "split": "development",
            "crs_authid": "EPSG:3857", "bounds": [0., 0., 1000., 1000.],
            "pixel_bounds": [0, 0, 1000, 1000], "source_raster_sha256": "a"*64}
    crs = {"type": "name", "properties": {"name": "EPSG:3857"}}
    rows, features, events, decisions = [], [], [], []
    classes = ["contour", "non_contour", "unsure", "probable_contour", "contour", "not_separately_stated", "contour"]
    actions = ["accept_original"]*4+["reject_original", "unresolved_no_explicit_decision", "accept_revised_geometry_with_reservation"]
    for i in range(8):
        sid = f"A{i:04}"
        x = 10.+i*25
        props = {"proposal_id": sid, "tile_id": tile["tile_id"], "source_uid": sid+"-a", "target_uid": sid+"-b",
                 "dataset_role": "review_only_not_training", "human_approved": False, "mode": "contextual_gap"}
        rows.append({**props, "pixel_points": [[x, 100.], [x+10, 100.]]})
        geometry = {"type": "LineString", "coordinates": [[x+.5, 899.5], [x+10.5, 899.5]]}
        features.append({"type": "Feature", "properties": props, "geometry": geometry})
        if i < 7:
            event = {"event_id": f"synthetic-e{i}", "proposal_id": sid, "origin": "explicit_user_chat",
                     "status": "current", "user_text": "Synthetic test fixture, not real human evidence.",
                     "case_actions": {} if i == 5 else {sid: actions[i]}, "case_semantics": {sid: classes[i]}}
            events.append(event)
            decisions.append({"proposal_id": sid, "tile_id": tile["tile_id"], "source_uid": props["source_uid"],
                              "target_uid": props["target_uid"], "source_raster_sha256": tile["source_raster_sha256"],
                              "original_geometry_sha256": geometry_digest(geometry), "geometry_decision": actions[i],
                              "semantic_decision": classes[i], "human_judgment_received": i != 5,
                              "evidence_event_ids": [event["event_id"]]})
    revised = copy.deepcopy(features[6])
    revised["properties"].update(proposal_id="A0006-R1", parent_proposal_id="A0006")
    a, b = revised["geometry"]["coordinates"]
    revised["geometry"]["coordinates"] = [a, [(a[0]+b[0])/2, a[1]+1], b]
    decisions[6].update(revision_id="A0006-R1", review_qualification="too curved but acceptable")
    events[6]["accepted_revision_ids"] = ["A0006-R1"]
    ledger = {"schema": "jap-map-assisted-human-review/1", "review_origin": "explicit_user_chat",
              "dataset_role": "review_only_not_training", "holdout_used": False, "native_crs": "EPSG:3857",
              "events": events, "decisions": decisions, "deferred_original_ids": [],
              "revisions": [{"revision_id": "A0006-R1", "parent_proposal_id": "A0006", "approved": True,
                             "accepted_event_id": events[6]["event_id"], "geometry_sha256": geometry_digest(revised["geometry"])}]}
    report = {"schema": "jap-map-assisted-contour-drawing/1", "holdout_used": False,
              "tiles": [tile], "proposals": rows, "proposal_count": len(rows)}
    collection = {"type": "FeatureCollection", "crs": crs, "features": features}
    revisions = {"type": "FeatureCollection", "crs": crs, "features": [revised]}
    return report, collection, ledger, revisions


class AssistedReviewTests(unittest.TestCase):
    def test_geometry_semantics_uncertainty_and_revisions_are_separate(self):
        args = review_fixture()
        before = copy.deepcopy(args)
        result = build_review_outputs(*args)
        self.assertEqual(args, before)
        self.assertEqual(result["summary"]["counts"], {
            "approved_contour": 2, "semantics_pending": 2, "non_contour": 1, "rejected_route": 1,
            "unresolved_route": 1, "pending_refinement": 0, "reviewed": 7, "unreviewed": 1,
            "deferred": 0, "accepted_geometry": 5, "original_proposal_count": 8})
        approved = result["collections"]["approved_contour"]["features"]
        self.assertEqual([f["properties"]["proposal_id"] for f in approved], ["A0000", "A0006-R1"])
        self.assertEqual(approved[-1]["geometry"], args[3]["features"][0]["geometry"])
        self.assertEqual(approved[-1]["properties"]["review_qualification"], "too curved but acceptable")
        self.assertTrue(all(not f["properties"]["human_approved"] for f in result["collections"]["semantics_pending"]["features"]))
        noncontour = result["collections"]["non_contour"]["features"][0]["properties"]
        self.assertTrue(noncontour["human_geometry_accepted"])
        self.assertFalse(noncontour["human_approved"])
        self.assertEqual(result["queue"]["eligible_ids"], ["A0007"])
        self.assertFalse(result["summary"]["model_fitted"])

    def test_unknown_drift_duplicate_or_wrong_crs_are_rejected(self):
        mutations = [
            lambda a: a[1]["features"][0]["geometry"]["coordinates"][0].__setitem__(0, 900),
            lambda a: a[2]["decisions"][0].__setitem__("source_uid", "wrong"),
            lambda a: a[2]["decisions"].append(copy.deepcopy(a[2]["decisions"][0])),
            lambda a: a[2]["decisions"][0].__setitem__("proposal_id", "missing"),
            lambda a: a[2].__setitem__("native_crs", "EPSG:4326"),
            lambda a: a[2].__setitem__("holdout_used", True),
            lambda a: a[2].__setitem__("dataset_role", "training"),
            lambda a: a[2]["decisions"][0].__setitem__("original_geometry_sha256", "0"*64),
        ]
        for mutation in mutations:
            args = review_fixture()
            mutation(args)
            with self.assertRaises(ValueError):
                build_review_outputs(*args)

    def test_superseded_ai_or_uncertain_evidence_cannot_become_an_approval(self):
        for mutation in (
            lambda a: a[2]["events"][0].__setitem__("status", "superseded"),
            lambda a: a[2]["events"][0].__setitem__("origin", "ai_visual_provisional"),
            lambda a: a[2]["decisions"][0].__setitem__("human_judgment_received", False),
            lambda a: a[2]["decisions"][5].update(geometry_decision="accept_original", human_judgment_received=True),
            lambda a: a[2]["decisions"][2].__setitem__("semantic_decision", "contour"),
        ):
            args = review_fixture(); mutation(args)
            with self.assertRaises(ValueError):
                build_review_outputs(*args)

    def test_revision_requires_exact_geometry_endpoints_and_specific_user_approval(self):
        for mutation in (
            lambda a: a[2]["revisions"][0].__setitem__("approved", False),
            lambda a: a[2]["events"][6].__setitem__("accepted_revision_ids", []),
            lambda a: a[2]["revisions"][0].__setitem__("geometry_sha256", "0"*64),
            lambda a: a[3]["features"][0]["properties"].__setitem__("parent_proposal_id", "A0000"),
            lambda a: a[3]["features"].clear(),
        ):
            args = review_fixture(); mutation(args)
            with self.assertRaises(ValueError):
                build_review_outputs(*args)
        args = review_fixture()
        args[3]["features"][0]["geometry"]["coordinates"][0][0] += 1
        args[2]["revisions"][0]["geometry_sha256"] = geometry_digest(args[3]["features"][0]["geometry"])
        with self.assertRaisesRegex(ValueError, "endpoints"):
            build_review_outputs(*args)

    def test_pending_refinement_and_deferred_cases_are_not_counted_as_approved(self):
        args = review_fixture()
        decision = args[2]["decisions"][6]
        decision["geometry_decision"] = "accept_connection_refine_curvature"
        args[2]["events"][6]["case_actions"]["A0006"] = decision["geometry_decision"]
        args[2]["revisions"] = []; args[3]["features"] = []
        args[2]["deferred_original_ids"] = ["A0007"]
        result = build_review_outputs(*args)
        self.assertEqual(result["summary"]["counts"]["approved_contour"], 1)
        self.assertEqual(result["summary"]["counts"]["pending_refinement"], 1)
        self.assertEqual(result["queue"]["eligible_ids"], [])

    def test_export_binds_files_is_read_only_and_refuses_overwrite(self):
        report, collection, ledger, revisions = review_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); packet = root/"packet"; packet.mkdir()
            files = {packet/"drawing-report.json": report, packet/"ai-proposals.geojson": collection,
                     root/"revisions.geojson": revisions}
            for path, value in files.items():
                path.write_text(json.dumps(value), encoding="utf-8")
            ledger.update(source_report_sha256=sha256_file(packet/"drawing-report.json"),
                          source_proposals_sha256=sha256_file(packet/"ai-proposals.geojson"),
                          revision_collection_sha256=sha256_file(root/"revisions.geojson"))
            path = root/"ledger.json"; path.write_text(json.dumps(ledger), encoding="utf-8")
            before = {str(p): p.read_bytes() for p in [*files, path]}
            result = export(packet, path, root/"out", revisions_path=root/"revisions.geojson", next_id="A0007")
            self.assertEqual(result["queue"]["next_batch_ids"], ["A0007"])
            self.assertEqual(before, {p: Path(p).read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                export(packet, path, root/"out", revisions_path=root/"revisions.geojson")
            with self.assertRaises(ValueError):
                export(packet, path, root/"wrong-next", revisions_path=root/"revisions.geojson", next_id="A0000")
            self.assertFalse((root/"wrong-next").exists())

    def test_publication_guard_rejects_private_paths_and_ai_events(self):
        for value in ({"text": "/Users/synthetic/Desktop/map.png"}, ["file:///tmp/private.png"]):
            with self.assertRaises(ValueError):
                assert_no_private_paths(value)
        with self.assertRaises(ValueError):
            public_event({"event_id": "ai", "type": "assistant_geometry_revision", "geometry_decision": "accept_original"})
        event = public_event({"event_id": "e", "type": "case_judgment", "proposal_id": "A0000",
                              "geometry_decision": "accept_original", "semantic_decision": "contour", "verbatim_text": "Synthetic"})
        self.assertEqual(event["case_actions"], {"A0000": "accept_original"})


def local_tail_fixture():
    report, collection, ledger, revisions = review_fixture()
    original = collection["features"][6]
    a, b = original["geometry"]["coordinates"]
    props = original["properties"]
    source = {"type": "Feature", "properties": {"segment_uid": props["source_uid"],
              "tile_id": "synthetic", "human_approved": False},
              "geometry": {"type": "LineString", "coordinates": [[a[0]-20, a[1]], a]}}
    target = {"type": "Feature", "properties": {"segment_uid": props["target_uid"],
              "tile_id": "synthetic", "human_approved": False},
              "geometry": {"type": "LineString", "coordinates": [b, [b[0]+20, b[1]]]}}
    base = {"type": "FeatureCollection", "crs": collection["crs"], "features": [source, target]}
    revised = revisions["features"][0]
    revised["geometry"]["coordinates"] = [[a[0]-4, a[1]], [(a[0]+b[0])/2, a[1]+1], [b[0]+3, b[1]]]
    spec = ledger["revisions"][0]
    spec.update(revision_kind="local_tail_replacement", geometry_sha256=geometry_digest(revised["geometry"]),
                tail_replacement_contract={
                    "schema": "jap-map-local-tail-replacement/1", "source_uid": props["source_uid"],
                    "target_uid": props["target_uid"], "source_geometry_sha256": geometry_digest(source["geometry"]),
                    "target_geometry_sha256": geometry_digest(target["geometry"]), "source_tail_trim_pixels": 4,
                    "target_tail_trim_pixels": 3, "join_points_pixels": [[a[0]-4.5, 100], [b[0]+2.5, 100]]})
    return report, collection, ledger, revisions, base


class LocalTailReviewTests(unittest.TestCase):
    def test_approved_patch_cuts_exact_sources_and_keeps_semantics(self):
        args = local_tail_fixture()
        before = copy.deepcopy(args)
        result = build_review_outputs(*args[:4], base_lines=args[4])
        self.assertEqual(args, before)
        patch = result["collections"]["approved_contour"]["features"][-1]
        lines = result["source_tail_replacements"]["features"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["geometry"]["coordinates"][-1], patch["geometry"]["coordinates"][0])
        self.assertEqual(lines[1]["geometry"]["coordinates"][0], patch["geometry"]["coordinates"][-1])
        self.assertFalse(patch["properties"]["append_only_safe"])
        self.assertTrue(patch["properties"]["requires_source_tail_replacement"])
        for feature in lines:
            self.assertFalse(feature["properties"]["human_approved"])
            self.assertFalse(feature["properties"]["whole_source_line_semantics_approved"])
            self.assertFalse(feature["properties"]["training_eligible"])

    def test_endpoint_rule_is_not_waived_without_source_contract(self):
        args = local_tail_fixture()
        with self.assertRaisesRegex(ValueError, "actual original base"):
            build_review_outputs(*args[:4])
        args[2]["revisions"][0].pop("revision_kind")
        with self.assertRaisesRegex(ValueError, "endpoints"):
            build_review_outputs(*args[:4], base_lines=args[4])

    def test_tail_source_and_join_mutations_fail_closed(self):
        mutations = [
            lambda a: a[2]["revisions"][0]["tail_replacement_contract"].__setitem__("source_uid", "wrong"),
            lambda a: a[2]["revisions"][0]["tail_replacement_contract"].__setitem__("source_geometry_sha256", "0"*64),
            lambda a: a[2]["revisions"][0]["tail_replacement_contract"].__setitem__("source_tail_trim_pixels", True),
            lambda a: a[2]["revisions"][0]["tail_replacement_contract"].__setitem__("source_tail_trim_pixels", 17),
            lambda a: a[2]["revisions"][0]["tail_replacement_contract"].__setitem__("source_tail_trim_pixels", 5),
            lambda a: a[2]["revisions"][0]["tail_replacement_contract"]["join_points_pixels"][0].__setitem__(0, 0),
            lambda a: a[4]["features"][0]["geometry"]["coordinates"][0].__setitem__(0, 0),
            lambda a: a[2]["revisions"][0].__setitem__("revision_kind", "unchecked_endpoint_move"),
            lambda a: a[2]["events"][6].__setitem__("accepted_revision_ids", []),
        ]
        for mutate in mutations:
            args = local_tail_fixture(); mutate(args)
            with self.assertRaises(ValueError):
                build_review_outputs(*args[:4], base_lines=args[4])

    def test_patch_excursion_is_not_allowed_even_with_updated_digest(self):
        args = local_tail_fixture()
        args[3]["features"][0]["geometry"]["coordinates"][1][1] += 10
        args[2]["revisions"][0]["geometry_sha256"] = geometry_digest(args[3]["features"][0]["geometry"])
        with self.assertRaisesRegex(ValueError, "local corridor"):
            build_review_outputs(*args[:4], base_lines=args[4])

    def test_tail_export_binds_base_file_and_emits_replacement_instructions(self):
        report, collection, ledger, revisions, base = local_tail_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); packet = root/"packet"; packet.mkdir()
            files = {packet/"drawing-report.json": report, packet/"ai-proposals.geojson": collection,
                     packet/"base-lines.geojson": base, root/"revisions.geojson": revisions}
            for path, value in files.items():
                path.write_text(json.dumps(value), encoding="utf-8")
            ledger.update(source_report_sha256=sha256_file(packet/"drawing-report.json"),
                          source_proposals_sha256=sha256_file(packet/"ai-proposals.geojson"),
                          source_base_lines_sha256=sha256_file(packet/"base-lines.geojson"),
                          revision_collection_sha256=sha256_file(root/"revisions.geojson"))
            ledger_path = root/"ledger.json"; ledger_path.write_text(json.dumps(ledger))
            result = export(packet, ledger_path, root/"out", revisions_path=root/"revisions.geojson", next_id="A0007")
            self.assertEqual(result["summary"]["source_tail_replacement_feature_count"], 2)
            self.assertTrue((root/"out/source-tail-replacements.geojson").is_file())
            self.assertTrue((root/"out/source-tail-application.json").is_file())
            base["features"][0]["geometry"]["coordinates"][0][0] += .1
            (packet/"base-lines.geojson").write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "base-line fingerprint"):
                export(packet, ledger_path, root/"changed", revisions_path=root/"revisions.geojson")
            self.assertFalse((root/"changed").exists())


if __name__ == "__main__":
    unittest.main()
