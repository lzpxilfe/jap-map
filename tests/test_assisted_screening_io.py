import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from histcontour_core.provenance import sha256_file
from scripts.apply_assisted_visual_review import apply
from scripts.generate_assisted_contour_drawing import world
from scripts.screen_assisted_contour_drawing import screen


class AssistedScreeningIOTests(unittest.TestCase):
    def fixture(self, root):
        raw = root/"raw"
        for name in ("sources", "images"):
            (raw/name).mkdir(parents=True)
        image = Image.new("RGB", (40, 40), "white")
        draw = ImageDraw.Draw(image)
        draw.line([(2, 20), (10, 20)], fill="black")
        draw.line([(20, 20), (35, 20)], fill="black")
        image.save(raw/"sources"/"fixture.png")
        for kind in ("source", "proposal"):
            image.save(raw/"images"/f"A0001-{kind}.png")
        tile = {"tile_id": "fixture", "sheet_id": "fixture-sheet", "split": "development", "crs_authid": "EPSG:5132",
                "bounds": [126., 36., 126.01, 36.01], "pixel_bounds": [0, 0, 40, 40],
                "raster_path": "sources/fixture.png", "source_raster_sha256": sha256_file(raw/"sources"/"fixture.png")}
        row = {"proposal_id": "A0001", "tile_id": "fixture", "mode": "contextual_gap", "gap_pixels": 10.,
               "source_uid": "a", "target_uid": "b", "source_endpoint": "a:1", "target_endpoint": "b:0",
               "start": [10., 20.], "end": [20., 20.], "pixel_points": [[10., 20.], [20., 20.]],
               "pixel_box": [0, 0, 40, 40], "priority": 2, "question": "Synthetic only",
               "dataset_role": "review_only_not_training", "human_approved": False,
               "quality": {"path_length_pixels": 10., "new_fraction_outside_original_1_5px": .8}}
        report = {"schema": "jap-map-assisted-contour-drawing/1", "holdout_used": False, "human_approvals": 0,
                  "attempt_count": 1, "proposal_count": 1, "tiles": [tile], "proposals": [row]}
        base = [{"type": "Feature", "properties": {"segment_uid": uid, "tile_id": "fixture"},
                 "geometry": {"type": "LineString", "coordinates": world(tile, points)}} for uid, points in (
                     ("a", [[2., 20.], [10., 20.]]), ("b", [[20., 20.], [35., 20.]]))]
        proposal = {"type": "Feature", "properties": {k: v for k, v in row.items() if k not in ("quality", "pixel_points")},
                    "geometry": {"type": "LineString", "coordinates": world(tile, row["pixel_points"])}}
        for name, value in (("drawing-report.json", report), ("base-lines.geojson", {"type": "FeatureCollection", "features": base}),
                            ("ai-proposals.geojson", {"type": "FeatureCollection", "features": [proposal]}),
                            ("configuration.json", {}), ("upstream-pin.json", {}), ("attempts.json", [])):
            (raw/name).write_text(json.dumps(value), encoding="utf-8")
        return raw, report

    def test_both_screening_stages_declare_native_crs_preserve_geometry_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, _ = self.fixture(root)
            before = {str(p.relative_to(raw)): p.read_bytes() for p in raw.rglob("*") if p.is_file()}
            screened = root/"screened"
            result = screen(raw/"drawing-report.json", screened)
            self.assertEqual(result["proposal_count"], 1)
            review = {"schema": "jap-map-assisted-visual-screen/1", "source_report_sha256": sha256_file(screened/"drawing-report.json"),
                      "reference_origin": "ai_visual_provisional", "human_approved": False,
                      "items": [{"proposal_id": "A0001", "decision": "ask_human", "note": "Synthetic question"}]}
            path = root/"visual.json"; path.write_text(json.dumps(review), encoding="utf-8")
            final = root/"visual"
            apply(screened/"drawing-report.json", path, final)
            for folder in (screened, final):
                for name in ("base-lines.geojson", "ai-proposals.geojson"):
                    actual = json.loads((folder/name).read_text())
                    original = json.loads((raw/name).read_text())
                    self.assertEqual(actual["crs"]["properties"]["name"], "EPSG:5132")
                    self.assertEqual([f["geometry"] for f in actual["features"]], [f["geometry"] for f in original["features"]])
            self.assertEqual(before, {str(p.relative_to(raw)): p.read_bytes() for p in raw.rglob("*") if p.is_file()})

    def test_automated_screen_rejects_human_approved_or_held_out_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, report = self.fixture(root)
            for index, kind in enumerate(("approved", "heldout")):
                changed = copy.deepcopy(report)
                if kind == "approved":
                    changed["proposals"][0]["human_approved"] = True
                else:
                    changed["tiles"][0]["sheet_id"] = "178-gongju"
                (raw/"drawing-report.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    screen(raw/"drawing-report.json", root/f"invalid-{index}")
                self.assertFalse((root/f"invalid-{index}").exists())


if __name__ == "__main__":
    unittest.main()
