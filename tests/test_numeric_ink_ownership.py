"""Numeric pixel hypotheses: synthetic contracts plus pinned local regressions.

The local regression inputs are machine evidence, not human truth labels.
Expected CCs document the inspected 505/100 seed-set mistakes; they do not
turn these hypotheses into training data or establish contour accuracy.
"""

import copy
from dataclasses import replace
from pathlib import Path
import json
import math
import unittest

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from histcontour_core.numeric_ink_ownership import NumericInkOwnershipConfig, infer_numeric_ink_ownership
from histcontour_core.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/"data/derived/contour-reconstruction-2026-09-14/component-text-recognition-v2"
PINS = {
    "R004": ("a3ab887ba5b81a85f0684cb22cee75ff72825928dc1fe78da6e251226ab8db85", "442503ea3ea5536f1248866bc9e190920dbcda70859e58252f8753f0c0d98e6f"),
    "R015": ("a0385d119f182f4f96e7f7e0d82962d49520e61188753093efe93f7e254d01f6", "a5a3c943d712e1bac3f3bac588737588f4c2a3b7fe399fe838f8a34d01deda71"),
    "R009": ("956a701a817431a77d2b1a8617dd947070d05cffa61195af1f417f3ed73e95fa", "2fa32ce4ba399046c98a0352f50afeaac7aa9f7f8bd7ba35a8a86de412782df3"),
}


def word_fixture(*, angle=0, extra_curve=False, weak_connection=False):
    image = Image.new("L", (112, 76), 255)
    draw = ImageDraw.Draw(image)
    stroke = 2 if angle else 1
    draw.line([(22, 30), (24, 28), (28, 28), (30, 31), (29, 33), (22, 40), (30, 40)], fill=25, width=stroke)
    draw.ellipse((36, 28, 44, 40), outline=25, width=stroke)
    draw.ellipse((50, 28, 58, 40), outline=25, width=stroke)
    word = np.asarray(image) < 160
    if extra_curve:
        draw.line([(63, 18), (60, 32), (65, 58)], fill=25, width=1)
    if weak_connection:
        draw.line([(5, 40), (22, 40)], fill=210, width=1)
    if angle:
        # Two-pixel fixture strokes preserve connectivity under nearest-neighbor
        # rotation. Keep paper white instead of PIL's default black border.
        image = image.resize((224, 152), Image.Resampling.NEAREST).rotate(
            angle, resample=Image.Resampling.NEAREST, center=(80, 68), fillcolor=255)
        word = np.asarray(Image.fromarray(np.uint8(word)*255).resize((224, 152), Image.Resampling.NEAREST).rotate(
            angle, resample=Image.Resampling.NEAREST, center=(80, 68), fillcolor=0)) > 0
    return np.asarray(image), word


def source_components(gray):
    contrast = ndi.maximum_filter(gray.astype(float), size=31)-gray
    labels, _ = ndi.label((gray <= 160) & (contrast >= 40), structure=np.ones((3, 3), bool))
    components = []
    for cid, window in enumerate(ndi.find_objects(labels), 1):
        ys, xs = np.nonzero(labels[window] == cid)
        components.append({"component_id": cid, "ink_pixels": len(xs),
            "bbox_crop_pixel_centers": [window[1].start, window[0].start, window[1].stop-1, window[0].stop-1]})
    return labels, components


def quad_for_mask(mask, *, angle=0, padding=2):
    ys, xs = np.nonzero(mask)
    points = np.column_stack((xs, ys))
    along = np.array([math.cos(math.radians(angle)), -math.sin(math.radians(angle))])
    normal = np.array([-along[1], along[0]])
    u, v = points @ along, points @ normal
    lo, hi = [u.min()-padding, v.min()-padding], [u.max()+padding, v.max()+padding]
    return [(a*along+b*normal).tolist() for a, b in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]


def candidate(gid, quad, ids, **extras):
    return {"group_id": gid, "quad_crop_pixel_centers": copy.deepcopy(quad), "group_component_ids": list(ids),
            "human_approved": False, "training_eligible": False, **extras}


def readings_for(groups, text="200", score=.95):
    return [{"group_id": group["group_id"], "scale": scale, "rec_text": text, "rec_score": score}
            for group in groups for scale in (2, 4)]


class NumericInkOwnershipTests(unittest.TestCase):
    def fixture(self, **kwargs):
        gray, word = word_fixture(**kwargs)
        labels, components = source_components(gray)
        ids = sorted(int(v) for v in np.unique(labels[word]) if v > 0)
        quad = quad_for_mask(word, angle=kwargs.get("angle", 0))
        groups = [candidate("first", quad, ids), candidate("duplicate", quad_for_mask(word, angle=kwargs.get("angle", 0), padding=2.5), ids)]
        return gray, word, labels, components, ids, groups

    def assert_partition(self, result):
        np.testing.assert_array_equal(result.glyph_candidate | result.ambiguous_ink | result.protected_throughgoing, result.source_ink)
        self.assertFalse(np.any(result.glyph_candidate & result.ambiguous_ink))
        self.assertFalse(np.any(result.glyph_candidate & result.protected_throughgoing))
        self.assertFalse(np.any(result.ambiguous_ink & result.protected_throughgoing))
        self.assertFalse(np.any(result.considered_ink & ~result.source_ink))

    def test_complete_word_dark_pixels_become_unapproved_candidates(self):
        gray, word, labels, components, ids, groups = self.fixture()
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        np.testing.assert_array_equal(result.glyph_candidate, labels > 0)
        self.assertGreater(result.glyph_candidate.sum(), 50)
        self.assertEqual(set(result.hypotheses[0]["glyph_candidate_component_ids"]), set(ids))
        self.assertFalse(result.provenance["glyph_truth_assigned"])
        self.assertFalse(result.provenance["human_approved"])
        self.assertFalse(result.provenance["training_eligible"])
        self.assertFalse(result.provenance["elevation_assigned"])
        self.assertFalse(result.provenance["erase_mask_generated"])
        self.assert_partition(result)

    def test_separator_readings_are_candidates_not_elevation_values(self):
        gray, word, labels, components, ids, groups = self.fixture()
        for text in ('20.0','20,0'):
            result=infer_numeric_ink_ownership(gray,labels,components,groups,readings_for(groups,text))
            np.testing.assert_array_equal(result.glyph_candidate,word)
            hypothesis=result.hypotheses[0]
            self.assertTrue(hypothesis['numeric_separator_present'])
            self.assertFalse(hypothesis['punctuation_ink_ownership_assigned'])
            self.assertFalse(hypothesis['elevation_assigned'])

    def test_multiple_separators_and_missing_fraction_remain_unresolved(self):
        gray, word, labels, components, ids, groups = self.fixture()
        for text in ('20.','2.0.0','20,,0'):
            result=infer_numeric_ink_ownership(gray,labels,components,groups,readings_for(groups,text))
            self.assertFalse(result.glyph_candidate.any())

    def test_small_component_filter_does_not_relax_integer_subword_guard(self):
        gray,word=word_fixture();gray=gray.copy();gray[34,33]=25
        labels,components=source_components(gray)
        ids=sorted(int(v) for v in np.unique(labels[word]) if v)
        full=candidate('full',quad_for_mask(word),ids)
        prefix=candidate('prefix',quad_for_mask(np.isin(labels,ids[:2])),ids[:2])
        suffix=candidate('suffix',quad_for_mask(np.isin(labels,ids[1:])),ids[1:])
        groups=[full,prefix,suffix]
        reads=readings_for([prefix],'20')+readings_for([suffix],'00')
        integer=infer_numeric_ink_ownership(gray,labels,components,groups,readings_for([full],'200')+reads)
        decimal=infer_numeric_ink_ownership(gray,labels,components,groups,readings_for([full],'20.0')+reads)
        self.assertFalse(integer.glyph_candidate.any())
        np.testing.assert_array_equal(decimal.glyph_candidate,word)
        self.assertFalse(decimal.glyph_candidate[34,33])
        self.assert_partition(decimal)

    def test_seed_cc_missing_middle_is_completed_from_actual_quad_ink(self):
        gray, word, labels, components, ids, groups = self.fixture()
        groups[0]["group_component_ids"] = [ids[0], ids[-1]]
        groups[1]["group_component_ids"] = [ids[0], ids[-1]]
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        self.assertTrue(result.glyph_candidate[labels == ids[1]].all())
        self.assertEqual(set(result.hypotheses[0]["glyph_candidate_component_ids"]), set(ids))
        self.assertTrue(all(ids[1] in row["added_nonseed_component_ids"] for row in result.hypotheses[0]["seed_set_corrections"]))
        middle = next(row for row in result.component_evidence if row["component_id"] == ids[1])
        self.assertTrue(middle["evaluated_even_if_not_a_seed"])
        self.assertTrue(middle["considered_by_hypotheses"])

    def test_wrong_curve_seed_is_protected_and_nonseed_glyphs_recovered(self):
        gray, word, labels, components, ids, groups = self.fixture(extra_curve=True)
        curve_id = next(int(v) for v in np.unique(labels) if v > 0 and v not in ids)
        groups[1] = candidate("wrong-wide", [[19, 16], [69, 16], [69, 59], [19, 59]], [ids[0], curve_id])
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        self.assertTrue(result.protected_throughgoing[labels == curve_id].all())
        self.assertFalse(result.glyph_candidate[labels == curve_id].any())
        self.assertTrue(result.glyph_candidate[np.isin(labels, ids)].all())
        correction = next(row for row in result.hypotheses[0]["seed_set_corrections"] if row["group_id"] == "wrong-wide")
        self.assertIn(curve_id, correction["seed_ids_not_owned_as_glyphs"])
        self.assertEqual(set(correction["added_nonseed_component_ids"]), set(ids[1:]))
        self.assert_partition(result)

    def test_rotated_word_uses_original_source_grid_without_box_fill(self):
        gray, word, labels, components, ids, groups = self.fixture(angle=35)
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        self.assertGreater(result.glyph_candidate.sum(), 50)
        self.assertFalse(result.glyph_candidate[gray == 255].any())
        self.assertEqual(set(result.hypotheses[0]["glyph_candidate_component_ids"]), set(ids))
        self.assert_partition(result)

    def test_weak_connected_ink_is_a_separate_warning_not_glyph_mask_growth(self):
        gray, word, labels, components, ids, groups = self.fixture(weak_connection=True)
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        self.assertGreater(result.glyph_candidate.sum(), 30)
        self.assertFalse(result.glyph_candidate[gray == 210].any())
        sensitive = [row for row in result.component_evidence if row["decision"] == "glyph_candidate" and row["weak_connectivity_review_required"]]
        self.assertTrue(sensitive)
        self.assertTrue(any(item["weaker_ink_external_connection"] for row in sensitive for item in row["connectivity_sensitivity"]))

    def test_strict_dark_connection_outside_quad_protects_entire_connected_cc(self):
        gray, word = word_fixture()
        image = Image.fromarray(gray)
        ImageDraw.Draw(image).line([(4, 40), (22, 40)], fill=25, width=1)
        gray = np.asarray(image)
        labels, components = source_components(gray)
        connected_id = int(labels[40, 4])
        ids = [row["component_id"] for row in components]
        quad = quad_for_mask(word)
        groups = [candidate("first", quad, ids), candidate("duplicate", quad_for_mask(word, padding=2.5), ids)]
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        self.assertTrue(result.protected_throughgoing[labels == connected_id].all())
        self.assertFalse(result.glyph_candidate[labels == connected_id].any())
        self.assert_partition(result)

    def test_partial_one_cc_search_never_assigns_the_whole_cc(self):
        gray, word = word_fixture()
        image = Image.fromarray(gray)
        draw = ImageDraw.Draw(image)
        draw.line([(30, 40), (40, 40)], fill=25)
        draw.line([(40, 40), (54, 40)], fill=25)
        gray = np.asarray(image)
        labels, components = source_components(gray)
        self.assertEqual(len(components), 1)
        quad = quad_for_mask(labels > 0)
        groups = [candidate(gid, quad, [1], candidate_kind="intra_component_hole_axis", partial_component_search=True,
                            full_component_containment=False) for gid in ("partial-first", "partial-second")]
        result = infer_numeric_ink_ownership(gray, labels, components, groups, readings_for(groups))
        self.assertFalse(result.glyph_candidate.any())
        self.assertTrue(result.ambiguous_ink.any())
        self.assertIn("partial_component_search_never_assigns_whole_component", result.provenance["decision_reason_counts"])

    def test_conflicting_numeric_words_override_glyph_proposals_and_correct_audits(self):
        gray, word, labels, components, ids, groups = self.fixture()
        others = [candidate("other-"+g["group_id"], g["quad_crop_pixel_centers"], ids) for g in groups]
        result = infer_numeric_ink_ownership(gray, labels, components, groups+others,
                                             readings_for(groups, "100")+readings_for(others, "200"))
        self.assertFalse(result.glyph_candidate.any())
        self.assertTrue(result.considered_ink.any())
        self.assertIn("conflicting_numeric_string_hypotheses", result.provenance["decision_reason_counts"])
        for hypothesis in result.hypotheses:
            self.assertEqual(hypothesis["glyph_candidate_component_ids"], [])
            self.assertTrue(all(not row["added_nonseed_component_ids"] for row in hypothesis["seed_set_corrections"]))
        self.assert_partition(result)

    def test_overlapping_subwords_require_exact_order_and_cover_every_digit(self):
        gray, word, labels, components, ids, groups = self.fixture()
        prefix = candidate("prefix", quad_for_mask(np.isin(labels, ids[:2])), ids[:2])
        suffix = candidate("suffix", quad_for_mask(np.isin(labels, ids[1:])), ids[1:])
        selected = groups[:1]+[prefix, suffix]
        reads = readings_for(groups[:1], "152")+readings_for([prefix], "15")+readings_for([suffix], "52")
        result = infer_numeric_ink_ownership(gray, labels, components, selected, reads)
        np.testing.assert_array_equal(result.glyph_candidate, word)
        full = next(h for h in result.hypotheses if h["numeric_string_hypothesis"] == "152")
        self.assertTrue(full["ordered_subword_corroboration_sufficient"])
        self.assertEqual(len(full["ordered_subword_support"]), 2)
        self.assert_partition(result)
        for bad_reads in (reads[:-2], readings_for(groups[:1], "152")+readings_for([prefix], "52")+readings_for([suffix], "15")):
            rejected = infer_numeric_ink_ownership(gray, labels, components, selected, bad_reads)
            self.assertFalse(rejected.glyph_candidate.any())

    def test_partial_component_subword_cannot_corroborate_full_word(self):
        gray, word, labels, components, ids, groups = self.fixture()
        prefix = candidate("prefix", quad_for_mask(np.isin(labels, ids[:2])), ids[:2], partial_component_search=True)
        suffix = candidate("suffix", quad_for_mask(np.isin(labels, ids[1:])), ids[1:])
        result = infer_numeric_ink_ownership(gray, labels, components, groups[:1]+[prefix, suffix],
            readings_for(groups[:1], "152")+readings_for([prefix], "15")+readings_for([suffix], "52"))
        self.assertFalse(result.glyph_candidate.any())

    def test_uncorroborated_disagreeing_low_score_and_nonnumeric_readings_abstain(self):
        gray, word, labels, components, ids, groups = self.fixture()
        variants = [readings_for(groups, "10", .3), readings_for(groups, "河川"), readings_for(groups, "1")]
        disagree = readings_for(groups)
        disagree[-1]["rec_text"] = "100"
        disagree[1]["rec_text"] = "100"
        variants.append(disagree)
        for readings in variants:
            with self.subTest(readings=readings):
                result = infer_numeric_ink_ownership(gray, labels, components, groups, readings)
                self.assertFalse(result.glyph_candidate.any())
                self.assertFalse(result.considered_ink.any())
        result = infer_numeric_ink_ownership(gray, labels, components, groups[:1], readings_for(groups[:1]))
        self.assertFalse(result.glyph_candidate.any())
        self.assertTrue(result.considered_ink.any())

    def test_unexamined_ink_stays_unknown_without_considered_veto(self):
        gray, word, labels, components, ids, groups = self.fixture()
        result = infer_numeric_ink_ownership(gray, labels, components, groups, [])
        np.testing.assert_array_equal(result.ambiguous_ink, labels > 0)
        self.assertFalse(result.considered_ink.any())
        self.assertEqual(result.component_evidence, ())
        self.assertEqual(result.hypotheses, ())

    def test_input_metadata_pixels_and_readings_are_unchanged_and_masks_readonly(self):
        gray, word, labels, components, ids, groups = self.fixture()
        reads = readings_for(groups)
        before_gray, before_labels = gray.copy(), labels.copy()
        before = copy.deepcopy((components, groups, reads))
        result = infer_numeric_ink_ownership(gray, labels, components, groups, reads)
        np.testing.assert_array_equal(gray, before_gray)
        np.testing.assert_array_equal(labels, before_labels)
        self.assertEqual((components, groups, reads), before)
        for name in ("source_ink", "glyph_candidate", "ambiguous_ink", "protected_throughgoing", "considered_ink"):
            self.assertFalse(getattr(result, name).flags.writeable)
            self.assertFalse(np.shares_memory(getattr(result, name), labels))

    def test_metadata_pixel_mismatch_and_white_pixel_labels_are_rejected(self):
        gray, word, labels, components, ids, groups = self.fixture()
        bad = copy.deepcopy(components)
        bad[0]["ink_pixels"] += 1
        with self.assertRaisesRegex(ValueError, "area or pixel bounds"):
            infer_numeric_ink_ownership(gray, labels, bad, groups, readings_for(groups))
        bad_labels = labels.copy()
        bad_labels[0, 0] = ids[0]
        with self.assertRaisesRegex(ValueError, "paper or unsupported"):
            infer_numeric_ink_ownership(gray, bad_labels, components, groups, readings_for(groups))

    def test_uint64_component_id_is_checked_before_int32_conversion(self):
        gray = np.full((12, 12), 255, np.uint8)
        gray[5, 5] = 25
        labels = np.zeros((12, 12), np.uint64)
        labels[5, 5] = 2**32
        with self.assertRaisesRegex(ValueError, "component IDs exceed"):
            infer_numeric_ink_ownership(gray, labels, [], [], [])

    def test_duplicate_scale_unknown_group_invalid_score_and_bounds_fail(self):
        gray, word, labels, components, ids, groups = self.fixture()
        reads = readings_for(groups)
        for changes in ({"group_id": "unknown"}, {"scale": 3}, {"rec_score": True}, {"rec_score": float("nan")}):
            bad = copy.deepcopy(reads)
            bad[0].update(changes)
            with self.assertRaises(ValueError): infer_numeric_ink_ownership(gray, labels, components, groups, bad)
        with self.assertRaises(ValueError): infer_numeric_ink_ownership(gray, labels, components, groups, reads+[reads[0]])
        with self.assertRaises(ValueError): infer_numeric_ink_ownership(gray, labels, components, groups, reads, config=replace(NumericInkOwnershipConfig(), max_point_quad_tests=1))
        with self.assertRaises(ValueError): infer_numeric_ink_ownership(gray, labels, components, groups, reads, config=replace(NumericInkOwnershipConfig(), max_image_pixels=1))


@unittest.skipUnless(FIXTURES.is_dir(), "fixed local development OCR evidence is not bundled in lightweight test environments")
class FixedNumericInkRegressionTests(unittest.TestCase):
    def run_region(self, rid):
        record_path, arrays_path = FIXTURES/rid/"candidates-and-readings.json", FIXTURES/rid/"component-pixels.npz"
        self.assertEqual((sha256_file(record_path), sha256_file(arrays_path)), PINS[rid])
        record = json.loads(record_path.read_text())
        source = Path(record["source_absolute_path"])
        self.assertEqual(sha256_file(source), record["source_raster_sha256"])
        with Image.open(source) as image: gray = np.asarray(image.crop(record["box"]).convert("L"))
        with np.load(arrays_path, allow_pickle=False) as arrays: labels = arrays["component_labels"]
        groups = [row["candidate"] for row in record["groups"]]
        readings = [{"group_id": row["candidate"]["group_id"], "scale": entry["scale"], **entry["recognition"]}
                    for row in record["groups"] for entry in row["passes"] if entry.get("status") == "completed"]
        before = gray.copy(), labels.copy()
        result = infer_numeric_ink_ownership(gray, labels, record["components"], groups, readings)
        np.testing.assert_array_equal(gray, before[0])
        np.testing.assert_array_equal(labels, before[1])
        self.assertEqual((sha256_file(record_path), sha256_file(arrays_path)), PINS[rid])
        self.assertEqual(sha256_file(source), record["source_raster_sha256"])
        return record, labels, result

    def test_r004_505_excludes_wrong_seed_88_and_recovers_82_79(self):
        record, labels, result = self.run_region("R004")
        hypothesis = next(h for h in result.hypotheses if h["numeric_string_hypothesis"] == "505")
        self.assertEqual(set(hypothesis["glyph_candidate_component_ids"]), {82, 79})
        self.assertTrue(result.glyph_candidate[np.isin(labels, [82, 79])].all())
        self.assertFalse(result.glyph_candidate[labels == 88].any())
        self.assertTrue(result.protected_throughgoing[labels == 88].all())
        wrong_id = next(g["candidate"]["group_id"] for g in record["groups"] if g["selection_rank"] == 3)
        correction = next(r for r in hypothesis["seed_set_corrections"] if r["group_id"] == wrong_id)
        self.assertIn(79, correction["added_nonseed_component_ids"])
        self.assertIn(88, correction["seed_ids_not_owned_as_glyphs"])
        last_five = next(r for r in result.component_evidence if r["component_id"] == 79)
        self.assertTrue(last_five["weak_connectivity_review_required"])
        self.assertEqual(result.provenance["counts"]["glyph_candidate_pixels"], 255)

    def test_r015_100_completes_nonseed_middle_114_and_marks_weak_links(self):
        record, labels, result = self.run_region("R015")
        hypothesis = next(h for h in result.hypotheses if h["numeric_string_hypothesis"] == "100")
        self.assertEqual(set(hypothesis["glyph_candidate_component_ids"]), {124, 114, 106})
        self.assertTrue(result.glyph_candidate[labels == 114].all())
        missing_id = next(g["candidate"]["group_id"] for g in record["groups"] if g["selection_rank"] == 19)
        correction = next(r for r in hypothesis["seed_set_corrections"] if r["group_id"] == missing_id)
        self.assertIn(114, correction["added_nonseed_component_ids"])
        sensitive = {r["component_id"] for r in result.component_evidence if r["decision"] == "glyph_candidate" and r["weak_connectivity_review_required"]}
        self.assertEqual(sensitive, {106, 114})
        self.assertEqual(result.provenance["counts"]["glyph_candidate_pixels"], 204)
        self.assertFalse(result.glyph_candidate[labels == 198].any())
        self.assertFalse(any(h["numeric_string_hypothesis"] == "200" for h in result.hypotheses))

    def test_r009_has_no_supported_numeric_hypothesis_or_considered_veto(self):
        _record, labels, result = self.run_region("R009")
        self.assertFalse(result.glyph_candidate.any())
        self.assertFalse(result.considered_ink.any())
        self.assertEqual(result.hypotheses, ())
        np.testing.assert_array_equal(result.ambiguous_ink, labels > 0)


if __name__ == "__main__":
    unittest.main()
