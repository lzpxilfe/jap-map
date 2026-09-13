import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.render_assisted_drawing_sheets import render


class AssistedReviewSheetTests(unittest.TestCase):
    def source(self, root):
        packet = root/"packet"
        (packet/"images").mkdir(parents=True)
        (packet/"drawing-report.json").write_text(json.dumps({
            "priority_ids": ["A0001"],
            "proposals": [{"proposal_id": "A0001", "mode": "contextual_gap", "gap_pixels": 12.}],
        }), encoding="utf-8")
        for suffix in ("source", "proposal"):
            Image.new("RGB", (161, 161), "white").save(packet/"images"/f"A0001-{suffix}.png")
        return packet

    def test_external_batch_preserves_packet_bytes_and_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packet = self.source(root)
            before = {str(p.relative_to(packet)): p.read_bytes() for p in packet.rglob("*") if p.is_file()}
            output = root/"feedback"/"batch-002"
            paths = render(packet, ["A0001"], case_columns=1, output=output)
            self.assertEqual(paths, [str(output/"decisions-01.png")])
            with Image.open(paths[0]) as image:
                self.assertEqual(image.size, (512, 300))
            self.assertEqual(before, {str(p.relative_to(packet)): p.read_bytes() for p in packet.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                render(packet, ["A0001"], output=output)

    def test_bad_batch_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packet = self.source(root)
            output = root/"invalid"
            for ids, columns in ((["missing"], 1), (["A0001", "A0001"], 1), (["A0001"], 3)):
                with self.assertRaises(ValueError):
                    render(packet, ids, case_columns=columns, output=output)
                self.assertFalse(output.exists())

    def test_default_remains_a_new_packet_subfolder(self):
        with tempfile.TemporaryDirectory() as temp:
            packet = self.source(Path(temp))
            paths = render(packet, output_name="questions")
            self.assertEqual(paths, [str(packet/"questions"/"decisions-01.png")])


if __name__ == "__main__":
    unittest.main()
