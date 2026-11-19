import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from baselines.run_pretrained import CSV_FIELDS, batches, prediction_rows, write_csv


def record(uid, duration, text="þess vegna", raw="Þess vegna."):
    return {"id": uid, "split": "test", "speaker_session_id": "007107",
            "duration_sec": duration, "text": text, "text_raw": raw}


class PredictionTableTest(unittest.TestCase):
    def test_rows_follow_manifest_order_and_normalize_hypothesis(self):
        records = [record("0000001", 1.0), record("0000002", 5.0)]
        rows = prediction_rows(records, {"0000002": "Já, takk!", "0000001": "Þess VEGNA."})
        self.assertEqual([r["id"] for r in rows], ["0000001", "0000002"])
        self.assertEqual(rows[0]["hypothesis"], "þess vegna")
        self.assertEqual(rows[1]["hypothesis"], "já takk")
        self.assertEqual(rows[1]["hypothesis_raw"], "Já, takk!")

    def test_missing_hypothesis_fails(self):
        with self.assertRaises(ValueError):
            prediction_rows([record("0000001", 1.0)], {})

    def test_batches_cover_every_clip_once(self):
        records = [record(f"{i:07d}", float(i)) for i in range(10)]
        seen = [r["id"] for b in batches(records, 3) for r in b]
        self.assertEqual(sorted(seen), sorted(r["id"] for r in records))

    def test_csv_keeps_leading_zeros_and_icelandic_text(self):
        rows = prediction_rows([record("0000042", 1.0)], {"0000042": "Ærið ðól"})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            write_csv(path, rows)
            with path.open(encoding="utf-8", newline="") as handle:
                back = list(csv.DictReader(handle))
        self.assertEqual(tuple(back[0]), CSV_FIELDS)
        self.assertEqual(back[0]["id"], "0000042")
        self.assertEqual(back[0]["hypothesis"], "ærið ðól")


if __name__ == "__main__":
    unittest.main()
