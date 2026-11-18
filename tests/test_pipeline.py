import csv
import gzip
import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from audit import SPLITS, audit_path, inspect_file, read_metadata, run_audit
from prepare import build, flags_for, normalize_text, overlap, read_jsonl, select
from verify import GENERATED, check_manifests
from verify_handoff import verify_package
from subset import clean_subset, raw_membership, split_integrity, stage_release

ONE_SECOND = {split: 1 / 3600 for split in SPLITS}
FIELDS = ("id", "speaker_id", "filename", "status", "sentence", "sentence_norm",
          "duration", "sample_rate", "is_valid", "empty")


def metadata_row(split="train", speaker="000010", uid="0000001", sentence="Þá."):
    return dict(zip(FIELDS, (uid, speaker, f"{speaker}-{uid}.flac", split, sentence, sentence,
                             "1.0", "16000", "1.0", "0")))


def flac(path, amplitude):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.tile(np.array([amplitude, -amplitude], dtype=np.int16), 8000)
    sf.write(path, samples, 16_000, subtype="PCM_16")
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def item(split, uid, speaker, reference, pcm):
    return {
        "id": uid, "split": split, "session": speaker,
        "audio_path": f"{split}/{speaker}/{speaker}-{uid}.flac",
        "raw_prompt": reference, "source_normalized": reference, "reference": reference,
        "pcm": pcm, "file_md5": None, "frames": 16_000, "blocked": [], "selected": False,
    }


class AuditTests(unittest.TestCase):
    def test_unicode_nfc_lowercase_and_punctuation(self):
        source = "  ÞETTA\u00a0— Æ-FIÐ! A\u0301 E\u0301 I\u0301 O\u0301 O\u0308 U\u0301 Y\u0301. €16 "
        self.assertEqual(normalize_text(source), "þetta æ fið á é í ó ö ú ý 16")

    def test_malformed_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required metadata"):
            read_metadata(b"id\tsentence\n1\tNo\n")
        header = "\t".join(FIELDS) + "\n"
        with self.assertRaisesRegex(ValueError, "Ragged TSV row"):
            read_metadata((header + "0000001\t000010\n").encode("utf-8"))
        values = metadata_row(split="validation")
        with self.assertRaisesRegex(ValueError, "Unexpected metadata identity"):
            read_metadata((header + "\t".join(values.values()) + "\n").encode("utf-8"))

    def test_missing_and_corrupt_recordings(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "missing.flac"
            missing = inspect_file(root, name, None)
            self.assertEqual(flags_for(missing, None), (["no_metadata", "missing"], ["no_release_checksum"]))
            (root / name).write_bytes(b"This is not a FLAC.")
            corrupt = inspect_file(root, name, "0" * 32)
            bad_metadata = metadata_row() | {"sentence": " ", "sentence_norm": "—", "is_valid": "0.0", "empty": "1"}
            self.assertEqual(flags_for(corrupt, bad_metadata), (
                ["missing_raw_transcript", "missing_normalized_transcript", "metadata_not_validated",
                 "metadata_empty_or_unknown", "decode_error", "release_checksum_mismatch"], [],
            ))

    def test_digital_silence_near_silence_and_quiet_but_eligible(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for amplitude, expected_dbfs, quiet, blocked, flags in (
                (0, None, 1, ["digital_silence"], []),
                (32, -60.206, 1, ["near_silence"], []),
                (128, -48.1648, 0, [], ["low_rms"]),
            ):
                with self.subTest(amplitude=amplitude):
                    path = root / f"{amplitude}.flac"
                    checksum = flac(path, amplitude)
                    audio = inspect_file(root, path.name, checksum)
                    self.assertEqual(audio["file_md5"], checksum)
                    self.assertEqual(audio["rms_dbfs"], expected_dbfs)
                    self.assertEqual(audio["quiet_fraction"], quiet)
                    self.assertEqual(flags_for(audio, metadata_row()), (blocked, flags))
            for duration in ("bad", "inf", "NAN"):
                row = metadata_row() | {"duration": duration}
                self.assertEqual(flags_for(audio, row), ([], ["audio_metadata_disagreement", "low_rms"]))


class SelectionTests(unittest.TestCase):
    def test_matching_duplicate_prefers_test_and_conflicting_pcm_is_withheld(self):
        test = item("test", "0000001", "000010", "þá", "matching")
        copy = item("train", "0000002", "000020", "þá", "matching")
        clip_a = item("train", "0000003", "000030", "æð", "conflict")
        clip_b = item("dev", "0000004", "000040", "öð", "conflict")
        train = item("train", "0000005", "000050", "ár", "independent-train")
        dev = item("dev", "0000006", "000060", "sæ", "independent-dev")
        with patch.dict("prepare.TARGET_HOURS", ONE_SECOND):
            chosen = select([train, copy, clip_a, dev, clip_b, test], 2105)
        self.assertEqual({split: [x["id"] for x in chosen[split]] for split in SPLITS},
                         {"train": ["0000005"], "dev": ["0000006"], "test": ["0000001"]})
        self.assertEqual(copy["blocked"], ["duplicate_pcm"])
        self.assertEqual(copy["related_audio_path"], test["audio_path"])
        self.assertEqual(clip_a["blocked"], ["duplicate_pcm_conflicting_reference"])
        self.assertEqual(clip_b["blocked"], ["duplicate_pcm_conflicting_reference"])

    def test_overlap_counts_and_evaluation_priority(self):
        evaluation = item("test", "0000001", "000010", "þá", "test")
        shared_session = item("train", "0000002", "000010", "æð", "train-a")
        shared_prompt = item("train", "0000003", "000030", "þá", "train-b")
        repeated_prompt = item("train", "0000004", "000040", "þá", "train-c")
        clean = item("train", "0000005", "000050", "ár", "train-d")
        dev = item("dev", "0000006", "000060", "sæ", "dev")
        groups = {"train": [shared_session, shared_prompt, repeated_prompt, clean], "dev": [dev], "test": [evaluation]}
        counts = overlap(groups)["train/test"]
        self.assertEqual(counts["session"], {"shared": 1, "train": 1, "test": 1})
        self.assertEqual(counts["reference"], {"shared": 1, "train": 2, "test": 1})
        self.assertEqual(counts["pcm"], {"shared": 0, "train": 0, "test": 0})
        with patch.dict("prepare.TARGET_HOURS", ONE_SECOND):
            chosen = select([row for group in groups.values() for row in group], 2105)
        self.assertEqual([row["id"] for row in chosen["train"]], ["0000005"])
        self.assertEqual(shared_session["blocked"], ["shared_session_with_selected_evaluation"])
        self.assertIn("shared_reference_with_selected_evaluation", shared_prompt["blocked"])

    def test_known_seed_order_and_original_partitions(self):
        clips = [
            item("train", "0000001", "000010", "ár", "1"),
            item("train", "0000002", "000010", "þá", "2"),
            item("train", "0000003", "000010", "öð", "3"),
            item("dev", "0000004", "000040", "sæ", "4"),
            item("test", "0000005", "000050", "íð", "5"),
        ]
        expected = {"train": ["0000003", "0000001"], "dev": ["0000004"], "test": ["0000005"]}
        with patch.dict("prepare.TARGET_HOURS", ONE_SECOND | {"train": 2 / 3600}):
            for source in (clips, clips[::-1]):
                chosen = select(deepcopy(source), 2105)
                self.assertEqual({split: [x["id"] for x in chosen[split]] for split in SPLITS}, expected)
                for split, rows in chosen.items():
                    self.assertTrue(all(row["split"] == split and row["audio_path"].startswith(split + "/")
                                        for row in rows))


class ArtifactTests(unittest.TestCase):
    def test_plain_and_gzip_audit_reproduce_expected_manifests_and_LF_vocab(self):
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace / "release"
            plain = workspace / "plain"
            packed = workspace / "packed"
            packed.mkdir()
            rows = [
                metadata_row("train", "000010", "0000001", "Þá."),
                metadata_row("dev", "000020", "0000002", "Öð?"),
                metadata_row("test", "000030", "0000003", "Æ!"),
            ]
            sums = []
            for index, row in enumerate(rows, 1):
                path = f"{row['status']}/{row['speaker_id']}/{row['filename']}"
                sums.append(f"{flac(root / path, index * 3000)}  r1/{path}")
            (root / "README").write_bytes(b"Fixture\n")
            (root / "md5s.txt").write_bytes(("\n".join(sums) + "\n").encode("utf-8"))
            with (root / "metadata.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, FIELDS, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            run_audit(root, plain, 1)
            (packed / "audio_audit.jsonl.gz").write_bytes(gzip.compress(audit_path(plain).read_bytes(), mtime=0))
            with patch.dict("prepare.TARGET_HOURS", ONE_SECOND), patch.dict("verify.TARGET_HOURS", ONE_SECOND):
                build(root, plain, 2105)
                build(root, packed, 2105)
                manifests = {split: list(read_jsonl(packed / f"{split}.jsonl")) for split in SPLITS}
                self.assertEqual(check_manifests(manifests, packed), 2)
            self.assertEqual([[row["id"] for row in manifests[key]] for key in SPLITS],
                             [["0000001"], ["0000002"], ["0000003"]])
            self.assertEqual(list(read_jsonl(packed / "evaluation_references.jsonl")), [
                {"id": "0000002", "split": "dev", "text": "öð", "text_raw": "Öð?"},
                {"id": "0000003", "split": "test", "text": "æ", "text_raw": "Æ!"},
            ])
            expected_vocab = '{\n  "[PAD]": 0,\n  "[UNK]": 1,\n  "|": 2,\n  "á": 3,\n  "þ": 4\n}\n'
            self.assertEqual((packed / "ctc_vocab.json").read_bytes(), expected_vocab.encode("utf-8"))
            summary = json.loads((packed / "quality_summary.json").read_bytes())
            self.assertEqual(summary["splits"]["dev"]["ctc_oov_characters"], {"ö": 1, "ð": 1})
            self.assertEqual(summary["splits"]["test"]["ctc_oov_characters"], {"æ": 1})
            for name in GENERATED:
                data = (packed / name).read_bytes()
                self.assertEqual(data, (plain / name).read_bytes(), name)
                self.assertNotIn(b"\r", data, name)
                self.assertTrue(data.endswith(b"\n"), name)

    def test_handoff_verifier_requires_packaged_audio_and_assets(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "extracted handoff"):
                verify_package(Path(temporary), None)


class RawSubsetTests(unittest.TestCase):
    def test_raw_scope_uses_first_official_id_independently_of_quality(self):
        first = "train/000010/000010-0000001.flac"
        chosen = "train/000010/000010-0000002.flac"
        unrelated = "train/000020/000020-0000003.flac"
        metadata = {
            chosen: metadata_row(uid="0000002"),
            unrelated: metadata_row(speaker="000020", uid="0000003"),
            first: metadata_row() | {"is_valid": "0.0"},
        }
        row = {"audio_path": chosen, "split": "train", "id": "0000002", "speaker_session_id": "000010"}
        expected = {first: "first_session_utterance", chosen: "frozen_v1"}
        self.assertEqual(raw_membership(metadata, {"train": [row]}), expected)
        self.assertEqual(raw_membership(dict(reversed(list(metadata.items()))), {"train": [row]}), expected)
        with self.assertRaisesRegex(ValueError, "Official identity"):
            raw_membership(metadata, {"dev": [row]})
        with self.assertRaisesRegex(ValueError, "Official identity"):
            raw_membership(metadata, {"train": [row | {"speaker_session_id": "000020"}]})

    def test_exact_and_normalized_test_overlap_count_utterances_separately(self):
        def utterance(speaker, original, normalized):
            return {"session": speaker, "raw_prompt": original, "reference": normalized,
                    "frames": 16_000, "audio": {"sample_rate": 16_000}}

        clips = {"train": [utterance("01", "Þá.", "þá")], "dev": [],
                 "test": [utterance("02", "Þá.", "þá"), utterance("02", "ÞÁ!", "þá"),
                          utterance("02", "Æ.", "æ")]}
        report = split_integrity(clips)
        self.assertEqual(report["test_in_train"], {
            "exact_original_sentence": {"test_utterances": 1, "distinct_sentences": 1},
            "normalized_scoring_text": {"test_utterances": 2, "distinct_sentences": 1},
        })
        self.assertEqual(report["splits"]["test"], {
            "speaker_session_ids": 1, "utterances": 3, "decoded_hours": 3 / 3600,
        })
        clips["test"][0]["session"] = "01"
        with self.assertRaisesRegex(ValueError, "Cross-split speaker/session overlap"):
            split_integrity(clips)

    def test_real_raw_flacs_rebuild_frozen_labels_and_log_a_quality_removal(self):
        with TemporaryDirectory() as temporary, patch.dict("prepare.TARGET_HOURS", ONE_SECOND):
            base = Path(temporary)
            source, raw, frozen, clean, output = (
                base / name for name in ("release", "raw", "frozen", "cleaned", "output")
            )
            rows = [
                metadata_row(sentence="Hljótt."),
                metadata_row(uid="0000002", sentence="Þá."),
                metadata_row("dev", "000020", "0000003", "Öð?"),
                metadata_row("test", "000030", "0000004", "Æ!"),
            ]
            paths = [f"{row['status']}/{row['speaker_id']}/{row['filename']}" for row in rows]
            sums = [f"{flac(source / path, amplitude)}  r1/{path}"
                    for path, amplitude in zip(paths, (0, 2000, 4000, 6000))]
            (source / "README").write_bytes(b"Fixture\n")
            (source / "md5s.txt").write_bytes(("\n".join(sums) + "\n").encode("utf-8"))
            with (source / "metadata.tsv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, FIELDS, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            run_audit(source, frozen, 1)
            build(source, frozen, 2105)
            raw.mkdir()
            for name in ("README", "metadata.tsv", "md5s.txt"):
                (raw / name).write_bytes((source / name).read_bytes())
            stage_release(source, raw, frozen)
            clean_subset(raw, clean, frozen, output, 1)

            self.assertEqual(sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*.flac")), sorted(paths))
            self.assertEqual(sorted(p.relative_to(clean).as_posix() for p in clean.rglob("*.flac")), sorted(paths[1:]))
            for path in paths[1:]:
                self.assertEqual((clean / path).read_bytes(), (source / path).read_bytes())
            with (output / "raw_subset_removals.tsv").open(encoding="utf-8", newline="") as stream:
                removals = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([(r["id"], r["role"], r["reasons"]) for r in removals],
                             [("0000001", "first_session_utterance", "digital_silence")])
            self.assertEqual(list(read_jsonl(output / "evaluation_references.jsonl")), [
                {"id": "0000003", "split": "dev", "text": "öð", "text_raw": "Öð?"},
                {"id": "0000004", "split": "test", "text": "æ", "text_raw": "Æ!"},
            ])
            self.assertEqual(json.loads((output / "ctc_vocab.json").read_bytes()),
                             {"[PAD]": 0, "[UNK]": 1, "|": 2, "á": 3, "þ": 4})
            (raw / paths[2]).write_bytes(b"git-lfs pointer instead of audio")
            with self.assertRaisesRegex(ValueError, "Missing or changed original FLAC bytes"):
                clean_subset(raw, clean, frozen, base / "failure", 1)


if __name__ == "__main__":
    unittest.main()
