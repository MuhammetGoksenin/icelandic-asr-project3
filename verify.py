from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from audit import SPLITS, audit_path, pcm_hash, read_metadata, sha256
from prepare import RATE, TARGET_HOURS, build, normalize_text, read_jsonl, seeded_order
from preprocess import (WHISPER_MODEL, WHISPER_REVISION, ctc_tokenizer, load_example, resolve_audio,
                        wav2vec2_example, whisper_components, whisper_example)

GENERATED = ("train.jsonl", "dev.jsonl", "test.jsonl", "evaluation_references.jsonl",
             "evaluation_references.tsv", "ctc_vocab.json", "exclusions.tsv", "listening_queue.tsv",
             "focused_review.tsv", "prompt_overlap.tsv", "quality_summary.json", "quality_report.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_clip(root, record, expected_md5=None):
    path = resolve_audio(root, record)
    source_file = path
    if expected_md5 is not None:
        contents = path.read_bytes()
        require(hashlib.md5(contents, usedforsecurity=False).hexdigest() == expected_md5,
                f"Original FLAC checksum differs for {record['id']}")
        source_file = io.BytesIO(contents)
    with sf.SoundFile(source_file) as source:
        require((source.samplerate, source.channels, source.format, source.subtype) == (RATE, 1, "FLAC", "PCM_16"),
                f"Unexpected audio format for {record['id']}")
        waveform = source.read(dtype="int32", always_2d=True)
    require(len(waveform) == record["num_samples"], f"Frame count differs for {record['id']}")
    require(pcm_hash(waveform, RATE) == record["pcm_sha256"], f"PCM hash differs for {record['id']}")


def check_manifests(splits, output, metadata=None):
    used = {field: set() for field in ("id", "audio_path", "pcm_sha256")}
    previous_splits = {field: set() for field in ("speaker_session_id", "text", "text_raw")}
    references = list(read_jsonl(output / "evaluation_references.jsonl"))
    with (output / "evaluation_references.tsv").open(encoding="utf-8", newline="") as stream:
        require(list(csv.DictReader(stream, delimiter="\t")) == references, "TSV and JSONL references differ")
    require(references == [{key: row[key] for key in ("id", "split", "text", "text_raw")}
                           for split in ("dev", "test") for row in splits[split]],
            "Evaluation references disagree with dev/test manifests")

    for split, rows in splits.items():
        require(rows, f"{split} manifest is empty")
        for row in rows:
            require(row["split"] == split, f"Manifest split differs for {row['id']}")
            require(row["text"] and row["text"] == normalize_text(row["text_source_normalized"]),
                    f"Invalid normalized text for {row['id']}")
            if metadata is not None:
                meta = metadata[row["audio_path"]]
                require((row["id"], row["speaker_session_id"], split) == (meta["id"], meta["speaker_id"], meta["status"]),
                        f"Official identity or split differs for {row['id']}")
                require(row["text_raw"] == meta["sentence"] and row["text_source_normalized"] == meta["sentence_norm"],
                        f"Metadata text differs for {row['id']}")
            require(row["num_samples"] / RATE == row["duration_sec"] and row["sample_rate"] == RATE,
                    f"Invalid manifest duration or sample rate for {row['id']}")
            for field, values in used.items():
                require(row[field] not in values, f"Repeated {field} for {row['id']}")
                values.add(row[field])
        for field, previous in previous_splits.items():
            current = {row[field] for row in rows}
            require(not current & previous, f"Cross-split {field} overlaps in {split}")
            previous.update(current)
        delta = sum(row["num_samples"] for row in rows) / RATE - TARGET_HOURS[split] * 3600
        require(abs(delta) <= 15, f"{split} selection is {delta:+.3f} seconds from target")
    return len(references)


def check_selected_audio(root, splits, file_md5s=None):
    def check(row):
        expected = file_md5s[row["audio_path"]] if file_md5s is not None else None
        return check_clip(root, row, expected)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for split, rows in splits.items():
            for _ in pool.map(check, rows, buffersize=512):
                pass
            print(f"Verified all {len(rows):,} {split} PCM hashes", flush=True)


def check_model_inputs(root, splits, output, seed, whisper_extractor, whisper_tokenizer):
    from transformers import Wav2Vec2FeatureExtractor

    wav2vec_extractor = Wav2Vec2FeatureExtractor(sampling_rate=RATE, do_normalize=True, return_attention_mask=True)
    wav2vec_tokenizer = ctc_tokenizer(output / "ctc_vocab.json")
    vocab = json.loads((output / "ctc_vocab.json").read_text(encoding="utf-8"))
    expected_chars = set().union(*(set(row["text"]) for row in splits["train"])) - {" "}
    require(set(vocab) == expected_chars | {"[PAD]", "[UNK]", "|"}, "CTC vocabulary differs from train characters")
    feature_checks = {}
    for split, rows in splits.items():
        targets = [row["text"] for row in rows]
        whisper_ids = whisper_tokenizer(targets).input_ids
        language_id = whisper_tokenizer.convert_tokens_to_ids("<|is|>")
        task_id = whisper_tokenizer.convert_tokens_to_ids("<|transcribe|>")
        require(all(ids[:len(whisper_tokenizer.prefix_tokens)] == whisper_tokenizer.prefix_tokens for ids in whisper_ids)
                and language_id in whisper_tokenizer.prefix_tokens and task_id in whisper_tokenizer.prefix_tokens,
                "Whisper Icelandic/transcribe prefix differs")
        require(whisper_tokenizer.batch_decode(whisper_ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False) == targets,
                f"{split} Whisper labels do not round-trip")
        ctc_ids = wav2vec_tokenizer(targets, add_special_tokens=False).input_ids
        manual_ids = [[vocab.get(char, vocab["[UNK]"]) for char in text.replace(" ", "|")] for text in targets]
        require(ctc_ids == manual_ids, f"{split} CTC tokenizer differs from character vocabulary")
        if not any(vocab["[UNK]"] in ids for ids in ctc_ids):
            require(wav2vec_tokenizer.batch_decode(ctc_ids, group_tokens=False, clean_up_tokenization_spaces=False) == targets,
                    f"{split} CTC labels do not round-trip")

        # Include duration extremes and seeded examples from both review states, when available.
        ordered = sorted(rows, key=lambda row: seeded_order(row, seed + 2))
        sample = {row["id"]: row for row in [
            min(rows, key=lambda row: row["num_samples"]), max(rows, key=lambda row: row["num_samples"]),
            *[row for row in ordered if row["flags"]][:4], *[row for row in ordered if not row["flags"]][:4],
        ]}
        for row in sample.values():
            example = row | {"waveform": load_example(root, row)}
            whisper = whisper_example(example, whisper_extractor, whisper_tokenizer)
            ctc = wav2vec2_example(example, wav2vec_extractor, wav2vec_tokenizer)
            require(whisper["input_features"].shape == (80, 3000) and np.isfinite(whisper["input_features"]).all()
                    and whisper["attention_mask"].shape == (3000,) and 0 < whisper["attention_mask"].sum() < 3000,
                    f"Invalid Whisper features for {row['id']}")
            require(ctc["input_values"].dtype == np.float32 and np.isfinite(ctc["input_values"]).all()
                    and ctc["input_values"].shape == (row["num_samples"],)
                    and ctc["attention_mask"].shape == (row["num_samples"],) and ctc["attention_mask"].all(),
                    f"Invalid wav2vec 2.0 inputs for {row['id']}")
            require(whisper["text"] == ctc["text"] == row["text"] and whisper["labels"] and ctc["labels"],
                    f"Missing shared target or labels for {row['id']}")
        feature_checks[split] = list(sample)
        print(f"Verified all {split} labels and {len(sample)} examples through both feature extractors", flush=True)
    return feature_checks


def verify(root, output, offline, *, replay_only=False):
    summary = json.loads((output / "quality_summary.json").read_text(encoding="utf-8"))
    metadata = read_metadata((root / "metadata.tsv").read_bytes())
    splits = {split: list(read_jsonl(output / f"{split}.jsonl")) for split in SPLITS}
    reference_count = check_manifests(splits, output, metadata)

    with TemporaryDirectory(prefix="samromur-verify-", dir=output.parent) as temporary:
        replay = Path(temporary)
        source_audit = audit_path(output)
        replay_audit = replay / source_audit.name
        try:
            os.link(source_audit, replay_audit)
        except OSError:
            shutil.copyfile(source_audit, replay_audit)
        build(root, replay, summary["seed"])
        digests = {}
        for name in GENERATED:
            data = (output / name).read_bytes()
            require(data == (replay / name).read_bytes(), f"Reproduction differs byte for byte for {name}")
            digests[name] = sha256(data)

    if replay_only:
        print(f"Audit replay matched all {len(digests)} artifacts byte for byte. "
              f"Checked {reference_count:,} shared evaluation references.", flush=True)
        return

    from transformers import __version__ as transformers_version

    check_selected_audio(root, splits)
    components = whisper_components(local_files_only=offline)
    feature_checks = check_model_inputs(root, splits, output, summary["seed"], *components)
    report = {
        "status": "passed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libsndfile": sf.__libsndfile_version__,
        "transformers": transformers_version,
        "whisper_assets": {"checkpoint": WHISPER_MODEL, "revision": WHISPER_REVISION},
        "selected_pcm_paths_verified": {split: len(rows) for split, rows in splits.items()},
        "labels_verified": {split: len(rows) for split, rows in splits.items()},
        "shared_reference_rows_verified": reference_count,
        "feature_smoke_test_ids": feature_checks,
        "reproduced_artifacts_sha256": digests,
        "human_listening": "pending",
    }
    (output / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("Verification passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify generated Samrómur artifacts and ASR inputs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--offline", action="store_true", help="Read cached Whisper assets without network access")
    parser.add_argument("--replay-only", action="store_true",
                        help="Compare regenerated artifacts using the audit and release documents only")
    args = parser.parse_args()
    verify(args.root, args.output, args.offline, replay_only=args.replay_only)
