"""Materialize and verify the frozen v1 raw and cleaned Samrómur subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from audit import SPLITS, inspect_file, inventory, read_checksums, read_metadata, sha256
from prepare import (SEED, make_items, mark_duplicates, read_jsonl, seeded_order, write_asr_artifacts,
                     write_tsv)
from verify import check_manifests, require

VERSION = "samromur-21.05-20h-v1"
RAW = Path("data/raw/samromur_21.05")
CLEANED = Path("data/cleaned") / VERSION
ARTIFACTS = (*[f"{split}.jsonl" for split in SPLITS], "evaluation_references.jsonl",
             "evaluation_references.tsv", "ctc_vocab.json")
SOURCE_DOCUMENTS = ("README", "metadata.tsv", "md5s.txt")


def load_inputs(raw, frozen):
    docs = {name: (raw / name).read_bytes() for name in SOURCE_DOCUMENTS}
    provenance = json.loads((frozen / "quality_summary.json").read_bytes())["source"]
    for name, key in zip(SOURCE_DOCUMENTS, ("readme_sha256", "metadata_sha256", "checksums_sha256")):
        require(sha256(docs[name]) == provenance[key], f"Original release document changed: {name}")
    metadata = read_metadata(docs["metadata.tsv"])
    checksums = read_checksums(docs["md5s.txt"])
    manifests = {split: list(read_jsonl(frozen / f"{split}.jsonl")) for split in SPLITS}
    check_manifests(manifests, frozen, metadata)
    return metadata, checksums, manifests


def raw_membership(metadata, manifests):
    """Add the first release utterance from each represented session, regardless of its quality."""
    membership = {}
    represented = set()
    for split, rows in manifests.items():
        for row in rows:
            path = row["audio_path"]
            meta = metadata.get(path)
            require(meta is not None and row["split"] == split and (split, row["id"], row["speaker_session_id"]) ==
                    (meta["status"], meta["id"], meta["speaker_id"]), f"Official identity differs: {path}")
            membership[path] = "frozen_v1"
            represented.add((split, meta["speaker_id"]))
    first = {}
    for path, row in metadata.items():
        session = (row["status"], row["speaker_id"])
        if session in represented and (session not in first or int(row["id"]) < int(metadata[first[session]]["id"])):
            first[session] = path
    for path in first.values():
        membership.setdefault(path, "first_session_utterance")
    return dict(sorted(membership.items()))


def write_release_file(dest, data):
    if dest.exists():
        require(dest.is_file() and dest.read_bytes() == data, f"Existing file differs: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as stream:
        stream.write(data)


def require_separate(*roots):
    normalized = [root.resolve() for root in roots]
    for left, right in combinations(normalized, 2):
        require(not left.is_relative_to(right) and not right.is_relative_to(left),
                f"Directories must be separate: {left} and {right}")


def stage_release(source, raw, frozen):
    require_separate(source, raw, frozen)
    metadata, checksums, manifests = load_inputs(raw, frozen)
    membership = raw_membership(metadata, manifests)
    for name in SOURCE_DOCUMENTS:
        require((source / name).read_bytes() == (raw / name).read_bytes(), f"Source document differs: {name}")
    for path in membership:
        data = (source / path).read_bytes()
        require(hashlib.md5(data, usedforsecurity=False).hexdigest() == checksums[path],
                f"Original FLAC checksum differs: {path}")
        write_release_file(raw / path, data)
    print(f"Staged {len(membership):,} checksum-matched original FLACs under {raw}", flush=True)


def split_integrity(selected):
    summaries = {}
    sessions = {}
    for split in SPLITS:
        rows = selected[split]
        sessions[split] = {row["session"] for row in rows}
        summaries[split] = {
            "speaker_session_ids": len(sessions[split]),
            "utterances": len(rows),
            "decoded_hours": sum(row["frames"] / row["audio"]["sample_rate"] for row in rows) / 3600,
        }
    pairs = {f"{left}/{right}": len(sessions[left] & sessions[right]) for left, right in combinations(SPLITS, 2)}
    require(not any(pairs.values()), f"Cross-split speaker/session overlap: {pairs}")
    overlaps = {}
    for field, name in (("raw_prompt", "exact_original_sentence"), ("reference", "normalized_scoring_text")):
        train = {row[field] for row in selected["train"]}
        matches = [row[field] for row in selected["test"] if row[field] in train]
        overlaps[name] = {"test_utterances": len(matches), "distinct_sentences": len(set(matches))}
    return {"dataset_version": VERSION, "splits": summaries,
            "pairwise_speaker_session_overlap": pairs, "test_in_train": overlaps}


def write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def clean_subset(raw, cleaned, frozen, output, workers):
    require_separate(raw, cleaned, frozen, output)
    metadata, checksums, manifests = load_inputs(raw, frozen)
    membership = raw_membership(metadata, manifests)
    available, _ = inventory(raw, set(membership))
    extras = available - membership.keys()
    require(not extras, f"Unexpected raw audio outside the subset: {sorted(extras)[:5]}")
    records = []
    with ThreadPoolExecutor(workers) as pool:
        for index, result in enumerate(pool.map(
            lambda path: inspect_file(raw, path, checksums[path], include_sha256=True), membership, buffersize=512
        ), 1):
            records.append(result)
            if index % 5000 == 0 or index == len(membership):
                print(f"Inspected {index:,} / {len(membership):,} raw subset FLACs", flush=True)
    by_path = {item["audio_path"]: item for item in make_items(records, metadata)}
    mark_duplicates(sorted(by_path.values(), key=lambda item: seeded_order(item, SEED)))
    selected = {split: [by_path[row["audio_path"]] for row in manifests[split]] for split in SPLITS}
    integrity = split_integrity(selected) if all(
        row["audio"]["state"] == "decoded" for rows in selected.values() for row in rows
    ) else None

    output.mkdir(parents=True, exist_ok=True)
    counts = {split: Counter() for split in SPLITS}
    inventory_rows = []
    removals = []
    for path, item in by_path.items():
        audio = item["audio"]
        role = membership[path]
        outcome = "removed_quality" if item["blocked"] else ("kept" if role == "frozen_v1" else "eligible_outside_v1")
        count = counts[item["split"]]
        count["raw_utterances"] += 1
        count["raw_bytes"] += audio.get("file_bytes", 0)
        count[outcome] += 1
        if outcome == "kept":
            count["cleaned_bytes"] += audio["file_bytes"]
        details = {
            "split": item["split"], "id": item["id"], "speaker_session_id": item["session"], "audio_path": path,
            "role": role, "outcome": outcome, "release_md5": audio["expected_md5"],
            "sha256": audio.get("file_sha256"), "bytes": audio.get("file_bytes"),
        }
        inventory_rows.append(details)
        if item["blocked"]:
            removals.append({**details, "reasons": "|".join(item["blocked"]), "state": audio["state"],
                             "rms_dbfs": audio.get("rms_dbfs"), "quiet_fraction": audio.get("quiet_fraction")})
    columns = ("split", "id", "speaker_session_id", "audio_path", "role", "outcome", "release_md5", "sha256", "bytes")
    write_tsv(output / "raw_subset_inventory.tsv", columns, inventory_rows)
    write_tsv(output / "raw_subset_removals.tsv", (*columns, "reasons", "state", "rms_dbfs", "quiet_fraction"), removals)
    incomplete = [row["audio_path"] for row in records if row.get("file_md5") != row["expected_md5"]]
    require(not incomplete, f"Missing or changed original FLAC bytes. Fetch or repair: {incomplete[:5]}")
    blocked_frozen = [(row["audio_path"], row["blocked"]) for group in selected.values() for row in group if row["blocked"]]
    require(not blocked_frozen, f"Frozen v1 clips failed cleaning: {blocked_frozen[:5]}")

    write_asr_artifacts(output, selected)
    for name in ARTIFACTS:
        require((output / name).read_bytes() == (frozen / name).read_bytes(), f"Frozen artifact changed: {name}")
    check_manifests({split: list(read_jsonl(output / f"{split}.jsonl")) for split in SPLITS}, output, metadata)
    if cleaned.exists():
        expected = {row["audio_path"] for group in selected.values() for row in group}
        extras = {path.relative_to(cleaned).as_posix() for path in cleaned.rglob("*.flac")} - expected
        require(not extras, f"Unexpected files in cleaned data: {sorted(extras)[:5]}")
    for group in selected.values():
        for row in group:
            path = row["audio_path"]
            contents = (raw / path).read_bytes()
            require(sha256(contents) == row["audio"]["file_sha256"], f"Raw audio changed during cleaning: {path}")
            write_release_file(cleaned / path, contents)
    write_json(output / "split_integrity.json", integrity)
    write_json(output / "raw_subset_cleaning.json", {
        "dataset_version": VERSION,
        "splits": {split: {key: counts[split][key] for key in (
            "raw_utterances", "raw_bytes", "kept", "cleaned_bytes", "removed_quality", "eligible_outside_v1"
        )} for split in SPLITS},
        "removal_reason_hits": dict(sorted(Counter(
            reason for row in removals for reason in row["reasons"].split("|")
        ).items())),
        "human_listening": "pending",
    })
    print(f"Rebuilt all {len(ARTIFACTS)} frozen artifacts and {sum(map(len, selected.values())):,} cleaned FLACs. "
          f"Applied quality exclusions to {len(removals)} raw subset inputs.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--frozen", type=Path, default=Path("output"))
    subcommands = parser.add_subparsers(dest="command", required=True)
    stage = subcommands.add_parser("stage", help="Copy the deterministic raw subset from an original release")
    stage.add_argument("--source-release", type=Path, required=True)
    clean = subcommands.add_parser("clean", help="Inspect raw FLACs and regenerate cleaned audio and text")
    clean.add_argument("--cleaned", type=Path, default=Path("reproduced/cleaned") / VERSION)
    clean.add_argument("--output", type=Path, default=Path("reproduced/subset"))
    clean.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.command == "stage":
        stage_release(args.source_release, args.raw, args.frozen)
    else:
        clean_subset(args.raw, args.cleaned, args.frozen, args.output, args.workers)
