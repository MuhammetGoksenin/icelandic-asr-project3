from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from audit import AUDIT_VERSION, QUIET_DBFS, SPLITS, audit_path, json_line, read_metadata, sha256

SEED = 2105
TARGET_HOURS = {"train": 15, "dev": 2, "test": 3}
PRIORITY = ("test", "dev", "train")
OVERLAP_FIELDS = ("session", "raw_prompt", "source_normalized", "reference", "pcm", "file_md5")
RATE = 16_000
MIN_SECONDS = 0.5
MAX_SECONDS = 30
MISSING_TEXT = {"", "nan", "none", "null", "n/a"}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text.lower())
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    return " ".join(text.split())


def reference_text(row):
    return normalize_text(row["sentence_norm"])


def seeded_order(item, seed):
    return hashlib.sha256(f"{seed}/{item['audio_path']}".encode("utf-8")).digest(), item["audio_path"]


def read_jsonl(path):
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            yield json.loads(line)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json_line(record))


def write_tsv(path, columns, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def flags_for(audio, row):
    blocked = []
    flags = []
    if row is None:
        blocked.append("no_metadata")
    else:
        if row["sentence"].strip().lower() in MISSING_TEXT:
            blocked.append("missing_raw_transcript")
        if row["sentence_norm"].strip().lower() in MISSING_TEXT or not reference_text(row):
            blocked.append("missing_normalized_transcript")
        if row["is_valid"] != "1.0":
            blocked.append("metadata_not_validated")
        if row["empty"] != "0":
            blocked.append("metadata_empty_or_unknown")
    if audio["state"] != "decoded":
        blocked.append(audio["state"])
    if audio["expected_md5"] is None:
        flags.append("no_release_checksum")
    elif audio.get("file_md5") and audio["file_md5"] != audio["expected_md5"]:
        blocked.append("release_checksum_mismatch")
    if audio["state"] != "decoded":
        return blocked, flags
    rate = audio["sample_rate"]
    duration = audio["frames"] / rate
    if (rate, audio["channels"], audio["format"], audio["subtype"]) != (RATE, 1, "FLAC", "PCM_16"):
        blocked.append("unexpected_audio_format")
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        blocked.append("outside_0.5_to_30_seconds")
    if row is not None:
        try:
            reported_duration = float(row["duration"])
        except ValueError:
            reported_duration = math.nan
        if (row["sample_rate"] != str(rate) or not math.isfinite(reported_duration)
                or abs(reported_duration - duration) > 0.05):
            flags.append("audio_metadata_disagreement")
    if duration == 0 or audio["peak_dbfs"] is None:
        blocked.append("digital_silence")
    elif audio["rms_dbfs"] <= -55 or audio["quiet_fraction"] >= 0.98:
        blocked.append("near_silence")
    else:
        if audio["rms_dbfs"] <= -35:
            flags.append("low_rms")
        if audio["quiet_fraction"] >= 0.75:
            flags.append("mostly_quiet")
        if max(audio["quiet_leading_sec"], audio["quiet_trailing_sec"]) >= 1.5:
            flags.append("long_quiet_edge")
    if audio.get("full_scale_fraction", 0) >= 0.001:
        flags.append("possible_clipping")
    return blocked, flags


def overlap(groups):
    result = {}
    for left, right in combinations(SPLITS, 2):
        a, b = groups[left], groups[right]
        pair = {}
        for field in OVERLAP_FIELDS:
            values_a = Counter(item[field] for item in a if item.get(field))
            values_b = Counter(item[field] for item in b if item.get(field))
            common = values_a.keys() & values_b.keys()
            pair[field] = {
                "shared": len(common),
                left: sum(values_a[key] for key in common),
                right: sum(values_b[key] for key in common),
            }
        result[f"{left}/{right}"] = pair
    return result


def prompt_overlap_rows(groups):
    by_split = {}
    for split, items in groups.items():
        by_reference = defaultdict(list)
        for item in items:
            if item["reference"]:
                by_reference[item["reference"]].append(item)
        by_split[split] = by_reference
    for left, right in combinations(SPLITS, 2):
        shared = by_split[left].keys() & by_split[right].keys()
        for reference in sorted(shared):
            for split in (left, right):
                for item in by_split[split][reference]:
                    yield {
                        "pair": f"{left}/{right}", "split": split, "id": item["id"],
                        "audio_path": item["audio_path"], "selected": item["selected"], "reference": reference,
                        "text_raw": item["raw_prompt"], "text_source_normalized": item["source_normalized"],
                    }


def nearest_prefix(items, target):
    count = 0
    total = 0
    for item in items:
        if total + item["frames"] >= target:
            if target - total >= total + item["frames"] - target:
                count += 1
            return items[:count]
        total += item["frames"]
        count += 1
    raise ValueError(f"Only {total / RATE / 3600:.3f} eligible hours for target {target / RATE / 3600:.3f}")


def select(items, seed):
    candidates = sorted(items, key=lambda item: seeded_order(item, seed))
    hashes = defaultdict(list)
    for item in candidates:
        if item["pcm"]:
            hashes[item["pcm"]].append(item)

    for group in hashes.values():
        labels = {item["reference"] for item in group if item["reference"]}
        if len(labels) > 1:
            for item in group:
                item["blocked"].append("duplicate_pcm_conflicting_reference")
            continue
        usable = [item for item in group if not item["blocked"]]
        if usable:
            # Stable evaluation-first ownership for exact duplicate recordings.
            keep = min(usable, key=lambda item: PRIORITY.index(item["split"]))
            for item in usable:
                if item is not keep:
                    item["blocked"].append("duplicate_pcm")
                    item["related_audio_path"] = keep["audio_path"]

    selected = {}
    previous = {field: {} for field in ("session", "raw_prompt", "reference")}
    for split in PRIORITY:
        choices = []
        for item in candidates:
            if item["split"] != split or item["blocked"]:
                continue
            for field, previous_paths in previous.items():
                if item[field] in previous_paths:
                    item["blocked"].append(f"shared_{field}_with_selected_evaluation")
                    item.setdefault("related_audio_path", previous_paths[item[field]])
            if not item["blocked"]:
                choices.append(item)
        try:
            selected[split] = nearest_prefix(choices, TARGET_HOURS[split] * 3600 * RATE)
        except ValueError as exc:
            raise ValueError(f"{split}: {exc}") from exc
        for item in selected[split]:
            item["selected"] = True
            for field, previous_paths in previous.items():
                previous_paths.setdefault(item[field], item["audio_path"])
    return selected


def make_items(audio_records, metadata):
    items = []
    for audio in audio_records:
        path = audio["audio_path"]
        row = metadata.get(path)
        blocked, flags = flags_for(audio, row)
        items.append({
            "audio_path": path,
            "split": audio["split"],
            "id": row["id"] if row else None,
            "session": row["speaker_id"] if row else None,
            "raw_prompt": row["sentence"] if row else "",
            "source_normalized": row["sentence_norm"] if row else "",
            "reference": reference_text(row) if row else "",
            "pcm": audio.get("pcm_sha256"),
            "file_md5": audio.get("file_md5"),
            "frames": audio.get("frames"),
            "blocked": blocked,
            "flags": flags,
            "selected": False,
            "audio": audio,
        })
    return items


def manifest_record(item):
    return {
        "id": item["id"],
        "split": item["split"],
        "speaker_session_id": item["session"],
        "audio_path": item["audio_path"],
        "sample_rate": RATE,
        "num_samples": item["frames"],
        "duration_sec": item["frames"] / RATE,
        "text_raw": item["raw_prompt"],
        "text_source_normalized": item["source_normalized"],
        "text": item["reference"],
        "pcm_sha256": item["pcm"],
        "flags": item["flags"],
    }


def write_focused_review(path, items):
    identity = ("split", "id", "audio_path")
    metrics = ("peak_dbfs", "rms_dbfs", "quiet_fraction", "quiet_leading_sec", "quiet_trailing_sec", "full_scale_fraction")
    records = []
    for item in items:
        audio = item["audio"]
        if item["selected"] and "possible_clipping" in item["flags"]:
            trigger = "full_scale_fraction >= 0.001"
            category = "selected_possible_clipping"
        elif "near_silence" in item["blocked"]:
            tests = []
            if audio["rms_dbfs"] <= -55:
                tests.append("rms_dbfs <= -55")
            if audio["quiet_fraction"] >= 0.98:
                tests.append("quiet_fraction >= 0.98")
            trigger = " | ".join(tests)
            category = "excluded_near_silence"
        else:
            continue
        records.append({
            **{field: item[field] for field in identity},
            "category": category,
            "trigger": trigger,
            "exclusion_reasons": "|".join(item["blocked"]),
            "review_flags": "|".join(item["flags"]),
            "duration_sec": audio["frames"] / audio["sample_rate"],
            **{key: audio[key] for key in metrics},
            "listening_status": "pending",
        })
    records.sort(key=lambda row: (row["category"] != "selected_possible_clipping",
                                  PRIORITY.index(row["split"]), row["audio_path"]))
    write_tsv(path, (*identity, "category", "trigger", "exclusion_reasons", "review_flags", "duration_sec",
                     *metrics, "listening_status"), records)


def aggregate_duplicates(values):
    groups = Counter(value for value in values if value)
    return {
        "groups": sum(count > 1 for count in groups.values()),
        "clips": sum(count for count in groups.values() if count > 1),
    }


def summarize(groups, selected, header, seed):
    result = {
        "seed": seed,
        "source": header,
        "official_overlap": overlap(groups),
        "selected_overlap": overlap(selected),
        "splits": {},
    }
    for split in SPLITS:
        items = groups[split]
        chosen = selected[split]
        decoded = [item["audio"] for item in items if item["audio"]["state"] == "decoded"]
        durations = [a["frames"] / a["sample_rate"] for a in decoded]
        result["splits"][split] = {
            "metadata_clips": sum(item["id"] is not None for item in items),
            "session_ids": len({item["session"] for item in items if item["session"]}),
            "audio_without_metadata": sum(item["id"] is None and item["audio"]["state"] != "missing" for item in items),
            "missing_raw_transcripts": sum("missing_raw_transcript" in item["blocked"] for item in items),
            "missing_normalized_transcripts": sum("missing_normalized_transcript" in item["blocked"] for item in items),
            "audio_states": dict(Counter(item["audio"]["state"] for item in items)),
            "checked_md5_matches": sum(bool(a.get("expected_md5")) and a.get("file_md5") == a["expected_md5"]
                                      for a in (item["audio"] for item in items)),
            "decoded_hours": sum(durations) / 3600,
            "duration_percentiles_seconds": dict(zip(
                ["min", "p05", "median", "p95", "max"], map(float, np.percentile(durations, [0, 5, 50, 95, 100]))
            )) if durations else {},
            "sample_rates": dict(Counter(a["sample_rate"] for a in decoded)),
            "channels": dict(Counter(a["channels"] for a in decoded)),
            "formats": dict(Counter(f"{a['format']}/{a['subtype']}" for a in decoded)),
            "pcm_duplicates": aggregate_duplicates(item["pcm"] for item in items),
            "file_duplicates": aggregate_duplicates(item["file_md5"] for item in items),
            "exclusions": dict(sorted(Counter(reason for item in items for reason in item["blocked"]).items())),
            "flags": dict(sorted(Counter(flag for item in items for flag in item["flags"]).items())),
            "excluded_paths": sum(bool(item["blocked"]) for item in items),
            "eligible_unsampled_paths": sum(not item["blocked"] and not item["selected"] for item in items),
            "selected_clips": len(chosen),
            "selected_hours": sum(item["frames"] for item in chosen) / RATE / 3600,
            "target_delta_seconds": (sum(item["frames"] for item in chosen) / RATE) - 3600 * TARGET_HOURS[split],
            "selected_sessions": len({item["session"] for item in chosen}),
            "selected_unique_references": len({item["reference"] for item in chosen}),
            "selected_flagged_clips": sum(bool(item["flags"]) for item in chosen),
            "selected_flags": dict(sorted(Counter(flag for item in chosen for flag in item["flags"]).items())),
        }
    all_items = [item for split in SPLITS for item in groups[split]]
    result["pcm_duplicates_total"] = aggregate_duplicates(item["pcm"] for item in all_items)
    result["file_duplicates_total"] = aggregate_duplicates(item["file_md5"] for item in all_items)
    return result


def write_report(path, summary):
    excluded = sum(split["excluded_paths"] for split in summary["splits"].values())
    near_silence = sum(split["exclusions"].get("near_silence", 0) for split in summary["splits"].values())
    clipping = sum(split["selected_flags"].get("possible_clipping", 0) for split in summary["splits"].values())
    lines = [
        "# Samrómur 21.05 quality report", "",
        f"Selection seed **{summary['seed']}**. The shipped README and UTF-8 metadata were read.",
        "Audio was inventoried under the three official partitions and decoded end to end with "
        f"libsndfile {summary['source']['libsndfile']}. The release checksum list was checked using relative paths.",
        "", "## Full corpus inventory", "",
        "| Split | Metadata clips | Decoded clips | Missing audio | Read errors | Decode failures | "
        "Audio without metadata | Missing raw / normalized text | Decoded hours | Session IDs | Matching release MD5s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in summary["splits"].items():
        counts = stats["audio_states"]
        lines.append(
            f"| {split} | {stats['metadata_clips']:,} | {counts.get('decoded', 0):,} | {counts.get('missing', 0):,} | "
            f"{counts.get('read_error', 0):,} | {counts.get('decode_error', 0):,} | {stats['audio_without_metadata']:,} | "
            f"{stats['missing_raw_transcripts']} / {stats['missing_normalized_transcripts']} | "
            f"{stats['decoded_hours']:.4f} | {stats['session_ids']:,} | {stats['checked_md5_matches']:,} |"
        )
    lines += ["", "| Split | Sample rate Hz (clips) | Channels (clips) | Format (clips) | "
              "Duration min / p05 / median / p95 / max (s) | PCM duplicate groups / clips within split |",
              "|---|---|---|---|---|---:|"]
    for split, stats in summary["splits"].items():
        format_counts = lambda counts: ", ".join(f"{key} ({value:,})" for key, value in counts.items())
        durations = " / ".join(f"{value:.3f}" for value in stats["duration_percentiles_seconds"].values())
        dup = stats["pcm_duplicates"]
        lines.append(f"| {split} | {format_counts(stats['sample_rates'])} | {format_counts(stats['channels'])} | "
                     f"{format_counts(stats['formats'])} | {durations} | {dup['groups']} / {dup['clips']} |")
    pcm, files = summary["pcm_duplicates_total"], summary["file_duplicates_total"]
    docs = summary["source"]["documentation_md5"]
    lines += [
        "",
        f"Corpus-wide exact PCM duplicates comprised **{pcm['groups']} groups / {pcm['clips']} clips**. "
        f"Identical file bytes (MD5) comprised **{files['groups']} groups / {files['clips']} clips**.",
        "The PCM hash covers sample rate, channel count and all decoded samples. The comparison covers exact "
        "matches. Edited, time-shifted or re-recorded audio may require listening or a separate similarity check.",
        f"Release documentation checksums matched for "
        f"**{sum(d['expected'] == d['actual'] for d in docs.values())}/{len(docs)}** files.",
        "", "## Split overlap", "",
        "The release uses the session ID as the speaker ID. Anonymous IDs can establish session separation. "
        "The same person could record in more than one session. The raw prompt comparison uses exact metadata "
        "strings, including a separate exact comparison of the release-normalized column. The reference "
        "comparison uses the shared text normalization described in the README.",
    ]
    for label, key in [("Official splits", "official_overlap"), ("Selected subsets", "selected_overlap")]:
        lines += ["", f"### {label}", "",
                  "| Pair | Shared session IDs | Shared raw prompts | Shared release-normalized prompts | Shared scoring references | "
                  "Shared decoded audio hashes | Shared file MD5s |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for pair, stats in summary[key].items():
            lines.append(f"| {pair} | " + " | ".join(str(stats[field]["shared"]) for field in OVERLAP_FIELDS) + " |")
    lines += [
        "", "The [JSON summary](quality_summary.json) gives both the number of shared keys and the number of "
        "affected utterances on each side of each split pair. [prompt_overlap.tsv](prompt_overlap.tsv) shows "
        "the corresponding prompts and whether each clip was selected.",
        "", "## Selected subsets", "",
        "| Split | Utterances | Hours | Target difference (s) | Session IDs | Unique references | Flagged clips | "
        "Excluded paths | Eligible but unsampled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, s in summary["splits"].items():
        lines.append(f"| {split} | {s['selected_clips']:,} | {s['selected_hours']:.5f} | "
                     f"{s['target_delta_seconds']:+.3f} | {s['selected_sessions']:,} | "
                     f"{s['selected_unique_references']:,} | {s['selected_flagged_clips']:,} | "
                     f"{s['excluded_paths']:,} | {s['eligible_unsampled_paths']:,} |")
    for label, key in [("Excluded paths by reason", "exclusions"), ("All decoded or inventoried paths with flags", "flags"),
                       ("Selected paths with flags", "selected_flags")]:
        lines += ["", f"### {label}", "", "| Reason | Train | Dev | Test |", "|---|---:|---:|---:|"]
        reasons = sorted({reason for s in summary["splits"].values() for reason in s[key]})
        for reason in reasons:
            lines.append(f"| `{reason}` | " + " | ".join(
                str(summary["splits"][split][key].get(reason, 0)) for split in SPLITS) + " |")
        if not reasons:
            lines.append("| None | 0 | 0 | 0 |")
    lines += [
        "", f"The exclusions cover **{excluded} unique paths**. An utterance can have multiple reasons or flags. "
        "See [exclusions.tsv](exclusions.tsv) for "
        "withheld paths and [listening_queue.tsv](listening_queue.tsv) for reviewable paths and controls.",
        "", "## Human listening", "",
        f"Signal rules used 20 ms windows below {QUIET_DBFS} dBFS to measure quiet time. "
        "They can flag quiet speech, pauses, noise or full-scale samples without establishing what was spoken. "
        "Human listening remains pending for intelligibility, prompt agreement, background voices or noise, "
        "audible clipping, cut-off speech, and questionable silence decisions.",
        f"The [focused review queue](focused_review.tsv) lists {near_silence} near-silence exclusions and "
        f"{clipping} selected possible-clipping flags with measurements. Thresholds are screening decisions. "
        "Low-volume speech may still be usable, and full-scale samples may be inaudible.",
        "Review the selected flagged evaluation clips first, then selected flagged train clips and the "
        "unflagged controls. Keep the same frozen dev and test IDs for both model pipelines.",
        "", "Source and license are documented in the [project README](../README.md).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build(root, output, seed):
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Use an output directory separate from the corpus directory")
    metadata_bytes = (root / "metadata.tsv").read_bytes()
    metadata = read_metadata(metadata_bytes)
    records = read_jsonl(audit_path(output))
    header = next(records)
    if (header["audit_version"] != AUDIT_VERSION or header["metadata_sha256"] != sha256(metadata_bytes)
            or header["readme_sha256"] != sha256((root / "README").read_bytes())
            or header["checksums_sha256"] != sha256((root / "md5s.txt").read_bytes())):
        raise ValueError("Audit version or corpus metadata changed. Run audit.py again.")
    items = make_items(records, metadata)
    paths = {item["audio_path"] for item in items}
    if len(paths) != len(items) or not metadata.keys() <= paths:
        raise ValueError("Audit paths are incomplete or repeated")
    groups = {split: [item for item in items if item["split"] == split] for split in SPLITS}
    selected = select(items, seed)
    summary = summarize(groups, selected, header, seed)
    if any(metric["shared"] for pair in summary["selected_overlap"].values() for metric in pair.values()):
        raise ValueError("Unexpected selected overlap")
    for split in SPLITS:
        write_jsonl(output / f"{split}.jsonl", map(manifest_record, selected[split]))
    references = [{key: record[key] for key in ("id", "split", "text", "text_raw")}
                  for split in ("dev", "test") for record in map(manifest_record, selected[split])]
    write_jsonl(output / "evaluation_references.jsonl", references)
    write_tsv(output / "evaluation_references.tsv", ("id", "split", "text", "text_raw"), references)
    write_tsv(output / "prompt_overlap.tsv", ("pair", "split", "id", "audio_path", "selected", "reference",
              "text_raw", "text_source_normalized"), prompt_overlap_rows(groups))

    identity = ("split", "id", "audio_path")
    write_tsv(output / "exclusions.tsv", (*identity, "reasons", "related_audio_path", "review_flags"), (
        {**{key: item[key] for key in identity}, "reasons": "|".join(item["blocked"]),
         "related_audio_path": item.get("related_audio_path", ""), "review_flags": "|".join(item["flags"])}
        for item in items if item["blocked"]
    ))
    write_focused_review(output / "focused_review.tsv", items)
    controls = {split: {item["audio_path"] for item in
                        sorted((x for x in selected[split] if not x["flags"]),
                               key=lambda item: seeded_order(item, seed + 1))[:20]} for split in SPLITS}
    review_metrics = ("rms_dbfs", "quiet_fraction", "quiet_leading_sec", "quiet_trailing_sec", "full_scale_fraction")
    review = []
    for item in items:
        reasons = item["flags"] + [r for r in item["blocked"] if r in {
            "near_silence", "digital_silence", "outside_0.5_to_30_seconds", "duplicate_pcm_conflicting_reference"}]
        if item["audio_path"] in controls[item["split"]]:
            reasons.append("unflagged_control")
        if reasons and item["audio"]["state"] == "decoded":
            review.append({**{key: item[key] for key in identity}, "selected": item["selected"],
                           "review_reasons": "|".join(reasons), "text": item["reference"],
                           **{key: item["audio"].get(key) for key in review_metrics}})
    review.sort(key=lambda row: (not row["selected"], row["review_reasons"] == "unflagged_control",
                                 PRIORITY.index(row["split"]), row["audio_path"]))
    write_tsv(output / "listening_queue.tsv", (*identity, "selected", "review_reasons", "text", *review_metrics), review)
    characters = sorted(set("".join(item["reference"] for item in selected["train"])) - {" "})
    vocab = {token: index for index, token in enumerate(["[PAD]", "[UNK]", "|", *characters])}
    (output / "ctc_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for split in ("dev", "test"):
        unknown = Counter(char for item in selected[split] for char in item["reference"] if char != " " and char not in vocab)
        summary["splits"][split]["ctc_oov_characters"] = dict(sorted(unknown.items()))
    (output / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    write_report(output / "quality_report.md", summary)
    for split, stats in summary["splits"].items():
        print(f"{split}: {stats['selected_clips']:,} clips, {stats['selected_hours']:.5f} h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build seeded Samrómur subsets and quality artifacts")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(args.root, args.output, args.seed)
