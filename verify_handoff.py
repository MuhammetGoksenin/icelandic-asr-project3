from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from pathlib import Path

import soundfile as sf
from transformers import __version__ as transformers_version

from audit import SPLITS, audit_path, sha256
from prepare import read_jsonl
from preprocess import WHISPER_MODEL, WHISPER_REVISION, whisper_components
from verify import GENERATED, check_manifests, check_model_inputs, check_selected_audio, require

REVIEW_COLUMNS = (
    "split", "id", "audio_path", "flags", "text_raw", "text_scoring",
    "audio_quality_status", "audio_quality_reviewer", "audio_quality_notes",
    "transcript_status", "transcript_reviewer", "transcript_notes",
)


def review_records(splits):
    for split in ("test", "dev"):
        for row in splits[split]:
            if row["flags"]:
                yield {
                    "split": split, "id": row["id"], "audio_path": f"audio/{row['audio_path']}",
                    "flags": "|".join(row["flags"]), "text_raw": row["text_raw"], "text_scoring": row["text"],
                    "audio_quality_status": "pending", "audio_quality_reviewer": "", "audio_quality_notes": "",
                    "transcript_status": "pending", "transcript_reviewer": "", "transcript_notes": "",
                }


def verify_package(package, report_path):
    audio_root = package / "audio"
    output = package / "output"
    assets = package / "assets" / "whisper"
    missing = [name for name in ("audio/train", "audio/dev", "audio/test", "assets/whisper")
               if not (package / name).is_dir()]
    if missing or not (assets / "sha256.json").is_file():
        raise ValueError("Use --package-root with the extracted handoff containing "
                         "audio/{train,dev,test}, output/, and assets/whisper/sha256.json.")
    splits = {split: list(read_jsonl(output / f"{split}.jsonl")) for split in SPLITS}
    baseline = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    require(baseline["status"] == "passed", "Source verification did not pass")
    digests = baseline["reproduced_artifacts_sha256"]
    require(set(digests) == set(GENERATED), "Source verification digest list differs")
    for name, expected in digests.items():
        require(sha256((output / name).read_bytes()) == expected, f"Frozen artifact differs: {name}")
    reference_count = check_manifests(splits, output)
    selected_paths = {row["audio_path"] for rows in splits.values() for row in rows}
    package_paths = {path.relative_to(audio_root).as_posix() for path in audio_root.rglob("*") if path.is_file()}
    require(selected_paths == package_paths, "Packaged audio paths differ from manifest selection")
    audit = read_jsonl(audit_path(output))
    next(audit)  # The first line holds release provenance.
    expected_md5s = {}
    for row in audit:
        if row["audio_path"] in selected_paths:
            require(row["file_md5"] and row["file_md5"] == row["expected_md5"],
                    f"Missing verified source MD5: {row['audio_path']}")
            expected_md5s[row["audio_path"]] = row["expected_md5"]
    require(expected_md5s.keys() == selected_paths, "Packaged source audit omits selected audio")
    check_selected_audio(audio_root, splits, expected_md5s)

    expected_review = list(review_records(splits))
    with (output / "flagged_dev_test_review.tsv").open(encoding="utf-8", newline="") as stream:
        actual_review = list(csv.DictReader(stream, delimiter="\t"))
    identity_columns = REVIEW_COLUMNS[:6]
    require([{key: row[key] for key in identity_columns} for row in actual_review]
            == [{key: row[key] for key in identity_columns} for row in expected_review],
            "Evaluation review queue differs from the selected flagged clips")
    review_counts = Counter(row["split"] for row in expected_review)
    summary = json.loads((output / "quality_summary.json").read_text(encoding="utf-8"))
    require(all(review_counts[split] == summary["splits"][split]["selected_flagged_clips"] for split in ("dev", "test")),
            "Evaluation review counts differ from quality report")

    asset_hashes = json.loads((assets / "sha256.json").read_text(encoding="utf-8"))
    for name, expected in asset_hashes.items():
        require(Path(name).name == name and sha256((assets / name).read_bytes()) == expected,
                f"Bundled Whisper asset differs: {name}")
    components = whisper_components(assets, local_files_only=True)
    examples = check_model_inputs(audio_root, splits, output, summary["seed"], *components)
    report = {
        "status": "passed",
        "dataset_input": "Packaged audio/ and output/ only",
        "whisper_input": "Bundled assets/whisper/ in local-files-only mode",
        "whisper_source": {"checkpoint": WHISPER_MODEL, "revision": WHISPER_REVISION},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libsndfile": sf.__libsndfile_version__,
        "transformers": transformers_version,
        "selected_audio_path_md5_and_pcm_checks": {split: len(rows) for split, rows in splits.items()},
        "label_checks": {split: len(rows) for split, rows in splits.items()},
        "shared_reference_rows_verified": reference_count,
        "unchanged_artifact_sha256": digests,
        "feature_smoke_test_ids": examples,
        "flagged_eval_review_rows": dict(review_counts),
        "human_listening": "pending",
    }
    if report_path:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Portable handoff verification passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify this extracted handoff without the full Samrómur release")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory extracted from the matching handoff ZIP")
    parser.add_argument("--report", type=Path, help="Optionally write a JSON record of this run")
    args = parser.parse_args()
    try:
        verify_package(args.package_root.resolve(), args.report)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Handoff verification failed: {exc}\n")
