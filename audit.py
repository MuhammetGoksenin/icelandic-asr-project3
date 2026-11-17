from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

SPLITS = ("train", "dev", "test")
AUDIT_VERSION = 1
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif", ".wma", ".au", ".caf", ".mp4", ".webm"}
QUIET_DBFS = -50


def json_line(record):
    return json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_metadata(metadata):
    with io.StringIO(metadata.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        required = {"id", "speaker_id", "filename", "status", "sentence", "sentence_norm", "duration",
                    "sample_rate", "is_valid", "empty"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Missing required metadata columns")
        rows = {}
        ids = set()
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Ragged TSV row {reader.line_num}")
            split, speaker, uid = row["status"], row["speaker_id"], row["id"]
            if split not in SPLITS or not re.fullmatch(r"[0-9]+", speaker) or not re.fullmatch(r"[0-9]+", uid):
                raise ValueError(f"Unexpected metadata identity on line {reader.line_num}")
            if row["filename"] != f"{speaker}-{uid}.flac":
                raise ValueError(f"Unexpected audio filename on line {reader.line_num}")
            path = f"{split}/{speaker}/{row['filename']}"
            if uid in ids:
                raise ValueError(f"Repeated utterance ID {uid}")
            ids.add(uid)
            rows[path] = row
    return rows


def read_checksums(data):
    checksums = {}
    for line in data.decode("utf-8-sig").splitlines():
        checksum, path = line.split(maxsplit=1)
        path = path.removeprefix("r1/").removeprefix("./")
        if not re.fullmatch(r"[0-9a-f]{32}", checksum) or path in checksums:
            raise ValueError(f"Invalid checksum entry for {path}")
        checksums[path] = checksum
    return checksums


def inventory(root, metadata_paths):
    for split in SPLITS:
        if not (root / split).is_dir():
            raise FileNotFoundError(root / split)
    files = {p.relative_to(root).as_posix() for split in SPLITS
             for p in (root / split).rglob("*") if p.is_file()}
    audio = {p for p in files if PurePosixPath(p).suffix.lower() in AUDIO_EXTENSIONS}
    return (files & metadata_paths) | audio, sorted(files - metadata_paths - audio)


def pcm_hash(samples, rate):
    channels = samples.shape[1]
    digest = hashlib.sha256(f"{rate}/{channels}\n".encode("ascii"))
    digest.update(memoryview(samples.astype("<i4", copy=False)).cast("B"))
    return digest.hexdigest()


def audio_metrics(samples, rate):
    frames, channels = samples.shape
    if not frames:
        return {}
    scale = 2**31
    squared = np.square(samples.astype(np.float64) / scale)
    rms = math.sqrt(float(squared.mean()))
    peak = max(int(samples.max()), -int(samples.min())) / scale
    # Include the final partial window and weight its contribution by its real length.
    window = max(1, rate // 50)
    starts = np.arange(0, frames, window)
    lengths = np.diff(starts, append=frames)
    energy = np.add.reduceat(squared.sum(axis=1), starts) / (lengths * channels)
    quiet = energy < 10 ** (QUIET_DBFS / 10)
    full_scale = (samples == np.iinfo(np.int32).min) | (samples >= scale - 2**16)
    return {
        "pcm_sha256": pcm_hash(samples, rate),
        "peak_dbfs": round(20 * math.log10(peak), 4) if peak else None,
        "rms_dbfs": round(20 * math.log10(rms), 4) if rms else None,
        "quiet_fraction": round(float(lengths[quiet].sum() / frames), 6),
        "quiet_leading_sec": round(float(np.cumsum(lengths)[np.argmax(~quiet) - 1] / rate), 6)
        if quiet[0] and not quiet.all() else (frames / rate if quiet.all() else 0),
        "quiet_trailing_sec": round(float(lengths[len(quiet) - np.argmax((~quiet)[::-1]):].sum() / rate), 6)
        if quiet[-1] and not quiet.all() else (frames / rate if quiet.all() else 0),
        "full_scale_fraction": round(float(full_scale.mean()), 6),
    }


def inspect_file(root, path, expected_md5):
    result = {"audio_path": path, "split": path.split("/")[0], "expected_md5": expected_md5}
    try:
        contents = (root / path).read_bytes()
    except FileNotFoundError:
        return result | {"state": "missing"}
    except OSError as exc:
        return result | {"state": "read_error", "error": exc.strerror or type(exc).__name__}
    result["file_md5"] = hashlib.md5(contents, usedforsecurity=False).hexdigest()
    result["file_bytes"] = len(contents)
    try:
        with sf.SoundFile(io.BytesIO(contents)) as audio:
            result.update(sample_rate=audio.samplerate, channels=audio.channels, format=audio.format,
                          subtype=audio.subtype, header_frames=audio.frames)
            samples = audio.read(dtype="int32", always_2d=True)
            result["frames"] = len(samples)
            if len(samples) != audio.frames or len(audio.read(1, dtype="int32")):
                return result | {"state": "decode_error", "error": "Decoded frame count differs from header"}
    except (sf.SoundFileError, ValueError, OSError) as exc:
        return result | {"state": "decode_error", "error": str(exc)}
    return result | {"state": "decoded"} | audio_metrics(samples, result["sample_rate"])


def run_audit(root, output, workers):
    if output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()):
        raise ValueError("Use an output directory separate from the corpus directory")
    readme = (root / "README").read_bytes()
    metadata = (root / "metadata.tsv").read_bytes()
    checksum_bytes = (root / "md5s.txt").read_bytes()
    rows = read_metadata(metadata)
    checksums = read_checksums(checksum_bytes)
    actual, other_files = inventory(root, rows.keys())
    paths = sorted(actual | rows.keys() | {p for p in checksums if p.split("/")[0] in SPLITS})
    header = {
        "audit_version": AUDIT_VERSION,
        "metadata_sha256": sha256(metadata),
        "readme_sha256": sha256(readme),
        "checksums_sha256": sha256(checksum_bytes),
        "documentation_md5": {name: {
            "expected": checksums.get(name),
            "actual": hashlib.md5(data, usedforsecurity=False).hexdigest(),
        } for name, data in {"README": readme, "metadata.tsv": metadata}.items()},
        "metadata_rows": len(rows),
        "audio_paths_found": len(actual),
        "other_split_files": other_files,
        "quiet_dbfs": QUIET_DBFS,
        "libsndfile": sf.__libsndfile_version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "audio_audit.jsonl.partial"
    with partial.open("w", encoding="utf-8", newline="\n") as stream, ThreadPoolExecutor(workers) as pool:
        stream.write(json_line(header))
        for start in range(0, len(paths), 512):
            work = paths[start:start + 512]
            results = pool.map(lambda path: inspect_file(root, path, checksums.get(path)), work)
            for result in results:
                stream.write(json_line(result))
            count = start + len(work)
            if count % 5120 == 0 or count == len(paths):
                print(f"Audited {count:,} / {len(paths):,} audio paths", flush=True)
    partial.replace(output / "audio_audit.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory and fully decode Samrómur 21.05 audio")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    run_audit(args.root, args.output, args.workers)
