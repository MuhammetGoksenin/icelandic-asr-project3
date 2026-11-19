"""Run published pretrained ASR checkpoints on the frozen v1 dev/test manifests.

These are baselines: no training by our group. Each run writes
  results/pretrained/predictions/<model_key>_<split>.csv
  results/pretrained/model_info/<model_key>_<split>.json

Usage (from the repository root):
  python baselines/run_pretrained.py --model whisper_small --split dev --num-beams 1 5
  python baselines/run_pretrained.py --model whisper_small --split test --num-beams 5
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare import normalize_text, read_jsonl  # noqa: E402

AUDIO_ROOT = ROOT / "data" / "cleaned" / "samromur-21.05-20h-v1"
MANIFESTS = ROOT / "output"
OUT = ROOT / "results" / "pretrained"
LANGUAGE = "is"


@dataclass(frozen=True)
class Checkpoint:
    key: str
    repo_id: str
    family: str  # "whisper" or "ctc"
    note: str


# Revisions are resolved at run time and written to model_info. After the
# first run, copy the resolved hash here so later runs load the same weights.
CHECKPOINTS = {
    "whisper_small": Checkpoint(
        "whisper_small", "openai/whisper-small", "whisper",
        "Multilingual Whisper small, zero-shot Icelandic (language='is').",
    ),
    "wav2vec2_is": Checkpoint(
        "wav2vec2_is", "language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h", "ctc",
        "wav2vec 2.0 XLSR-53 + CTC head, fine-tuned by LVL on 967 h of Samromur Milljon. "
        "Possible overlap with our test clips; see README.",
    ),
    "whisper_small_is": Checkpoint(
        "whisper_small_is", "Valdimarb13/whisper-small-icelandic", "whisper",
        "Whisper small fine-tuned on language-and-voice-lab/samromur_asr by a third party.",
    ),
}
REVISIONS: dict[str, str | None] = {key: None for key in CHECKPOINTS}

CSV_FIELDS = ("id", "split", "speaker_session_id", "duration_sec",
              "reference", "reference_raw", "hypothesis_raw", "hypothesis")


def load_manifest(split: str) -> list[dict]:
    return list(read_jsonl(MANIFESTS / f"{split}.jsonl"))


def batches(records: list[dict], batch_size: int):
    """Sort by duration so batches pad little; output order is restored later."""
    ordered = sorted(records, key=lambda r: r["duration_sec"], reverse=True)
    for start in range(0, len(ordered), batch_size):
        yield ordered[start:start + batch_size]


def prediction_rows(records: list[dict], hypotheses: dict[str, str]) -> list[dict]:
    """Join hypotheses to manifest rows by id, in manifest order. Fails on gaps."""
    missing = [r["id"] for r in records if r["id"] not in hypotheses]
    if missing:
        raise ValueError(f"{len(missing)} clips have no hypothesis, e.g. {missing[:3]}")
    return [{
        "id": r["id"],
        "split": r["split"],
        "speaker_session_id": r["speaker_session_id"],
        "duration_sec": r["duration_sec"],
        "reference": r["text"],
        "reference_raw": r["text_raw"],
        "hypothesis_raw": hypotheses[r["id"]],
        "hypothesis": normalize_text(hypotheses[r["id"]]),
    } for r in records]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8, LF, ids stay strings so leading zeros survive
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def corpus_scores(rows: list[dict]) -> dict:
    import jiwer
    refs = [r["reference"] for r in rows]
    hyps = [r["hypothesis"] for r in rows]
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}


def environment() -> dict:
    import torch
    import transformers
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                          if torch.cuda.is_available() else None),
    }
    try:
        import jiwer
        info["jiwer"] = jiwer.__version__
    except (ImportError, AttributeError):
        pass
    return info


def resolve_revision(repo_id: str, pinned: str | None) -> str:
    if pinned:
        return pinned
    from huggingface_hub import HfApi
    return HfApi().model_info(repo_id).sha


def load_audio(record: dict):
    from preprocess import load_example
    return load_example(AUDIO_ROOT, record)


def transcribe_whisper(ckpt, revision, records, *, batch_size, num_beams, device, dtype):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ckpt.repo_id, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt.repo_id, revision=revision, dtype=dtype).to(device).eval()
    hyps = {}
    for batch in batches(records, batch_size):
        audio = [load_audio(r) for r in batch]
        feats = processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(
                feats.input_features.to(device, dtype),
                language=LANGUAGE, task="transcribe",
                num_beams=num_beams, max_new_tokens=225,
            )
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        hyps.update({r["id"]: t.strip() for r, t in zip(batch, texts)})
    settings = {"decoding": "beam" if num_beams > 1 else "greedy", "num_beams": num_beams,
                "language": LANGUAGE, "task": "transcribe", "max_new_tokens": 225}
    return hyps, model, settings


def transcribe_ctc(ckpt, revision, records, *, batch_size, device, dtype, **_):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(ckpt.repo_id, revision=revision)
    model = Wav2Vec2ForCTC.from_pretrained(
        ckpt.repo_id, revision=revision, dtype=dtype).to(device).eval()
    hyps = {}
    for batch in batches(records, batch_size):
        audio = [load_audio(r) for r in batch]
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt",
                           padding=True, return_attention_mask=True)
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device, dtype),
                           attention_mask=inputs.attention_mask.to(device)).logits
        texts = processor.batch_decode(logits.argmax(dim=-1))
        hyps.update({r["id"]: t.strip() for r, t in zip(batch, texts)})
    settings = {"decoding": "greedy CTC (argmax), no language model"}
    return hyps, model, settings


def run(model_key: str, split: str, *, num_beams: int, batch_size: int, limit: int | None):
    import torch

    ckpt = CHECKPOINTS[model_key]
    revision = resolve_revision(ckpt.repo_id, REVISIONS[model_key])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    records = load_manifest(split)[:limit] if limit else load_manifest(split)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fn = transcribe_whisper if ckpt.family == "whisper" else transcribe_ctc
    hyps, model, settings = fn(ckpt, revision, records, batch_size=batch_size,
                               num_beams=num_beams, device=device, dtype=dtype)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    audio_seconds = sum(r["duration_sec"] for r in records)

    rows = prediction_rows(records, hyps)
    tag = f"{model_key}_{split}" + (f"_beam{num_beams}" if ckpt.family == "whisper" else "")
    if limit:
        tag += f"_first{limit}"
    write_csv(OUT / "predictions" / f"{tag}.csv", rows)

    info = {
        "model_key": model_key,
        "checkpoint": ckpt.repo_id,
        "revision": revision,
        "family": ckpt.family,
        "note": ckpt.note,
        "trained_by_our_group": False,
        "parameters": sum(p.numel() for p in model.parameters()),
        "split": split,
        "utterances": len(rows),
        "audio_hours": round(audio_seconds / 3600, 5),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        **settings,
        "wall_seconds_incl_audio_loading": round(seconds, 2),
        "rtf_incl_audio_loading": round(seconds / audio_seconds, 4),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                               if device == "cuda" else None),
        "text_normalization": "prepare.normalize_text (lowercase, NFC, punctuation/symbols -> space)",
        "scores_preliminary": corpus_scores(rows),
        "environment": environment(),
    }
    path = OUT / "model_info" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tag}: WER {info['scores_preliminary']['wer']:.4f} "
          f"CER {info['scores_preliminary']['cer']:.4f} RTF {info['rtf_incl_audio_loading']}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--num-beams", type=int, nargs="+", default=[1],
                        help="Whisper only. Several values on dev to choose one; one value on test.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Smoke test on the first N clips only.")
    args = parser.parse_args(argv)
    beams = args.num_beams if CHECKPOINTS[args.model].family == "whisper" else [1]
    if args.split == "test" and len(beams) > 1:
        parser.error("Choose the beam size on dev; pass exactly one value for test.")
    for beam in beams:
        run(args.model, args.split, num_beams=beam, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
