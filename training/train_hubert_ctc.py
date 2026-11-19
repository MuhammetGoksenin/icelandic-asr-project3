"""Fine-tune a multilingual HuBERT encoder with a CTC head on the frozen v1 Icelandic train split.

This is our own training (course baseline family "HuBERT + CTC head, fine-tuned for
your language"). The base encoder utter-project/mHuBERT-147 has no ASR head; we add
a randomly initialised CTC head over the train-only character vocabulary in
output/ctc_vocab.json and fine-tune on train. Dev selects the best checkpoint.
Test is decoded once, with the selected checkpoint, by --evaluate-test.

Outputs (results/hubert_ctc/):
  predictions/hubert_ctc_{dev,test}.csv    per-clip predictions, same columns as the baselines
  model_info/hubert_ctc_{dev,test}.json    checkpoint, hyperparameters, timing, GPU, scores
  training_log.json, training_curves.png   loss and dev WER/CER per evaluation
The fine-tuned weights go to models/hubert_ctc/best (git-ignored; see README for where we publish them).

Usage (from the repository root):
  python training/train_hubert_ctc.py --train                 # train + dev selection
  python training/train_hubert_ctc.py --evaluate-test         # once, after freezing everything
  python training/train_hubert_ctc.py --train --limit 64 --epochs 1   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.run_pretrained import (  # noqa: E402
    corpus_scores, environment, load_audio, load_manifest, prediction_rows, write_csv,
)

OUT = ROOT / "results" / "hubert_ctc"
MODEL_DIR = ROOT / "models" / "hubert_ctc"
VOCAB = ROOT / "output" / "ctc_vocab.json"
KEY = "hubert_ctc"


@dataclass
class Config:
    base_model: str = "utter-project/mHuBERT-147"
    base_revision: str | None = None  # resolved at run time and recorded; pin it after the first run
    seed: int = 2105
    epochs: int = 30
    # Run 1 (lr 1e-4, grad_accum 2, patience 5) stayed on the CTC blank plateau
    # (dev WER 1.0, train loss ~12.7) and early stopping ended it after 6 epochs /
    # ~1k updates. Run 2: 3x higher LR, twice as many updates per epoch, and
    # patience 10 so the model gets past the plateau before stopping can trigger.
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.005
    batch_size: int = 16
    grad_accum: int = 1
    max_seconds: float = 20.0
    freeze_feature_encoder: bool = True
    mask_time_prob: float = 0.05
    layerdrop: float = 0.05
    ctc_zero_infinity: bool = True
    early_stopping_patience: int = 10
    # Quota guard: if dev WER is still >= this after `plateau_guard_epoch` epochs,
    # the model never left the CTC blank plateau. Stop instead of burning GPU hours.
    plateau_guard_epoch: int = 6
    plateau_guard_wer: float = 0.98
    fp16: bool = True


# ---------- data ----------

class AudioDataset:
    """Reads FLACs on the fly so 15 h of audio never sits in RAM."""

    def __init__(self, records, extractor, tokenizer):
        self.records, self.extractor, self.tokenizer = records, extractor, tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        features = self.extractor(load_audio(record), sampling_rate=16_000)
        return {
            "input_values": features.input_values[0],
            "labels": self.tokenizer(record["text"], add_special_tokens=False).input_ids,
        }


class CTCCollator:
    """Pads audio with zeros (+ attention mask) and labels with -100 so CTC ignores padding."""

    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, batch):
        import torch
        use_mask = bool(self.extractor.return_attention_mask)
        audio = self.extractor.pad([{"input_values": b["input_values"]} for b in batch],
                                   padding=True, return_attention_mask=use_mask, return_tensors="pt")
        width = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for row, b in enumerate(batch):
            labels[row, :len(b["labels"])] = torch.tensor(b["labels"], dtype=torch.long)
        out = {"input_values": audio["input_values"], "labels": labels}
        if use_mask:
            out["attention_mask"] = audio["attention_mask"]
        return out


def decode_ids(tokenizer, pred_ids):
    """Greedy CTC decoding: the tokenizer collapses repeats, drops [PAD] and maps '|' to spaces."""
    return [t.strip() for t in tokenizer.batch_decode(pred_ids)]


# ---------- model ----------

def components(cfg: Config, revision: str):
    from transformers import Wav2Vec2FeatureExtractor

    sys.path.insert(0, str(ROOT))
    from preprocess import ctc_tokenizer

    tokenizer = ctc_tokenizer(VOCAB)
    # Keep the checkpoint's own setting: group-norm HuBERT/wav2vec2 base models are trained
    # without an attention mask on zero-padded input, so the mask is only passed if it says so.
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.base_model, revision=revision)
    return tokenizer, extractor


def build_model(cfg: Config, revision: str, tokenizer):
    from transformers import HubertForCTC

    model = HubertForCTC.from_pretrained(
        cfg.base_model, revision=revision,
        vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean", ctc_zero_infinity=cfg.ctc_zero_infinity,
        mask_time_prob=cfg.mask_time_prob, layerdrop=cfg.layerdrop,
        ignore_mismatched_sizes=True,
    )
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()
    return model


def resolve_revision(repo_id, pinned):
    if pinned:
        return pinned
    from huggingface_hub import HfApi
    return HfApi().model_info(repo_id).sha


# ---------- train ----------

class PlateauGuard:
    """Stops training when dev WER never drops below `wer` by epoch `epoch`."""

    def __init__(self, epoch, wer):
        from transformers import TrainerCallback

        guard = self

        class _Callback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                guard.history.append(metrics.get("eval_wer", 1.0))
                if state.epoch is not None and state.epoch >= guard.epoch and min(guard.history) >= guard.wer:
                    guard.tripped = True
                    control.should_training_stop = True
                    print(f"PLATEAU GUARD: dev WER still {min(guard.history):.3f} after epoch "
                          f"{state.epoch:.0f}; stopping to save GPU quota.", flush=True)

        self.epoch, self.wer, self.history, self.tripped = epoch, wer, [], False
        self.callback = _Callback()


def train(cfg: Config, limit: int | None, resume: bool):
    import numpy as np
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    revision = resolve_revision(cfg.base_model, cfg.base_revision)
    tokenizer, extractor = components(cfg, revision)

    train_records = [r for r in load_manifest("train") if r["duration_sec"] <= cfg.max_seconds]
    dev_records = load_manifest("dev")
    if limit:
        train_records, dev_records = train_records[:limit], dev_records[:max(8, limit // 4)]

    model = build_model(cfg, revision, tokenizer)
    guard = PlateauGuard(cfg.plateau_guard_epoch, cfg.plateau_guard_wer)

    def compute_metrics(pred):
        import jiwer
        ids = np.where(pred.predictions == -100, tokenizer.pad_token_id, pred.predictions)
        hyps = decode_ids(tokenizer, ids)
        refs = [r["text"] for r in dev_records]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        seed=cfg.seed,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=CTCCollator(extractor),
        train_dataset=AudioDataset(train_records, extractor, tokenizer),
        eval_dataset=AudioDataset(dev_records, extractor, tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1),
        callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience), guard.callback],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume or None)
    train_seconds = time.perf_counter() - start

    if guard.tripped:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n",
                                               encoding="utf-8")
        plot_curves(trainer.state.log_history, OUT / "training_curves.png")
        raise SystemExit("Training stopped by the plateau guard: no usable model, test not decoded.")

    best = MODEL_DIR / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    extractor.save_pretrained(str(best))

    history = trainer.state.log_history
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training_log.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    plot_curves(history, OUT / "training_curves.png")

    training = {
        "config": asdict(cfg) | {"base_revision": revision},
        "train_utterances": len(train_records),
        "train_hours": round(sum(r["duration_sec"] for r in train_records) / 3600, 5),
        "dropped_longer_than_max_seconds": len(load_manifest("train")) - len(train_records) if not limit else None,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev_wer": trainer.state.best_metric,
        "epochs_run": trainer.state.epoch,
        "global_steps": trainer.state.global_step,
        "train_seconds": round(train_seconds, 1),
        "gpus_used": torch.cuda.device_count(),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                               if torch.cuda.is_available() else None),
        "smoke_test_limit": limit,
    }
    (OUT / "training_summary.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: training[k] for k in ("best_dev_wer", "epochs_run", "train_seconds")}))

    # final dev predictions from the selected checkpoint
    decode_split("dev", best, training, limit)
    return training


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    dev = [(h["step"], h["eval_loss"], h["eval_wer"], h["eval_cer"]) for h in history if "eval_wer" in h]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    if train:
        left.plot(*zip(*train), label="train loss")
    if dev:
        steps, loss, wer, cer = zip(*dev)
        left.plot(steps, loss, marker="o", label="dev loss")
        right.plot(steps, [w * 100 for w in wer], marker="o", label="dev WER")
        right.plot(steps, [c * 100 for c in cer], marker="o", label="dev CER")
    left.set(xlabel="step", ylabel="CTC loss", title="Loss")
    right.set(xlabel="step", ylabel="%", title="Dev error rate")
    for ax in (left, right):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- decode ----------

def decode_split(split: str, model_path: Path, training: dict | None, limit: int | None = None,
                 batch_size: int = 16):
    import torch
    from transformers import HubertForCTC, Wav2Vec2FeatureExtractor

    sys.path.insert(0, str(ROOT))
    from preprocess import ctc_tokenizer

    from baselines.run_pretrained import batches

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = ctc_tokenizer(model_path / "vocab.json" if (model_path / "vocab.json").exists() else VOCAB)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path))
    model = HubertForCTC.from_pretrained(str(model_path)).to(device).eval()
    records = load_manifest(split)[:limit] if limit else load_manifest(split)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    hyps = {}
    for batch in batches(records, batch_size):
        use_mask = bool(extractor.return_attention_mask)
        inputs = extractor([load_audio(r) for r in batch], sampling_rate=16_000, padding=True,
                           return_attention_mask=use_mask, return_tensors="pt")
        mask = inputs.attention_mask.to(device) if use_mask else None
        with torch.inference_mode(), torch.autocast(device, enabled=device == "cuda"):
            logits = model(inputs.input_values.to(device), attention_mask=mask).logits
        hyps.update(zip((r["id"] for r in batch), decode_ids(tokenizer, logits.argmax(-1).cpu())))
    seconds = time.perf_counter() - start

    rows = prediction_rows(records, hyps)
    tag = f"{KEY}_{split}" + (f"_first{limit}" if limit else "")
    write_csv(OUT / "predictions" / f"{tag}.csv", rows)
    audio_seconds = sum(r["duration_sec"] for r in records)
    if training is None and (OUT / "training_summary.json").exists():
        training = json.loads((OUT / "training_summary.json").read_text(encoding="utf-8"))
    info = {
        "model_key": KEY,
        "checkpoint": "our fine-tune of " + (training or {}).get("config", {}).get("base_model", "?"),
        "base_revision": (training or {}).get("config", {}).get("base_revision"),
        "trained_by_our_group": True,
        "family": "ctc",
        "decoding": "greedy CTC (argmax), no language model",
        "split": split,
        "utterances": len(rows),
        "audio_hours": round(audio_seconds / 3600, 5),
        "device": device,
        "dtype": "float16 autocast" if device == "cuda" else "float32",
        "batch_size": batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "wall_seconds_incl_audio_loading": round(seconds, 2),
        "rtf_incl_audio_loading": round(seconds / audio_seconds, 4),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                               if device == "cuda" else None),
        "text_normalization": "prepare.normalize_text (lowercase, NFC, punctuation/symbols -> space)",
        "scores_preliminary": corpus_scores(rows),
        "training": training,
        "environment": environment(),
    }
    path = OUT / "model_info" / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{tag}: WER {info['scores_preliminary']['wer']:.4f} CER {info['scores_preliminary']['cer']:.4f}")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Fine-tune on train, select on dev.")
    mode.add_argument("--evaluate-test", action="store_true", help="Decode test once with models/hubert_ctc/best.")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint (new Kaggle session).")
    parser.add_argument("--limit", type=int, help="Smoke test on the first N train clips.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args(argv)

    cfg = Config()
    for name in ("epochs", "batch_size", "learning_rate"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    if args.train:
        train(cfg, args.limit, args.resume)
    else:
        decode_split("test", MODEL_DIR / "best", None)


if __name__ == "__main__":
    main()
