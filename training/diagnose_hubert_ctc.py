"""~10-minute sanity check before spending GPU hours on HuBERT + CTC training.

Two full runs stayed on the CTC blank plateau (dev WER 1.0). This script tells
whether the pipeline itself is broken or the model just needs a different recipe:

  1. prints the feature-extractor, model and tokenizer settings that matter for CTC
  2. checks that labels round-trip through the tokenizer
  3. tries to OVERFIT 32 train clips on one GPU with no augmentation, in two modes:
       A) full fine-tune (encoder + head, like train_hubert_ctc.py)
       B) frozen encoder, head only (fairseq's freeze_finetune_updates idea)
     printing loss, blank ratio and a decoded sample every 50 steps.

A working pipeline memorises 32 clips: loss falls well below the plateau and the
decoded text starts to look like the reference. If neither mode leaves the
plateau, the problem is in the data/labels/model setup, not the hyperparameters.

Run from the repository root on a GPU (pulls only the 32 clips it needs):
  CUDA_VISIBLE_DEVICES=0 python training/diagnose_hubert_ctc.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.run_pretrained import load_audio, load_manifest  # noqa: E402
from training.train_hubert_ctc import VOCAB, Config, decode_ids  # noqa: E402

N_CLIPS = 32
BATCH = 8
STEPS = 400
PRINT_EVERY = 50

try:
    import torch as _t
    DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


def pull_audio(records):
    paths = ",".join(f"data/cleaned/samromur-21.05-20h-v1/{r['audio_path']}" for r in records)
    subprocess.run(["git", "lfs", "pull", f"--include={paths}", "--exclude="], cwd=ROOT, check=True)


def describe(extractor, model, tokenizer, records):
    cfg = model.config
    print("== feature extractor:", json.dumps({k: getattr(extractor, k, None) for k in
          ("do_normalize", "return_attention_mask", "sampling_rate", "padding_value")}))
    print("== model:", json.dumps({k: getattr(cfg, k, None) for k in
          ("vocab_size", "pad_token_id", "ctc_loss_reduction", "ctc_zero_infinity", "feat_extract_norm",
           "do_stable_layer_norm", "mask_time_prob", "layerdrop", "final_dropout", "hidden_size")}))
    print("== tokenizer: size", len(tokenizer), "pad", tokenizer.pad_token_id, "unk", tokenizer.unk_token_id)
    for record in records[:3]:
        ids = tokenizer(record["text"], add_special_tokens=False).input_ids
        back = tokenizer.decode(ids, group_tokens=False)
        unk = ids.count(tokenizer.unk_token_id)
        print(f"   label {record['id']}: {record['text']!r} -> {len(ids)} ids, unk={unk}, back={back!r}")
        assert back.strip() == record["text"], "labels do not round-trip"
    audio = load_audio(records[0])
    print(f"== audio sample: len={len(audio)}, min={audio.min():.3f}, max={audio.max():.3f}, "
          f"std={audio.std():.4f}")


def overfit(mode, cfg, revision, tokenizer, extractor, records, audio):
    import numpy as np
    import torch
    from transformers import HubertForCTC

    torch.manual_seed(cfg.seed)
    model = HubertForCTC.from_pretrained(
        cfg.base_model, revision=revision, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean", ctc_zero_infinity=True, mask_time_prob=0.0, layerdrop=0.0,
        ignore_mismatched_sizes=True).to(DEVICE).train()
    model.freeze_feature_encoder()
    if mode == "B_head_only":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("lm_head")
    lr = 3e-4 if mode == "A_full" else 1e-3
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    print(f"\n== overfit {mode}: {sum(p.numel() for p in params):,} trainable params, lr {lr}")

    labels = [tokenizer(r["text"], add_special_tokens=False).input_ids for r in records]
    use_mask = bool(extractor.return_attention_mask)
    start, best = time.time(), float("inf")
    for step in range(1, STEPS + 1):
        idx = np.random.default_rng(step).choice(len(records), BATCH, replace=False)
        feats = extractor([audio[i] for i in idx], sampling_rate=16_000, padding=True,
                          return_attention_mask=use_mask, return_tensors="pt")
        width = max(len(labels[i]) for i in idx)
        lab = torch.full((BATCH, width), -100, dtype=torch.long)
        for row, i in enumerate(idx):
            lab[row, :len(labels[i])] = torch.tensor(labels[i])
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            out = model(feats.input_values.to(DEVICE),
                        attention_mask=feats.attention_mask.to(DEVICE) if use_mask else None,
                        labels=lab.to(DEVICE))
        optim.zero_grad()
        scaler.scale(out.loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optim)
        scaler.update()
        best = min(best, out.loss.item())
        if step % PRINT_EVERY == 0 or step == 1:
            pred = out.logits.argmax(-1).cpu()
            blank = (pred == tokenizer.pad_token_id).float().mean().item()
            hyp = decode_ids(tokenizer, pred[:1].numpy())[0]
            print(f"  step {step:4d}  loss {out.loss.item():7.3f}  blank {blank:5.1%}  "
                  f"ref {records[idx[0]]['text'][:40]!r}  hyp {hyp[:40]!r}  ({time.time() - start:.0f}s)",
                  flush=True)
    del model, optim
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best


def main():
    import numpy as np
    from transformers import HubertForCTC

    from training.train_hubert_ctc import components, resolve_revision

    cfg = Config()
    train = [r for r in load_manifest("train") if 2.0 <= r["duration_sec"] <= 6.0]
    records = [train[i] for i in np.random.default_rng(cfg.seed).choice(len(train), N_CLIPS, replace=False)]
    pull_audio(records)

    revision = resolve_revision(cfg.base_model, cfg.base_revision)
    tokenizer, extractor = components(cfg, revision)
    probe = HubertForCTC.from_pretrained(cfg.base_model, revision=revision, vocab_size=len(tokenizer),
                                         pad_token_id=tokenizer.pad_token_id, ignore_mismatched_sizes=True)
    describe(extractor, probe, tokenizer, records)
    del probe
    audio = [load_audio(r) for r in records]

    results = {mode: overfit(mode, cfg, revision, tokenizer, extractor, records, audio)
               for mode in ("A_full", "B_head_only")}
    print("\n== lowest loss per mode:", json.dumps({k: round(v, 3) for k, v in results.items()}))
    print("Plateau in the full runs was ~6.5. Well below that + readable hyp = pipeline OK.")


if __name__ == "__main__":
    main()
