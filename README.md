# Person 3 — Whisper Small Fine-Tuning (Icelandic ASR)

## Overview

This contribution fine-tunes the multilingual `openai/whisper-small` model for Icelandic automatic speech recognition using the team's cleaned **Samrómur 21.05** dataset. The training run completed **3 epochs / 1,968 optimizer steps** on an **NVIDIA Tesla T4**, taking approximately **5 hours 12 minutes**.

## Training setup

| Setting | Value |
|---|---|
| Training / validation / held-out test samples | 10,488 / 1,314 / 1,884 |
| Learning rate | 1e-5 |
| Per-device training batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Precision | FP16 |
| Epochs | 3 |

## Recorded validation results

| Checkpoint step | WER | CER |
|---|---:|---:|
| 200 | 50.94% | 15.98% |
| 600 | 38.21% | 10.85% |
| 1000 | 34.51% | 9.70% |
| 1400 | 31.59% | 8.67% |
| 1600 | **30.62%** | 8.41% |
| 1968 | 30.70% | **8.40%** |

The **best recorded validation WER** was **30.62% at step 1600**. The **lowest recorded validation CER** was **8.40% at step 1968**. These are different checkpoints, and neither figure is a held-out test result. See `training_metrics.json` for the complete validation history.

## Available model and checkpoint limitation

The training run completed, but the best step-1600 checkpoint was not preserved. The **usable exported model is checkpoint 1000**, with validation **WER 34.51%** and **CER 9.70%**. Do not attribute the 30.62% WER to the exported model.

The recovered model is stored outside GitHub in Google Drive:

`MyDrive/NLP_Project3_Whisper_FineTuning/recovered_model_step1000/`

Ask Person 3 for access to the **whole folder** (model weights, model/generation configuration, tokenizer and processor files). A shareable Drive URL is not included here because one has not been provided. Model weights are intentionally not committed to GitHub.

## Notebook

`NLP_Project3_Person3_Whisper_Clean.ipynb` contains the implementation and the saved output of the completed training run. **Do not use Run all** unless intentionally repeating training; Step 10B is retained as an archived training record.

## Remaining evaluation

A teammate with available GPU capacity should load the **recovered step-1000 model** and evaluate it on the team's **unchanged held-out test split** (`output/test.jsonl`). Please record:

- Test-set WER and CER, with the exact normalization and decoding settings used.
- Total inference duration, per-utterance latency or throughput, and GPU model.
- Per-utterance reference and prediction pairs, with sample IDs, for paired comparisons.
- Confidence intervals and statistical tests **after** other systems' predictions on the same test samples are available.

**Status:** No held-out test WER/CER, latency, confidence interval, or p-value has been reported for this recovered model yet. Do not invent these results.

## Provenance

The JSON metrics were reconstructed from the notebook's saved Step 10B training output because the originally reported `training_results.json` could not be located after the runtime reset.
