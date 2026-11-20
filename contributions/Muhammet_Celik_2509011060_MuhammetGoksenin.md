# Muhammet Çelik

- Student ID **2509011060**
- GitHub **MuhammetGoksenin**
- Hugging Face **GokseninMuhammet**

## Pretrained baselines (Person 2)

I wrote [baselines/run_pretrained.py](../baselines/run_pretrained.py), which runs three published checkpoints on the frozen v1 dev and test manifests without any training by our group: `openai/whisper-small` (multilingual, zero-shot Icelandic), `language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h` (wav2vec 2.0 with a CTC head), and `Valdimarb13/whisper-small-icelandic`. Each run records the exact checkpoint revision, decoding settings, precision, batch size, GPU and package versions next to its per-clip predictions in [results/pretrained/](../results/pretrained/).

I chose the Whisper beam size on the dev split (beam 5 won for both Whisper models) and decoded the test split once with the frozen setting. The [notebook](../notebooks/person2_pretrained_baselines.ipynb) pulls the audio from this repository through Git LFS and needs no manual uploads; I ran it on Kaggle (one Tesla T4). [tests/test_pretrained.py](../tests/test_pretrained.py) checks that predictions join to the manifest by ID, keep leading zeros and Icelandic characters, and use the shared text normalization. I fixed the decoding of `Valdimarb13/whisper-small-icelandic`, whose old generation config has no language tables.

I documented which course baselines do not support Icelandic (Canary, Parakeet, Qwen3-ASR) and the possible training-data overlap of the published Icelandic checkpoints with our test clips.

## HuBERT + CTC (our training)

No published Icelandic HuBERT ASR checkpoint was available, so I fine-tuned the multilingual `utter-project/mHuBERT-147` encoder with a new CTC head over our train-only character vocabulary ([training/train_hubert_ctc.py](../training/train_hubert_ctc.py), [notebook](../notebooks/hubert_ctc_training.ipynb)). The best epoch was chosen by dev WER and the test split was decoded once: **test WER 68.46 %, CER 23.31 %**.

The first two full runs collapsed to all-blank output. I wrote [training/diagnose_hubert_ctc.py](../training/diagnose_hubert_ctc.py), which showed that full fine-tuning from a random head could not even memorize 32 clips, while a two-stage schedule (frozen encoder first, then unfrozen) could. I also added a plateau guard that stops a run whose dev WER stays near 100 %, which kept the failed runs to about one hour of GPU each. [tests/test_hubert_ctc.py](../tests/test_hubert_ctc.py) checks label padding, attention-mask handling and greedy CTC decoding.

The fine-tuned weights are published at [GokseninMuhammet/mhubert-147-icelandic-ctc-samromur](https://huggingface.co/GokseninMuhammet/mhubert-147-icelandic-ctc-samromur) (revision `6cf95c5810843ceeac3044038760f40eeb992879`), with a model card that lists the data, the training recipe, the results and the limitations.

## Documentation

I wrote the README section "Pretrained baselines and HuBERT + CTC", including the justification of WER and CER and the license note for the model weights, and added the new folders to the repository contents table.
