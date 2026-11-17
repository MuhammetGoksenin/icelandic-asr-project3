# Samrómur 21.05 Icelandic ASR dataset preparation

This repository contains the shared 20-hour Samrómur selection for Whisper and wav2vec 2.0. Seed **2105** selected clips within the official train, dev, and test partitions. The dataset is Icelandic audio-to-text ASR.

| Split | Clips | Decoded hours |
|---|---:|---:|
| [Train](output/train.jsonl) | 10,488 | 15.00002 |
| [Dev](output/dev.jsonl) | 1,314 | 2.00088 |
| [Test](output/test.jsonl) | 1,884 | 2.99971 |

## Get started

Use your local copy of the full Samrómur 21.05 release or extract the separately provided team handoff ZIP. Set the audio root to the release directory or the handoff's `audio/` directory. Both have `train/`, `dev/`, and `test/` directories and work with the same manifests.

The preparation environment was verified on Windows with Python 3.14.10. Run from the repository root.

```powershell
python -m venv .venv
$venvPy = '.\.venv\Scripts\python.exe'
& $venvPy -m pip install --only-binary=:all: -r requirements-windows-py314.lock.txt
& $venvPy -m pip check
```

On macOS or Linux with Python 3.14, use `python3.14 -m venv .venv`, `.venv/bin/python`, and [requirements.txt](requirements.txt). See the [Windows dependency lock](requirements-windows-py314.lock.txt).

### Load and preprocess a clip

```python
from pathlib import Path

from transformers import Wav2Vec2FeatureExtractor

from preprocess import (
    ctc_tokenizer,
    iter_examples,
    wav2vec2_example,
    whisper_components,
    whisper_example,
)

audio_root = Path(r"C:\path\to\samromur_21.05")  # Or the extracted handoff's audio directory.
output = Path("output")
example = next(iter_examples(audio_root, output / "dev.jsonl"))

extractor, tokenizer = whisper_components()
whisper = whisper_example(example, extractor, tokenizer)

ctc_extractor = Wav2Vec2FeatureExtractor(
    sampling_rate=16_000, do_normalize=True, return_attention_mask=True
)
wav2vec = wav2vec2_example(example, ctc_extractor, ctc_tokenizer(output / "ctc_vocab.json"))
```

[preprocess.py](preprocess.py) loads each FLAC as a mono 16 kHz float32 waveform. Whisper uses the pinned multilingual feature configuration and tokenizer, with Icelandic as the language and `transcribe` as the task. The first call fetches the public preprocessing assets. The wav2vec 2.0 helper uses the [CTC vocabulary derived from train](output/ctc_vocab.json), which covers all characters in the frozen dev/test references. For a wav2vec 2.0 checkpoint with a fixed CTC vocabulary, pass its matching extractor and tokenizer.

Use the shared [evaluation references (TSV)](output/evaluation_references.tsv) or [evaluation references (JSONL)](output/evaluation_references.jsonl) for both models. Join predictions by `split` and `id`, preserving leading zeros. Each manifest preserves the raw and release-normalized prompts. The scoring text lowercases Unicode, applies NFC, treats punctuation and symbols as spaces, and collapses whitespace. Apply `normalize_text` from [prepare.py](prepare.py) to model hypotheses for comparable WER or CER. Icelandic characters and diacritics are preserved.

## Human listening remains pending

Start with the **541 flagged evaluation clips**, comprising **315 test** and **226 dev** clips. Open the [flagged dev/test review checklist](output/flagged_dev_test_review.tsv), import `id` as text, and divide the rows among the four teammates. The `audio_path` column is relative to the extracted handoff. For a full-release audio root, drop the leading `audio/` component.

1. **Basic audio quality.** Any teammate can listen for audible speech, long gaps, noise, other voices, clipping, and abrupt cutoffs. Record `pass`, `issue`, or `uncertain` under `audio_quality_status`, with a timestamped note where useful.
2. **Transcript agreement.** A reviewer who understands Icelandic should compare the recording with `text_raw` and check for missing, extra, or misread words. Record `match`, `mismatch`, or `uncertain` under `transcript_status`, with the phrase or timestamp. Leave the transcript status `pending` until that review takes place.
3. **Shared evaluation.** Record recommended exclusions or corrections by split and ID. Keep the current evaluation IDs and references fixed for both models.

After the flagged dev/test clips, use the [full listening queue](output/listening_queue.tsv) for flagged train clips and unflagged controls. The [exclusion log](output/exclusions.tsv) records clips withheld during preparation.

## Reproduce and inspect the audit

To repeat the full audit and seed-2105 selection, use the full Samrómur 21.05 release. Write fresh results to a separate folder so they can be compared with the tracked outputs.

```powershell
$release = 'C:\path\to\samromur_21.05'
& $venvPy -X utf8 audit.py --root $release --output reproduced
& $venvPy -X utf8 prepare.py --root $release --output reproduced --seed 2105
& $venvPy -X utf8 verify.py --root $release --output reproduced
```

[audit.py](audit.py) inventories and decodes the full release. [prepare.py](prepare.py) applies exclusion rules, assesses overlap, and builds the seeded subsets and references. [verify.py](verify.py) checks the result against the full release. [verify_handoff.py](verify_handoff.py) is the verifier used in the extracted team handoff with its bundled audio and Whisper assets.

- [Quality report](output/quality_report.md), [machine-readable summary](output/quality_summary.json), and [prompt overlap](output/prompt_overlap.tsv)
- [Per-recording audio audit (gzip-compressed JSONL)](output/audio_audit.jsonl.gz). Python's `gzip.open` reads the full audit directly.
- [Preparation verification](output/verification.json) and [portable handoff verification](output/handoff_verification.json)

## Source and attribution

Samrómur Icelandic Speech 21.05 by David Erik Mollberg et al. is distributed under CC BY 4.0. This repository selects IDs and derives normalized scoring references from that release. Cite Mollberg et al., “Samrómur: Crowd-sourcing Data Collection for Icelandic Speech Recognition,” LREC 2020. [CLARIN release](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/189) · [OpenSLR 112](https://openslr.org/112/) · [LREC paper](https://aclanthology.org/2020.lrec-1.425/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
