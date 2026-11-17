# Samrómur 21.05 Icelandic ASR dataset preparation

**Status.** The selected IDs and references are frozen. Human listening and in-repository delivery of raw and cleaned audio remain pending. The [data inventory and storage plan](DATA_STORAGE.md) describes what is available for each workflow.

This repository contains the shared 20-hour Samrómur selection for Whisper and wav2vec 2.0. Seed **2105** selected clips within the official train, dev, and test partitions. The dataset is Icelandic audio-to-text ASR.

| Split | Clips | Decoded hours |
|---|---:|---:|
| [Train](output/train.jsonl) | 10,488 | 15.00002 |
| [Dev](output/dev.jsonl) | 1,314 | 2.00088 |
| [Test](output/test.jsonl) | 1,884 | 2.99971 |

## Get started

Use the revised **Samromur_ASR_Handoff_LF.zip** for artifact checksum verification. It contains the same audio, IDs, and references as the earlier ZIP. The Windows verification environment used Python 3.14.10, NumPy 2.5.5, SoundFile 0.14.0 with libsndfile 1.2.2, and Transformers 5.23.0.

Use your local copy of the full Samrómur 21.05 release or extract the separately provided team handoff ZIP. Set the audio root to the release directory or the handoff's `audio/` directory. Both have `train/`, `dev/`, and `test/` directories and work with the same manifests.

The preparation environment was verified on Windows with Python 3.14.10. Run from the repository root.

```powershell
python -m venv .venv
$venvPy = '.\.venv\Scripts\python.exe'
& $venvPy -m pip install --only-binary=:all: -r requirements-windows-py314.lock.txt
& $venvPy -m pip check
```

On macOS or Linux with Python 3.14, use `python3.14 -m venv .venv`, `.venv/bin/python`, and [requirements.txt](requirements.txt). See the [Windows dependency lock](requirements-windows-py314.lock.txt).

## Repository contents

| Path | Purpose |
|---|---|
| [README.md](README.md) | Setup, review, and reproduction instructions |
| [DATA_STORAGE.md](DATA_STORAGE.md) | Measured data sizes, availability, and storage decision |
| [LICENSE](LICENSE) | MIT license for project code |
| [.gitignore](.gitignore) and [.gitattributes](.gitattributes) | Local-file exclusions and byte-preserving checkout of generated artifacts |
| [audit.py](audit.py) and [prepare.py](prepare.py) | Release audit, cleaning, and seeded subset generation |
| [preprocess.py](preprocess.py) | Audio loader and Whisper and wav2vec 2.0 preprocessing |
| [verify.py](verify.py) and [verify_handoff.py](verify_handoff.py) | Verification against the release or the extracted handoff |
| [requirements.txt](requirements.txt) and [requirements-windows-py314.lock.txt](requirements-windows-py314.lock.txt) | Pinned dependencies and tested Windows environment |
| [tests/](tests/) | Independent tests with known inputs and expected outputs |
| [output/](output/) | Frozen manifests and references, vocabulary, audit, review queues, and verification records |
| [contributions/](contributions/) | Individual work records authored and committed by team members |

## Load and preprocess a clip

The multilingual Whisper preprocessing checkpoint is **openai/whisper-tiny** at revision **169d4a4341b33bc18d8881c4b69c2e104e1cc0af**.

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

The [focused review queue](output/focused_review.tsv) documents **53 near-silence exclusions and seven selected possible-clipping flags**. All 53 met the RMS threshold, and 15 also met the quiet-fraction threshold. The seven selected clips have 0.1095% to 0.3238% full-scale samples. Quiet speech may still be intelligible and full-scale samples may be inaudible. Use the release paths to listen and record proposed changes by ID.

The exclusion log contains **150 unique utterances**, with 127 train, 18 dev, and five test. Of those, 94 were withheld only for the release validation flag, one for both validation and digital silence, two for the metadata empty status, and 53 for near-silence.

## Cleaning policy

- The audit checks the FLAC bytes against the release MD5 list and fully decodes each recording. It records format, sample count, sound levels, and a SHA-256 of the PCM that includes sample rate and channel count. The hash detects exact duplicates. Edited or near-duplicate recordings need other checks.
- Selection excludes absent metadata or prompt text, validation values other than **1.0**, empty values other than **0**, missing or unreadable audio, decode failures, checksum mismatches, audio outside mono 16 kHz FLAC PCM16, or decoded durations outside 0.5–30 seconds. A missing release checksum is a review flag.
- A quiet window has RMS below −50 dBFS over 20 ms, with the trailing partial window weighted by its sample count. Digital silence means zero frames or all-zero PCM. Near-silence means whole-clip RMS at or below −55 dBFS or at least 98% quiet time. These exclusions are automated decisions pending listening.
- Among otherwise usable signals, flag RMS at or below −35 dBFS, at least 75% quiet time, a quiet edge of at least 1.5 seconds, or at least 0.1% full-scale samples. Flag invalid duration metadata, a measured duration difference greater than 0.05 seconds, or a sample-rate disagreement.
- Withhold exact PCM groups with conflicting scoring text. For matching labels, retain one eligible copy in test/dev/train priority order, breaking ties by seeded order. The same split priority screens later selections against selected session IDs, raw prompts, and scoring references. The IDs establish session separation. A person could have recorded in several sessions.
- The sampler sorts by SHA-256 of **seed/audio_path** and takes the prefix closest to each duration target from its official partition. This revision preserves the existing IDs and references. Any later membership or reference change needs a dataset version and a documented reason before model comparisons use it.

## Reproduce and inspect the audit

Use a clean extraction of the release. The [data inventory](DATA_STORAGE.md) documents extra copies found outside the official paths in the local extraction.

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

To replay selection using the committed audit, copy the gzip into a new output folder. The release root must supply the exact **README**, **metadata.tsv**, and **md5s.txt**. The verification command additionally needs the selected audio and pinned Whisper preprocessing assets.

~~~powershell
New-Item -ItemType Directory -Force reproduced/replay
Copy-Item output/audio_audit.jsonl.gz reproduced/replay/
& $venvPy -X utf8 prepare.py --root $release --output reproduced/replay --seed 2105
& $venvPy -X utf8 verify.py --root $release --output reproduced/replay
~~~

Both scripts accept **audio_audit.jsonl** or **audio_audit.jsonl.gz**. A plain audit in the same directory takes precedence. Generated artifacts use UTF-8 with LF line endings. Verification reports record the platform used. Pass **--offline** to **verify.py** once the pinned assets are already local.

Run the independent fixture tests from the repository root. Run the handoff verifier against the inner folder extracted from the LF ZIP. It needs the packaged **audio/** split directories, **output/**, and **assets/whisper/** with its checksum file.

~~~powershell
& $venvPy -X utf8 -m unittest discover -s tests -v
& $venvPy -X utf8 verify_handoff.py --package-root 'C:\path\to\Samromur_ASR_Handoff'
~~~

The fixture tests and gzip replay passed on Windows with Python 3.14.10. A fresh audit of all 100,000 official paths matched the committed decompressed audit. Real-data verification checked every selected clip and label and ran both feature extractors on 30 clips. Handoff verification used a fresh extraction and local Whisper assets. Linux and macOS were outside these runs.

## Source and attribution

The repository code is [MIT-licensed](LICENSE). The release audio, prompts, and derived references use the corpus license.

Samrómur Icelandic Speech 21.05 by David Erik Mollberg et al. is distributed under CC BY 4.0. This repository selects IDs and derives normalized scoring references from that release. Cite Mollberg et al., “Samrómur: Crowd-sourcing Data Collection for Icelandic Speech Recognition,” LREC 2020. [CLARIN release](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/189) · [OpenSLR 112](https://openslr.org/112/) · [LREC paper](https://aclanthology.org/2020.lrec-1.425/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
