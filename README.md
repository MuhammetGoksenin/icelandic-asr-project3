# Samrómur 21.05 Icelandic ASR dataset preparation

**Dataset.** Use **samromur-21.05-20h-v1**, the frozen seed-2105 selection verified at commit 55f5a68. Its 15/2/3-hour targets, selected IDs, and references are unchanged. The professor accepts the bounded raw subset and Git LFS in this public repository. [Download the audio](#download-the-audio) · [Raw scope and storage](DATA_STORAGE.md)

This repository contains the shared 20-hour Samrómur selection for Whisper and wav2vec 2.0. Seed **2105** selected clips within the official train, dev, and test partitions. The dataset is Icelandic audio-to-text ASR.

| Split | Speaker/session IDs | Utterances | Decoded hours |
|---|---:|---:|---:|
| [Train](output/train.jsonl) | 2,796 | 10,488 | 15.00002 |
| [Dev](output/dev.jsonl) | 963 | 1,314 | 2.00088 |
| [Test](output/test.jsonl) | 1,026 | 1,884 | 2.99971 |

The [split integrity report](output/split_integrity.json) confirms **zero shared speaker/session IDs** for train/dev, train/test, and dev/test. The selected test set has **zero utterances** whose exact original sentence is in selected train and **zero** whose normalized scoring text is in selected train. These are separate comparisons.

The [publisher's README](data/raw/samromur_21.05/README) documents the official partitions as speaker-separated and uses session IDs as speaker IDs. The cleaner checks each selected utterance against the original partition and session in the release metadata. It selects within those partitions. The identifiers establish separation of recorded sessions and cannot identify a person across sessions. Any future new split design must keep each complete speaker/session group together.

Use **dev** to select checkpoints, settings, and hyperparameters. Use **test** after freezing those choices for the final model comparison with shared v1 references. [Required baseline checklist](BASELINES.md)

## Download the audio

With Git LFS installed, clone without downloading audio and then fetch the paths you need. The example uses PowerShell.

~~~powershell
git lfs install
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/RanimAlyaseen/icelandic-asr-project3.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location icelandic-asr-project3

# For model preprocessing, fetch the cleaned v1 audio
git lfs pull --include="data/cleaned/samromur-21.05-20h-v1/**" --exclude=""

# For raw-to-cleaned reproduction or review of shipped exclusions, fetch the raw subset
git lfs pull --include="data/raw/samromur_21.05/**" --exclude=""
~~~

For a single split, append its name before the wildcard, for example **data/cleaned/samromur-21.05-20h-v1/dev/**. Git LFS reuses cached objects shared by the raw and cleaned roots. The [storage notes](DATA_STORAGE.md) give byte counts and quota details.

A fresh public clone on Windows fetched every object with credential helpers disabled. The checkout decoded and verified all 17,307 raw FLACs, matched the 13,686 cleaned files to their release bytes, and reproduced the scoped reports and six v1 ASR artifacts. All 12 independent tests and the 12-artifact gzip replay also passed there. [Verification record](output/lfs_verification.json)

## Get started

Use the hydrated cleaned LFS root, the local full release, or the **audio/** directory of the optional LF handoff. Each has train, dev, and test directories and works with the frozen manifests. The Windows verification environment used Python 3.14.10, NumPy 2.5.5, SoundFile 0.14.0 with libsndfile 1.2.2, and Transformers 5.23.0.

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
| [BASELINES.md](BASELINES.md) | Required modeling checklist and official language support evidence |
| [DATA_STORAGE.md](DATA_STORAGE.md) | Measured data sizes, availability, and storage decision |
| [LICENSE](LICENSE) | MIT terms for project code and CC BY 4.0 dataset attribution |
| [.gitignore](.gitignore) and [.gitattributes](.gitattributes) | Local-file exclusions and byte-preserving checkout of generated artifacts |
| [audit.py](audit.py) and [prepare.py](prepare.py) | Release audit, cleaning, and seeded subset generation |
| [subset.py](subset.py) | Bounded raw membership and raw-to-cleaned processing with split checks |
| [preprocess.py](preprocess.py) | Audio loader and Whisper and wav2vec 2.0 preprocessing |
| [verify.py](verify.py) and [verify_handoff.py](verify_handoff.py) | Verification against the release or the extracted handoff |
| [requirements.txt](requirements.txt) and [requirements-windows-py314.lock.txt](requirements-windows-py314.lock.txt) | Pinned dependencies and tested Windows environment |
| [tests/](tests/) | Independent tests with known inputs and expected outputs |
| [output/](output/) | Frozen manifests and references, vocabulary, audit, review queues, and verification records |
| [data/](data/) | Raw subset and original documents under **raw/samromur_21.05/** and cleaned v1 FLACs under **cleaned/samromur-21.05-20h-v1/** |
| [review/](review/) | Manual listening decisions kept outside the generated outputs |
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

audio_root = Path("data/cleaned/samromur-21.05-20h-v1")
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

Record decisions for the **53 near-silence exclusions and seven retained possible-clipping cases** in the [60-recording manual review log](review/focused_manual_review.tsv). The [generated measurements](output/focused_review.tsv) remain available as evidence. Preprocessing writes the generated outputs separately. Keep leading zeros when importing IDs.

1. **Find and play the FLAC.** Append the row's **audio_path** to the hydrated raw subset or to the root of your full local release. The raw subset includes all seven retained clipping cases and two of the near-silence cases, **0008401** and **0076273**. Use the full release for the other 51 near-silence recordings.
2. **Record audio quality.** Any teammate can listen for audible speech, gaps, competing voices, clipping, noise, and abrupt cutoffs. Enter your name or initials as reviewer and record **usable**, **issue**, or **uncertain**. Add a timestamp and what you heard where useful. A low RMS score can occur in intelligible speech.
3. **Check transcript agreement.** A reviewer who understands Icelandic can find the original prompt in the [release metadata](data/raw/samromur_21.05/metadata.tsv) under the same ID in the **sentence** column. Record **match**, **mismatch**, or **uncertain**, and note the phrase or timestamp. Keep each unreviewed decision **pending**.
4. **Recommend an action.** Note whether to retain the clip, keep it excluded, or propose a change. Keep v1 frozen. A verified membership or transcript correction needs a documented reason and a new dataset version before its use in benchmark comparisons.

For example, run this from the repository root in PowerShell and replace the release path and ID for each clip.

~~~powershell
$release = 'C:\path\to\samromur_21.05'
$id = '0025341'
$row = Import-Csv -LiteralPath './review/focused_manual_review.tsv' -Delimiter ([char]9) |
    Where-Object { $_.id -eq $id }
if (-not $row) { throw "Review ID not found: $id" }
Invoke-Item -LiteralPath (Join-Path $release $row.audio_path)
~~~

The [541-clip evaluation checklist](output/flagged_dev_test_review.tsv) covers 315 test and 226 dev recordings. Its paths start with **audio/** so they resolve from the handoff root. For the original release root, drop that prefix. The [full listening queue](output/listening_queue.tsv) also covers flagged train clips and unflagged controls.

The [focused review queue](output/focused_review.tsv) documents **53 near-silence exclusions and seven selected possible-clipping flags**. All 53 met the RMS threshold, and 15 also met the quiet-fraction threshold. The seven selected clips have 0.1095% to 0.3238% full-scale samples. Quiet speech may still be intelligible and full-scale samples may be inaudible. Use the release paths to listen and record proposed changes by ID.

The exclusion log contains **150 unique utterances**, with 127 train, 18 dev, and five test. Of those, 94 were withheld only for the release validation flag, one for both validation and digital silence, two for the metadata empty status, and 53 for near-silence.

## Cleaning policy

- The audit checks the FLAC bytes against the release MD5 list and fully decodes each recording. It records format, sample count, sound levels, and a SHA-256 of the PCM that includes sample rate and channel count. The hash detects exact duplicates. Edited or near-duplicate recordings need other checks.
- Selection excludes absent metadata or prompt text, validation values other than **1.0**, empty values other than **0**, missing or unreadable audio, decode failures, checksum mismatches, audio outside mono 16 kHz FLAC PCM16, or decoded durations outside 0.5–30 seconds. A missing release checksum is a review flag.
- A quiet window has RMS below −50 dBFS over 20 ms, with the trailing partial window weighted by its sample count. Digital silence means zero frames or all-zero PCM. Near-silence means whole-clip RMS at or below −55 dBFS or at least 98% quiet time. These exclusions are automated decisions pending listening.
- Among otherwise usable signals, flag RMS at or below −35 dBFS, at least 75% quiet time, a quiet edge of at least 1.5 seconds, or at least 0.1% full-scale samples. Flag invalid duration metadata, a measured duration difference greater than 0.05 seconds, or a sample-rate disagreement.
- Withhold exact PCM groups with conflicting scoring text. For matching labels, retain one eligible copy in test/dev/train priority order, breaking ties by seeded order. The same split priority screens later selections against selected session IDs, raw prompts, and scoring references. The IDs establish session separation. A person could have recorded in several sessions.
- The sampler sorts by SHA-256 of **seed/audio_path** and takes the prefix closest to each duration target from its official partition. This revision preserves the existing IDs and references. Any later membership or reference change needs a dataset version and a documented reason before model comparisons use it.

## Regenerate cleaned audio from the raw subset

The raw input is the frozen v1 membership plus the first release utterance by numeric ID from every represented split and session. Membership uses the original metadata and ignores audio flags. It does not construct new splits. The raw files and full release metadata preserve the original bytes. The publisher says human listeners verified the published audio against the prompts. Every selected v1 row carries the release validation flag and its original text alongside derived scoring text. [Upstream release README](data/raw/samromur_21.05/README)

With the raw LFS files hydrated, run:

~~~powershell
& $venvPy -X utf8 subset.py clean
~~~

This reads and decodes all raw inputs, verifies release MD5s, applies the same automated cleaning rules, and fails if a frozen clip or artifact changes. It writes the **13,686** cleaned FLACs under **reproduced/cleaned/samromur-21.05-20h-v1/**, plus rebuilt manifests, references, vocabulary, and logs under **reproduced/subset/**. The output is independent of the committed compressed audio audit. The [scoped removal log](output/raw_subset_removals.tsv) records two near-silence exclusions and one unvalidated utterance, while **3,618** eligible session examples stay outside frozen v1. The 150 historical exclusions describe the entire release.

To reconstruct the inputs from a local full release, run **subset.py stage --source-release** followed by the release directory path. It verifies the three documents and each original FLAC checksum before copying the bounded membership into the raw data directory.

## Replay the frozen artifacts from a fresh checkout

The [upstream README](data/raw/samromur_21.05/README), [metadata.tsv](data/raw/samromur_21.05/metadata.tsv), and [md5s.txt](data/raw/samromur_21.05/md5s.txt) are committed byte for byte. The release MD5 list confirms the README and metadata. All three SHA-256 hashes match the committed audit header.

After cloning and installing dependencies as shown above, run these commands from the repository root.

~~~powershell
& $venvPy -X utf8 -m unittest discover -s tests -v
& $venvPy -X utf8 verify.py --replay-only --root data/raw/samromur_21.05 --output output
~~~

Replay uses the committed gzip audit and the three release documents. It compares regenerated manifests, references, vocabulary, and quality reports byte for byte in a temporary folder. It leaves the existing real-audio verification record intact. The decoded audio measurements come from the stored audit.

To keep a copy of the rebuilt artifacts, use a new output directory.

~~~powershell
New-Item -ItemType Directory -Force reproduced/replay
Copy-Item output/audio_audit.jsonl.gz reproduced/replay/
& $venvPy -X utf8 prepare.py --root data/raw/samromur_21.05 --output reproduced/replay --seed 2105
~~~

## Re-audit the original audio

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

Re-decoding the entire release needs all 100,000 local FLACs. Running **verify.py** without **--replay-only** also opens the selected audio and checks features with the pinned Whisper assets. Both **prepare.py** and **verify.py** accept a plain or gzip audit, with a plain audit taking precedence. Generated artifacts use UTF-8 with LF line endings. Pass **--offline** to **verify.py** once the pinned assets are already local.

Run the handoff verifier against the inner folder extracted from the LF ZIP. It needs the packaged **audio/** split directories, **output/**, and **assets/whisper/** with its checksum file.

~~~powershell
& $venvPy -X utf8 verify_handoff.py --package-root 'C:\path\to\Samromur_ASR_Handoff'
~~~

Earlier Windows verification re-decoded all 100,000 official paths and matched the committed decompressed audit. It checked all selected clips and labels and ran both feature extractors on 30 clips. The earlier handoff check used a fresh extraction and local Whisper assets. The fresh-checkout replay above is a separate check of regeneration from stored audit results.

## Remaining work

Team listening and Icelandic transcript-agreement review remain pending. The [confirmed baseline checklist](BASELINES.md) records the open applicability decisions for named checkpoints. The professor's minimum training duration remains unconfirmed. Each teammate documents and commits their own contribution from their own GitHub account.

## Source and attribution

The repository's original code and documentation use [MIT](LICENSE). The Samrómur audio, prompts, metadata, and derived scoring references use CC BY 4.0.

Samrómur Icelandic Speech 21.05 by David Erik Mollberg et al. is distributed under CC BY 4.0. This repository selects IDs and derives normalized scoring references from that release. Cite Mollberg et al., “Samrómur: Crowd-sourcing Data Collection for Icelandic Speech Recognition,” LREC 2020. [CLARIN release](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/189) · [OpenSLR 112](https://openslr.org/112/) · [LREC paper](https://aclanthology.org/2020.lrec-1.425/) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
