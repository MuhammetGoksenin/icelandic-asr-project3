# Samrómur 21.05 quality report

Selection seed **2105**. The shipped README and UTF-8 metadata were read.
Audio was inventoried under the three official partitions and decoded end to end with libsndfile 1.2.2. The release checksum list was checked using relative paths.

## Full corpus inventory

| Split | Metadata clips | Decoded clips | Missing audio | Read errors | Decode failures | Audio without metadata | Missing raw / normalized text | Decoded hours | Session IDs | Matching release MD5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 80,005 | 80,005 | 0 | 0 | 0 | 0 | 0 / 0 | 114.5724 | 4,643 | 80,005 |
| dev | 9,995 | 9,995 | 0 | 0 | 0 | 0 | 0 / 0 | 15.2767 | 2,108 | 9,995 |
| test | 10,000 | 10,000 | 0 | 0 | 0 | 0 | 0 / 0 | 15.8585 | 1,641 | 10,000 |

| Split | Sample rate Hz (clips) | Channels (clips) | Format (clips) | Duration min / p05 / median / p95 / max (s) | PCM duplicate groups / clips within split |
|---|---|---|---|---|---:|
| train | 16000 (80,005) | 1 (80,005) | FLAC/PCM_16 (80,005) | 1.068 / 2.880 / 4.920 / 8.209 / 19.980 | 0 / 0 |
| dev | 16000 (9,995) | 1 (9,995) | FLAC/PCM_16 (9,995) | 1.440 / 3.029 / 5.340 / 8.498 / 15.000 | 0 / 0 |
| test | 16000 (10,000) | 1 (10,000) | FLAC/PCM_16 (10,000) | 1.560 / 3.240 / 5.580 / 8.640 / 19.980 | 0 / 0 |

Corpus-wide exact PCM duplicates comprised **0 groups / 0 clips**. Identical file bytes (MD5) comprised **0 groups / 0 clips**.
The PCM hash covers sample rate, channel count and all decoded samples. The comparison covers exact matches. Edited, time-shifted or re-recorded audio may require listening or a separate similarity check.
Release documentation checksums matched for **2/2** files.

## Split overlap

The release uses the session ID as the speaker ID. Anonymous IDs can establish session separation. The same person could record in more than one session. The raw prompt comparison uses exact metadata strings, including a separate exact comparison of the release-normalized column. The reference comparison uses the shared text normalization described in the README.

### Official splits

| Pair | Shared session IDs | Shared raw prompts | Shared release-normalized prompts | Shared scoring references | Shared decoded audio hashes | Shared file MD5s |
|---|---:|---:|---:|---:|---:|---:|
| train/dev | 0 | 0 | 0 | 0 | 0 | 0 |
| train/test | 0 | 0 | 0 | 1 | 0 | 0 |
| dev/test | 0 | 0 | 0 | 0 | 0 | 0 |

### Selected subsets

| Pair | Shared session IDs | Shared raw prompts | Shared release-normalized prompts | Shared scoring references | Shared decoded audio hashes | Shared file MD5s |
|---|---:|---:|---:|---:|---:|---:|
| train/dev | 0 | 0 | 0 | 0 | 0 | 0 |
| train/test | 0 | 0 | 0 | 0 | 0 | 0 |
| dev/test | 0 | 0 | 0 | 0 | 0 | 0 |

The [JSON summary](quality_summary.json) gives both the number of shared keys and the number of affected utterances on each side of each split pair. [prompt_overlap.tsv](prompt_overlap.tsv) shows the corresponding prompts and whether each clip was selected.

## Selected subsets

| Split | Utterances | Hours | Target difference (s) | Session IDs | Unique references | Flagged clips | Excluded paths | Eligible but unsampled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10,488 | 15.00002 | +0.054 | 2,796 | 10,240 | 1,363 | 127 | 69,390 |
| dev | 1,314 | 2.00088 | +3.173 | 963 | 1,314 | 226 | 18 | 8,663 |
| test | 1,884 | 2.99971 | -1.037 | 1,026 | 1,884 | 315 | 5 | 8,111 |

### Excluded paths by reason

| Reason | Train | Dev | Test |
|---|---:|---:|---:|
| `digital_silence` | 1 | 0 | 0 |
| `metadata_empty_or_unknown` | 1 | 0 | 1 |
| `metadata_not_validated` | 83 | 10 | 2 |
| `near_silence` | 43 | 8 | 2 |

### All decoded or inventoried paths with flags

| Reason | Train | Dev | Test |
|---|---:|---:|---:|
| `long_quiet_edge` | 7856 | 1344 | 1309 |
| `low_rms` | 2195 | 355 | 434 |
| `mostly_quiet` | 1163 | 207 | 141 |
| `possible_clipping` | 24 | 0 | 17 |

### Selected paths with flags

| Reason | Train | Dev | Test |
|---|---:|---:|---:|
| `long_quiet_edge` | 1079 | 183 | 245 |
| `low_rms` | 306 | 52 | 83 |
| `mostly_quiet` | 141 | 26 | 18 |
| `possible_clipping` | 5 | 0 | 2 |

The exclusions cover **150 unique paths**. An utterance can have multiple reasons or flags. See [exclusions.tsv](exclusions.tsv) for withheld paths and [listening_queue.tsv](listening_queue.tsv) for reviewable paths and controls.

## Human listening

Signal rules used 20 ms windows below -50 dBFS to measure quiet time. They can flag quiet speech, pauses, noise or full-scale samples without establishing what was spoken. Human listening remains pending for intelligibility, prompt agreement, background voices or noise, audible clipping, cut-off speech, and questionable silence decisions.
The [focused review queue](focused_review.tsv) lists 53 near-silence exclusions and 7 selected possible-clipping flags with measurements. Thresholds are screening decisions. Low-volume speech may still be usable, and full-scale samples may be inaudible.
Review the selected flagged evaluation clips first, then selected flagged train clips and the unflagged controls. Keep the same frozen dev and test IDs for both model pipelines.

Source and license are documented in the [project README](../README.md).
