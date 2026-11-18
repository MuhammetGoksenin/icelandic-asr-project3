# Raw and cleaned data

The professor accepts a documented raw subset and Git LFS in this public repository. The raw root is **data/raw/samromur_21.05/** and the cleaned root is **data/cleaned/samromur-21.05-20h-v1/**. The original release documents and the frozen manifests use ordinary Git.

## Raw scope and cleaned output

Raw audio is the original **released** FLAC. The upstream release converted earlier recording formats. The bounded input contains each frozen v1 utterance plus the smallest numeric utterance ID in each represented official split and session. If that ID is already selected, include it once. This adds a first-utterance example from every represented session using metadata order before considering audio quality or validation flags. Unrepresented sessions stay outside the scope. [Subset inventory](output/raw_subset_inventory.tsv)

| Audio | Raw candidates | Raw bytes | Cleaned v1 | Cleaned bytes |
|---|---:|---:|---:|---:|
| Train | 12,677 | 985,998,416 | 10,488 | 809,559,603 |
| Dev | 2,027 | 170,000,879 | 1,314 | 109,222,018 |
| Test | 2,603 | 228,670,716 | 1,884 | 163,760,007 |
| **Total** | **17,307** | **1,384,670,011 (1.29 GiB)** | **13,686** | **1,082,541,628 (1.01 GiB)** |

The [three original release documents](data/raw/samromur_21.05/) add 38,814,399 bytes in ordinary Git. They include the entire release metadata and checksum list. The full upstream release contains 100,000 official recordings and totals 7,924,073,547 bytes (7.38 GiB) including its three companion files. A previous local extraction contained 22 extra FLAC copies under **dev/train/** and three extra copies of the release documents under **dev/**. These paths were outside the official release inventory and the bounded project subset.

The cleaner decodes and screens all 17,307 inputs and removes two near-silence recordings and one unvalidated recording. Another 3,618 session examples passed quality checks and remained outside frozen v1. The historic 150 exclusions refer to the full release. [Scoped cleaning report](output/raw_subset_cleaning.json) · [Scoped removal log](output/raw_subset_removals.tsv)

## Reproduction

| Inputs | What can be reproduced |
|---|---|
| Ordinary Git files and installed dependencies | Run the tests and replay the full-corpus seed-2105 selection from its committed gzip audit and original release documents |
| Raw FLACs fetched through Git LFS | Independently decode the 17,307 candidates, regenerate the cleaned FLACs and six frozen ASR artifacts, and reproduce the subset removal log |
| Cleaned FLACs fetched through Git LFS | Load the frozen v1 clips and use the same committed references for all models |
| Full external release | Re-decode all 100,000 official recordings and reproduce the original full-release audit from the audio |

[Download, cleaning, and verification commands](README.md#download-the-audio)

## Git LFS storage and quota

Both locations contain individually accessible FLACs when hydrated. Their **30,993 paths** refer to **17,307 distinct SHA-256 object IDs** because every cleaned file has the same bytes and object ID as its raw counterpart. The unique payload is **1,384,670,011 bytes**. With both locations hydrated, the working tree has **2,467,211,639 audio bytes**.

An anonymous fresh clone fetched all 17,307 objects by the raw path and hydrated both locations from one cache. Its downloaded FLACs reproduced the scoped audit and cleaning artifacts. [Public-clone verification](output/lfs_verification.json)

The signed-in billing page showed GitHub Free, no LFS usage for the month, and a **$0** LFS budget before upload. GitHub Free includes 10 GiB of LFS storage and 10 GiB of monthly download bandwidth. A raw download consumes about 1.29 GiB and a cleaned-only download about 1.01 GiB on a cold cache. A public download counts toward the owner's bandwidth allowance. [GitHub LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

Redistributed release files and derived references use CC BY 4.0. The project code and original documentation use MIT. [Attribution and terms](LICENSE)
