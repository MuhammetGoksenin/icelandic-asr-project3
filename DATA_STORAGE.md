# Raw and cleaned data delivery

The professor requires both raw and cleaned data in the repository. The current GitHub checkout contains code, frozen manifests and references, and a compressed audit. Repository-contained audio delivery remains pending. The separately distributed handoff supports local team use.

## What the pipeline uses

The raw dataset for this experiment is the published Samrómur 21.05 release. Raw means the original **released** FLACs and companion metadata. The upstream release converted earlier recording formats.

| Measured content | Files or recordings | Bytes |
|---|---:|---:|
| Raw official train FLACs | 80,005 | 6,189,063,586 |
| Raw official dev FLACs | 9,995 | 830,660,665 |
| Raw official test FLACs | 10,000 | 865,534,897 |
| Raw metadata.tsv, md5s.txt, and README | 3 | 38,814,399 |
| **Full raw release** | **100,003** | **7,924,073,547 (7.38 GiB)** |
| Cleaned selected train FLACs | 10,488 | 809,559,603 |
| Cleaned selected dev FLACs | 1,314 | 109,222,018 |
| Cleaned selected test FLACs | 1,884 | 163,760,007 |
| **Selected audio** | **13,686** | **1,082,541,628 (1.01 GiB)** |
| Selected manifests and two reference formats | 5 | 7,802,843 |

The largest official FLAC is 247,461 bytes. The revised 20-hour LF handoff ZIP is 1,099,892,127 bytes, including audit data, verification evidence, preprocessing assets, and scripts. The local extraction also contains 22 FLAC copies under **dev/train/** and three duplicated release text files under **dev/**. All 25 matched their official counterparts byte for byte. They were outside the release count. A fresh audit in a separate clean view of the 100,000 official paths matched the committed audit.

The cleaned dataset selects and labels existing FLACs. The frozen manifests under [output/](output/) encode membership and relative source paths. Evaluation references and the train-only vocabulary record how text is represented. Feature extraction happens when the audio is loaded. The cleaned handoff materializes the selected FLACs alongside these artifacts.

## Reproduction from a checkout

| Inputs available | What can be reproduced |
|---|---|
| Checkout and installed dependencies | Run the synthetic tests and inspect IDs, references, audit measurements, and reports |
| Checkout, committed audit, and original three release documents | Replay the full-corpus seed-2105 selection and generate its text artifacts |
| Checkout and full release | Re-decode the corpus, recreate the audit and selection, and verify the selected audio against the release. Feature checks additionally require the pinned Whisper configuration and tokenizer. |
| Extracted LF handoff | Verify every packaged FLAC and both preprocessing paths using the bundled assets. The full release is required to listen to excluded clips and rebuild the audit from audio. |

An independent full-release audit needs all 100,000 FLACs under their official split paths and the three exact release documents. A smaller raw scope requires a documented input snapshot, a new audit and selection version, and a revised reproducibility claim.

## Storage recommendation before an upload

Ask the professor whether **Git LFS objects fetched from this repository** meet the requirement. If accepted and the account has enough included capacity, track a versioned raw release under **data/raw/** in checksummed archive shards. Add separate selected-audio shards under **data/cleaned/** with the frozen manifests. Keep the three raw companion files in ordinary Git. Shards of about 1 GiB or less allow a clean checkout to fetch just the cleaned data for preprocessing or both datasets for grading and reproduction. Commit a shard-to-path index and checksums, then test a fresh clone and LFS fetch before calling data delivery complete. [GitHub LFS checkout behavior and limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

- **Ordinary Git.** Each FLAC fits under the 100 MiB blob limit, but the full release would add roughly 7.38 GiB before a separate cleaned snapshot. GitHub strongly recommends repositories under 5 GB and caps individual pushes at 2 GiB. Splitting pushes would still leave each full clone with a multi-gigabyte Git history. [Large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) · [Push limit](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)
- **Git LFS.** A full raw snapshot plus a separate selected-audio copy totals about **8.39 GiB** before archive overhead. GitHub documents **10 GiB each** of included storage and monthly download bandwidth on individual Free and Pro plans. Four complete downloads would use about **33.6 GiB**. Four cleaned-only downloads would use about **4.03 GiB**. Adding one full raw download brings that to about **11.41 GiB**. Verify the owner's available allowance and expected grading downloads before choosing this route. [Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
- **Account and approval status.** Git LFS is installed locally. The authenticated billing usage request returned HTTP 404, so the remaining owner quota and budget settings are unverified. Confirm both in Billing and licensing, and confirm the professor accepts LFS, before creating or uploading shards. Paid billing was left untouched. If bandwidth is insufficient or LFS is declined, get an agreed storage arrangement or explicit approval for a smaller raw scope. [Account billing settings](https://github.com/settings/billing)

Redistributed release files and derived references carry CC BY 4.0 attribution. The project code uses the repository's MIT license. [Corpus license and attribution](README.md#source-and-attribution)
