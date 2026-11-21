# MFCC + GMM-HMM contribution

I implemented the project's classical Icelandic ASR baseline. The recognizer extracts 39-dimensional MFCC, delta, and delta-delta vectors from the cleaned 16 kHz audio, fits diagonal Gaussian mixtures for Icelandic characters, and decodes with a character HMM and train-only smoothed bigram transitions. An explicit boundary state handles repeated letters.

The training script reads the frozen manifests and references, fits on all 10,488 train utterances, compares language-model weights on dev, writes per-utterance predictions and run metadata, and requires an explicit freeze confirmation before test evaluation. Dev selected weight 1.5 (WER 101.46%, CER 87.51%). After freezing, the single 1,884-utterance test run scored WER 101.47% and CER 87.74%. Configuration, dependencies, instructions, and a model card are in [`mfcc_gmm_hmm/`](../mfcc_gmm_hmm/). Trained weights are reproducibly accessed through Git LFS under `models/mfcc_gmm_hmm/`.

Uniform transcript-to-frame alignment is an approximate initialization; this baseline does not use a curated Icelandic pronunciation lexicon and has high error rates. A one-train/one-dev utterance smoke run checked the pipeline before the full fit. Full predictions, run timing, and scores are committed under `results/mfcc_gmm_hmm/` for Saleh's scoring and statistical comparison.
