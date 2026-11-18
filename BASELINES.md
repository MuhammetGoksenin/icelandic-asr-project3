# Project 3 baseline checklist

The professor requires every applicable baseline. This checklist records requirements and model-card evidence. The modeling team owns implementation and results.

| Required baseline | Icelandic scope |
|---|---|
| MFCC or log-mel features with GMM-HMM | Trainable architecture. Use Icelandic acoustic labels and an Icelandic lexicon for the chosen recipe. |
| CTC encoder | Trainable architecture. Provide a CTC head and Icelandic token inventory. |
| Attention encoder-decoder, including hybrid CTC/attention | Trainable with Icelandic output tokens. Track the hybrid requirement explicitly. |
| RNN-T transducer | Trainable with Icelandic output tokens. |
| Language-fine-tuned wav2vec 2.0 with a CTC head | An Icelandic checkpoint is published. Check for overlap between its training data and this project's test set when interpreting results. [Language and Voice Lab model card](https://huggingface.co/language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h) |
| Language-fine-tuned HuBERT with a CTC head | An Icelandic fine-tuned checkpoint has yet to be identified for this project. Meta's commonly used English CTC checkpoint is documented for LibriSpeech. HuBERT can be adapted with an Icelandic CTC head. [Meta HuBERT CTC model card](https://huggingface.co/facebook/hubert-large-ls960-ft) |
| Multilingual Whisper | OpenAI's multilingual tokenizer lists Icelandic as **is**. [Multilingual Whisper card](https://huggingface.co/openai/whisper-tiny) · [OpenAI language mapping](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) |
| nvidia/canary-1b-v2 | Icelandic is outside NVIDIA's documented language list for this release. Confirm the professor's expected treatment of this checkpoint. [NVIDIA model card](https://huggingface.co/nvidia/canary-1b-v2) |
| nvidia/parakeet-tdt-0.6b-v3 | Icelandic is outside NVIDIA's documented language list for this release. Confirm the expected treatment. [NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B | Icelandic is outside Qwen's documented language list for both releases. Confirm the expected treatment. [1.7B model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · [0.6B model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |

A bare encoder needs an ASR head and tokenizer to transcribe. Keep the results of published fine-tuned checkpoints separate from the team's own training or adaptation. Map each run to the checklist without double-counting a result across overlapping categories. Keep the named checkpoints on the list pending the course's applicability ruling.

Use the frozen v1 train split for fitting, dev for checkpoint and hyperparameter selection, and test after those decisions are frozen. Score against the same evaluation IDs and references. The professor's minimum training duration remains unconfirmed.
