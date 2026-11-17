from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from prepare import MAX_SECONDS, RATE, read_jsonl

WHISPER_MODEL = "openai/whisper-tiny"
WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


def resolve_audio(root, record):
    path = PurePosixPath(record["audio_path"])
    expected = (record["split"], record["speaker_session_id"],
                f"{record['speaker_session_id']}-{record['id']}.flac")
    if path.parts != expected or path.as_posix() != record["audio_path"]:
        raise ValueError(f"Invalid manifest path: {record['audio_path']}")
    root = Path(root).resolve()
    audio = root.joinpath(*path.parts).resolve()
    if not audio.is_relative_to(root):
        raise ValueError(f"Manifest path escapes corpus: {path}")
    return audio


def load_example(root, record):
    """Read a selected utterance into a one-dimensional 16 kHz float32 waveform."""
    path = resolve_audio(root, record)
    with sf.SoundFile(path) as source:
        if (source.samplerate, source.channels, source.format, source.subtype) != (RATE, 1, "FLAC", "PCM_16"):
            raise ValueError(f"Unexpected audio format: {record['audio_path']}")
        waveform = source.read(dtype="float32")
    if len(waveform) != record["num_samples"] or record["sample_rate"] != RATE:
        raise ValueError(f"Audio length or sample rate differs from manifest: {record['audio_path']}")
    return waveform


def iter_examples(root, manifest):
    for record in read_jsonl(Path(manifest)):
        yield record | {"waveform": load_example(root, record)}


def whisper_components(checkpoint=WHISPER_MODEL, revision=None, *, local_files_only=False):
    """Load a multilingual checkpoint's feature configuration and Icelandic tokenizer."""
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    if revision is None and checkpoint == WHISPER_MODEL:
        revision = WHISPER_REVISION
    source_args = {"revision": revision, "local_files_only": local_files_only}
    return (
        WhisperFeatureExtractor.from_pretrained(checkpoint, **source_args),
        AutoTokenizer.from_pretrained(checkpoint, language="icelandic", task="transcribe", **source_args),
    )


def whisper_example(example, extractor, tokenizer=None):
    waveform = example["waveform"]
    if len(waveform) > min(MAX_SECONDS, extractor.chunk_length) * RATE:
        raise ValueError(f"Whisper chunk exceeded for {example['id']}")
    features = extractor(waveform, sampling_rate=RATE, return_attention_mask=True, return_tensors="np")
    result = {
        "id": example["id"],
        "input_features": features.input_features[0],
        "attention_mask": features.attention_mask[0],
        "text": example["text"],
    }
    if tokenizer is not None:
        result["labels"] = tokenizer(example["text"]).input_ids
    return result


def ctc_tokenizer(vocab_path):
    """Build a CTC tokenizer for the included train-only character vocabulary."""
    from transformers import Wav2Vec2CTCTokenizer

    return Wav2Vec2CTCTokenizer(vocab_file=str(vocab_path), unk_token="[UNK]", pad_token="[PAD]",
                               bos_token=None, eos_token=None, word_delimiter_token="|")


def wav2vec2_example(example, extractor, tokenizer=None):
    features = extractor(example["waveform"], sampling_rate=RATE, return_attention_mask=True, return_tensors="np")
    result = {
        "id": example["id"],
        "input_values": np.asarray(features.input_values[0], dtype=np.float32),
        "attention_mask": features.attention_mask[0],
        "text": example["text"],
    }
    if tokenizer is not None:
        result["labels"] = tokenizer(example["text"], add_special_tokens=False).input_ids
    return result
