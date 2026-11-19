import unittest

import numpy as np

from preprocess import ctc_tokenizer
from training.train_hubert_ctc import VOCAB, CTCCollator, decode_ids


class FakeExtractor:
    return_attention_mask = False

    def pad(self, features, padding, return_attention_mask, return_tensors):
        import torch
        width = max(len(f["input_values"]) for f in features)
        values = torch.zeros(len(features), width)
        mask = torch.zeros(len(features), width, dtype=torch.long)
        for i, f in enumerate(features):
            values[i, :len(f["input_values"])] = torch.tensor(f["input_values"])
            mask[i, :len(f["input_values"])] = 1
        out = {"input_values": values}
        if return_attention_mask:
            out["attention_mask"] = mask
        return out


class HubertCTCTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ctc_tokenizer(VOCAB)

    def test_labels_cover_icelandic_text_without_unknowns(self):
        ids = self.tokenizer("þú ert ágæt", add_special_tokens=False).input_ids
        self.assertNotIn(self.tokenizer.unk_token_id, ids)

    def test_collator_pads_labels_with_ignore_index(self):
        batch = [{"input_values": np.ones(5, np.float32), "labels": [3, 4, 5]},
                 {"input_values": np.ones(3, np.float32), "labels": [6]}]
        out = CTCCollator(FakeExtractor())(batch)
        self.assertEqual(out["labels"].tolist(), [[3, 4, 5], [6, -100, -100]])
        self.assertEqual(tuple(out["input_values"].shape), (2, 5))
        self.assertNotIn("attention_mask", out)  # group-norm base models are trained without a mask

    def test_collator_keeps_mask_when_extractor_asks_for_it(self):
        extractor = FakeExtractor()
        extractor.return_attention_mask = True
        out = CTCCollator(extractor)([{"input_values": np.ones(4, np.float32), "labels": [3]}])
        self.assertIn("attention_mask", out)

    def test_greedy_decoding_collapses_repeats_and_spaces(self):
        t = self.tokenizer
        pad, bar = t.pad_token_id, t.convert_tokens_to_ids("|")
        j, a = t.convert_tokens_to_ids("j"), t.convert_tokens_to_ids("á")
        ids = np.array([[j, j, pad, a, a, bar, bar, j, a]])
        self.assertEqual(decode_ids(t, ids), ["já já"])


if __name__ == "__main__":
    unittest.main()
