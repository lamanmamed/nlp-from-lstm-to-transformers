import unittest

import torch

from nlp_models.translation.data import Vocabulary, pad_sequences
from nlp_models.translation.model import Seq2SeqTranslator


class TranslationTests(unittest.TestCase):
    def test_padding(self):
        result = pad_sequences([[1, 2], [3]], pad_value=0)
        self.assertEqual(result.tolist(), [[1, 2], [3, 0]])

    def test_vocabulary_has_special_tokens(self):
        vocab = Vocabulary([["hello"]] * 11, extra_tokens=("<start>", "<end>"))
        self.assertEqual(vocab.pad_id, 0)
        self.assertIn("hello", vocab.token_to_id)
        self.assertIn("<start>", vocab.token_to_id)

    def test_all_translation_configurations_keep_target_shape(self):
        source = torch.tensor([[2, 3, 0], [4, 5, 6]])
        target = torch.tensor([[1, 2], [1, 3]])
        for attention in [False, True]:
            for bidirectional in [False, True]:
                model = Seq2SeqTranslator(
                    10,
                    12,
                    embedding_size=8,
                    hidden_size=10,
                    dropout=0.0,
                    use_attention=attention,
                    bidirectional_encoder=bidirectional,
                )
                logits = model(source, target)
                self.assertEqual(logits.shape, (2, 2, 12))


if __name__ == "__main__":
    unittest.main()
