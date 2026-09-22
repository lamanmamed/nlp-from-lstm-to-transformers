import unittest
from types import SimpleNamespace

import torch

from nlp_models.text_generation.decoding import beam_search, beam_search_no_repeat, greedy_search


class ToyLanguageModel:
    """Deterministic logits make the search behavior easy to test."""

    def __call__(self, input_ids):
        vocab_size = 5
        logits = torch.full((1, input_ids.shape[1], vocab_size), -8.0)
        last = int(input_ids[0, -1])
        if last == 0:
            logits[0, -1] = torch.tensor([-8.0, 4.0, 3.8, -8.0, -8.0])
        elif last == 1:
            logits[0, -1] = torch.tensor([-8.0, -8.0, 0.0, 0.1, -8.0])
        elif last == 2:
            logits[0, -1] = torch.tensor([-8.0, -8.0, -8.0, 4.0, -8.0])
        else:
            logits[0, -1] = torch.tensor([-8.0, -8.0, -8.0, -8.0, 5.0])
        return SimpleNamespace(logits=logits)


class DecodingTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyLanguageModel()
        self.prompt = torch.tensor([[0]])

    def test_greedy_search_extends_to_eos(self):
        sequence, score = greedy_search(self.model, self.prompt, max_length=6, eos_token_id=4)
        self.assertEqual(int(sequence[0, -1]), 4)
        self.assertIsInstance(score, float)

    def test_beam_search_returns_one_of_the_best_paths(self):
        sequence, score = beam_search(
            self.model, self.prompt, beam_size=2, max_length=6, eos_token_id=4
        )
        self.assertEqual(int(sequence[0, -1]), 4)
        self.assertIsInstance(score, float)

    def test_recent_token_blocking_changes_repetition_options(self):
        sequence, _ = beam_search_no_repeat(
            self.model,
            self.prompt,
            beam_size=2,
            max_length=5,
            recent_token_window=3,
        )
        generated = sequence[0].tolist()
        self.assertEqual(len(generated), len(set(generated)))


if __name__ == "__main__":
    unittest.main()
