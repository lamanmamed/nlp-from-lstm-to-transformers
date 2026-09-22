import unittest

import torch

from nlp_models.sentiment.data import combine_review_and_aspect, polarity_to_id
from nlp_models.sentiment.pooling import masked_mean


class SentimentTests(unittest.TestCase):
    def test_review_and_aspect_are_separated_explicitly(self):
        self.assertEqual(
            combine_review_and_aspect("great food", "food"),
            "great food [SEP] food",
        )

    def test_polarity_mapping(self):
        self.assertEqual([polarity_to_id(x) for x in ["negative", "neutral", "positive"]], [0, 1, 2])

    def test_masked_mean_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = masked_mean(hidden, mask)
        torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
