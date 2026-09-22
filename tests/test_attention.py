import unittest

import torch

from nlp_models.translation.attention import LuongAttention


class AttentionTests(unittest.TestCase):
    def test_attention_weights_sum_to_one_over_source_positions(self):
        torch.manual_seed(0)
        encoder = torch.randn(2, 4, 6)
        decoder = torch.randn(2, 3, 6)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        combined, weights = LuongAttention()(
            encoder, decoder, source_mask=mask, return_weights=True
        )
        self.assertEqual(combined.shape, (2, 3, 12))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))
        self.assertTrue(torch.all(weights[0, :, 3] < 1e-7))


if __name__ == "__main__":
    unittest.main()
