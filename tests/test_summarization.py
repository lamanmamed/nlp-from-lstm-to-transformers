import random
import unittest

from nlp_models.summarization.preprocessing import (
    add_summarization_prefix,
    sample_keywords,
    token_count_stats,
)


class SummarizationTests(unittest.TestCase):
    def test_t5_prefix(self):
        self.assertEqual(add_summarization_prefix("Some news"), "summarize: Some news")

    def test_keyword_sampling_is_bounded_and_reproducible(self):
        text = "one two three four five six"
        a = sample_keywords(text, count=3, rng=random.Random(7))
        b = sample_keywords(text, count=3, rng=random.Random(7))
        self.assertEqual(a, b)
        self.assertEqual(len(a.split()), 3)

    def test_token_stats(self):
        mean, std = token_count_stats(["one two", "one two three four"])
        self.assertEqual(mean, 3.0)
        self.assertEqual(std, 1.0)


if __name__ == "__main__":
    unittest.main()
