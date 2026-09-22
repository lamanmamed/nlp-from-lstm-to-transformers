import unittest

import numpy as np

from nlp_models.social_media.evaluation import average_predictions, mean_squared_error, normalize_rating


class SocialMediaTests(unittest.TestCase):
    def test_ratings_are_normalized_to_zero_one_scale(self):
        np.testing.assert_allclose(normalize_rating([0, 2.5, 5]), [0, 0.5, 1])

    def test_ensemble_uses_mean_prediction(self):
        result = average_predictions([[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
        np.testing.assert_allclose(result, [0.3, 0.7])

    def test_mse(self):
        self.assertAlmostEqual(mean_squared_error([0, 1], [0.2, 0.8]), 0.04)


if __name__ == "__main__":
    unittest.main()
