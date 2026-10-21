import unittest
import pandas as pd
from softelo_release.suite import compare_frames


class ReferenceComparisonTests(unittest.TestCase):
    def test_model_order_does_not_change_comparison(self):
        expected = pd.DataFrame({'model': ['a', 'b'], 'elo': [10., 20.]})
        self.assertTrue(compare_frames(expected.iloc[::-1], expected, ['model'], ['elo'])[0]['passed'])

    def test_different_model_set_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_frames(pd.DataFrame({'model': ['a'], 'elo': [10.]}),
                           pd.DataFrame({'model': ['b'], 'elo': [10.]}), ['model'], ['elo'])

    def test_changed_result_fails_without_relative_tolerance(self):
        expected = pd.DataFrame({'model': ['a'], 'elo': [1500.]})
        actual = pd.DataFrame({'model': ['a'], 'elo': [1500.001]})
        self.assertFalse(compare_frames(actual, expected, ['model'], ['elo'])[0]['passed'])

    def test_missing_temperature_is_not_a_match_to_finite(self):
        expected = pd.DataFrame({'model': ['a'], 'beta': [0.5]})
        actual = pd.DataFrame({'model': ['a'], 'beta': [float('nan')]})
        self.assertFalse(compare_frames(actual, expected, ['model'], ['beta'])[0]['passed'])
