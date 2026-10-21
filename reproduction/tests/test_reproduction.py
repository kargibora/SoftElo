"""Small semantic checks; these do not rerun the paper experiments."""
import unittest

import numpy as np
import pandas as pd

from softelo_release import historical_core as core
from softelo_release.reproduce import canonical_frame


class HistoricalProtocolTests(unittest.TestCase):
    def test_collapse_both_human_and_judge_labels(self):
        frame = pd.DataFrame([dict(model_a="a", model_b="b", human_pref=.25,
                                   judge_pref=.75, scores_a={"x": 9.}, scores_b={"x": 1.})])
        prepared = canonical_frame(frame)
        self.assertEqual(prepared.human_pref.iloc[0], .5)
        self.assertEqual(prepared.judge_pref.iloc[0], .5)

    def test_soft_tie_policy_changes_evidence(self):
        frame = pd.DataFrame([dict(model_a="a", model_b="b", judge_pref=.5,
                                   scores_a={"x": 9.}, scores_b={"x": 1.})])
        y, _, w = core.encode_target_soft_outcomes(frame, "a", ["x"], .5, {"b": 0.}, soft_tie_strategy="half")
        self.assertAlmostEqual(float(np.average(y, weights=w)), .5)
        y_gap, _, _ = core.encode_target_soft_outcomes(frame, "a", ["x"], .5, {"b": 0.})
        self.assertGreater(y_gap[0], .98)

    def test_constant_orientation(self):
        frame = pd.DataFrame([dict(model_a="a", model_b="b", judge_pref=0.,
                                   scores_a={}, scores_b={})])
        _, labels, _ = core._build_constant_matches(frame, ["a", "b"], "judge_pref", .75)
        self.assertEqual(labels[0], .75)
        y, _, _ = core._encode_constant_target_for_held_out(frame, "b", .75, {"a": 0.})
        self.assertEqual(y[0], .25)

    def test_historical_quantile_preserved(self):
        scores = np.arange(27.)
        self.assertEqual(core._conformal_quantile(scores, .1), 26.)
        # The conventional exact order statistic would instead be 25.
        self.assertEqual(np.sort(scores)[int(np.ceil(28 * .9)) - 1], 25.)

    def test_ranges_are_percentiles(self):
        repeats = pd.DataFrame({"alpha": [.1] * 5, "empirical_coverage": [.9] * 5,
                                "median_interval_width": [0., 10., 20., 30., 40.],
                                "qhat": [1.] * 5, "n_total": [55] * 5})
        summary = core.build_alpha_summary(repeats, [])
        self.assertEqual(summary.split_median_width_lo.iloc[0], 1.)
        self.assertEqual(summary.split_median_width_hi.iloc[0], 39.)


if __name__ == "__main__":
    unittest.main()
