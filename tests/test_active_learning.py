"""Tests for suggest_next (active learning)."""

import numpy as np
import pandas as pd

from surrogate import SurrogateModel


class TestSuggestNextGP:
    def test_alm_returns_candidates(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)

        rng = np.random.default_rng(7)
        candidates = pd.DataFrame(
            rng.uniform(0, 1, (50, 3)),
            columns=["a", "b", "c"],
        )
        result = model.suggest_next(candidates, method="ALM", n_suggestions=3)

        assert len(result) == 3
        assert "_score" in result.columns
        # Scores should be sorted descending
        assert result["_score"].is_monotonic_decreasing

    def test_mice_returns_candidates(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)

        rng = np.random.default_rng(7)
        candidates = pd.DataFrame(
            rng.uniform(0, 1, (50, 3)),
            columns=["a", "b", "c"],
        )
        result = model.suggest_next(candidates, method="MICE", n_suggestions=1)
        assert len(result) == 1


class TestSuggestNextDGP:
    def test_alm_returns_candidates(self, multi_output_data):
        X, Y = multi_output_data
        model = SurrogateModel(model_type="dgp")
        model.fit(X, Y, n_iter=50, n_imputations=3)

        rng = np.random.default_rng(7)
        candidates = pd.DataFrame(
            rng.uniform(0, 1, (50, 2)),
            columns=["x1", "x2"],
        )
        result = model.suggest_next(candidates, method="ALM", n_suggestions=2)

        assert len(result) == 2
        assert "_score" in result.columns
