"""End-to-end tests for SurrogateModel fit / predict / score."""

import numpy as np
import pytest

from surrogate import SurrogateModel


class TestGPTier:
    """Single-output, low-dim → should use GP."""

    def test_fit_predict(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)

        assert model._resolved_type == "gp"
        result = model.predict(X)
        assert set(result) == {"mean", "std", "lower_95", "upper_95"}
        assert result["mean"].shape == (len(X), 1)
        assert (result["std"].values >= 0).all()

    def test_predictions_at_training_points_low_std(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)
        result = model.predict(X)
        # At training points, std should be small relative to output range
        y_range = Y.values.max() - Y.values.min()
        mean_std = result["std"].values.mean()
        assert mean_std < 0.3 * y_range

    def test_score_returns_r2(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)
        scores = model.score()
        assert "r2" in scores
        assert "rmse" in scores
        # LOO R² should be reasonable on a smooth function
        assert scores["r2"]["out"] > 0.5

    def test_sample_shape(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)
        samples = model.sample(X, n_samples=20)
        assert samples.shape == (20, len(X), 1)

    def test_auto_selects_gp_for_single_output_low_dim(self, numeric_only_data):
        X, Y = numeric_only_data
        model = SurrogateModel()  # auto
        model.fit(X, Y)
        assert model._resolved_type == "gp"


class TestDGPTier:
    """Multi-output → should use DGP."""

    @pytest.fixture()
    def fitted_model(self, multi_output_data):
        X, Y = multi_output_data
        model = SurrogateModel(model_type="dgp")
        model.fit(X, Y, n_iter=50, n_imputations=3)
        return model, X, Y

    def test_fit_predict(self, fitted_model):
        model, X, Y = fitted_model
        assert model._resolved_type == "dgp"
        result = model.predict(X)
        assert result["mean"].shape == (len(X), 2)
        assert list(result["mean"].columns) == ["y1", "y2"]

    def test_auto_selects_dgp_for_multi_output(self, multi_output_data):
        X, Y = multi_output_data
        model = SurrogateModel()
        model.fit(X, Y, n_iter=50, n_imputations=3)
        assert model._resolved_type == "dgp"

    def test_sample_preserves_shape(self, fitted_model):
        model, X, _ = fitted_model
        samples = model.sample(X, n_samples=10)
        assert samples.shape[1] == len(X)
        assert samples.shape[2] == 2

    def test_predict_correlation(self, fitted_model):
        model, X, _ = fitted_model
        corr = model.predict_correlation(X.head(3), n_samples=50)
        assert corr.shape == (3, 2, 2)
        # Diagonal should be 1
        for i in range(3):
            np.testing.assert_allclose(np.diag(corr[i]), 1.0, atol=1e-10)

    def test_score(self, fitted_model):
        model, _, _ = fitted_model
        scores = model.score()
        assert "r2" in scores
        assert "y1" in scores["r2"]
        assert "y2" in scores["r2"]


class TestAutoTierSelection:
    def test_high_dim_forces_dgp(self):
        """Even single-output, if encoded dim > threshold → DGP."""
        model = SurrogateModel(dim_threshold=3)
        # Simulate: 5-dim input, single output
        assert model._select_tier(n_outputs=1, enc_dim=5) == "dgp"

    def test_low_dim_single_output_uses_gp(self):
        model = SurrogateModel()
        assert model._select_tier(n_outputs=1, enc_dim=5) == "gp"

    def test_multi_output_always_dgp(self):
        model = SurrogateModel()
        assert model._select_tier(n_outputs=3, enc_dim=2) == "dgp"


class TestNotFittedErrors:
    def test_predict_raises(self):
        model = SurrogateModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            import pandas as pd

            model.predict(pd.DataFrame({"a": [1]}))

    def test_score_raises(self):
        model = SurrogateModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.score()

    def test_suggest_next_raises(self):
        model = SurrogateModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            import pandas as pd

            model.suggest_next(pd.DataFrame({"a": [1]}))
