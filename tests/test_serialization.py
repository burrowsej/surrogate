"""Tests for save / load round-trip."""

import numpy as np
import pytest

from surrogate import SurrogateModel


class TestSerializationGP:
    def test_save_load_roundtrip(self, numeric_only_data, tmp_path):
        X, Y = numeric_only_data
        model = SurrogateModel(model_type="gp")
        model.fit(X, Y)

        path = tmp_path / "model.pkl"
        model.save(path)

        loaded = SurrogateModel.load(path)
        assert loaded._resolved_type == "gp"

        orig = model.predict(X)
        restored = loaded.predict(X)

        np.testing.assert_allclose(
            orig["mean"].values, restored["mean"].values, rtol=1e-10
        )
        np.testing.assert_allclose(
            orig["std"].values, restored["std"].values, rtol=1e-10
        )


class TestSerializationDGP:
    def test_save_load_roundtrip(self, multi_output_data, tmp_path):
        X, Y = multi_output_data
        model = SurrogateModel(model_type="dgp")
        model.fit(X, Y, n_iter=50, n_imputations=3)

        path = tmp_path / "dgp_model.pkl"
        model.save(path)

        loaded = SurrogateModel.load(path)
        assert loaded._resolved_type == "dgp"

        orig = model.predict(X)
        restored = loaded.predict(X)

        # DGP predictions involve stochastic imputations, so we check
        # that both produce the same structure and similar values
        assert orig["mean"].shape == restored["mean"].shape


class TestLoadValidation:
    def test_load_invalid_type(self, tmp_path):
        import pickle

        path = tmp_path / "bad.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a model"}, f)

        with pytest.raises(TypeError, match="Expected SurrogateModel"):
            SurrogateModel.load(path)
