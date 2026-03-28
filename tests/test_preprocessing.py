"""Tests for DataFrameEncoder and OutputScaler."""

import numpy as np
import pandas as pd
import pytest

from surrogate.preprocessing import DataFrameEncoder, OutputScaler


class TestDataFrameEncoder:
    def test_fit_transform_numeric_only(self, numeric_only_data):
        X, _ = numeric_only_data
        enc = DataFrameEncoder()
        arr = enc.fit_transform(X)

        assert arr.shape == (len(X), 3)
        assert arr.min() >= 0.0 - 1e-9
        assert arr.max() <= 1.0 + 1e-9
        assert enc.encoded_dim == 3

    def test_fit_transform_with_categorical(self, single_output_data):
        X, _ = single_output_data
        enc = DataFrameEncoder()
        arr = enc.fit_transform(X)

        # 2 continuous + 2 one-hot categories (A, B)
        assert arr.shape[1] == 4
        assert enc.encoded_dim == 4
        assert enc.categorical_columns == ["cat"]
        assert enc.continuous_columns == ["x1", "x2"]

    def test_inverse_transform_roundtrip(self, single_output_data):
        X, _ = single_output_data
        enc = DataFrameEncoder()
        arr = enc.fit_transform(X)
        X_back = enc.inverse_transform(arr)

        assert list(X_back.columns) == list(X.columns)
        pd.testing.assert_frame_equal(
            X_back[["x1", "x2"]].astype(float),
            X[["x1", "x2"]].astype(float),
            atol=1e-10,
        )
        assert (X_back["cat"].values == X["cat"].values).all()

    def test_explicit_categorical_columns(self, single_output_data):
        X, _ = single_output_data
        enc = DataFrameEncoder(categorical_columns=["cat"])
        enc.fit(X)
        assert enc.categorical_columns == ["cat"]

    def test_not_fitted_raises(self):
        enc = DataFrameEncoder()
        with pytest.raises(RuntimeError, match="not been fitted"):
            enc.transform(pd.DataFrame({"a": [1]}))

    def test_get_bounds(self, single_output_data):
        X, _ = single_output_data
        enc = DataFrameEncoder()
        enc.fit(X)
        bounds = enc.get_bounds()
        assert bounds.shape == (2, enc.encoded_dim)
        np.testing.assert_array_equal(bounds[0], 0.0)
        np.testing.assert_array_equal(bounds[1], 1.0)


class TestOutputScaler:
    def test_fit_transform_inverse(self, single_output_data):
        _, Y = single_output_data
        sc = OutputScaler()
        arr = sc.fit_transform(Y)

        np.testing.assert_allclose(arr.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(arr.std(axis=0, ddof=0), 1.0, atol=1e-10)

        Y_back = sc.inverse_transform(arr)
        pd.testing.assert_frame_equal(Y_back, Y, atol=1e-10)

    def test_inverse_transform_std(self, single_output_data):
        _, Y = single_output_data
        sc = OutputScaler()
        sc.fit(Y)

        std_scaled = np.array([1.0])
        std_orig = sc.inverse_transform_std(std_scaled)
        expected = Y.values.std(axis=0, ddof=0)
        np.testing.assert_allclose(std_orig, expected, rtol=1e-10)

    def test_constant_column(self):
        Y = pd.DataFrame({"c": [5.0, 5.0, 5.0]})
        sc = OutputScaler()
        arr = sc.fit_transform(Y)
        np.testing.assert_array_equal(arr, 0.0)
