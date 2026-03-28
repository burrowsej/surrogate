import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DataFrameEncoder:
    """Encodes a mixed-type DataFrame into a numeric numpy array for dgpsi.

    Continuous columns are scaled to [0, 1] via MinMaxScaler.
    Categorical columns (dtype object/category) are one-hot encoded.

    Args:
        categorical_columns: Column names to treat as categorical.
            Auto-detected if ``None``.
    """

    def __init__(self, categorical_columns=None):
        self._categorical_columns = categorical_columns
        self._continuous_columns = None
        self._fitted = False
        self._scaler = None
        self._ohe = None
        self._original_columns = None
        self._ohe_feature_names = None

    @property
    def encoded_dim(self):
        if not self._fitted:
            raise RuntimeError("DataFrameEncoder has not been fitted yet.")
        n_cont = len(self._continuous_columns)
        n_ohe = self._ohe.get_feature_names_out().shape[0] if self._ohe is not None else 0
        return n_cont + n_ohe

    @property
    def continuous_columns(self):
        return list(self._continuous_columns)

    @property
    def categorical_columns(self):
        return list(self._categorical_columns) if self._categorical_columns else []

    def fit(self, X: pd.DataFrame):
        self._original_columns = list(X.columns)

        if self._categorical_columns is None:
            self._categorical_columns = [
                c
                for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])
            ]
        self._continuous_columns = [c for c in X.columns if c not in self._categorical_columns]

        if self._continuous_columns:
            self._scaler = MinMaxScaler()
            self._scaler.fit(X[self._continuous_columns].values)

        if self._categorical_columns:
            self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="error")
            self._ohe.fit(X[self._categorical_columns])
            self._ohe_feature_names = list(self._ohe.get_feature_names_out())

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DataFrameEncoder has not been fitted yet. Call fit() first.")

        parts = []
        if self._continuous_columns:
            scaled = self._scaler.transform(X[self._continuous_columns].values)
            parts.append(scaled)
        if self._ohe is not None:
            encoded = self._ohe.transform(X[self._categorical_columns])
            parts.append(encoded)

        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DataFrameEncoder has not been fitted yet.")

        n_cont = len(self._continuous_columns)
        result = pd.DataFrame()

        if self._continuous_columns:
            cont_arr = arr[:, :n_cont]
            inv = self._scaler.inverse_transform(cont_arr)
            for i, col in enumerate(self._continuous_columns):
                result[col] = inv[:, i]

        if self._ohe is not None:
            ohe_arr = arr[:, n_cont:]
            cat_decoded = self._ohe.inverse_transform(ohe_arr)
            for i, col in enumerate(self._categorical_columns):
                result[col] = cat_decoded[:, i]

        return result[self._original_columns]

    def get_bounds(self) -> np.ndarray:
        """Return bounds in encoded space.

        Returns:
            Array of shape ``(2, encoded_dim)`` with ``[lower, upper]`` rows.
        """
        n_cont = len(self._continuous_columns)
        n_ohe = len(self._ohe_feature_names) if self._ohe is not None else 0
        d = n_cont + n_ohe
        lower = np.zeros(d)
        upper = np.ones(d)
        return np.stack([lower, upper], axis=0)


class OutputScaler:
    """Standardises output columns (zero mean, unit variance) with inverse support."""

    def __init__(self):
        self._mean = None
        self._std = None
        self._columns = None
        self._fitted = False

    def fit(self, Y: pd.DataFrame):
        self._columns = list(Y.columns)
        self._mean = Y.values.mean(axis=0)
        self._std = Y.values.std(axis=0, ddof=0)
        self._std[self._std == 0] = 1.0
        self._fitted = True
        return self

    def transform(self, Y: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OutputScaler has not been fitted yet.")
        return (Y.values - self._mean) / self._std

    def fit_transform(self, Y: pd.DataFrame) -> np.ndarray:
        return self.fit(Y).transform(Y)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("OutputScaler has not been fitted yet.")
        values = arr * self._std + self._mean
        return pd.DataFrame(values, columns=self._columns)

    def inverse_transform_std(self, std_arr: np.ndarray) -> np.ndarray:
        """Scale standard deviations back to original output units.

        Args:
            std_arr: Standard deviations in scaled space.

        Returns:
            Standard deviations in original output units.
        """
        return std_arr * self._std
