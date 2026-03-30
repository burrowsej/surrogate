"""SurrogateModel — high-level DataFrame-in/DataFrame-out surrogate modelling."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .architectures import build_dgp, build_gp
from .preprocessing import DataFrameEncoder, OutputScaler

logger = logging.getLogger(__name__)


class SurrogateModel:
    """A surrogate model wrapping dgpsi with a pandas-native interface.

    Automatically selects between a single-layer GP (fast, single-output, low-dim)
    and a Deep GP (multi-output, high-dim, nonlinear) based on the data.

    Args:
        categorical_columns: Column names to treat as categorical.
            Auto-detected if ``None``.
        model_type: Force a model tier (``"gp"`` or ``"dgp"``) or let the
            class choose with ``"auto"``.
        depth: Number of DGP layers (ignored when ``model_type="gp"``).
        kernel: Kernel name — ``"matern25"`` or ``"sexp"``.
        dim_threshold: Encoded input dimensions above which *auto* selects
            DGP even for single-output problems.
    """

    def __init__(
        self,
        categorical_columns: list[str] | None = None,
        model_type: Literal["auto", "gp", "dgp"] = "auto",
        depth: int = 2,
        kernel: str = "matern25",
        dim_threshold: int = 15,
    ):
        self.categorical_columns = categorical_columns
        self.model_type = model_type
        self.depth = depth
        self.kernel = kernel
        self.dim_threshold = dim_threshold

        # Populated by fit()
        self._encoder: DataFrameEncoder | None = None
        self._scaler: OutputScaler | None = None
        self._resolved_type: str | None = None
        self._gp_model = None  # dgpsi.gp  (Tier 1)
        self._dgp_emulator = None  # dgpsi.emulator (Tier 2)
        self._dgp_obj = None  # dgpsi.dgp object (kept for VIGF & LOO)
        self._train_X: np.ndarray | None = None
        self._train_Y: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        n_iter: int = 500,
        ess_burn: int = 10,
        parallel: bool = True,
        n_imputations: int = 10,
    ) -> SurrogateModel:
        """Train the surrogate on input/output DataFrames.

        Args:
            X: Input (independent) variables. May contain both numeric and
                categorical columns.
            Y: Output (dependent) variables. All columns must be continuous
                (numeric); categorical outputs are not supported.
            n_iter: SEM iterations (DGP) or ignored (GP).
            ess_burn: ESS burn-in per SEM step.
            parallel: Use multi-core training for DGP.
            n_imputations: Number of stochastic imputations for the DGP emulator.
        """
        # Encode inputs
        self._encoder = DataFrameEncoder(categorical_columns=self.categorical_columns)
        X_enc = self._encoder.fit_transform(X)

        # Scale outputs
        self._scaler = OutputScaler()
        Y_enc = self._scaler.fit_transform(Y)

        self._train_X = X_enc
        self._train_Y = Y_enc

        # Tier selection
        n_outputs = Y_enc.shape[1]
        enc_dim = self._encoder.encoded_dim
        self._resolved_type = self._select_tier(n_outputs, enc_dim)

        logger.info(
            "Fitting %s  (inputs=%d encoded dims, outputs=%d)",
            self._resolved_type.upper(),
            enc_dim,
            n_outputs,
        )

        if self._resolved_type == "gp":
            self._gp_model = build_gp(X_enc, Y_enc, kernel_name=self.kernel)
        else:
            self._dgp_emulator, self._dgp_obj = build_dgp(
                X_enc,
                Y_enc,
                depth=self.depth,
                kernel_name=self.kernel,
                n_iter=n_iter,
                ess_burn=ess_burn,
                parallel=parallel,
                n_imputations=n_imputations,
            )

        self._fitted = True
        return self

    def _select_tier(self, n_outputs: int, enc_dim: int) -> str:
        if self.model_type in ("gp", "dgp"):
            return self.model_type
        # auto
        if n_outputs > 1:
            return "dgp"
        if enc_dim > self.dim_threshold:
            return "dgp"
        return "gp"

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Return mean, std, and 95% credible-interval bounds.

        Args:
            X: Input variables to predict at.

        Returns:
            Dict with keys ``"mean"``, ``"std"``, ``"lower_95"``,
            ``"upper_95"``. Each value is a DataFrame in original output scale.
        """
        self._check_fitted()
        X_enc = self._encoder.transform(X)
        mean_enc, var_enc = self._predict_raw(X_enc)
        std_enc = np.sqrt(np.clip(var_enc, 0, None))

        mean_df = self._scaler.inverse_transform(mean_enc)
        std_orig = self._scaler.inverse_transform_std(std_enc)
        std_df = pd.DataFrame(std_orig, columns=self._scaler._columns)

        lower = mean_df.values - 1.96 * std_orig
        upper = mean_df.values + 1.96 * std_orig

        return {
            "mean": mean_df,
            "std": std_df,
            "lower_95": pd.DataFrame(lower, columns=self._scaler._columns),
            "upper_95": pd.DataFrame(upper, columns=self._scaler._columns),
        }

    def _predict_raw(self, X_enc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) arrays in *scaled* output space."""
        if self._resolved_type == "gp":
            mu, var = self._gp_model.predict(X_enc)
            return mu, var
        else:
            mu, var = self._dgp_emulator.predict(X_enc, method="mean_var")
            # dgpsi returns a list per output for DGP emulator
            if isinstance(mu, list):
                mu = np.column_stack(mu)
                var = np.column_stack(var)
            return np.asarray(mu), np.asarray(var)

    # ------------------------------------------------------------------
    # Sample (jointly drawn across outputs)
    # ------------------------------------------------------------------
    def sample(self, X: pd.DataFrame, n_samples: int = 50) -> np.ndarray:
        """Draw joint posterior samples.

        Args:
            X: Input variables to sample at.
            n_samples: Number of posterior draws.

        Returns:
            Array of shape ``(n_samples, n_points, n_outputs)`` in original
            output scale. Cross-output correlation is preserved within each
            sample.
        """
        self._check_fitted()
        X_enc = self._encoder.transform(X)
        n_points = X_enc.shape[0]
        n_out = self._train_Y.shape[1]

        if self._resolved_type == "gp":
            raw = self._gp_model.predict(X_enc, method="sampling", sample_size=n_samples)
            # raw shape: (n_points, n_samples) for single-output GP
            samples = raw.T[:, :, np.newaxis]  # → (n_samples, n_points, 1)
        else:
            raw = self._dgp_emulator.predict(
                X_enc, method="sampling", sample_size=n_samples
            )
            # raw is list of k arrays, each (n_points, N*sample_size)
            total = raw[0].shape[1]
            samples = np.empty((total, n_points, n_out))
            for k, arr in enumerate(raw):
                samples[:, :, k] = arr.T

        # Inverse-scale each sample
        mean = self._scaler._mean
        std = self._scaler._std
        samples = samples * std + mean
        return samples

    # ------------------------------------------------------------------
    # Output correlation
    # ------------------------------------------------------------------
    def predict_correlation(
        self, X: pd.DataFrame, n_samples: int = 200
    ) -> np.ndarray:
        """Estimate output correlation matrices from posterior samples.

        Args:
            X: Input variables to estimate correlations at.
            n_samples: Number of posterior draws used for estimation.

        Returns:
            Array of shape ``(n_points, n_outputs, n_outputs)``.
        """
        samples = self.sample(X, n_samples=n_samples)
        n_points = samples.shape[1]
        n_out = samples.shape[2]
        corr = np.empty((n_points, n_out, n_out))
        for i in range(n_points):
            s = samples[:, i, :]  # (n_samples, n_out)
            c = np.corrcoef(s, rowvar=False)
            corr[i] = c if c.ndim == 2 else np.ones((1, 1))
        return corr

    # ------------------------------------------------------------------
    # Score (LOO cross-validation)
    # ------------------------------------------------------------------
    def score(self) -> dict:
        """Leave-one-out cross-validation diagnostics.

        Returns:
            Dict with keys ``"r2"`` and ``"rmse"``, each a dict mapping
            output column names to float values.
        """
        self._check_fitted()
        cols = self._scaler._columns
        n = self._train_X.shape[0]

        if self._resolved_type == "gp":
            mu, var = self._gp_model.loo()
            mu = np.asarray(mu).reshape(-1, 1)
            var = np.asarray(var).reshape(-1, 1)
        else:
            result = self._dgp_emulator.loo(self._train_X)
            if isinstance(result, tuple):
                mu, var = result
                if isinstance(mu, list):
                    mu = np.column_stack(mu)
                    var = np.column_stack(var)
            else:
                mu = np.column_stack(result) if isinstance(result, list) else result
                var = None
            mu = np.asarray(mu)

        Y = self._train_Y
        residuals = Y - mu
        ss_res = np.sum(residuals**2, axis=0)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
        r2 = 1.0 - ss_res / np.where(ss_tot == 0, 1.0, ss_tot)
        rmse = np.sqrt(ss_res / n)

        # Inverse-scale RMSE to original units
        rmse_orig = self._scaler.inverse_transform_std(rmse)

        return {
            "r2": dict(zip(cols, r2.tolist())),
            "rmse": dict(zip(cols, rmse_orig.tolist())),
        }

    # ------------------------------------------------------------------
    # Active learning — suggest next evaluation point(s)
    # ------------------------------------------------------------------
    def suggest_next(
        self,
        candidates: pd.DataFrame,
        method: Literal["ALM", "MICE", "VIGF"] = "ALM",
        n_suggestions: int = 1,
    ) -> pd.DataFrame:
        """Select the most informative candidate point(s) to evaluate next.

        Args:
            candidates: Candidate input designs in original (unencoded) space.
            method: Sequential design criterion.
            n_suggestions: Number of points to return.

        Returns:
            Top-*n* candidates as a DataFrame with an extra ``"_score"``
            column.
        """
        self._check_fitted()
        X_cand = self._encoder.transform(candidates)

        if self._resolved_type == "gp":
            scores = self._gp_model.metric(
                X_cand, method=method, score_only=True
            )
        else:
            scores = self._dgp_emulator.metric(
                X_cand,
                method=method,
                obj=self._dgp_obj if method == "VIGF" else None,
                score_only=True,
            )

        scores = np.asarray(scores)
        if scores.ndim == 2:
            agg_scores = scores.mean(axis=1)
        else:
            agg_scores = scores.ravel()

        top_idx = np.argsort(agg_scores)[::-1][:n_suggestions]

        result = candidates.iloc[top_idx].copy()
        result["_score"] = agg_scores[top_idx]
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    _MODELS_DIR = Path("models")

    def save(self, path: str | Path | None = None) -> None:
        """Persist the fitted model to disk via pickle.

        Args:
            path: File path to write the model to. If a bare filename is given
                (no directory component), the file is saved inside the
                ``models/`` directory. Defaults to ``models/surrogate.pkl``.
        """
        self._check_fitted()
        path = Path(path) if path is not None else self._MODELS_DIR / "surrogate.pkl"
        if path.parent == Path("."):
            path = self._MODELS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> SurrogateModel:
        """Load a previously saved model.

        Args:
            path: File path to load the model from.

        Returns:
            The loaded ``SurrogateModel`` instance.
        """
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SurrogateModel, got {type(obj).__name__}")
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        tier = self._resolved_type or "?"
        return f"SurrogateModel(type={tier}, status={status})"
