"""Shared test fixtures  -- synthetic data generators."""

import numpy as np
import pandas as pd
import pytest


def _branin(x1, x2):
    """Modified Branin function (inputs expected in [0, 1])."""
    x1_s = 15 * x1 - 5
    x2_s = 15 * x2
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2_s - b * x1_s**2 + c * x1_s - r) ** 2 + s * (1 - t) * np.cos(x1_s) + s


@pytest.fixture()
def single_output_data():
    """Low-dim single-output dataset (Branin + categorical)."""
    rng = np.random.default_rng(42)
    n = 30
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    cat = rng.choice(["A", "B"], size=n)

    y = _branin(x1, x2) + np.where(cat == "A", 0.0, 5.0)

    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    Y = pd.DataFrame({"y": y})
    return X, Y


@pytest.fixture()
def multi_output_data():
    """Multi-output dataset (2 correlated outputs)."""
    rng = np.random.default_rng(42)
    n = 30
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)

    y1 = _branin(x1, x2)
    y2 = 2.0 * y1 + 10 * x1 + rng.normal(0, 0.5, n)

    X = pd.DataFrame({"x1": x1, "x2": x2})
    Y = pd.DataFrame({"y1": y1, "y2": y2})
    return X, Y


@pytest.fixture()
def numeric_only_data():
    """Purely numeric low-dim data (no categoricals)."""
    rng = np.random.default_rng(99)
    n = 25
    x = rng.uniform(0, 1, (n, 3))
    y = np.sin(x[:, 0] * 5) + x[:, 1] ** 2 - x[:, 2]
    X = pd.DataFrame(x, columns=["a", "b", "c"])
    Y = pd.DataFrame({"out": y})
    return X, Y
