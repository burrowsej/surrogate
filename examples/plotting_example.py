"""Plotting example - visualise model diagnostics.

Requires the optional plot dependencies::

    uv sync --group plot

This file uses ``# %%`` cell markers so it can be run interactively as a
notebook in VS Code. It also runs as a normal script.

Note:
    dgpsi uses multiprocessing internally. When running ``.fit()`` with
    ``parallel=True`` (the default) outside an ``if __name__ == '__main__':``
    guard, you may see a ``RuntimeError`` about the bootstrapping phase.
    Either wrap the call in a guard or pass ``parallel=False``.
"""

# %%
import numpy as np
import pandas as pd

from surrogate import SurrogateModel
from surrogate.plotting import (
    calibration_plot,
    correlation_heatmap,
    parity_plot,
    slice_plot,
)

# %% [markdown]
# ## 1. Generate synthetic data and fit model

# %%
rng = np.random.default_rng(42)
n = 40
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)
cat = rng.choice(["low", "high"], size=n)

base = np.sin(5 * x1) + x2**2
y1 = base + np.where(cat == "high", 3.0, 0.0)
y2 = 2.0 * base + 0.5 * x1 + rng.normal(0, 0.2, n)

X = pd.DataFrame({"x1": x1, "x2": x2, "category": cat})
Y = pd.DataFrame({"output_1": y1, "output_2": y2})

model = SurrogateModel()
model.fit(X, Y, n_iter=100, n_imputations=5, parallel=False)
print(f"Model: {model}")

# %% [markdown]
# ## 2. Parity plot (LOO predicted vs actual)

# %%
fig = parity_plot(model)
fig.savefig("models/parity.png", dpi=150)

# %% [markdown]
# ## 3. Uncertainty calibration

# %%
fig = calibration_plot(model)
fig.savefig("models/calibration.png", dpi=150)

# %% [markdown]
# ## 4. 1D slice plot - sweep x1

# %%
X_centre = pd.DataFrame({"x1": [0.5], "x2": [0.5], "category": ["low"]})
fig = slice_plot(model, X_centre, column="x1")
fig.savefig("models/slice_x1.png", dpi=150)

# %% [markdown]
# ## 5. Output correlation heatmap

# %%
X_test = pd.DataFrame(
    {
        "x1": [0.3, 0.7],
        "x2": [0.5, 0.2],
        "category": ["low", "high"],
    }
)
fig = correlation_heatmap(model, X_test, point_index=0)
fig.savefig("models/correlation.png", dpi=150)
