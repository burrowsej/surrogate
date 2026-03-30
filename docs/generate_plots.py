"""Generate plot images for the documentation site.

Run with:  uv run python docs/generate_plots.py

Requires the plot dependency group:  uv sync --group plot
"""

from pathlib import Path

import numpy as np
import pandas as pd

from surrogate import SurrogateModel
from surrogate.plotting import (
    calibration_plot,
    convergence_plot,
    correlation_heatmap,
    parity_plot,
    slice_plot,
)

OUT = Path(__file__).parent / "assets" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 180

# -- Synthetic data and model --------------------------------------------------

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
print(f"Fitted: {model}")

# -- Parity plot ---------------------------------------------------------------

fig = parity_plot(model)
fig.savefig(OUT / "parity.png", dpi=DPI)
print("Saved parity.png")

# -- Calibration plot -----------------------------------------------------------

fig = calibration_plot(model)
fig.savefig(OUT / "calibration.png", dpi=DPI)
print("Saved calibration.png")

# -- Slice plot -----------------------------------------------------------------

X_centre = pd.DataFrame({"x1": [0.5], "x2": [0.5], "category": ["low"]})
fig = slice_plot(model, X_centre, column="x1")
fig.savefig(OUT / "slice.png", dpi=DPI)
print("Saved slice.png")

# -- Correlation heatmap --------------------------------------------------------

X_test = pd.DataFrame(
    {"x1": [0.3, 0.7], "x2": [0.5, 0.2], "category": ["low", "high"]}
)
fig = correlation_heatmap(model, X_test, point_index=0)
fig.savefig(OUT / "correlation.png", dpi=DPI)
print("Saved correlation.png")

# -- Convergence plot (mock 5-step active learning loop) ------------------------

metrics = []
X_al = X.copy()
Y_al = Y.copy()
model_al = SurrogateModel()
model_al.fit(X_al, Y_al, n_iter=100, n_imputations=5, parallel=False)
metrics.append(model_al.score())

candidates = pd.DataFrame(
    {
        "x1": rng.uniform(0, 1, 200),
        "x2": rng.uniform(0, 1, 200),
        "category": rng.choice(["low", "high"], size=200),
    }
)

for i in range(5):
    suggestion = model_al.suggest_next(candidates, method="ALM", n_suggestions=1)
    x_new = suggestion.drop(columns=["_score"])
    y_new = pd.DataFrame(
        {
            "output_1": np.sin(5 * x_new["x1"])
            + x_new["x2"] ** 2
            + np.where(x_new["category"] == "high", 3.0, 0.0),
            "output_2": 2.0 * (np.sin(5 * x_new["x1"]) + x_new["x2"] ** 2)
            + 0.5 * x_new["x1"],
        }
    )
    X_al = pd.concat([X_al, x_new], ignore_index=True)
    Y_al = pd.concat([Y_al, y_new], ignore_index=True)
    model_al.fit(X_al, Y_al, n_iter=100, n_imputations=5, parallel=False)
    metrics.append(model_al.score())

fig = convergence_plot(metrics, metric="r2")
fig.savefig(OUT / "convergence.png", dpi=DPI)
print("Saved convergence.png")

print(f"\nAll plots saved to {OUT}")
