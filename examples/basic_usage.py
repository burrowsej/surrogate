"""Basic usage example  -- fit, predict, score, suggest next, save/load.

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

# %% [markdown]
# ## 1. Generate synthetic data

# %%
rng = np.random.default_rng(42)
n = 40
x1 = rng.uniform(0, 1, n)
x2 = rng.uniform(0, 1, n)
cat = rng.choice(["low", "high"], size=n)

# Two correlated outputs (simulating an expensive model)
base = np.sin(5 * x1) + x2**2
y1 = base + np.where(cat == "high", 3.0, 0.0)
y2 = 2.0 * base + 0.5 * x1 + rng.normal(0, 0.2, n)

X = pd.DataFrame({"x1": x1, "x2": x2, "category": cat})
Y = pd.DataFrame({"output_1": y1, "output_2": y2})

print("Training data:")
print(X.head())
print(Y.head())

# %% [markdown]
# ## 2. Fit surrogate

# %%
model = SurrogateModel()  # auto-selects DGP for multi-output
model.fit(X, Y, n_iter=100, n_imputations=5, parallel=False)
print(f"Model: {model}")

# %% [markdown]
# ## 3. Predict at new points

# %%
X_test = pd.DataFrame(
    {
        "x1": [0.3, 0.7],
        "x2": [0.5, 0.2],
        "category": ["low", "high"],
    }
)
result = model.predict(X_test)
print("Predictions:")
print(result["mean"])
print("\nUncertainty (std):")
print(result["std"])

# %% [markdown]
# ## 4. LOO cross-validation score

# %%
scores = model.score()
print("LOO Cross-Validation:")
for name in Y.columns:
    print(f"  {name}: R²={scores['r2'][name]:.3f}, RMSE={scores['rmse'][name]:.3f}")

# %% [markdown]
# ## 5. Output correlation

# %%
corr = model.predict_correlation(X_test, n_samples=100)
print("Output correlation at first test point:")
print(corr[0])

# %% [markdown]
# ## 6. Active learning  -- suggest next experiment

# %%
candidates = pd.DataFrame(
    {
        "x1": rng.uniform(0, 1, 200),
        "x2": rng.uniform(0, 1, 200),
        "category": rng.choice(["low", "high"], size=200),
    }
)
suggestions = model.suggest_next(candidates, method="ALM", n_suggestions=3)
print("Top 3 suggested next experiments:")
print(suggestions)

# %% [markdown]
# ## 7. Save and reload

# %%
model.save("models/my_surrogate.pkl")
loaded = SurrogateModel.load("models/my_surrogate.pkl")
reloaded_pred = loaded.predict(X_test)
print(
    "Predictions after reload match:",
    np.allclose(result["mean"].values, reloaded_pred["mean"].values, atol=1e-6),
)
