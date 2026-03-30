# Getting started

## Installation

Requires Python 3.10+.

```bash
uv venv
uv sync
```

For diagnostic plots, install the optional `plot` group:

```bash
uv sync --group plot
```

## Minimal example

```python
import numpy as np
import pandas as pd
from surrogate import SurrogateModel

# Some training data
rng = np.random.default_rng(42)
X = pd.DataFrame({"x1": rng.uniform(0, 1, 30), "x2": rng.uniform(0, 1, 30)})
Y = pd.DataFrame({"y": np.sin(5 * X["x1"]) + X["x2"] ** 2})

# Fit (auto-selects GP for single-output, low-dim)
model = SurrogateModel()
model.fit(X, Y)

# Predict with uncertainty
X_new = pd.DataFrame({"x1": [0.3, 0.7], "x2": [0.5, 0.2]})
result = model.predict(X_new)
print(result["mean"])
print(result["std"])

# LOO cross-validation
scores = model.score()
print(f"R² = {scores['r2']}")
print(f"RMSE = {scores['rmse']}")
```

## What happens under the hood

1. **Encoding** - Continuous columns are scaled to [0, 1]. Categorical columns
   (auto-detected or specified) are one-hot encoded.
2. **Tier selection** - `"auto"` mode picks a single-layer GP when the problem
   is single-output and low-dimensional, otherwise a Deep GP.
3. **Training** - The GP is fitted analytically. The DGP uses stochastic
   expectation-maximisation (SEM).
4. **Prediction** - Results are inverse-transformed back to the original output
   scale.

## Multiprocessing note

dgpsi uses multiprocessing internally. When calling `.fit()` with
`parallel=True` (the default) in a notebook or script without an
`if __name__ == '__main__':` guard, you may get a `RuntimeError`. Either wrap
the call in a guard or pass `parallel=False`.

## Next steps

- [Basic usage guide](guide/basic-usage.md) - full walkthrough of fit, predict,
  score, and save/load
- [Plotting guide](guide/plotting.md) - diagnostic visualisations
- [Active learning guide](guide/active-learning.md) - sequential experiment design
