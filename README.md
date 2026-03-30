# surrogate

DataFrame-in/DataFrame-out surrogate modelling with uncertainty quantification,
powered by Deep Gaussian Processes via [dgpsi](https://github.com/mingdeyu/DGP).

Automatically selects between a single-layer GP (fast, single-output, low-dim)
and a Deep GP (multi-output, high-dim, nonlinear) based on the data.

## Features

- **pandas-native interface** — pass DataFrames in, get DataFrames out
- **Automatic model selection** — GP or DGP chosen based on output count and input dimensionality
- **Mixed input types** — continuous and categorical columns handled automatically
- **Uncertainty quantification** — predictions include mean, std, and 95% credible intervals
- **Output correlation** — estimate cross-output correlations from posterior samples
- **Active learning** — suggest the most informative next experiment (ALM, MICE, VIGF)
- **LOO cross-validation** — R² and RMSE diagnostics without re-fitting
- **Save/load** — persist fitted models to disk

## Installation

Requires Python 3.10+.

```bash
uv venv
uv sync
```

## Quick start

```python
import pandas as pd
from surrogate import SurrogateModel

# Fit
model = SurrogateModel()
model.fit(X, Y)

# Predict with uncertainty
result = model.predict(X_test)
result["mean"]      # DataFrame of predicted means
result["std"]       # DataFrame of predicted standard deviations
result["lower_95"]  # Lower 95% credible bound
result["upper_95"]  # Upper 95% credible bound

# LOO cross-validation
scores = model.score()
scores["r2"]    # {"output_1": 0.98, ...}
scores["rmse"]  # {"output_1": 0.12, ...}

# Active learning — suggest next experiment
suggestions = model.suggest_next(candidates, method="ALM", n_suggestions=3)

# Save and load
model.save("models/my_model.pkl")
loaded = SurrogateModel.load("models/my_model.pkl")
```

See [examples/basic_usage.py](examples/basic_usage.py) for a full walkthrough.

## Running the example

```bash
uv run python examples/basic_usage.py
```

## Running tests

```bash
uv run pytest
```

## API overview

### `SurrogateModel(model_type="auto", depth=2, kernel="matern25")`

| Parameter             | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| `categorical_columns` | Column names to treat as categorical (auto-detected if `None`) |
| `model_type`          | `"auto"`, `"gp"`, or `"dgp"`                                  |
| `depth`               | Number of DGP layers (ignored for GP)                          |
| `kernel`              | `"matern25"` or `"sexp"`                                       |
| `dim_threshold`       | Encoded dims above which auto selects DGP for single-output   |

### Methods

| Method                | Description                                           |
|-----------------------|-------------------------------------------------------|
| `fit(X, Y)`          | Train on input/output DataFrames                      |
| `predict(X)`         | Mean, std, and 95% credible intervals                 |
| `sample(X, n)`       | Joint posterior samples `(n, points, outputs)`        |
| `predict_correlation(X)` | Output correlation matrices per point             |
| `score()`            | LOO cross-validation (R², RMSE)                       |
| `suggest_next(candidates)` | Active learning — best next point(s)            |
| `save(path)` / `load(path)` | Persist and restore fitted models              |

**Note:** All output columns (`Y`) must be continuous (numeric). Categorical
outputs are not supported.

## License

[MIT](LICENSE)

## Acknowledgments

See [NOTICE.md](NOTICE.md) for dgpsi attribution and citations.
