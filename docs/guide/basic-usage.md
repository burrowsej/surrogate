# Basic usage

This guide walks through the core workflow: fitting a surrogate model,
making predictions with uncertainty, evaluating accuracy, and saving/loading
models.

## Training data

surrogate expects pandas DataFrames for both inputs (`X`) and outputs (`Y`).

- **X** may contain numeric and categorical columns. Categorical columns are
  auto-detected (dtype `object` or `category`) or can be specified explicitly.
- **Y** must be entirely numeric (continuous). Categorical outputs are not
  supported.

```python
import numpy as np
import pandas as pd
from surrogate import SurrogateModel

rng = np.random.default_rng(42)
n = 40
X = pd.DataFrame({
    "x1": rng.uniform(0, 1, n),
    "x2": rng.uniform(0, 1, n),
    "category": rng.choice(["low", "high"], size=n),
})

base = np.sin(5 * X["x1"]) + X["x2"] ** 2
Y = pd.DataFrame({
    "output_1": base + np.where(X["category"] == "high", 3.0, 0.0),
    "output_2": 2.0 * base + 0.5 * X["x1"] + rng.normal(0, 0.2, n),
})
```

## Fitting

```python
model = SurrogateModel()
model.fit(X, Y, n_iter=100, parallel=False)
```

In `"auto"` mode (the default) the model selects a single-layer GP for
single-output, low-dimensional problems and a Deep GP otherwise. You can
force a tier with `model_type="gp"` or `model_type="dgp"`.

## Prediction

`predict()` returns a dict of DataFrames in the original output scale:

```python
X_test = pd.DataFrame({
    "x1": [0.3, 0.7],
    "x2": [0.5, 0.2],
    "category": ["low", "high"],
})
result = model.predict(X_test)

result["mean"]       # predicted means
result["std"]        # predicted standard deviations
result["lower_95"]   # lower 95% credible bound
result["upper_95"]   # upper 95% credible bound
```

## Joint posterior samples

`sample()` draws from the joint posterior, preserving cross-output
correlation:

```python
samples = model.sample(X_test, n_samples=100)
# shape: (100, 2, 2) - (samples, points, outputs)
```

## Output correlation

Estimate the correlation matrix between outputs at each input point:

```python
corr = model.predict_correlation(X_test, n_samples=200)
# shape: (n_points, n_outputs, n_outputs)
print(corr[0])  # correlation matrix at first test point
```

## LOO cross-validation

`score()` returns leave-one-out R-squared and RMSE without re-fitting:

```python
scores = model.score()
for name in Y.columns:
    print(f"{name}: R²={scores['r2'][name]:.3f}, RMSE={scores['rmse'][name]:.3f}")
```

For full LOO predictions (useful for custom diagnostics):

```python
loo = model.loo_predict()
loo["mean"]    # LOO predicted means
loo["std"]     # LOO predicted standard deviations
loo["actual"]  # actual training values
```

See the [plotting guide](plotting.md) for visual diagnostics like this LOO
parity plot:

<img src="../../assets/plots/parity.png" alt="Parity plot" style="max-height: 280px; width: auto;">

## Save and load

Models are saved to the `models/` directory by default:

```python
model.save("my_surrogate.pkl")           # saved as models/my_surrogate.pkl
model.save("models/my_surrogate.pkl")    # same result
model.save()                             # saved as models/surrogate.pkl

loaded = SurrogateModel.load("models/my_surrogate.pkl")
```

The full example is in
[examples/basic_usage.py](https://github.com/eburrows/surrogate/blob/main/examples/basic_usage.py).
