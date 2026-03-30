# Active learning

Active learning (sequential experiment design) uses the surrogate to choose
the next most informative experiment, reducing the total number of expensive
evaluations needed.

## Workflow

1. Fit a surrogate on an initial dataset.
2. Call `suggest_next()` with a set of candidate points.
3. Evaluate the suggested point(s) with your real model or experiment.
4. Add the new data and re-fit.
5. Repeat until the model is accurate enough.

```python
import numpy as np
import pandas as pd
from surrogate import SurrogateModel

rng = np.random.default_rng(42)

# Initial training data (small)
X = pd.DataFrame({"x1": rng.uniform(0, 1, 10), "x2": rng.uniform(0, 1, 10)})
Y = pd.DataFrame({"y": np.sin(5 * X["x1"]) + X["x2"] ** 2})

model = SurrogateModel()
model.fit(X, Y, parallel=False)

# Candidate pool
candidates = pd.DataFrame({
    "x1": rng.uniform(0, 1, 200),
    "x2": rng.uniform(0, 1, 200),
})

metrics = [model.score()]

for i in range(10):
    # Suggest next point
    suggestion = model.suggest_next(candidates, method="ALM", n_suggestions=1)
    x_new = suggestion.drop(columns=["_score"])

    # Evaluate (replace with your real model/experiment)
    y_new = pd.DataFrame({
        "y": np.sin(5 * x_new["x1"]) + x_new["x2"] ** 2,
    })

    # Augment and re-fit
    X = pd.concat([X, x_new], ignore_index=True)
    Y = pd.concat([Y, y_new], ignore_index=True)
    model.fit(X, Y, parallel=False)
    metrics.append(model.score())

    print(f"Iteration {i + 1}: R² = {metrics[-1]['r2']['y']:.4f}")
```

## Design criteria

`suggest_next()` supports three methods via the `method` argument:

| Method   | Full name                                | Best for |
|----------|------------------------------------------|----------|
| `"ALM"`  | Active Learning MacKay                   | Reducing overall predictive variance |
| `"MICE"` | Mutual Information for Computer Experiments | Reducing global mean squared error |
| `"VIGF"` | Variance of Improvement for Global Fit   | DGP models with structured uncertainty |

`"ALM"` is the default and works well in most situations. `"VIGF"` is only
available for DGP models and requires the underlying dgpsi object.

## Multiple suggestions

Request several points at once with `n_suggestions`:

```python
suggestions = model.suggest_next(candidates, method="ALM", n_suggestions=5)
```

The returned DataFrame includes a `_score` column with the acquisition
function value for each point, sorted from highest to lowest.

## Tracking convergence

Store `model.score()` at each iteration and use the convergence plot to
decide when to stop:

```python
from surrogate.plotting import convergence_plot

fig = convergence_plot(metrics, metric="r2")
```

<img src="../../assets/plots/convergence.png" alt="Convergence plot" style="max-height: 280px; width: auto;">

See [Plotting - convergence plot](plotting.md#convergence-plot) for details.
