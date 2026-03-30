# Plotting

surrogate includes optional diagnostic plots for evaluating model quality.
Install the plot dependencies first:

```bash
uv sync --group plot
```

All plot functions return a matplotlib `Figure` and accept an optional `ax`
argument for embedding in custom layouts.

## Parity plot

The standard first check after fitting. Points on the diagonal indicate
perfect predictions; systematic deviations reveal bias.

```python
from surrogate.plotting import parity_plot

fig = parity_plot(model)
fig.savefig("parity.png", dpi=150)
```

<img src="../../assets/plots/parity.png" alt="Parity plot" style="max-height: 280px; width: auto;">

## Calibration plot

Checks whether the reported uncertainty intervals are trustworthy. A
well-calibrated model follows the diagonal. Below the diagonal means
overconfident (intervals too narrow); above means underconfident.

```python
from surrogate.plotting import calibration_plot

fig = calibration_plot(model)
```

<img src="../../assets/plots/calibration.png" alt="Calibration plot" style="max-height: 280px; width: auto;">

## Slice plot

Sweeps one input variable while holding the rest fixed, showing the
predicted mean and 95% credible interval. Wide ribbons indicate sparse
training data in that region.

```python
from surrogate.plotting import slice_plot

X_centre = pd.DataFrame({"x1": [0.5], "x2": [0.5], "category": ["low"]})
fig = slice_plot(model, X_centre, column="x1")
```

<img src="../../assets/plots/slice.png" alt="Slice plot" style="max-height: 280px; width: auto;">

You can control the sweep range with `lower` and `upper`, and filter outputs
with `output_columns`.

## Correlation heatmap

Shows how outputs co-vary at a given input point according to the posterior.
Most useful with three or more outputs.

```python
from surrogate.plotting import correlation_heatmap

fig = correlation_heatmap(model, X_test, point_index=0)
```

<img src="../../assets/plots/correlation.png" alt="Correlation heatmap" style="max-height: 280px; width: auto;">

## Convergence plot

Tracks LOO metric vs training set size during an active learning loop.
A plateau indicates diminishing returns from additional experiments.

```python
from surrogate.plotting import convergence_plot

# metrics is a list of model.score() dicts, one per iteration
fig = convergence_plot(metrics, metric="r2")
```

<img src="../../assets/plots/convergence.png" alt="Convergence plot" style="max-height: 280px; width: auto;">

## Styling

All plots use the Okabe-Ito colourblind-safe palette and a clean seaborn
`"ticks"` theme. The style is applied automatically on the first plot call.

The full example is in
[examples/plotting_example.py](https://github.com/eburrows/surrogate/blob/main/examples/plotting_example.py).
