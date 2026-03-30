# surrogate

Surrogate modelling with uncertainty quantification, powered by Deep Gaussian
Processes via [dgpsi](https://github.com/mingdeyu/DGP).

---

## What it does

**surrogate** wraps [dgpsi](https://github.com/mingdeyu/DGP) behind a
pandas-friendly interface so you can work with DataFrames directly instead of
manually encoding and scaling arrays.

It picks between a single-layer GP and a Deep GP depending on the number of
outputs and the input dimensionality.

## Key features

- Pass pandas DataFrames in, get DataFrames out - no array wrangling
- Auto-selects GP or DGP based on the data
- Handles mixed continuous/categorical inputs
- Predictions with mean, std, and 95% credible intervals
- Cross-output correlation from posterior samples
- Built-in active learning (ALM, MICE, VIGF)
- LOO cross-validation scores without re-fitting
- Save and load fitted models

## Quick example

```python
import pandas as pd
from surrogate import SurrogateModel

model = SurrogateModel()
model.fit(X, Y)

result = model.predict(X_test)
result["mean"]       # predicted means
result["std"]        # predicted standard deviations
result["lower_95"]   # lower 95% credible bound
result["upper_95"]   # upper 95% credible bound
```

See the [getting started](getting-started.md) guide for installation and a full
walkthrough.
