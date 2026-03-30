# What is surrogate modelling?

If you have a simulation or experiment that takes ages to run, you probably
can't afford to evaluate it at every point you care about. A surrogate model
(sometimes called an emulator, metamodel, or response surface) is a fast
statistical stand-in trained on a small number of real evaluations. You run
the expensive thing a handful of times, fit a surrogate, then use the
surrogate for exploration, optimisation, or sensitivity analysis.

The active learning loop looks like:

1. Run a few experiments.
2. Fit a surrogate.
3. Use the surrogate's uncertainty to pick the most useful next experiment.
4. Run it, add the result, re-fit. Repeat until good enough.

## Other approaches

There are plenty of ways to build surrogates. This library uses **Gaussian
Processes** (GPs) and **Deep GPs** because they give you uncertainty estimates
for free, which is what makes the active learning loop work. But they're not
the only option:

- **Polynomial response surfaces** work well for smooth, low-dimensional
  problems but fall over quickly as complexity grows.
- **Radial basis functions** are popular in engineering optimisation.
- **Neural networks** can fit almost anything given enough data, but they need
  a lot of it and don't give you uncertainty without extra work (ensembles,
  MC dropout, etc.).
- **Polynomial chaos expansions** are common in UQ for smooth responses.
- **Support vector regression** does fine in moderate dimensions.

## Why GPs?

GPs work well with small datasets (tens to hundreds of points) and give you
calibrated uncertainty bounds out of the box. That makes them a natural fit
for problems where data is expensive and you need to know where the model is
confident vs where it's guessing.
