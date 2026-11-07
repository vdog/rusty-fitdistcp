# fitdistcp (Python bindings)

PyO3 bindings over `fitdistcp-core`. The Python API exposes the core numerics
following the design document §5.2 convention: the `pdf` / `cdf` / `ppf` / `rvs`
distribution functions for the ten Phase-1 models, with the optional arguments
`p`, `means`, `waicscores`, `logscores`, `y` and the predictor arguments `t`,
`t0`, `n0` for the `_p1` models.

## Relationship to the R API

The reference `fitdistcp` package is **R-only** — it ships no Python package.
The Python API therefore uses the §5.2 `pdf`/`cdf`/`ppf`/`rvs` naming convention
rather than the R `d/p/q/r` convention, but it **mirrors the R API's semantics
and return fields** so the numerics are identical across the two bindings:

| Python (this package)      | R (reference)   | Meaning                       |
|----------------------------|-----------------|-------------------------------|
| `pdf_<model>(x, y, ...)`   | `d<model>_cp`   | MLE + CP predictive densities |
| `cdf_<model>(x, y, ...)`   | `p<model>_cp`   | MLE + CP predictive CDFs      |
| `ppf_<model>(x, p, ...)`   | `q<model>_cp`   | MLE + CP predictive quantiles |
| `rvs_<model>(n, x, ...)`   | `r<model>_cp`   | MLE + CP random deviates      |

## Models

| Model            | suffix    |
|------------------|-----------|
| Normal           | `norm`    |
| Normal + pred.   | `norm_p1` |
| GEV              | `gev`     |
| GEV + pred.      | `gev_p1`  |
| GPD              | `gpd_k1`  |
| Gumbel           | `gumbel`  |
| Log-normal       | `lnorm`   |
| Weibull          | `weibull` |
| Gamma            | `gamma`   |
| Exponential      | `exp`     |

## Signatures

```python
ppf_<model>(x, p, [means=None])                       # quantiles
pdf_<model>(x, y, [means=None])                       # densities
cdf_<model>(x, y, [means=None])                       # CDFs
rvs_<model>(n, x, [means=None])                       # random deviates

# predictor models (norm_p1, gev_p1) additionally accept:
ppf_norm_p1(x, p, [t=None, t0=None, n0=None, means=None])
pdf_norm_p1(x, y, [t=None, t0=None, n0=None, means=None])
cdf_norm_p1(x, y, [t=None, t0=None, n0=None, means=None])
rvs_norm_p1(n, x, [t=None, t0=None, n0=None, means=None])
```

`x` is the prediction grid (densities/CDFs) or the observed data (quantiles /
deviates), `y` is the observed data, `p` is the probability level(s), and `n` is
the number of deviates. All array arguments are NumPy `float64` arrays.

## Return values

Each function returns a Python `dict` with the design §5.2 / R return field
names:

- `ml_params`, `cp_params`, `ml_se`, `cp_se` — parameter estimates and SEs;
- `ml_pdf` / `cp_pdf` (from `pdf_*`), `ml_cdf` / `cp_cdf` (from `cdf_*`),
  `ml_quantiles` / `cp_quantiles` (from `ppf_*`), `ml_deviates` / `cp_deviates`
  (from `rvs_*`);
- `ml_value`, `aic`, `maic`, `maic_vector`, `waic`, `log_scores`;
- `convergence` (string), `small_n` (bool), `cp_method` (string);
- model-specific extras: `ml_max`, `revert2ml` (GEV / GPD).

## Documented discrepancies

1. **No reference Python package.** The reference is R-only; the Python API
   mirrors the R semantics/return fields under the §5.2 naming convention.
2. **`waic` / `log_scores` are `None`.** The core does not yet compute these;
   the `waicscores` / `logscores` optional args are accepted and passed through
   but the values are not yet populated (W13 scope boundary).
3. **`cp_params` / `cp_se` equal the MLE values.** The DMGS expansion corrects
   the predictive distribution, not the point estimates (the reference likewise
   reports the MLE parameters for the CP model).
4. **Predictor handling (`_p1` models) is W14.** The `_p1` wrappers accept
   `t`/`t0`/`n0` and pass them through; full predictor semantics are left to W14.
5. **`means` is accepted and passed through** to the core `PredictionRequest`;
   the core does not yet consume it.

## Build & test (inside the dev container)

```sh
maturin develop
python -c "import fitdistcp; print(fitdistcp.core_version())"
pytest
```
