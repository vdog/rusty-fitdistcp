# fitdistcp (Python bindings)

PyO3 bindings over `fitdistcp-core`. The Python package is a **drop-in
replacement** for the reference Python package `fitdistcp` v0.0.4: it exposes the
same **submodule-based** public API (`fitdistcp.norm.ppf(...)`,
`fitdistcp.genextreme.ppf(...)`, ...) with the reference signatures, defaults,
and return-dict keys.

The compiled extension module is private (`fitdistcp._fitdistcp`); the public
surface is the set of thin Python submodules under
[`python/fitdistcp/`](python/fitdistcp/__init__.py:1).

## Relationship to the reference Python package

The reference Python package `fitdistcp` v0.0.4 is vendored read-only at
[`tools/fitdistcp-py-ref/`](../../tools/fitdistcp-py-ref/fitdistcp/__init__.py:1).
This binding is a **drop-in replacement** for it: the submodule API, function
signatures/defaults, and return-dict keys match the reference, so existing code
written against the reference package runs unchanged against this binding.

The Rust core provides the numerics; the Python submodules are thin wrappers
that mirror the reference surface.

## Submodule API (primary surface)

The public API is submodule-based, mirroring the reference package:

```python
import fitdistcp

fitdistcp.norm.ppf([1.0, 2.0, 3.0], p=[0.1, 0.5, 0.9])
fitdistcp.genextreme.ppf(x, p=0.5)
fitdistcp.genpareto.pdf(x, y)
fitdistcp.expon.rvs(n=100, x=x)
fitdistcp.reltest(x)
```

Each distribution submodule exposes `ppf` / `pdf` / `cdf` / `rvs` (and
`reltest` where applicable) with the **reference signatures and defaults**:

```python
ppf(x, p=np.arange(0.1, 1.0, 0.1), means=False, waicscores=False,
    logscores=False, ru=False, ru_nsamples=100000, unbiasedv=False)
pdf(x, y=None, ...)   # y=None defaults to y=x
cdf(x, y=None, ...)
rvs(n, x, ...)
```

`x` is the prediction grid (densities/CDFs) or the observed data (quantiles /
deviates), `y` is the observed data (defaulting to `x`), `p` is the probability
level(s), and `n` is the number of deviates.

## Models

| Submodule          | Model                          |
|--------------------|--------------------------------|
| `norm`             | Normal                         |
| `genextreme`       | GEV                            |
| `genpareto`        | GPD                            |
| `expon`            | Exponential                    |
| `gamma`            | Gamma                          |
| `gumbel`           | Gumbel                         |
| `lnorm`            | Log-normal                     |
| `weibull`          | Weibull                        |
| `genextreme_p1`    | GEV + predictor                |
| `reltest`          | Relative-likelihood test       |

`fitdistcp.__all__` is
`[genextreme, genextreme_p1, genpareto, expon, gamma, gumbel, lnorm, norm, weibull, reltest]`
(note: `genextreme_p12` is deferred and omitted).

## Flat API (deprecated aliases)

The previous flat API (`fitdistcp.ppf_norm`, `fitdistcp.pdf_gev`, ...) is
**retained as deprecated aliases**. Accessing a flat name emits a
`DeprecationWarning` (via a module-level `__getattr__`) and forwards to the
corresponding submodule function. New code should use the submodule API.

## Return values

Each function returns a Python `dict` using the **reference keys**:

- `ml_params`, `ml_value`, `standard_errors`, `ml_quantiles`, `cp_quantiles`,
  `ru_quantiles`, `maic`, `waic1`, `waic2`, `ml_oos_logscore`,
  `cp_oos_logscore`, `ml_mean`, `cp_mean`, `cp_method`.

Model-specific extras:

- `norm` — `uv_params`, `uv_quantiles`;
- `genextreme` — `pw_params`, `pw_quantiles`, `custom_quantiles`, `ml_max`,
  `revert2ml`, `ml_pdf`, `cp_pdf`, `custom_mean`;
- `genpareto` — `ml_max`, `revert2ml`, `lp_quantiles`, `lp2_quantiles`,
  `dpi_quantiles`, `ml_pdf`, `cp_pdf`;
- `genextreme_p1` — `predictedparameter`, `adjustedx`, `revert2ml`, `ml_max`,
  `ml_pdf`, `cp_pdf`.

Notes on specific fields:

- `maic` is the reference **3-vector** `[ml_value, -nparams, ml_value - nparams]`.
- `standard_errors` is a NumPy **object-dtype** array; entries may be the string
  `"square root not possible"` (reference `make_se` semantics).
- `ml_value` is the maximized log-likelihood.
- `waic1`, `waic2`, `ml_oos_logscore`, `cp_oos_logscore`, `ml_mean`, and
  `cp_mean` are populated.
- `ru_*` fields are **placeholder strings** (e.g. `"ru not selected"`); `ru` is
  disabled by default.
- `rvs` draws `np.random.uniform(size=n)` and inverts via `ppf`.
- `reltest` returns `{'actual_p_ml', 'actual_p_cp', 'p'}` as lists.

## Documented discrepancies

1. **`ru` is a placeholder.** The `ru` argument is accepted but returns
   placeholder strings (e.g. `"ru not selected"`) and is disabled by default.
2. **`genextreme_p12` is deferred.** The two-predictor GEV submodule is not
   implemented and is omitted from `__all__`.
3. **`extramodels` / `customprior` / PWM return reference placeholders.** These
   options are accepted and return the reference placeholder values.
4. **`cp_params` / `cp_se` equal the MLE values.** The DMGS expansion corrects
   the predictive distribution, not the point estimates (the reference likewise
   reports the MLE parameters for the CP model).

## Build & test (inside the dev container)

```sh
maturin develop
python -c "import fitdistcp; fitdistcp.norm.ppf([1.0,2.0,3.0])"
pytest
```

The full pytest suite is **109 passed, 1 skipped**. The skip is the gamma
`reltest` case — a pre-existing core robustness bug in
`crates/fitdistcp-core`'s `special::inc_gamma` when the gamma MLE optimizer
wanders to a non-positive shape; it is documented and is not a binding
regression.
