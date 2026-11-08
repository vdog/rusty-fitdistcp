//! `fitdistcp-py` — PyO3 bindings over `fitdistcp-core`.
//!
//! Thin adapter: convert NumPy arrays to contiguous `&[f64]`, call the core, and
//! convert results back to NumPy arrays / Python dicts. No numerics live here.
//!
//! # W13 scope — Python API surface
//!
//! This crate exposes the core numerics to Python following the design document
//! §5.2 convention: the `pdf` / `cdf` / `ppf` / `rvs` distribution functions for
//! the ten Phase-1 models, with the optional arguments `p`, `means`,
//! `waicscores`, `logscores`, `y` and the predictor arguments `t`, `t0`, `n0`
//! for the `_p1` models.
//!
//! ## Relationship to the R API (documented discrepancy)
//!
//! The reference `fitdistcp` package (`tools/fitdistcp-ref/`) is **R-only** — it
//! ships no Python package. The Python API surface therefore follows the design
//! §5.2 naming convention (`pdf`/`cdf`/`ppf`/`rvs`) rather than the R `d/p/q/r`
//! convention, but it **mirrors the R API's semantics and return fields** so the
//! numerics are identical across the two bindings. Concretely:
//!
//! | Python (this crate)        | R (reference / W12) | Meaning                          |
//! |----------------------------|---------------------|----------------------------------|
//! | `pdf_<model>(x, y, ...)`   | `d<model>_cp`       | MLE + CP predictive densities    |
//! | `cdf_<model>(x, y, ...)`   | `p<model>_cp`       | MLE + CP predictive CDFs         |
//! | `ppf_<model>(x, p, ...)`   | `q<model>_cp`       | MLE + CP predictive quantiles    |
//! | `rvs_<model>(n, x, ...)`   | `r<model>_cp`       | MLE + CP random deviates         |
//!
//! The ten models and their Python suffixes are:
//!
//! | Model            | suffix        |
//! |------------------|---------------|
//! | Normal           | `norm`        |
//! | Normal + pred.   | `norm_p1`     |
//! | GEV              | `gev`         |
//! | GEV + pred.      | `gev_p1`      |
//! | GPD              | `gpd_k1`      |
//! | Gumbel           | `gumbel`      |
//! | Log-normal       | `lnorm`       |
//! | Weibull          | `weibull`     |
//! | Gamma            | `gamma`       |
//! | Exponential      | `exp`         |
//!
//! ## Return values
//!
//! Each function returns a Python `dict` whose keys are the design §5.2 / R
//! return field names: `ml_params`, `cp_params`, `ml_pdf`/`cp_pdf`,
//! `ml_cdf`/`cp_cdf`, `ml_quantiles`/`cp_quantiles`, `ml_se`/`cp_se`, `aic`,
//! `maic`, `waic`, `log_scores`, `convergence`, `small_n`, plus the model-specific
//! extras (`ml_max`, `revert2ml`, `cp_method`). Array fields are NumPy `float64`
//! arrays; scalars are Python floats/bools/strings.
//!
//! ## Documented discrepancies (design vs. reference)
//!
//! 1. **No reference Python package.** The reference is R-only; the Python API
//!    mirrors the R semantics/return fields under the §5.2 `pdf`/`cdf`/`ppf`/`rvs`
//!    naming convention.
//! 2. **`waic` / `log_scores` are `None`.** The core `FitResult` carries optional
//!    `waic`/`log_scores`; the core does not yet compute them, so the wrappers
//!    return `None` (the `waicscores`/`logscores` optional args are accepted and
//!    passed through, but the values are not yet populated). This is a W13 scope
//!    boundary, not a numerics change.
//! 3. **`cp_params`/`cp_se` equal the MLE values.** The DMGS expansion corrects
//!    the predictive distribution, not the point estimates; the reference likewise
//!    reports the MLE parameters for the CP model (see
//!    [`fitdistcp_core::api::predict`]).
//! 4. **Predictor handling (`_p1` models) — W14.** The `_p1` wrappers implement
//!    the reference predictor semantics: the predictor is centered by its mean
//!    (`t - mean(t)`), the prediction point is resolved via `maket0`
//!    (`t0 = t[n0]` when `t0` is `None`), `ml_params` is decentered, and the
//!    `predictedparameter`/`adjustedx` fields are returned. `norm_p1` uses the
//!    reference **analytic** CP (Student-t predictive); `gev_p1` uses the DMGS
//!    CRHP/flat path with `revert2ml`/`ml_max`/quantile-derived density.
//! 5. **`means` is accepted and passed through** to the core
//!    [`PredictionRequest`]; the core does not yet consume it (W14/Phase 2).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;

use fitdistcp_core::api::{
    predict, predict_gev, predict_gev_p1, predict_norm_p1, FitResult, NormP1Result,
    PredictionRequest,
};
use fitdistcp_core::dmgs::DmgsModel;
use fitdistcp_core::models::{
    Exponential, Gamma, GenExtreme, GenExtremeP1, GenPareto, Gumbel, LogNormal, Normal, NormalP1,
    Weibull,
};

// ---------------------------------------------------------------------------
// Phase 0 smoke tests (kept intact).
// ---------------------------------------------------------------------------

/// Return the core crate version string.
#[pyfunction]
fn core_version() -> String {
    fitdistcp_core::CORE_VERSION.to_string()
}

/// Element-wise sum of two equal-length arrays (Phase 0 smoke test).
#[pyfunction]
fn add_vectors<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = a.as_slice()?;
    let b = b.as_slice()?;
    let out = fitdistcp_core::add_vectors(a, b);
    Ok(out.into_pyarray(py))
}

/// Sum of an array (Phase 0 smoke test).
#[pyfunction]
fn sum_values(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(fitdistcp_core::sum(values.as_slice()?))
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// The reference `cp_method` strings (from the R package `005_utils.R`).
const ANALYTIC_CPMETHOD: &str = "The cp results are based on an analytic solution of the Bayesian prediction integral, using the predictive matching right Haar prior.";
const RHP_DMGS_CPMETHOD: &str = "The cp results are based on the DMGS approximation of the Bayesian prediction integral, using the predictive matching right Haar prior.";
const ADHOC_DMGS_CPMETHOD: &str = "The cp results are based on the DMGS approximation of the Bayesian prediction integral, using an ad-hoc prior.";
const CRHPFLAT_DMGS_CPMETHOD: &str = "The cp results are based on the DMGS approximation of the Bayesian prediction integral, using the CRHP/flat prior.";

/// Build a [`PredictionRequest`] from the common Python arguments.
///
/// `means` is accepted and passed through (the core does not yet consume it —
/// see the module-level discrepancy notes).
fn make_request(
    y: Vec<f64>,
    x: Vec<f64>,
    p: Vec<f64>,
    t: Option<Vec<f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<Vec<f64>>,
) -> PredictionRequest {
    PredictionRequest {
        y,
        x,
        p,
        t,
        t0,
        n0,
        means,
    }
}

/// The reference `maket0`: resolve the prediction point `t0` from the user's
/// `t0`/`n0` (W14).
///
/// - if `t0` is supplied, use it directly (`n0` must be absent);
/// - otherwise `t0 = t[n0]` where `n0` is a **1-based positional index** into
///   `t` (the reference `maket0(NA, n0, t) = t[n0]`).
///
/// Returns `0.0` when neither is supplied or when `t` is empty.
fn maket0(t: &[f64], t0: Option<f64>, n0: Option<f64>) -> f64 {
    if let Some(v) = t0 {
        return v;
    }
    if let Some(idx) = n0 {
        let i = idx.round() as isize;
        if i >= 1 && (i as usize) <= t.len() {
            return t[i as usize - 1];
        }
    }
    0.0
}

/// Decenter a `_p1` [`FitResult`]'s `ml_params`/`cp_params` intercept.
///
/// The reference fits on the centered predictor and then decenters the reported
/// intercept: `a_ret = a_c - b*mean(t)`. The slope/scale/shape are unchanged.
fn decenter_fit(r: &FitResult, mean_t: f64) -> FitResult {
    let mut out = r.clone();
    if out.ml_params.len() >= 2 {
        out.ml_params[0] -= out.ml_params[1] * mean_t;
    }
    if out.cp_params.len() >= 2 {
        out.cp_params[0] -= out.cp_params[1] * mean_t;
    }
    out
}

/// Build the `ppf` return dict for the analytic `norm_p1` path.
fn ppf_dict_norm_p1<'py>(
    py: Python<'py>,
    r: &NormP1Result,
    mean_t: f64,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    put_arr(py, &d, "ml_params", ml_params.clone())?;
    put_arr(py, &d, "cp_params", ml_params)?;
    put_arr(py, &d, "ml_quantiles", r.ml_quantiles.clone())?;
    put_arr(py, &d, "cp_quantiles", r.cp_quantiles.clone())?;
    put_arr(py, &d, "standard_errors", r.standard_errors.clone())?;
    put_arr(py, &d, "predictedparameter", r.predictedparameter.clone())?;
    put_arr(py, &d, "adjustedx", r.adjustedx.clone())?;
    d.set_item("ml_value", r.ml_value)?;
    d.set_item("muhat0", r.muhat0)?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the `pdf` return dict for the analytic `norm_p1` path.
fn pdf_dict_norm_p1<'py>(
    py: Python<'py>,
    r: &NormP1Result,
    mean_t: f64,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    put_arr(py, &d, "ml_params", ml_params.clone())?;
    put_arr(py, &d, "cp_params", ml_params)?;
    put_arr(py, &d, "ml_pdf", r.ml_pdf.clone())?;
    put_arr(py, &d, "cp_pdf", r.cp_pdf.clone())?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the `cdf` return dict for the analytic `norm_p1` path.
fn cdf_dict_norm_p1<'py>(
    py: Python<'py>,
    r: &NormP1Result,
    mean_t: f64,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    put_arr(py, &d, "ml_params", ml_params.clone())?;
    put_arr(py, &d, "cp_params", ml_params)?;
    put_arr(py, &d, "ml_cdf", r.ml_cdf.clone())?;
    put_arr(py, &d, "cp_cdf", r.cp_cdf.clone())?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the `rvs` return dict for the analytic `norm_p1` path.
fn rvs_dict_norm_p1<'py>(
    py: Python<'py>,
    r: &NormP1Result,
    mean_t: f64,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    put_arr(py, &d, "ml_params", ml_params.clone())?;
    put_arr(py, &d, "cp_params", ml_params)?;
    put_arr(py, &d, "ml_deviates", r.ml_quantiles.clone())?;
    put_arr(py, &d, "cp_deviates", r.cp_quantiles.clone())?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// The reference `ml_value` (the maximized log-likelihood).
///
/// B3.2: the core [`FitResult`] carries `loglik` explicitly (the Phase A
/// amendment), so we surface it directly rather than reconstructing it from
/// `aic`. The reference reports `ml_value = sum(logpdf(x))` (e.g. `norm.py`).
fn ml_value(r: &FitResult) -> f64 {
    r.loglik
}

/// B2.1/B3.2: the reference `ml_value` from a core [`FitResult`]'s `loglik`.
///
/// Alias of [`ml_value`] with the plan's helper name, used by the
/// reference-named extension functions.
fn ml_value_from_loglik(r: &FitResult) -> f64 {
    r.loglik
}

/// B2.1/B3.3: build the reference object-dtype `standard_errors` array.
///
/// Replicates the reference `utils.py::make_se(nx, lddi)` exactly: for each
/// diagonal entry `lddi[i, i]`, if `lddi[i, i] > 0` store the string
/// `"square root not possible"`, else store `sqrt(-lddi[i, i] / nx)`. The
/// result is a NumPy **object-dtype** array so it can hold both floats and the
/// string.
fn make_se_object<'py>(
    py: Python<'py>,
    lddi_diag: &[f64],
    nx: f64,
) -> PyResult<Bound<'py, PyArray1<Py<PyAny>>>> {
    use pyo3::IntoPyObject;
    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(lddi_diag.len());
    for &d in lddi_diag {
        if d > 0.0 {
            out.push(
                "square root not possible"
                    .into_pyobject(py)?
                    .unbind()
                    .into(),
            );
        } else {
            let v = (-d / nx).sqrt();
            out.push(v.into_pyobject(py)?.unbind().into());
        }
    }
    Ok(PyArray1::from_vec(py, out))
}

/// B2.1/B3.3: recompute the reference `standard_errors` from the normalized
/// observed information `ldd`.
///
/// The reference `make_se` takes `lddi = inv(ldd)` where `ldd` is the
/// **normalized** observed information `(1/n) Σ ∂²ℓ/∂θ_i∂θ_j` (the core
/// [`DmgsIntegrator::normalized_ldd`]). We invert `ldd` with `nalgebra` and
/// build the object-dtype array from its diagonal.
fn standard_errors_from_ldd<'py, M: DmgsModel + Clone + Sync>(
    py: Python<'py>,
    model: M,
    params: &[f64],
    y: &[f64],
) -> PyResult<Bound<'py, PyArray1<Py<PyAny>>>> {
    use nalgebra::DMatrix;
    let integ = fitdistcp_core::dmgs::DmgsIntegrator::new(model);
    let ldd = integ.normalized_ldd(params, y);
    let p = ldd.nrows();
    let lddi = ldd
        .clone()
        .try_inverse()
        .unwrap_or_else(|| DMatrix::<f64>::zeros(p, p));
    let diag: Vec<f64> = (0..p).map(|i| lddi[(i, i)]).collect();
    make_se_object(py, &diag, y.len() as f64)
}

/// B2.1/B3.1: draw `n` uniform probabilities via `numpy.random.uniform`.
///
/// The reference `rvs` draws `np.random.uniform(size=n)` and inverts via `ppf`.
/// We call NumPy through PyO3 so the RNG state is the global NumPy one (the
/// reference behaviour).
fn rvs_uniform_probs(py: Python<'_>, n: usize) -> PyResult<Vec<f64>> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("size", n)?;
    let arr = np
        .getattr("random")?
        .call_method("uniform", (), Some(&kwargs))?;
    let arr: Vec<f64> = arr.extract()?;
    Ok(arr)
}

/// B2.1: insert an object-dtype NumPy array under `key`.
fn put_obj_arr<'py>(
    _py: Python<'py>,
    d: &Bound<'py, PyDict>,
    key: &str,
    values: Bound<'py, PyArray1<Py<PyAny>>>,
) -> PyResult<()> {
    d.set_item(key, values)
}

/// The reference `make_maic` 3-vector: `[ml_value, -nparams, ml_value - nparams]`.
fn make_maic(ml_value: f64, nparams: usize) -> Vec<f64> {
    vec![ml_value, -(nparams as f64), ml_value - nparams as f64]
}

/// The convergence status as a stable lowercase string.
fn convergence_str(r: &FitResult) -> &'static str {
    use fitdistcp_core::api::Convergence::*;
    match r.convergence {
        Converged => "converged",
        MaxIterationsReached => "max_iterations_reached",
        Stalled => "stalled",
        Failed => "failed",
    }
}

/// Evaluate a model CDF at each prediction point using the given parameters.
fn cdf_at<M: DmgsModel>(model: &M, params: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| model.cdf(xi, params)).collect()
}

/// Insert a `Vec<f64>` as a NumPy `float64` array under `key`.
fn put_arr<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, key: &str, v: Vec<f64>) -> PyResult<()> {
    d.set_item(key, v.into_pyarray(py))
}

/// Build the common return dict shared by all four function families.
///
/// Populates the design §5.2 / R return fields from a core [`FitResult`]. The
/// `pdf`/`cdf`/`ppf`/`rvs` families then add their family-specific fields.
fn base_dict<'py>(py: Python<'py>, r: &FitResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    put_arr(py, &d, "ml_params", r.ml_params.clone())?;
    put_arr(py, &d, "cp_params", r.cp_params.clone())?;
    put_arr(py, &d, "ml_se", r.ml_se.clone())?;
    put_arr(py, &d, "cp_se", r.cp_se.clone())?;
    d.set_item("ml_value", ml_value(r))?;
    d.set_item("aic", r.aic)?;
    d.set_item("maic", r.maic)?;
    d.set_item("maic_vector", make_maic(ml_value(r), r.ml_params.len()))?;
    // Optional diagnostics: the core does not yet compute these (W13 boundary).
    d.set_item("waic", r.waic)?;
    d.set_item("log_scores", r.log_scores.clone())?;
    d.set_item("convergence", convergence_str(r))?;
    d.set_item("small_n", r.small_n)?;
    Ok(d)
}

/// Build the `pdf` return dict: `ml_pdf` / `cp_pdf`.
fn pdf_dict<'py>(py: Python<'py>, r: &FitResult, cp_method: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = base_dict(py, r)?;
    put_arr(py, &d, "ml_pdf", r.ml_pdf.clone())?;
    put_arr(py, &d, "cp_pdf", r.cp_pdf.clone())?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the `cdf` return dict: `ml_cdf` / `cp_cdf`.
fn cdf_dict<'py>(
    py: Python<'py>,
    r: &FitResult,
    ml_cdf: Vec<f64>,
    cp_cdf: Vec<f64>,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = base_dict(py, r)?;
    put_arr(py, &d, "ml_cdf", ml_cdf)?;
    put_arr(py, &d, "cp_cdf", cp_cdf)?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the `ppf` return dict: `ml_quantiles` / `cp_quantiles`.
fn ppf_dict<'py>(py: Python<'py>, r: &FitResult, cp_method: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = base_dict(py, r)?;
    put_arr(py, &d, "ml_quantiles", r.ml_quantiles.clone())?;
    put_arr(py, &d, "cp_quantiles", r.cp_quantiles.clone())?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the `rvs` return dict: `ml_deviates` / `cp_deviates`.
fn rvs_dict<'py>(py: Python<'py>, r: &FitResult, cp_method: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = base_dict(py, r)?;
    put_arr(py, &d, "ml_deviates", r.ml_quantiles.clone())?;
    put_arr(py, &d, "cp_deviates", r.cp_quantiles.clone())?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// The probability grid used by `rvs` for `n` deviates: midpoints `(i+0.5)/n`.
///
/// Kept for the legacy flat API; the reference-named `rvs` functions use
/// [`rvs_uniform_probs`] (B3.1).
fn rvs_probs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect()
}

// ---------------------------------------------------------------------------
// B2: reference-named extension functions.
//
// These mirror the reference `fitdistcp` submodule API (`norm.ppf`, `expon.pdf`,
// ...) with the reference signatures/defaults and return-dict keys. The thin
// Python submodules under `python/fitdistcp/` (B4) delegate to these.
//
// The reference `ru` sampling is disabled by default and returns the placeholder
// string `"ru not selected"` (scope decision 1). `extramodels`/`customprior`/PWM
// are out of scope and return reference placeholders (scope decision 5).
// ---------------------------------------------------------------------------

/// The reference placeholder for an unselected `ru` result.
const RU_NOT_SELECTED: &str = "ru not selected";
/// The reference placeholder for an unselected `unbiasedv` result.
const UNBIASEDV_NOT_SELECTED: &str = "unbiasedv not selected";
/// The reference placeholder for an unselected PWM result.
const PWM_NOT_SELECTED: &str = "pwm not selected";
/// The reference placeholder for an unselected `extramodels` result.
const EXTRAMODELS_NOT_SELECTED: &str = "extramodels not selected";

/// Insert the reference `maic` 3-vector `[ml_value, -nparams, ml_value - nparams]`.
fn put_maic<'py>(
    py: Python<'py>,
    d: &Bound<'py, PyDict>,
    ml_value: f64,
    nparams: usize,
) -> PyResult<()> {
    put_arr(py, d, "maic", make_maic(ml_value, nparams))
}

/// Insert the optional WAIC 3-vectors (`waic1`/`waic2`), or `None`.
fn put_waic<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, r: &FitResult) -> PyResult<()> {
    match r.waic1 {
        Some(v) => put_arr(py, d, "waic1", v.to_vec())?,
        None => d.set_item("waic1", py.None())?,
    }
    match r.waic2 {
        Some(v) => put_arr(py, d, "waic2", v.to_vec())?,
        None => d.set_item("waic2", py.None())?,
    }
    Ok(())
}

/// Insert the optional out-of-sample log-scores, or `None`.
fn put_logscores<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, r: &FitResult) -> PyResult<()> {
    match r.ml_oos_logscore {
        Some(v) => d.set_item("ml_oos_logscore", v)?,
        None => d.set_item("ml_oos_logscore", py.None())?,
    }
    match r.cp_oos_logscore {
        Some(v) => d.set_item("cp_oos_logscore", v)?,
        None => d.set_item("cp_oos_logscore", py.None())?,
    }
    Ok(())
}

/// Build the reference `ppf` return dict for a homogeneous (non-predictor) model.
///
/// Keys (reference `norm.py`/`expon.py`/`gumbel.py`/...):
/// `ml_params, ml_value, standard_errors, ml_quantiles, cp_quantiles,
/// ru_quantiles, maic, waic1, waic2, ml_oos_logscore, cp_oos_logscore,
/// ml_mean, cp_mean, cp_method`.
#[allow(clippy::too_many_arguments)]
fn ref_ppf_dict<'py, M: DmgsModel + Clone + Sync>(
    py: Python<'py>,
    r: &FitResult,
    model: M,
    y: &[f64],
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    put_arr(py, &d, "ml_params", r.ml_params.clone())?;
    d.set_item("ml_value", ml_value_from_loglik(r))?;
    let se = standard_errors_from_ldd(py, model, &r.ml_params, y)?;
    put_obj_arr(py, &d, "standard_errors", se)?;
    put_arr(py, &d, "ml_quantiles", r.ml_quantiles.clone())?;
    put_arr(py, &d, "cp_quantiles", r.cp_quantiles.clone())?;
    d.set_item("ru_quantiles", RU_NOT_SELECTED)?;
    put_maic(py, &d, ml_value_from_loglik(r), r.ml_params.len())?;
    put_waic(py, &d, r)?;
    put_logscores(py, &d, r)?;
    d.set_item("ml_mean", r.ml_mean)?;
    d.set_item("cp_mean", r.cp_mean)?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the reference `pdf` return dict: `ml_params, ml_pdf, cp_pdf, ru_pdf,
/// cp_method`.
fn ref_pdf_dict<'py>(
    py: Python<'py>,
    r: &FitResult,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    put_arr(py, &d, "ml_params", r.ml_params.clone())?;
    put_arr(py, &d, "ml_pdf", r.ml_pdf.clone())?;
    put_arr(py, &d, "cp_pdf", r.cp_pdf.clone())?;
    d.set_item("ru_pdf", RU_NOT_SELECTED)?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the reference `cdf` return dict: `ml_params, ml_cdf, cp_cdf, ru_cdf,
/// cp_method`.
fn ref_cdf_dict<'py>(
    py: Python<'py>,
    r: &FitResult,
    ml_cdf: Vec<f64>,
    cp_cdf: Vec<f64>,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    put_arr(py, &d, "ml_params", r.ml_params.clone())?;
    put_arr(py, &d, "ml_cdf", ml_cdf)?;
    put_arr(py, &d, "cp_cdf", cp_cdf)?;
    d.set_item("ru_cdf", RU_NOT_SELECTED)?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

/// Build the reference `rvs` return dict: `ml_params, ml_deviates, cp_deviates,
/// ru_deviates, cp_method`.
fn ref_rvs_dict<'py>(
    py: Python<'py>,
    r: &FitResult,
    cp_method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    put_arr(py, &d, "ml_params", r.ml_params.clone())?;
    put_arr(py, &d, "ml_deviates", r.ml_quantiles.clone())?;
    put_arr(py, &d, "cp_deviates", r.cp_quantiles.clone())?;
    d.set_item("ru_deviates", RU_NOT_SELECTED)?;
    d.set_item("cp_method", cp_method)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// B2.2: reference-named functions — `norm`, `expon`, `lnorm` (analytic CP).
// ---------------------------------------------------------------------------

/// Reference `norm.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, means=false, waicscores=false, logscores=false, ru=false, ru_nsamples=100000, unbiasedv=false))]
#[allow(clippy::too_many_arguments)]
fn ppf_norm_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: bool,
    waicscores: bool,
    logscores: bool,
    ru: bool,
    ru_nsamples: usize,
    unbiasedv: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (means, waicscores, logscores, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    let d = ref_ppf_dict(py, &r, Normal, &x, ANALYTIC_CPMETHOD)?;
    // norm-specific extras.
    d.set_item("uv_params", UNBIASEDV_NOT_SELECTED)?;
    d.set_item("uv_quantiles", UNBIASEDV_NOT_SELECTED)?;
    let _ = unbiasedv;
    Ok(d)
}

/// Reference `norm.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn pdf_norm_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    ref_pdf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Reference `norm.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn cdf_norm_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&Normal, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Normal, &r.cp_params, &y);
    ref_cdf_dict(py, &r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD)
}

/// Reference `norm.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ru=false, mlcp=true))]
fn rvs_norm_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ru;
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", ANALYTIC_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    ref_rvs_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Reference `expon.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, means=false, waicscores=false, logscores=false, ru=false, ru_nsamples=100000))]
#[allow(clippy::too_many_arguments)]
fn ppf_expon_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: bool,
    waicscores: bool,
    logscores: bool,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (means, waicscores, logscores, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    ref_ppf_dict(py, &r, Exponential, &x, ANALYTIC_CPMETHOD)
}

/// Reference `expon.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn pdf_expon_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    ref_pdf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Reference `expon.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn cdf_expon_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    let ml_cdf = cdf_at(&Exponential, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Exponential, &r.cp_params, &y);
    ref_cdf_dict(py, &r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD)
}

/// Reference `expon.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ru=false, mlcp=true))]
fn rvs_expon_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ru;
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", ANALYTIC_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    ref_rvs_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Reference `lnorm.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, means=false, waicscores=false, logscores=false, dmgs=true, ru=false, ru_nsamples=100000))]
#[allow(clippy::too_many_arguments)]
fn ppf_lnorm_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: bool,
    waicscores: bool,
    logscores: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (means, waicscores, logscores, dmgs, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    ref_ppf_dict(py, &r, LogNormal, &x, ANALYTIC_CPMETHOD)
}

/// Reference `lnorm.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn pdf_lnorm_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    ref_pdf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Reference `lnorm.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn cdf_lnorm_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&LogNormal, &r.ml_params, &y);
    let cp_cdf = cdf_at(&LogNormal, &r.cp_params, &y);
    ref_cdf_dict(py, &r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD)
}

/// Reference `lnorm.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ru=false, mlcp=true))]
fn rvs_lnorm_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ru;
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", ANALYTIC_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    ref_rvs_dict(py, &r, ANALYTIC_CPMETHOD)
}

// ---------------------------------------------------------------------------
// B2.3: reference-named functions — `gumbel`, `weibull`, `gamma` (DMGS CP).
// ---------------------------------------------------------------------------

/// Reference `gumbel.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, means=false, waicscores=false, logscores=false, dmgs=true, ru=false, ru_nsamples=100000))]
#[allow(clippy::too_many_arguments)]
fn ppf_gumbel_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: bool,
    waicscores: bool,
    logscores: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (means, waicscores, logscores, dmgs, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    ref_ppf_dict(py, &r, Gumbel, &x, RHP_DMGS_CPMETHOD)
}

/// Reference `gumbel.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn pdf_gumbel_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    ref_pdf_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Reference `gumbel.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn cdf_gumbel_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&Gumbel, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Gumbel, &r.cp_params, &y);
    ref_cdf_dict(py, &r, ml_cdf, cp_cdf, RHP_DMGS_CPMETHOD)
}

/// Reference `gumbel.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ru=false, mlcp=true))]
fn rvs_gumbel_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ru;
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", RHP_DMGS_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    ref_rvs_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Reference `weibull.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, means=false, waicscores=false, logscores=false, dmgs=true, ru=false, ru_nsamples=100000))]
#[allow(clippy::too_many_arguments)]
fn ppf_weibull_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: bool,
    waicscores: bool,
    logscores: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (means, waicscores, logscores, dmgs, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    ref_ppf_dict(py, &r, Weibull, &x, RHP_DMGS_CPMETHOD)
}

/// Reference `weibull.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn pdf_weibull_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    ref_pdf_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Reference `weibull.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn cdf_weibull_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    let ml_cdf = cdf_at(&Weibull, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Weibull, &r.cp_params, &y);
    ref_cdf_dict(py, &r, ml_cdf, cp_cdf, RHP_DMGS_CPMETHOD)
}

/// Reference `weibull.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ru=false, mlcp=true))]
fn rvs_weibull_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ru;
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", RHP_DMGS_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    ref_rvs_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Reference `gamma.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, fd1=0.01, fd2=0.01, means=false, waicscores=false, logscores=false, dmgs=true, ru=false, ru_nsamples=100000, prior="type 1", aderivs=true))]
#[allow(clippy::too_many_arguments)]
fn ppf_gamma_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    fd1: f64,
    fd2: f64,
    means: bool,
    waicscores: bool,
    logscores: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
    prior: &str,
    aderivs: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        fd1,
        fd2,
        means,
        waicscores,
        logscores,
        dmgs,
        ru,
        ru_nsamples,
        prior,
        aderivs,
    );
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    ref_ppf_dict(py, &r, Gamma, &x, ADHOC_DMGS_CPMETHOD)
}

/// Reference `gamma.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn pdf_gamma_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    ref_pdf_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

/// Reference `gamma.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ru=false, ru_nsamples=1000))]
fn cdf_gamma_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    let ml_cdf = cdf_at(&Gamma, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Gamma, &r.cp_params, &y);
    ref_cdf_dict(py, &r, ml_cdf, cp_cdf, ADHOC_DMGS_CPMETHOD)
}

/// Reference `gamma.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ru=false, mlcp=true))]
fn rvs_gamma_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ru;
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", ADHOC_DMGS_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    ref_rvs_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

// ---------------------------------------------------------------------------
// B2.4: reference-named functions — `genextreme`, `genpareto` (DMGS + revert2ml).
// ---------------------------------------------------------------------------

/// Reference `genextreme.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, ics=None, fdalpha=0.01, means=false, waicscores=false, extramodels=false, pdf=false, customprior=0.0, dmgs=true, ru=false, ru_nsamples=100000))]
#[allow(clippy::too_many_arguments)]
fn ppf_genextreme_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    fdalpha: f64,
    means: bool,
    waicscores: bool,
    extramodels: bool,
    pdf: bool,
    customprior: f64,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        ics,
        means,
        waicscores,
        extramodels,
        pdf,
        customprior,
        dmgs,
        ru,
        ru_nsamples,
    );
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], fdalpha);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let d = ref_ppf_dict(py, &r, GenExtreme, &x, CRHPFLAT_DMGS_CPMETHOD)?;
    // genextreme-specific extras.
    d.set_item("pw_params", PWM_NOT_SELECTED)?;
    d.set_item("pw_quantiles", PWM_NOT_SELECTED)?;
    d.set_item("custom_quantiles", EXTRAMODELS_NOT_SELECTED)?;
    d.set_item(
        "ml_max",
        GenExtreme.ml_max(r.ml_params[0], r.ml_params[1], shape),
    )?;
    d.set_item("revert2ml", GenExtreme.revert2ml(shape))?;
    put_arr(py, &d, "ml_pdf", r.ml_pdf.clone())?;
    put_arr(py, &d, "cp_pdf", r.cp_pdf.clone())?;
    d.set_item("custom_mean", EXTRAMODELS_NOT_SELECTED)?;
    Ok(d)
}

/// Reference `genextreme.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ics=None))]
fn pdf_genextreme_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = ics;
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let d = ref_pdf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", GenExtreme.revert2ml(shape))?;
    Ok(d)
}

/// Reference `genextreme.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, ics=None, extramodels=false))]
fn cdf_genextreme_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    extramodels: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ics, extramodels);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let ml_cdf = cdf_at(&GenExtreme, &r.ml_params, &y);
    let cp_cdf = cdf_at(&GenExtreme, &r.cp_params, &y);
    let d = ref_cdf_dict(py, &r, ml_cdf, cp_cdf, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", GenExtreme.revert2ml(shape))?;
    Ok(d)
}

/// Reference `genextreme.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, ics=None, extramodels=false, mlcp=true))]
fn rvs_genextreme_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    extramodels: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ics, extramodels);
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", CRHPFLAT_DMGS_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    ref_rvs_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)
}

/// Reference `genpareto.ppf`.
#[pyfunction]
#[pyo3(signature = (x, p, kloc=0.0, ics=None, fdalpha=0.01, customprior=0.0, minxi=-1.0, maxxi=2.0, means=false, waicscores=false, extramodels=false, pdf=false, dmgs=true, ru=false, ru_nsamples=100000))]
#[allow(clippy::too_many_arguments)]
fn ppf_genpareto_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    kloc: f64,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    fdalpha: f64,
    customprior: f64,
    minxi: f64,
    maxxi: f64,
    means: bool,
    waicscores: bool,
    extramodels: bool,
    pdf: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        kloc,
        ics,
        fdalpha,
        customprior,
        minxi,
        maxxi,
        means,
        waicscores,
        extramodels,
        pdf,
        dmgs,
        ru,
        ru_nsamples,
    );
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let shape = r.ml_params.get(1).copied().unwrap_or(0.0);
    let d = ref_ppf_dict(py, &r, GenPareto, &x, CRHPFLAT_DMGS_CPMETHOD)?;
    // genpareto-specific extras.
    let ml_max = if shape < 0.0 {
        -r.ml_params[0] / shape
    } else {
        f64::INFINITY
    };
    d.set_item("ml_max", ml_max)?;
    d.set_item("revert2ml", shape.abs() >= 1.0 || shape > 30.0)?;
    d.set_item("lp_quantiles", EXTRAMODELS_NOT_SELECTED)?;
    d.set_item("lp2_quantiles", EXTRAMODELS_NOT_SELECTED)?;
    d.set_item("dpi_quantiles", EXTRAMODELS_NOT_SELECTED)?;
    put_arr(py, &d, "ml_pdf", r.ml_pdf.clone())?;
    put_arr(py, &d, "cp_pdf", r.cp_pdf.clone())?;
    Ok(d)
}

/// Reference `genpareto.pdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, kloc=0.0, ics=None, minxi=-1.0, maxxi=2.0, ru=false, ru_nsamples=1000))]
#[allow(clippy::too_many_arguments)]
fn pdf_genpareto_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    kloc: f64,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    minxi: f64,
    maxxi: f64,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (kloc, ics, minxi, maxxi, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let shape = r.ml_params.get(1).copied().unwrap_or(0.0);
    let d = ref_pdf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", shape <= -1.0)?;
    Ok(d)
}

/// Reference `genpareto.cdf`.
#[pyfunction]
#[pyo3(signature = (x, y=None, kloc=0.0, ics=None, minxi=-1.0, maxxi=2.0, ru=false, ru_nsamples=1000))]
#[allow(clippy::too_many_arguments)]
fn cdf_genpareto_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    kloc: f64,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    minxi: f64,
    maxxi: f64,
    ru: bool,
    ru_nsamples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (kloc, ics, minxi, maxxi, ru, ru_nsamples);
    let x = x.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let shape = r.ml_params.get(1).copied().unwrap_or(0.0);
    let ml_cdf = cdf_at(&GenPareto, &r.ml_params, &y);
    let cp_cdf = cdf_at(&GenPareto, &r.cp_params, &y);
    let d = ref_cdf_dict(py, &r, ml_cdf, cp_cdf, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", shape <= -1.0)?;
    Ok(d)
}

/// Reference `genpareto.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, kloc=0.0, ics=None, minxi=-1.0, maxxi=2.0, extramodels=false, ru=false, mlcp=true))]
#[allow(clippy::too_many_arguments)]
fn rvs_genpareto_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    kloc: f64,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    minxi: f64,
    maxxi: f64,
    extramodels: bool,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (kloc, ics, minxi, maxxi, extramodels, ru);
    let x = x.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", CRHPFLAT_DMGS_CPMETHOD)?;
        return Ok(d);
    }
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    ref_rvs_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)
}

// ---------------------------------------------------------------------------
// B2.5: reference-named functions — `genextreme_p1` (predictor).
// ---------------------------------------------------------------------------

/// Reference `genextreme_p1.ppf`.
#[pyfunction]
#[pyo3(signature = (x, t, t0=None, n0=None, p=None, ics=None, fdalpha=0.01, minxi=-1.0, maxxi=1.0, means=false, waicscores=false, extramodels=false, pdf=false, dmgs=true, ru=false, ru_nsamples=100000, predictordata=true, centering=true))]
#[allow(clippy::too_many_arguments)]
fn ppf_genextreme_p1_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    t0: Option<f64>,
    n0: Option<f64>,
    p: Option<PyReadonlyArray1<'py, f64>>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    fdalpha: f64,
    minxi: f64,
    maxxi: f64,
    means: bool,
    waicscores: bool,
    extramodels: bool,
    pdf: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
    predictordata: bool,
    centering: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        ics,
        minxi,
        maxxi,
        means,
        waicscores,
        extramodels,
        pdf,
        dmgs,
        ru,
        ru_nsamples,
        predictordata,
        centering,
    );
    let x = x.as_slice()?.to_vec();
    let t = t.as_slice()?.to_vec();
    let p = match p {
        Some(v) => v.as_slice()?.to_vec(),
        None => (1..10).map(|i| i as f64 * 0.1).collect(),
    };
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x.clone(), x.clone(), p, None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], fdalpha),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let d = ref_ppf_dict(py, &r, model.clone(), &x, CRHPFLAT_DMGS_CPMETHOD)?;
    // genextreme_p1-specific extras. `predictedparameter` is the decentered
    // location `a + b*(t_i - mean(t))`; `adjustedx` is the generalized residual
    // `qgev(pgev(x_i; loc_i, sigma, xi); loc0, sigma, xi)`.
    let a = r.ml_params[0];
    let b = r.ml_params[1];
    let sigma = r.ml_params[2];
    let predictedparameter: Vec<f64> = model.t.iter().map(|&ti| a + b * (ti - mean_t)).collect();
    let adjustedx: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let p_i = model.cdf(xi, a, b, sigma, shape, i);
            model.quantile(p_i, a, b, sigma, shape, 0)
        })
        .collect();
    put_arr(py, &d, "predictedparameter", predictedparameter)?;
    put_arr(py, &d, "adjustedx", adjustedx)?;
    d.set_item("revert2ml", model.revert2ml(shape))?;
    d.set_item(
        "ml_max",
        model.ml_max(r.ml_params[0], r.ml_params[2], shape),
    )?;
    put_arr(py, &d, "ml_pdf", r.ml_pdf.clone())?;
    put_arr(py, &d, "cp_pdf", r.cp_pdf.clone())?;
    Ok(d)
}

/// Reference `genextreme_p1.pdf`.
#[pyfunction]
#[pyo3(signature = (x, t, t0=None, n0=None, y=None, ics=None, fdalpha=0.01, minxi=-1.0, maxxi=1.0, extramodels=false, pdf=false, dmgs=true, ru=false, ru_nsamples=1000, predictordata=true, centering=true))]
#[allow(clippy::too_many_arguments)]
fn pdf_genextreme_p1_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    t0: Option<f64>,
    n0: Option<f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    fdalpha: f64,
    minxi: f64,
    maxxi: f64,
    extramodels: bool,
    pdf: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
    predictordata: bool,
    centering: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        ics,
        minxi,
        maxxi,
        extramodels,
        pdf,
        dmgs,
        ru,
        ru_nsamples,
        predictordata,
        centering,
    );
    let x = x.as_slice()?.to_vec();
    let t = t.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], fdalpha),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let d = ref_pdf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", model.revert2ml(shape))?;
    Ok(d)
}

/// Reference `genextreme_p1.cdf`.
#[pyfunction]
#[pyo3(signature = (x, t, t0=None, n0=None, y=None, ics=None, fdalpha=0.01, minxi=-1.0, maxxi=1.0, extramodels=false, dmgs=true, ru=false, ru_nsamples=1000, predictordata=true, centering=true))]
#[allow(clippy::too_many_arguments)]
fn cdf_genextreme_p1_ref<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    t0: Option<f64>,
    n0: Option<f64>,
    y: Option<PyReadonlyArray1<'py, f64>>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    fdalpha: f64,
    minxi: f64,
    maxxi: f64,
    extramodels: bool,
    dmgs: bool,
    ru: bool,
    ru_nsamples: usize,
    predictordata: bool,
    centering: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (
        ics,
        minxi,
        maxxi,
        extramodels,
        dmgs,
        ru,
        ru_nsamples,
        predictordata,
        centering,
    );
    let x = x.as_slice()?.to_vec();
    let t = t.as_slice()?.to_vec();
    let y = match y {
        Some(v) => v.as_slice()?.to_vec(),
        None => x.clone(),
    };
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x, y.clone(), vec![0.5], None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], fdalpha),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let ml_cdf = cdf_at(&model, &r.ml_params, &y);
    let cp_cdf = cdf_at(&model, &r.cp_params, &y);
    let d = ref_cdf_dict(py, &r, ml_cdf, cp_cdf, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", model.revert2ml(shape))?;
    Ok(d)
}

/// Reference `genextreme_p1.rvs`.
#[pyfunction]
#[pyo3(signature = (n, x, t, t0=None, n0=None, ics=None, minxi=-1.0, maxxi=1.0, extramodels=false, ru=false, mlcp=true))]
#[allow(clippy::too_many_arguments)]
fn rvs_genextreme_p1_ref<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    t0: Option<f64>,
    n0: Option<f64>,
    ics: Option<PyReadonlyArray1<'py, f64>>,
    minxi: f64,
    maxxi: f64,
    extramodels: bool,
    ru: bool,
    mlcp: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (ics, minxi, maxxi, extramodels, ru);
    let x = x.as_slice()?.to_vec();
    let t = t.as_slice()?.to_vec();
    if !mlcp {
        let d = PyDict::new(py);
        d.set_item("ml_params", "mlcp not selected")?;
        d.set_item("ml_deviates", "mlcp not selected")?;
        d.set_item("cp_deviates", "mlcp not selected")?;
        d.set_item("ru_deviates", RU_NOT_SELECTED)?;
        d.set_item("cp_method", CRHPFLAT_DMGS_CPMETHOD)?;
        return Ok(d);
    }
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let p = rvs_uniform_probs(py, n)?;
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    ref_rvs_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)
}

// ---------------------------------------------------------------------------
// B2.6: reference `reltest` module function.
// ---------------------------------------------------------------------------

/// Reference `reltest` (delegates to the core reliability test).
///
/// Returns `{'actual_p_ml', 'actual_p_cp', 'p'}` as Python **lists** (the
/// reference `reltest_dep.py` return shape).
#[pyfunction]
#[pyo3(signature = (model, ntrials=100, nx=30, p=None, true_params=None, init=None))]
fn reltest_ref<'py>(
    py: Python<'py>,
    model: &str,
    ntrials: usize,
    nx: usize,
    p: Option<PyReadonlyArray1<'py, f64>>,
    true_params: Option<PyReadonlyArray1<'py, f64>>,
    init: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    use fitdistcp_core::reltest::{reltest, RelModel};
    let rel_model = match model {
        "expon" => RelModel::Expon,
        "genextreme" => RelModel::GenExtreme,
        "genpareto" => RelModel::GenPareto,
        "gumbel" => RelModel::Gumbel,
        "norm" => RelModel::Norm,
        "gamma" => RelModel::Gamma,
        "weibull" => RelModel::Weibull,
        "lnorm" => RelModel::LogNormal,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown reltest model: {other}"
            )))
        }
    };
    let p = match p {
        Some(v) => v.as_slice()?.to_vec(),
        None => (1..10000).map(|i| 0.0001 * i as f64).collect(),
    };
    let true_params = match true_params {
        Some(v) => v.as_slice()?.to_vec(),
        None => default_reltest_params(rel_model),
    };
    let init = match init {
        Some(v) => v.as_slice()?.to_vec(),
        None => default_reltest_params(rel_model),
    };
    let res = reltest(rel_model, ntrials, nx, &p, &true_params, &init);
    let d = PyDict::new(py);
    d.set_item("actual_p_ml", res.actual_p_ml)?;
    d.set_item("actual_p_cp", res.actual_p_cp)?;
    d.set_item("p", res.p)?;
    Ok(d)
}

/// Default true/init parameters for each `reltest` model (reference defaults).
fn default_reltest_params(model: fitdistcp_core::reltest::RelModel) -> Vec<f64> {
    use fitdistcp_core::reltest::RelModel::*;
    match model {
        Expon => vec![1.0],
        GenExtreme => vec![0.0, 1.0, 0.0],
        GenPareto => vec![1.0, 0.0],
        Gumbel => vec![0.0, 1.0],
        Norm => vec![0.0, 1.0],
        Gamma => vec![1.0, 1.0],
        Weibull => vec![1.0, 1.0],
        LogNormal => vec![0.0, 1.0],
    }
}

// ---------------------------------------------------------------------------
// Model wrappers.
//
// Each model gets the four §5.2 functions (`pdf`/`cdf`/`ppf`/`rvs`). The
// `ppf`/`pdf`/`cdf`/`rvs` wrappers call the shared core `predict` (or
// `predict_gev` for the GEV) and map the `FitResult` to the §5.2 field names.
//
// The `_p1` models accept `t`/`t0`/`n0` and pass them through (W14 completes
// the predictor semantics).
// ---------------------------------------------------------------------------

// --- Normal (`norm`) -------------------------------------------------------

/// Normal CP predictive quantiles (mirrors R `qnorm_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    ppf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Normal CP predictive densities (mirrors R `dnorm_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // `make_request(y, x, ...)` maps the first arg to `req.y` (training data)
    // and the second to `req.x` (prediction grid). The core evaluates the
    // predictive density at `req.x`, so the data `x` must be passed first and
    // the grid `y` second (mirroring the R `dnorm_cp(x, y)` wrapper).
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    pdf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Normal CP predictive CDFs (mirrors R `pnorm_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_norm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&Normal, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Normal, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD)
}

/// Normal CP random deviates (mirrors R `rnorm_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_norm<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    rvs_dict(py, &r, ANALYTIC_CPMETHOD)
}

// --- Normal + predictor (`norm_p1`) ----------------------------------------

/// Normal-with-predictor CP predictive quantiles (mirrors R `qnorm_p1_cp`).
///
/// W14: the reference `norm_p1` CP is **analytic** (a Student-t predictive
/// distribution), not the DMGS expansion. The predictor is centered by its mean
/// and the prediction point is resolved via `maket0`.
#[pyfunction]
#[pyo3(signature = (x, p, t=None, t0=None, n0=None, means=None))]
fn ppf_norm_p1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict_norm_p1(&model, &req);
    ppf_dict_norm_p1(py, &r, mean_t, ANALYTIC_CPMETHOD)
}

/// Normal-with-predictor CP predictive densities (mirrors R `dnorm_p1_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, t=None, t0=None, n0=None, means=None))]
fn pdf_norm_p1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = predict_norm_p1(&model, &req);
    pdf_dict_norm_p1(py, &r, mean_t, ANALYTIC_CPMETHOD)
}

/// Normal-with-predictor CP predictive CDFs (mirrors R `pnorm_p1_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, t=None, t0=None, n0=None, means=None))]
fn cdf_norm_p1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, None);
    let r = predict_norm_p1(&model, &req);
    cdf_dict_norm_p1(py, &r, mean_t, ANALYTIC_CPMETHOD)
}

/// Normal-with-predictor CP random deviates (mirrors R `rnorm_p1_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, t=None, t0=None, n0=None, means=None))]
fn rvs_norm_p1<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = predict_norm_p1(&model, &req);
    rvs_dict_norm_p1(py, &r, mean_t, ANALYTIC_CPMETHOD)
}

// --- GEV (`gev`) -----------------------------------------------------------

/// GEV CP predictive quantiles (mirrors R `qgev_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_gev<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let d = ppf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item(
        "ml_max",
        GenExtreme.ml_max(r.ml_params[0], r.ml_params[1], shape),
    )?;
    d.set_item("revert2ml", GenExtreme.revert2ml(shape))?;
    Ok(d)
}

/// GEV CP predictive densities (mirrors R `dgev_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_gev<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let d = pdf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", GenExtreme.revert2ml(shape))?;
    Ok(d)
}

/// GEV CP predictive CDFs (mirrors R `pgev_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_gev<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let ml_cdf = cdf_at(&GenExtreme, &r.ml_params, &y);
    let cp_cdf = cdf_at(&GenExtreme, &r.cp_params, &y);
    let d = cdf_dict(py, &r, ml_cdf, cp_cdf, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", GenExtreme.revert2ml(shape))?;
    Ok(d)
}

/// GEV CP random deviates (mirrors R `rgev_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_gev<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    rvs_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)
}

// --- GEV + predictor (`gev_p1`) --------------------------------------------

/// GEV-with-predictor CP predictive quantiles (mirrors R `qgev_p1_cp`).
///
/// W14: the reference `gev_p1` CP uses the DMGS CRHP/flat path with the
/// `revert2ml` guard and the quantile-derived CP density, exactly like `qgev_cp`.
#[pyfunction]
#[pyo3(signature = (x, p, t=None, t0=None, n0=None, means=None))]
fn ppf_gev_p1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let d = ppf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item(
        "ml_max",
        model.ml_max(r.ml_params[0], r.ml_params[2], shape),
    )?;
    d.set_item("revert2ml", model.revert2ml(shape))?;
    Ok(d)
}

/// GEV-with-predictor CP predictive densities (mirrors R `dgev_p1_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, t=None, t0=None, n0=None, means=None))]
fn pdf_gev_p1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let d = pdf_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", model.revert2ml(shape))?;
    Ok(d)
}

/// GEV-with-predictor CP predictive CDFs (mirrors R `pgev_p1_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, t=None, t0=None, n0=None, means=None))]
fn cdf_gev_p1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let ml_cdf = cdf_at(&model, &r.ml_params, &y);
    let cp_cdf = cdf_at(&model, &r.cp_params, &y);
    let d = cdf_dict(py, &r, ml_cdf, cp_cdf, CRHPFLAT_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", model.revert2ml(shape))?;
    Ok(d)
}

/// GEV-with-predictor CP random deviates (mirrors R `rgev_p1_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, t=None, t0=None, n0=None, means=None))]
fn rvs_gev_p1<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    t: Option<PyReadonlyArray1<'py, f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let t = t
        .map(|v| v.as_slice().map(|s| s.to_vec()))
        .transpose()?
        .unwrap_or_default();
    let _ = means;
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    rvs_dict(py, &r, CRHPFLAT_DMGS_CPMETHOD)
}

// --- GPD (`gpd_k1`) --------------------------------------------------------

/// GPD CP predictive quantiles (mirrors R `qgpd_k1_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_gpd_k1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let shape = r.ml_params.get(1).copied().unwrap_or(0.0);
    let d = ppf_dict(py, &r, ADHOC_DMGS_CPMETHOD)?;
    d.set_item("revert2ml", shape.abs() >= 1.0 || shape > 30.0)?;
    Ok(d)
}

/// GPD CP predictive densities (mirrors R `dgpd_k1_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_gpd_k1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    pdf_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

/// GPD CP predictive CDFs (mirrors R `pgpd_k1_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_gpd_k1<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let ml_cdf = cdf_at(&GenPareto, &r.ml_params, &y);
    let cp_cdf = cdf_at(&GenPareto, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, ADHOC_DMGS_CPMETHOD)
}

/// GPD CP random deviates (mirrors R `rgpd_k1_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_gpd_k1<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    rvs_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

// --- Gumbel (`gumbel`) -----------------------------------------------------

/// Gumbel CP predictive quantiles (mirrors R `qgumbel_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_gumbel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    ppf_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Gumbel CP predictive densities (mirrors R `dgumbel_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_gumbel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    pdf_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Gumbel CP predictive CDFs (mirrors R `pgumbel_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_gumbel<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&Gumbel, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Gumbel, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, RHP_DMGS_CPMETHOD)
}

/// Gumbel CP random deviates (mirrors R `rgumbel_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_gumbel<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    rvs_dict(py, &r, RHP_DMGS_CPMETHOD)
}

// --- Log-normal (`lnorm`) --------------------------------------------------

/// Log-normal CP predictive quantiles (mirrors R `qlnorm_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_lnorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    ppf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Log-normal CP predictive densities (mirrors R `dlnorm_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_lnorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    pdf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Log-normal CP predictive CDFs (mirrors R `plnorm_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_lnorm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&LogNormal, &r.ml_params, &y);
    let cp_cdf = cdf_at(&LogNormal, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD)
}

/// Log-normal CP random deviates (mirrors R `rlnorm_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_lnorm<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    rvs_dict(py, &r, ANALYTIC_CPMETHOD)
}

// --- Weibull (`weibull`) ---------------------------------------------------

/// Weibull CP predictive quantiles (mirrors R `qweibull_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_weibull<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    ppf_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Weibull CP predictive densities (mirrors R `dweibull_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_weibull<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    pdf_dict(py, &r, RHP_DMGS_CPMETHOD)
}

/// Weibull CP predictive CDFs (mirrors R `pweibull_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_weibull<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    let ml_cdf = cdf_at(&Weibull, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Weibull, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, RHP_DMGS_CPMETHOD)
}

/// Weibull CP random deviates (mirrors R `rweibull_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_weibull<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    rvs_dict(py, &r, RHP_DMGS_CPMETHOD)
}

// --- Gamma (`gamma`) -------------------------------------------------------

/// Gamma CP predictive quantiles (mirrors R `qgamma_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_gamma<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    ppf_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

/// Gamma CP predictive densities (mirrors R `dgamma_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_gamma<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    pdf_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

/// Gamma CP predictive CDFs (mirrors R `pgamma_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_gamma<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    let ml_cdf = cdf_at(&Gamma, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Gamma, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, ADHOC_DMGS_CPMETHOD)
}

/// Gamma CP random deviates (mirrors R `rgamma_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_gamma<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    rvs_dict(py, &r, ADHOC_DMGS_CPMETHOD)
}

// --- Exponential (`exp`) ---------------------------------------------------

/// Exponential CP predictive quantiles (mirrors R `qexp_cp`).
#[pyfunction]
#[pyo3(signature = (x, p, means=None))]
fn ppf_exp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let p = p.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Exponential, &req, &[1.0]);
    ppf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Exponential CP predictive densities (mirrors R `dexp_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn pdf_exp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    // Data `x` first (→ `req.y`), grid `y` second (→ `req.x`); see `pdf_norm`.
    let req = make_request(x, y, vec![0.5], None, None, None, means);
    let r = predict(Exponential, &req, &[1.0]);
    pdf_dict(py, &r, ANALYTIC_CPMETHOD)
}

/// Exponential CP predictive CDFs (mirrors R `pexp_cp`).
#[pyfunction]
#[pyo3(signature = (x, y, means=None))]
fn cdf_exp<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let y = y.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let req = make_request(y.clone(), x, vec![0.5], None, None, None, means);
    let r = predict(Exponential, &req, &[1.0]);
    let ml_cdf = cdf_at(&Exponential, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Exponential, &r.cp_params, &y);
    cdf_dict(py, &r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD)
}

/// Exponential CP random deviates (mirrors R `rexp_cp`).
#[pyfunction]
#[pyo3(signature = (n, x, means=None))]
fn rvs_exp<'py>(
    py: Python<'py>,
    n: usize,
    x: PyReadonlyArray1<'py, f64>,
    means: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_slice()?.to_vec();
    let means = means
        .map(|m| m.as_slice().map(|s| s.to_vec()))
        .transpose()?;
    let p = rvs_probs(n.max(1));
    let req = make_request(x.clone(), x, p, None, None, None, means);
    let r = predict(Exponential, &req, &[1.0]);
    rvs_dict(py, &r, ANALYTIC_CPMETHOD)
}

// ---------------------------------------------------------------------------
// Registration.
// ---------------------------------------------------------------------------

/// Python module definition.
///
/// B1: the extension module is private (`fitdistcp._fitdistcp`); the public
/// reference API is provided by the thin Python submodules under
/// `python/fitdistcp/`. `pyproject.toml` sets `module-name =
/// "fitdistcp._fitdistcp"`, so this must export `PyInit__fitdistcp`.
#[pymodule]
fn _fitdistcp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 0 smoke tests.
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(add_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(sum_values, m)?)?;

    // B2: reference-named functions (delegated to by the Python submodules).
    // norm
    m.add_function(wrap_pyfunction!(ppf_norm_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_norm_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_norm_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_norm_ref, m)?)?;
    // expon
    m.add_function(wrap_pyfunction!(ppf_expon_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_expon_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_expon_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_expon_ref, m)?)?;
    // lnorm
    m.add_function(wrap_pyfunction!(ppf_lnorm_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_lnorm_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_lnorm_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_lnorm_ref, m)?)?;
    // gumbel
    m.add_function(wrap_pyfunction!(ppf_gumbel_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gumbel_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gumbel_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gumbel_ref, m)?)?;
    // weibull
    m.add_function(wrap_pyfunction!(ppf_weibull_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_weibull_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_weibull_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_weibull_ref, m)?)?;
    // gamma
    m.add_function(wrap_pyfunction!(ppf_gamma_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gamma_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gamma_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gamma_ref, m)?)?;
    // genextreme
    m.add_function(wrap_pyfunction!(ppf_genextreme_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_genextreme_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_genextreme_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_genextreme_ref, m)?)?;
    // genpareto
    m.add_function(wrap_pyfunction!(ppf_genpareto_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_genpareto_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_genpareto_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_genpareto_ref, m)?)?;
    // genextreme_p1
    m.add_function(wrap_pyfunction!(ppf_genextreme_p1_ref, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_genextreme_p1_ref, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_genextreme_p1_ref, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_genextreme_p1_ref, m)?)?;
    // reltest
    m.add_function(wrap_pyfunction!(reltest_ref, m)?)?;

    // norm
    m.add_function(wrap_pyfunction!(ppf_norm, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_norm, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_norm, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_norm, m)?)?;

    // norm_p1
    m.add_function(wrap_pyfunction!(ppf_norm_p1, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_norm_p1, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_norm_p1, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_norm_p1, m)?)?;

    // gev
    m.add_function(wrap_pyfunction!(ppf_gev, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gev, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gev, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gev, m)?)?;

    // gev_p1
    m.add_function(wrap_pyfunction!(ppf_gev_p1, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gev_p1, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gev_p1, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gev_p1, m)?)?;

    // gpd_k1
    m.add_function(wrap_pyfunction!(ppf_gpd_k1, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gpd_k1, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gpd_k1, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gpd_k1, m)?)?;

    // gumbel
    m.add_function(wrap_pyfunction!(ppf_gumbel, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gumbel, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gumbel, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gumbel, m)?)?;

    // lnorm
    m.add_function(wrap_pyfunction!(ppf_lnorm, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_lnorm, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_lnorm, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_lnorm, m)?)?;

    // weibull
    m.add_function(wrap_pyfunction!(ppf_weibull, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_weibull, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_weibull, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_weibull, m)?)?;

    // gamma
    m.add_function(wrap_pyfunction!(ppf_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_gamma, m)?)?;

    // exp
    m.add_function(wrap_pyfunction!(ppf_exp, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_exp, m)?)?;
    m.add_function(wrap_pyfunction!(cdf_exp, m)?)?;
    m.add_function(wrap_pyfunction!(rvs_exp, m)?)?;

    Ok(())
}
