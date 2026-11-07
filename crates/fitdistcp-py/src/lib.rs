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
/// The core reports `aic = 2k - 2*loglik`, so `loglik = k - aic/2`.
fn ml_value(r: &FitResult) -> f64 {
    r.aic / -2.0 + r.ml_params.len() as f64
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
fn rvs_probs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect()
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
#[pymodule]
fn fitdistcp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 0 smoke tests.
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(add_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(sum_values, m)?)?;

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
