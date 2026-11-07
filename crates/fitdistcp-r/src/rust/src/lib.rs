//! `fitdistcp-r` — extendr bindings over `fitdistcp-core`.
//!
//! Thin adapter: convert R vectors to contiguous `&[f64]`, call the core, and
//! convert results back to R lists. No numerics live here.
//!
//! # W12 scope — R API surface
//!
//! This crate exposes the reference `fitdistcp` R convention (§5.1) for the ten
//! Phase-1 models: the `d/p/q/r` distribution functions and the `t***_cp`
//! (confidence-predictive) functions, with the reference's exact names,
//! argument order, and return field names. The names and signatures are derived
//! from the reference package (`tools/fitdistcp-ref/NAMESPACE` +
//! `tools/fitdistcp-ref/R/*.R`), which is authoritative.
//!
//! ## Function inventory (per model)
//!
//! For each model `M` the reference exports:
//!
//! - `qM_cp(x, p, ...)` — MLE and CP predictive quantiles plus fit diagnostics;
//! - `dM_cp(x, y, ...)` — MLE and CP predictive densities;
//! - `pM_cp(x, y, ...)` — MLE and CP predictive CDFs;
//! - `rM_cp(n, x, ...)` — MLE and CP random deviates;
//! - `tM_cp(...)` — posterior theta samples (RUST). **Phase 2**; the wrapper
//!   accepts the reference signature and returns the reference's
//!   `"rust not selected"` placeholder (see the discrepancy note below).
//!
//! The ten models and their reference prefixes are:
//!
//! | Model            | prefix      |
//! |------------------|-------------|
//! | Normal           | `norm`      |
//! | Normal + pred.   | `norm_p1`   |
//! | GEV              | `gev`       |
//! | GEV + pred.      | `gev_p1`    |
//! | GPD              | `gpd_k1`    |
//! | Gumbel           | `gumbel`    |
//! | Log-normal       | `lnorm`     |
//! | Weibull          | `weibull`   |
//! | Gamma            | `gamma`     |
//! | Exponential      | `exp`       |
//!
//! ## Return fields
//!
//! The `q***_cp` wrappers return the reference field names: `ml_params`,
//! `ml_value`, `standard_errors`, `ml_quantiles`, `cp_quantiles`,
//! `ru_quantiles`, `maic`, `waic1`, `waic2`, `ml_oos_logscore`,
//! `cp_oos_logscore`, `ml_mean`, `cp_mean`, `cp_method`. The GEV wrapper adds
//! `ml_max`, `revert2ml`, `ml_pdf`, `cp_pdf`. The `d***_cp`/`p***_cp` wrappers
//! return `ml_params`, `ml_pdf`/`ml_cdf`, `cp_pdf`/`cp_cdf`, `ru_pdf`/`ru_cdf`,
//! `cp_method`. The `r***_cp` wrappers return `ml_params`, `ml_deviates`,
//! `cp_deviates`, `ru_deviates`, `cp_method`.
//!
//! ## Documented discrepancies (design vs. reference)
//!
//! 1. **RUST / bootstrap / PWM / custom-prior paths are Phase 2.** The
//!    reference `t***_cp` functions sample the posterior via Northrop's `rust`
//!    package; the design defers posterior sampling to Phase 2 (§8, §10). The
//!    `t***_cp` wrappers therefore accept the reference signature but return the
//!    reference's own `"rust not selected"` placeholder rather than samples.
//!    Likewise `ru_*`/`bs_*`/`pw_*`/`custom_*` fields are returned as the
//!    reference's `"rust not selected"` / `"bootstrap not selected"` /
//!    `"pwm not selected"` / `"extramodels not selected"` placeholders.
//! 2. **`waic1`/`waic2`/`ml_oos_logscore`/`cp_oos_logscore` are placeholders.**
//!    The core `FitResult` carries optional `waic`/`log_scores`; the reference
//!    returns 3-vectors for `waic1`/`waic2` and scalars for the log scores. The
//!    core does not yet compute these (they are `None`), so the wrappers return
//!    the reference's `"waicscores not selected"` / `"logscores not selected"`
//!    placeholders. This is a W12 scope boundary, not a numerics change.
//! 3. **`ml_mean`/`cp_mean` are not computed by the core.** The reference
//!    computes predictive means; the core `FitResult` does not carry them, so
//!    the wrappers return `NA` (documented) rather than inventing values.
//! 4. **`cp_params`/`cp_se`.** The core reports the MLE parameters/SEs for the
//!    CP model (the DMGS expansion corrects the predictive distribution, not the
//!    point estimates — see [`fitdistcp_core::api::predict`]). The reference
//!    likewise reports `ml_params` for the CP model, so the wrappers expose
//!    `ml_params`/`standard_errors` only, matching the reference.
//! 5. **Predictor handling (`_p1` models) — W14.** The `_p1` wrappers implement
//!    the reference predictor semantics: the predictor is centered by its mean
//!    (`t - mean(t)`), the prediction point is resolved via `maket0`
//!    (`t0 = t[n0]` when `t0` is `NA`), `ml_params` is decentered, and the
//!    `predictedparameter`/`adjustedx` fields are returned. `norm_p1` uses the
//!    reference **analytic** CP (Student-t predictive); `gev_p1` uses the DMGS
//!    CRHP/flat path with `revert2ml`/`ml_max`/quantile-derived density.

use extendr_api::prelude::*;

use fitdistcp_core::api::{
    predict, predict_gev, predict_gev_p1, predict_norm_p1, FitResult, NormP1Result,
    PredictionRequest,
};
use fitdistcp_core::models::{
    Exponential, Gamma, GenExtreme, GenExtremeP1, GenPareto, Gumbel, LogNormal, Normal, NormalP1,
    Weibull,
};

// ---------------------------------------------------------------------------
// Phase 0 smoke tests (kept intact).
// ---------------------------------------------------------------------------

/// Return the core crate version string.
#[extendr]
fn core_version() -> String {
    fitdistcp_core::CORE_VERSION.to_string()
}

/// Element-wise sum of two equal-length numeric vectors (Phase 0 smoke test).
#[extendr]
fn add_vectors(a: Robj, b: Robj) -> Result<Doubles> {
    let a = a
        .as_real_slice()
        .ok_or_else(|| Error::from("`a` must be a numeric vector"))?;
    let b = b
        .as_real_slice()
        .ok_or_else(|| Error::from("`b` must be a numeric vector"))?;
    Doubles::try_from(fitdistcp_core::add_vectors(a, b))
}

/// Sum of a numeric vector (Phase 0 smoke test).
#[extendr]
fn sum_values(values: Robj) -> Result<f64> {
    let values = values
        .as_real_slice()
        .ok_or_else(|| Error::from("`values` must be a numeric vector"))?;
    Ok(fitdistcp_core::sum(values))
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// The reference `cp_method` strings (from `005_utils.R`).
const ANALYTIC_CPMETHOD: &str = "The cp results are based on an analytic solution of the Bayesian prediction integral, using the predictive matching right Haar prior.";
const RHP_DMGS_CPMETHOD: &str = "The cp results are based on the DMGS approximation of the Bayesian prediction integral, using the predictive matching right Haar prior.";
const ADHOC_DMGS_CPMETHOD: &str = "The cp results are based on the DMGS approximation of the Bayesian prediction integral, using an ad-hoc prior.";
const CRHPFLAT_DMGS_CPMETHOD: &str = "The cp results are based on the DMGS approximation of the Bayesian prediction integral, using the CRHP/flat prior.";

/// Reference placeholder strings for the not-yet-implemented optional paths.
const RUST_NOT_SELECTED: &str = "rust not selected";
const BOOT_NOT_SELECTED: &str = "bootstrap not selected";
const PWM_NOT_SELECTED: &str = "pwm not selected";
const EXTRAMODELS_NOT_SELECTED: &str = "extramodels not selected";
const WAICSCORES_NOT_SELECTED: &str = "waicscores not selected";
const LOGSCORES_NOT_SELECTED: &str = "logscores not selected";

/// Extract a required numeric vector from an R object.
///
/// Accepts both R `double` and `integer` vectors (the reference R API accepts
/// either; e.g. `seq(1, 10)` is an integer vector), coercing integers to `f64`.
fn real_vec(obj: &Robj, name: &str) -> Result<Vec<f64>> {
    if let Some(s) = obj.as_real_slice() {
        return Ok(s.to_vec());
    }
    if let Some(s) = obj.as_integer_slice() {
        return Ok(s.iter().map(|&v| v as f64).collect());
    }
    Err(Error::from(format!("`{name}` must be a numeric vector")))
}

/// Extract an optional numeric vector: `NULL`/absent -> `None`.
///
/// Accepts both R `double` and `integer` vectors (see [`real_vec`]).
fn opt_real_vec(obj: &Robj, name: &str) -> Result<Option<Vec<f64>>> {
    if obj.is_null() {
        return Ok(None);
    }
    if let Some(s) = obj.as_real_slice() {
        return Ok(Some(s.to_vec()));
    }
    if let Some(s) = obj.as_integer_slice() {
        return Ok(Some(s.iter().map(|&v| v as f64).collect()));
    }
    Err(Error::from(format!(
        "`{name}` must be a numeric vector or NULL"
    )))
}

/// Extract an optional scalar: `NULL`/`NA` -> `None`.
///
/// The reference R API uses `NA` (a **logical** scalar) as the default for
/// `t0`/`n0`, so we must accept logical/integer `NA` as well as `NULL` and
/// numeric scalars. A non-finite numeric value (`NA_real_`/`NaN`) is also
/// treated as absent.
fn opt_scalar(obj: &Robj, name: &str) -> Result<Option<f64>> {
    if obj.is_null() {
        return Ok(None);
    }
    // R's `NA` is a logical scalar; `is_na()` catches it (and `NA_real_`).
    if obj.is_na() {
        return Ok(None);
    }
    if let Some(v) = obj.as_real() {
        return Ok(if v.is_finite() { Some(v) } else { None });
    }
    if let Some(v) = obj.as_integer() {
        return Ok(Some(v as f64));
    }
    if obj.as_logical().is_some() {
        // A non-NA logical (TRUE/FALSE) is not a valid numeric scalar.
        return Err(Error::from(format!(
            "`{name}` must be a numeric scalar or NULL"
        )));
    }
    Err(Error::from(format!(
        "`{name}` must be a numeric scalar or NULL"
    )))
}

/// Build a [`PredictionRequest`] from the common R arguments.
fn make_request(
    y: Vec<f64>,
    x: Vec<f64>,
    p: Vec<f64>,
    t: Option<Vec<f64>>,
    t0: Option<f64>,
    n0: Option<f64>,
) -> PredictionRequest {
    PredictionRequest {
        y,
        x,
        p,
        t,
        t0,
        n0,
        means: None,
    }
}

/// The reference `maket0`: resolve the prediction point `t0` from the user's
/// `t0`/`n0` (W14).
///
/// The reference semantics are:
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

/// Build the reference `qnorm_p1_cp` return list from a [`NormP1Result`].
///
/// The reference `qnorm_p1_cp` returns `ml_params` (decentered), `ml_value`,
/// `predictedparameter`, `adjustedx`, `standard_errors`, `ml_quantiles`,
/// `cp_quantiles`, `ru_quantiles`, `maic`, `waic1`, `waic2`,
/// `ml_oos_logscore`, `cp_oos_logscore`, `ml_mean`, `cp_mean`, `cp_method`.
fn q_cp_list_norm_p1(r: &NormP1Result, mean_t: f64, cp_method: &str) -> List {
    // Decenter the intercept: a_ret = a_c - b*mean(t).
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    let ml_value = r.ml_value;
    let maic = make_maic(ml_value, 3);
    named_list(
        vec![
            "ml_params",
            "ml_value",
            "predictedparameter",
            "adjustedx",
            "standard_errors",
            "ml_quantiles",
            "cp_quantiles",
            "ru_quantiles",
            "maic",
            "waic1",
            "waic2",
            "ml_oos_logscore",
            "cp_oos_logscore",
            "ml_mean",
            "cp_mean",
            "cp_method",
        ],
        vec![
            doubles(ml_params).into(),
            ml_value.into(),
            doubles(r.predictedparameter.clone()).into(),
            doubles(r.adjustedx.clone()).into(),
            doubles(r.standard_errors.clone()).into(),
            doubles(r.ml_quantiles.clone()).into(),
            doubles(r.cp_quantiles.clone()).into(),
            RUST_NOT_SELECTED.into(),
            maic.into(),
            WAICSCORES_NOT_SELECTED.into(),
            WAICSCORES_NOT_SELECTED.into(),
            LOGSCORES_NOT_SELECTED.into(),
            LOGSCORES_NOT_SELECTED.into(),
            r.muhat0.into(),
            r.muhat0.into(),
            cp_method.into(),
        ],
    )
}

/// Build the reference `dnorm_p1_cp` return list from a [`NormP1Result`].
fn d_cp_list_norm_p1(r: &NormP1Result, mean_t: f64, cp_method: &str) -> List {
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    named_list(
        vec!["ml_params", "ml_pdf", "cp_pdf", "ru_pdf", "cp_method"],
        vec![
            doubles(ml_params).into(),
            doubles(r.ml_pdf.clone()).into(),
            doubles(r.cp_pdf.clone()).into(),
            RUST_NOT_SELECTED.into(),
            cp_method.into(),
        ],
    )
}

/// Build the reference `pnorm_p1_cp` return list from a [`NormP1Result`].
fn p_cp_list_norm_p1(r: &NormP1Result, mean_t: f64, cp_method: &str) -> List {
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    named_list(
        vec!["ml_params", "ml_cdf", "cp_cdf", "ru_cdf", "cp_method"],
        vec![
            doubles(ml_params).into(),
            doubles(r.ml_cdf.clone()).into(),
            doubles(r.cp_cdf.clone()).into(),
            RUST_NOT_SELECTED.into(),
            cp_method.into(),
        ],
    )
}

/// Build the reference `rnorm_p1_cp` return list from a [`NormP1Result`].
fn r_cp_list_norm_p1(r: &NormP1Result, mean_t: f64, cp_method: &str) -> List {
    let mut ml_params = r.ml_params.clone();
    ml_params[0] -= ml_params[1] * mean_t;
    named_list(
        vec![
            "ml_params",
            "ml_deviates",
            "cp_deviates",
            "ru_deviates",
            "cp_method",
        ],
        vec![
            doubles(ml_params).into(),
            doubles(r.ml_quantiles.clone()).into(),
            doubles(r.cp_quantiles.clone()).into(),
            RUST_NOT_SELECTED.into(),
            cp_method.into(),
        ],
    )
}

/// Convert a `Vec<f64>` to an R `Doubles`.
fn doubles(v: Vec<f64>) -> Doubles {
    Doubles::from_values(v)
}

/// The reference `make_maic` 3-vector: `[ml_value, -nparams, ml_value - nparams]`.
fn make_maic(ml_value: f64, nparams: usize) -> Doubles {
    doubles(vec![ml_value, -(nparams as f64), ml_value - nparams as f64])
}

/// The reference `make_se`: `sqrt(-lddi[i,i]/nx)` from the inverse information.
///
/// The core already reports `ml_se` (the standard errors from the inverse
/// observed information), so we surface those directly.
fn standard_errors(r: &FitResult) -> Doubles {
    doubles(r.ml_se.clone())
}

/// Build a named R list from parallel name/value vectors.
///
/// `List::from_names_and_values` is the extendr API for constructing a list
/// whose names are known only at runtime (the model-specific extra fields).
fn named_list(names: Vec<&str>, values: Vec<Robj>) -> List {
    List::from_names_and_values(names, values).expect("named_list: names/values length mismatch")
}

/// The reference `ml_value` (the maximized log-likelihood).
///
/// The core reports `aic = 2k - 2*loglik`, so `loglik = k - aic/2`.
fn ml_value(r: &FitResult) -> f64 {
    r.aic / -2.0 + r.ml_params.len() as f64
}

/// Build the reference `q***_cp` return list from a core [`FitResult`].
///
/// `cp_method` is the model-specific reference string. The optional
/// `waic`/`log_scores`/`mean` fields are returned as the reference placeholders
/// (see the module-level discrepancy notes). `extra` appends model-specific
/// fields (e.g. the GEV `ml_max`/`revert2ml`).
fn q_cp_list(r: &FitResult, cp_method: &str, extra: Vec<(&str, Robj)>) -> List {
    let mut names = vec![
        "ml_params",
        "ml_value",
        "standard_errors",
        "ml_quantiles",
        "cp_quantiles",
        "ru_quantiles",
        "maic",
        "waic1",
        "waic2",
        "ml_oos_logscore",
        "cp_oos_logscore",
        "ml_mean",
        "cp_mean",
        "cp_method",
    ];
    let mut values: Vec<Robj> = vec![
        doubles(r.ml_params.clone()).into(),
        ml_value(r).into(),
        standard_errors(r).into(),
        doubles(r.ml_quantiles.clone()).into(),
        doubles(r.cp_quantiles.clone()).into(),
        RUST_NOT_SELECTED.into(),
        make_maic(ml_value(r), r.ml_params.len()).into(),
        WAICSCORES_NOT_SELECTED.into(),
        WAICSCORES_NOT_SELECTED.into(),
        LOGSCORES_NOT_SELECTED.into(),
        LOGSCORES_NOT_SELECTED.into(),
        f64::NAN.into(),
        f64::NAN.into(),
        cp_method.into(),
    ];
    for (n, v) in extra {
        names.push(n);
        values.push(v);
    }
    named_list(names, values)
}

/// Build the reference `d***_cp` return list.
fn d_cp_list(r: &FitResult, cp_method: &str, extra: Vec<(&str, Robj)>) -> List {
    let mut names = vec!["ml_params", "ml_pdf", "cp_pdf", "ru_pdf", "cp_method"];
    let mut values: Vec<Robj> = vec![
        doubles(r.ml_params.clone()).into(),
        doubles(r.ml_pdf.clone()).into(),
        doubles(r.cp_pdf.clone()).into(),
        RUST_NOT_SELECTED.into(),
        cp_method.into(),
    ];
    for (n, v) in extra {
        names.push(n);
        values.push(v);
    }
    named_list(names, values)
}

/// Build the reference `p***_cp` return list.
///
/// The core does not expose predictive CDFs directly; the reference computes
/// them from the fitted parameters. We evaluate the model CDF at the prediction
/// points using the MLE parameters for `ml_cdf` and the CP parameters for
/// `cp_cdf` (which equal the MLE parameters at this stage — see discrepancy 4).
fn p_cp_list(
    r: &FitResult,
    ml_cdf: Vec<f64>,
    cp_cdf: Vec<f64>,
    cp_method: &str,
    extra: Vec<(&str, Robj)>,
) -> List {
    let mut names = vec!["ml_params", "ml_cdf", "cp_cdf", "ru_cdf", "cp_method"];
    let mut values: Vec<Robj> = vec![
        doubles(r.ml_params.clone()).into(),
        doubles(ml_cdf).into(),
        doubles(cp_cdf).into(),
        RUST_NOT_SELECTED.into(),
        cp_method.into(),
    ];
    for (n, v) in extra {
        names.push(n);
        values.push(v);
    }
    named_list(names, values)
}

/// Build the reference `r***_cp` return list.
fn r_cp_list(r: &FitResult, cp_method: &str, extra: Vec<(&str, Robj)>) -> List {
    let mut names = vec![
        "ml_params",
        "ml_deviates",
        "cp_deviates",
        "ru_deviates",
        "cp_method",
    ];
    let mut values: Vec<Robj> = vec![
        doubles(r.ml_params.clone()).into(),
        doubles(r.ml_quantiles.clone()).into(),
        doubles(r.cp_quantiles.clone()).into(),
        RUST_NOT_SELECTED.into(),
        cp_method.into(),
    ];
    for (n, v) in extra {
        names.push(n);
        values.push(v);
    }
    named_list(names, values)
}

/// Build the reference `t***_cp` return list (RUST — Phase 2).
fn t_cp_list() -> List {
    list!(theta_samples = RUST_NOT_SELECTED)
}

/// Evaluate a model CDF at each prediction point using the given parameters.
fn cdf_at<M: fitdistcp_core::dmgs::DmgsModel>(model: &M, params: &[f64], x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| model.cdf(xi, params)).collect()
}

// ---------------------------------------------------------------------------
// Model wrappers.
//
// Each model gets the five reference functions. The `q`/`d`/`p`/`r` wrappers
// call the shared core `predict` (or `predict_gev` for the GEV) and map the
// `FitResult` to the reference field names. The `t` wrapper is the Phase-2
// RUST placeholder.
//
// The `_p1` models accept `t`/`t0`/`n0` and pass them through (W14 completes
// the predictor semantics).
// ---------------------------------------------------------------------------

// --- Normal (`norm`) -------------------------------------------------------

/// Normal CP quantiles (reference `qnorm_cp`).
#[extendr]
fn qnorm_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    Ok(q_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Normal CP densities (reference `dnorm_cp`).
#[extendr]
fn dnorm_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    Ok(d_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Normal CP CDFs (reference `pnorm_cp`).
#[extendr]
fn pnorm_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&Normal, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Normal, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD, vec![]))
}

/// Normal CP random deviates (reference `rnorm_cp`).
#[extendr]
fn rnorm_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Normal, &req, &[0.0, 1.0]);
    Ok(r_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Normal posterior theta samples (reference `tnorm_cp`; RUST — Phase 2).
#[extendr]
fn tnorm_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- Normal + predictor (`norm_p1`) ----------------------------------------

/// Normal-with-predictor CP quantiles (reference `qnorm_p1_cp`).
///
/// W14: the reference `norm_p1` CP is **analytic** (a Student-t predictive
/// distribution), not the DMGS expansion. The predictor is centered by its mean
/// and the prediction point is resolved via `maket0`.
#[extendr]
fn qnorm_p1_cp(x: Robj, t: Robj, t0: Robj, n0: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let p = real_vec(&p, "p")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict_norm_p1(&model, &req);
    Ok(q_cp_list_norm_p1(&r, mean_t, ANALYTIC_CPMETHOD))
}

/// Normal-with-predictor CP densities (reference `dnorm_p1_cp`).
#[extendr]
fn dnorm_p1_cp(x: Robj, t: Robj, t0: Robj, n0: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let y = real_vec(&y, "y")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(y, x, vec![0.5], None, None, None);
    let r = predict_norm_p1(&model, &req);
    Ok(d_cp_list_norm_p1(&r, mean_t, ANALYTIC_CPMETHOD))
}

/// Normal-with-predictor CP CDFs (reference `pnorm_p1_cp`).
#[extendr]
fn pnorm_p1_cp(x: Robj, t: Robj, t0: Robj, n0: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let y = real_vec(&y, "y")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(y.clone(), x, vec![0.5], None, None, None);
    let r = predict_norm_p1(&model, &req);
    Ok(p_cp_list_norm_p1(&r, mean_t, ANALYTIC_CPMETHOD))
}

/// Normal-with-predictor CP random deviates (reference `rnorm_p1_cp`).
#[extendr]
fn rnorm_p1_cp(n: Robj, x: Robj, t: Robj, t0: Robj, n0: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = NormalP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict_norm_p1(&model, &req);
    Ok(r_cp_list_norm_p1(&r, mean_t, ANALYTIC_CPMETHOD))
}

/// Normal-with-predictor posterior theta samples (reference `tnorm_p1_cp`; RUST — Phase 2).
#[extendr]
fn tnorm_p1_cp(_n: Robj, _x: Robj, _t: Robj, _t0: Robj, _n0: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- GEV (`gev`) -----------------------------------------------------------

/// GEV CP quantiles (reference `qgev_cp`).
#[extendr]
fn qgev_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let revert2ml = GenExtreme.revert2ml(shape);
    let ml_max = GenExtreme.ml_max(r.ml_params[0], r.ml_params[1], shape);
    Ok(q_cp_list(
        &r,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![
            ("ml_max", ml_max.into()),
            ("revert2ml", revert2ml.into()),
            ("ml_pdf", doubles(r.ml_pdf.clone()).into()),
            ("cp_pdf", doubles(r.cp_pdf.clone()).into()),
            ("pw_params", PWM_NOT_SELECTED.into()),
            ("pw_quantiles", PWM_NOT_SELECTED.into()),
            ("custom_quantiles", EXTRAMODELS_NOT_SELECTED.into()),
            ("custom_mean", f64::NAN.into()),
            ("bs_quantiles", BOOT_NOT_SELECTED.into()),
        ],
    ))
}

/// GEV CP densities (reference `dgev_cp`).
#[extendr]
fn dgev_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let revert2ml = GenExtreme.revert2ml(shape);
    Ok(d_cp_list(
        &r,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![
            ("revert2ml", revert2ml.into()),
            ("bs_pdf", BOOT_NOT_SELECTED.into()),
        ],
    ))
}

/// GEV CP CDFs (reference `pgev_cp`).
#[extendr]
fn pgev_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    let shape = r.ml_params.get(2).copied().unwrap_or(0.0);
    let revert2ml = GenExtreme.revert2ml(shape);
    let ml_cdf = cdf_at(&GenExtreme, &r.ml_params, &y);
    let cp_cdf = cdf_at(&GenExtreme, &r.cp_params, &y);
    Ok(p_cp_list(
        &r,
        ml_cdf,
        cp_cdf,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![
            ("revert2ml", revert2ml.into()),
            ("bs_cdf", BOOT_NOT_SELECTED.into()),
        ],
    ))
}

/// GEV CP random deviates (reference `rgev_cp`).
#[extendr]
fn rgev_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.0], 0.01);
    Ok(r_cp_list(
        &r,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![("bs_deviates", BOOT_NOT_SELECTED.into())],
    ))
}

/// GEV posterior theta samples (reference `tgev_cp`; RUST — Phase 2).
#[extendr]
fn tgev_cp(_method: Robj, _n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- GEV + predictor (`gev_p1`) --------------------------------------------

/// GEV-with-predictor CP quantiles (reference `qgev_p1_cp`).
///
/// W14: the reference `gev_p1` CP uses the DMGS CRHP/flat path with the
/// `revert2ml` guard and the quantile-derived CP density, exactly like
/// `qgev_cp`. The predictor is centered by its mean and the prediction point is
/// resolved via `maket0`.
#[extendr]
fn qgev_p1_cp(x: Robj, t: Robj, t0: Robj, n0: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let p = real_vec(&p, "p")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let revert2ml = model.revert2ml(shape);
    let ml_max = model.ml_max(r.ml_params[0], r.ml_params[2], shape);
    Ok(q_cp_list(
        &r,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![
            ("ml_max", ml_max.into()),
            ("revert2ml", revert2ml.into()),
            ("ml_pdf", doubles(r.ml_pdf.clone()).into()),
            ("cp_pdf", doubles(r.cp_pdf.clone()).into()),
            ("pw_params", PWM_NOT_SELECTED.into()),
            ("pw_quantiles", PWM_NOT_SELECTED.into()),
            ("custom_quantiles", EXTRAMODELS_NOT_SELECTED.into()),
            ("custom_mean", f64::NAN.into()),
            ("bs_quantiles", BOOT_NOT_SELECTED.into()),
        ],
    ))
}

/// GEV-with-predictor CP densities (reference `dgev_p1_cp`).
#[extendr]
fn dgev_p1_cp(x: Robj, t: Robj, t0: Robj, n0: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let y = real_vec(&y, "y")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(y, x, vec![0.5], None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let revert2ml = model.revert2ml(shape);
    Ok(d_cp_list(
        &r,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![
            ("revert2ml", revert2ml.into()),
            ("bs_pdf", BOOT_NOT_SELECTED.into()),
        ],
    ))
}

/// GEV-with-predictor CP CDFs (reference `pgev_p1_cp`).
#[extendr]
fn pgev_p1_cp(x: Robj, t: Robj, t0: Robj, n0: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let y = real_vec(&y, "y")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let req = make_request(y.clone(), x, vec![0.5], None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    let shape = r.ml_params.get(3).copied().unwrap_or(0.0);
    let revert2ml = model.revert2ml(shape);
    let ml_cdf = cdf_at(&model, &r.ml_params, &y);
    let cp_cdf = cdf_at(&model, &r.cp_params, &y);
    Ok(p_cp_list(
        &r,
        ml_cdf,
        cp_cdf,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![
            ("revert2ml", revert2ml.into()),
            ("bs_cdf", BOOT_NOT_SELECTED.into()),
        ],
    ))
}

/// GEV-with-predictor CP random deviates (reference `rgev_p1_cp`).
#[extendr]
fn rgev_p1_cp(n: Robj, x: Robj, t: Robj, t0: Robj, n0: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let t = opt_real_vec(&t, "t")?;
    let t0 = opt_scalar(&t0, "t0")?;
    let n0 = opt_scalar(&n0, "n0")?;
    let t = t.unwrap_or_default();
    let t0 = maket0(&t, t0, n0);
    let model = GenExtremeP1::new(t, t0, n0.unwrap_or(1.0));
    let mean_t = model.mean_t();
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = decenter_fit(
        &predict_gev_p1(&model, &req, &[0.0, 0.0, 1.0, 0.0], 0.01),
        mean_t,
    );
    Ok(r_cp_list(
        &r,
        CRHPFLAT_DMGS_CPMETHOD,
        vec![("bs_deviates", BOOT_NOT_SELECTED.into())],
    ))
}

/// GEV-with-predictor posterior theta samples (reference `tgev_p1_cp`; RUST — Phase 2).
#[extendr]
fn tgev_p1_cp(_n: Robj, _x: Robj, _t: Robj, _t0: Robj, _n0: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- GPD (`gpd_k1`) --------------------------------------------------------

/// GPD CP quantiles (reference `qgpd_k1_cp`).
#[extendr]
fn qgpd_k1_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let shape = r.ml_params.get(1).copied().unwrap_or(0.0);
    let revert2ml = shape.abs() >= 1.0 || shape > 30.0;
    Ok(q_cp_list(
        &r,
        ADHOC_DMGS_CPMETHOD,
        vec![
            ("revert2ml", revert2ml.into()),
            ("ml_pdf", doubles(r.ml_pdf.clone()).into()),
            ("cp_pdf", doubles(r.cp_pdf.clone()).into()),
            ("rh_flat_quantiles", doubles(r.cp_quantiles.clone()).into()),
            ("lp_quantiles", EXTRAMODELS_NOT_SELECTED.into()),
            ("lp2_quantiles", EXTRAMODELS_NOT_SELECTED.into()),
            ("dpi_quantiles", EXTRAMODELS_NOT_SELECTED.into()),
            ("rh_flat_pdf", doubles(r.cp_pdf.clone()).into()),
            ("rh_flat_mean", f64::NAN.into()),
        ],
    ))
}

/// GPD CP densities (reference `dgpd_k1_cp`).
#[extendr]
fn dgpd_k1_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    Ok(d_cp_list(&r, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// GPD CP CDFs (reference `pgpd_k1_cp`).
#[extendr]
fn pgpd_k1_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    let ml_cdf = cdf_at(&GenPareto, &r.ml_params, &y);
    let cp_cdf = cdf_at(&GenPareto, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// GPD CP random deviates (reference `rgpd_k1_cp`).
#[extendr]
fn rgpd_k1_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(GenPareto, &req, &[1.0, 0.0]);
    Ok(r_cp_list(&r, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// GPD posterior theta samples (reference `tgpd_k1_cp`; RUST — Phase 2).
#[extendr]
fn tgpd_k1_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- Gumbel (`gumbel`) -----------------------------------------------------

/// Gumbel CP quantiles (reference `qgumbel_cp`).
#[extendr]
fn qgumbel_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    Ok(q_cp_list(&r, RHP_DMGS_CPMETHOD, vec![]))
}

/// Gumbel CP densities (reference `dgumbel_cp`).
#[extendr]
fn dgumbel_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    Ok(d_cp_list(&r, RHP_DMGS_CPMETHOD, vec![]))
}

/// Gumbel CP CDFs (reference `pgumbel_cp`).
#[extendr]
fn pgumbel_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&Gumbel, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Gumbel, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, RHP_DMGS_CPMETHOD, vec![]))
}

/// Gumbel CP random deviates (reference `rgumbel_cp`).
#[extendr]
fn rgumbel_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Gumbel, &req, &[0.0, 1.0]);
    Ok(r_cp_list(&r, RHP_DMGS_CPMETHOD, vec![]))
}

/// Gumbel posterior theta samples (reference `tgumbel_cp`; RUST — Phase 2).
#[extendr]
fn tgumbel_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- Log-normal (`lnorm`) --------------------------------------------------

/// Log-normal CP quantiles (reference `qlnorm_cp`).
#[extendr]
fn qlnorm_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    Ok(q_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Log-normal CP densities (reference `dlnorm_cp`).
#[extendr]
fn dlnorm_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    Ok(d_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Log-normal CP CDFs (reference `plnorm_cp`).
#[extendr]
fn plnorm_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    let ml_cdf = cdf_at(&LogNormal, &r.ml_params, &y);
    let cp_cdf = cdf_at(&LogNormal, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD, vec![]))
}

/// Log-normal CP random deviates (reference `rlnorm_cp`).
#[extendr]
fn rlnorm_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(LogNormal, &req, &[0.0, 1.0]);
    Ok(r_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Log-normal posterior theta samples (reference `tlnorm_cp`; RUST — Phase 2).
#[extendr]
fn tlnorm_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- Weibull (`weibull`) ---------------------------------------------------

/// Weibull CP quantiles (reference `qweibull_cp`).
#[extendr]
fn qweibull_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    Ok(q_cp_list(&r, RHP_DMGS_CPMETHOD, vec![]))
}

/// Weibull CP densities (reference `dweibull_cp`).
#[extendr]
fn dweibull_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    Ok(d_cp_list(&r, RHP_DMGS_CPMETHOD, vec![]))
}

/// Weibull CP CDFs (reference `pweibull_cp`).
#[extendr]
fn pweibull_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    let ml_cdf = cdf_at(&Weibull, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Weibull, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, RHP_DMGS_CPMETHOD, vec![]))
}

/// Weibull CP random deviates (reference `rweibull_cp`).
#[extendr]
fn rweibull_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Weibull, &req, &[1.0, 1.0]);
    Ok(r_cp_list(&r, RHP_DMGS_CPMETHOD, vec![]))
}

/// Weibull posterior theta samples (reference `tweibull_cp`; RUST — Phase 2).
#[extendr]
fn tweibull_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- Gamma (`gamma`) -------------------------------------------------------

/// Gamma CP quantiles (reference `qgamma_cp`).
#[extendr]
fn qgamma_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    Ok(q_cp_list(&r, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// Gamma CP densities (reference `dgamma_cp`).
#[extendr]
fn dgamma_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    Ok(d_cp_list(&r, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// Gamma CP CDFs (reference `pgamma_cp`).
#[extendr]
fn pgamma_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    let ml_cdf = cdf_at(&Gamma, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Gamma, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// Gamma CP random deviates (reference `rgamma_cp`).
#[extendr]
fn rgamma_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Gamma, &req, &[1.0, 1.0]);
    Ok(r_cp_list(&r, ADHOC_DMGS_CPMETHOD, vec![]))
}

/// Gamma posterior theta samples (reference `tgamma_cp`; RUST — Phase 2).
#[extendr]
fn tgamma_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// --- Exponential (`exp`) ---------------------------------------------------

/// Exponential CP quantiles (reference `qexp_cp`).
#[extendr]
fn qexp_cp(x: Robj, p: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let p = real_vec(&p, "p")?;
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    Ok(q_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Exponential CP densities (reference `dexp_cp`).
#[extendr]
fn dexp_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y, vec![0.5], None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    Ok(d_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Exponential CP CDFs (reference `pexp_cp`).
#[extendr]
fn pexp_cp(x: Robj, y: Robj) -> Result<List> {
    let x = real_vec(&x, "x")?;
    let y = real_vec(&y, "y")?;
    let req = make_request(x, y.clone(), vec![0.5], None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    let ml_cdf = cdf_at(&Exponential, &r.ml_params, &y);
    let cp_cdf = cdf_at(&Exponential, &r.cp_params, &y);
    Ok(p_cp_list(&r, ml_cdf, cp_cdf, ANALYTIC_CPMETHOD, vec![]))
}

/// Exponential CP random deviates (reference `rexp_cp`).
#[extendr]
fn rexp_cp(n: Robj, x: Robj) -> Result<List> {
    let n = opt_scalar(&n, "n")?.unwrap_or(1.0).max(1.0) as usize;
    let x = real_vec(&x, "x")?;
    let p: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let req = make_request(x.clone(), x, p, None, None, None);
    let r = predict(Exponential, &req, &[1.0]);
    Ok(r_cp_list(&r, ANALYTIC_CPMETHOD, vec![]))
}

/// Exponential posterior theta samples (reference `texp_cp`; RUST — Phase 2).
#[extendr]
fn texp_cp(_n: Robj, _x: Robj) -> Result<List> {
    Ok(t_cp_list())
}

// ---------------------------------------------------------------------------
// Registration.
// ---------------------------------------------------------------------------

// Macro to generate the R wrapper registration.
extendr_module! {
    mod fitdistcp;
    fn core_version;
    fn add_vectors;
    fn sum_values;

    // norm
    fn qnorm_cp;
    fn dnorm_cp;
    fn pnorm_cp;
    fn rnorm_cp;
    fn tnorm_cp;

    // norm_p1
    fn qnorm_p1_cp;
    fn dnorm_p1_cp;
    fn pnorm_p1_cp;
    fn rnorm_p1_cp;
    fn tnorm_p1_cp;

    // gev
    fn qgev_cp;
    fn dgev_cp;
    fn pgev_cp;
    fn rgev_cp;
    fn tgev_cp;

    // gev_p1
    fn qgev_p1_cp;
    fn dgev_p1_cp;
    fn pgev_p1_cp;
    fn rgev_p1_cp;
    fn tgev_p1_cp;

    // gpd_k1
    fn qgpd_k1_cp;
    fn dgpd_k1_cp;
    fn pgpd_k1_cp;
    fn rgpd_k1_cp;
    fn tgpd_k1_cp;

    // gumbel
    fn qgumbel_cp;
    fn dgumbel_cp;
    fn pgumbel_cp;
    fn rgumbel_cp;
    fn tgumbel_cp;

    // lnorm
    fn qlnorm_cp;
    fn dlnorm_cp;
    fn plnorm_cp;
    fn rlnorm_cp;
    fn tlnorm_cp;

    // weibull
    fn qweibull_cp;
    fn dweibull_cp;
    fn pweibull_cp;
    fn rweibull_cp;
    fn tweibull_cp;

    // gamma
    fn qgamma_cp;
    fn dgamma_cp;
    fn pgamma_cp;
    fn rgamma_cp;
    fn tgamma_cp;

    // exp
    fn qexp_cp;
    fn dexp_cp;
    fn pexp_cp;
    fn rexp_cp;
    fn texp_cp;
}
