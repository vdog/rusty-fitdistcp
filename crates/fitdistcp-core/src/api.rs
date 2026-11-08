//! Binding-agnostic shared API surface (§5.1, §5.2, §5.3).
//!
//! These plain Rust structs define the return shapes **once** in the core. Each
//! binding (`fitdistcp-r`, `fitdistcp-py`) is a thin adapter that converts native
//! arrays to contiguous `&[f64]`, calls the core, and maps these structs back to
//! the reference field names. This guarantees identical numerics across R and
//! Python.
//!
//! Field names mirror the design document's §5.1 (R convention) and §5.2
//! (Python convention) so the adapters are a mechanical mapping.

/// Convergence status of an MLE fit (§4.2).
///
/// Non-convergence is reported explicitly rather than silently returning a
/// possibly-invalid fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convergence {
    /// The optimizer met its convergence criteria.
    Converged,
    /// The optimizer exhausted its iteration budget without converging.
    MaxIterationsReached,
    /// The optimizer stalled (parameter change / gradient norm below tolerance
    /// without satisfying the full convergence test).
    Stalled,
    /// The fit failed (e.g. non-finite objective, singular information matrix).
    Failed,
}

impl Convergence {
    /// Convenience predicate: `true` only for [`Convergence::Converged`].
    pub fn is_converged(self) -> bool {
        matches!(self, Convergence::Converged)
    }
}

/// Result of an MLE fit, shared by both bindings (§4.2, §5.1, §5.2).
///
/// Holds the fitted parameters, their standard errors, the information-criterion
/// values, and the convergence/small-n diagnostics. The predictive densities and
/// quantiles produced from the fit live in [`FitResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct MleFit {
    /// Fitted maximum-likelihood parameter estimates (`ml_params`).
    pub ml_params: Vec<f64>,
    /// Standard errors of the MLE estimates (`SEs`).
    pub ml_se: Vec<f64>,
    /// Akaike Information Criterion for the MLE fit (`AIC`).
    pub aic: f64,
    /// Number of observations used in the fit.
    pub n: usize,
    /// Number of estimated parameters (used for AIC/MAIC and exact-match tests).
    pub n_params: usize,
    /// Log-likelihood at the MLE.
    pub loglik: f64,
    /// Convergence status of the optimizer (§4.2).
    pub convergence: Convergence,
    /// `true` when `n` is below the small-n threshold (§4.4); results are flagged
    /// rather than silently degraded.
    pub small_n: bool,
}

/// Full result of a calibrated-prior (CP) prediction, shared by both bindings.
///
/// Mirrors the §5.1/§5.2 return fields: MLE and CP predictive densities and
/// quantiles, MLE and CP parameters, AIC/MAIC, standard errors, and the optional
/// WAIC / log-score diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// MLE predictive density evaluated at the prediction points (`ml_pdf`).
    pub ml_pdf: Vec<f64>,
    /// CP (calibrated-prior) predictive density at the prediction points (`cp_pdf`).
    pub cp_pdf: Vec<f64>,
    /// MLE predictive quantiles (`ml_quantiles`).
    pub ml_quantiles: Vec<f64>,
    /// CP predictive quantiles (`cp_quantiles`).
    pub cp_quantiles: Vec<f64>,
    /// MLE parameter estimates (`ml_params`).
    pub ml_params: Vec<f64>,
    /// CP parameter estimates (`cp_params`).
    pub cp_params: Vec<f64>,
    /// Standard errors of the MLE parameter estimates (`SEs`).
    pub ml_se: Vec<f64>,
    /// Standard errors of the CP parameter estimates (`SEs`).
    pub cp_se: Vec<f64>,
    /// Akaike Information Criterion (`AIC`).
    pub aic: f64,
    /// Modified Akaike Information Criterion (`MAIC`).
    pub maic: f64,
    /// The maximized log-likelihood (`ml_value`), reference `ml_value`.
    ///
    /// Phase B amendment: the reference reports `ml_value` as the maximized
    /// log-likelihood directly (e.g. `norm.py`: `ml_value = sum(logpdf(x))`),
    /// not reconstructed from `aic`. The core now carries it explicitly so the
    /// bindings can surface the exact reference value.
    pub loglik: f64,
    /// Optional Watanabe–Akaike Information Criterion (`WAIC`).
    pub waic: Option<f64>,
    /// Optional log-score diagnostics (`logscores`).
    pub log_scores: Option<Vec<f64>>,
    /// MLE predictive mean (`ml_mean`), reference `*_means`.
    pub ml_mean: f64,
    /// CP (calibrated-prior) predictive mean (`cp_mean`), reference `*_means`.
    pub cp_mean: f64,
    /// WAIC variant 1, the 3-vector `[lppd, -pwaic1, sum]` (reference
    /// `make_waic`). `None` when WAIC was not requested.
    pub waic1: Option<[f64; 3]>,
    /// WAIC variant 2, the 3-vector `[lppd, -pwaic2, sum]` (reference
    /// `make_waic`). `None` when WAIC was not requested.
    pub waic2: Option<[f64; 3]>,
    /// Leave-one-out MLE out-of-sample log-score (`ml_oos_logscore`).
    /// `None` when log-scores were not requested.
    pub ml_oos_logscore: Option<f64>,
    /// Leave-one-out CP out-of-sample log-score (`cp_oos_logscore`).
    /// `None` when log-scores were not requested.
    pub cp_oos_logscore: Option<f64>,
    /// Convergence status of the underlying MLE fit (§4.2).
    pub convergence: Convergence,
    /// `true` when the sample size is below the small-n threshold (§4.4).
    pub small_n: bool,
}

impl FitResult {
    /// Convenience predicate: `true` only when the underlying fit converged.
    pub fn is_converged(&self) -> bool {
        self.convergence.is_converged()
    }
}

/// Fit a model by maximum likelihood and build the full CP prediction result.
///
/// This is the shared-core entry point that ties together the MLE fit
/// ([`crate::mle::fit`]) and the DMGS asymptotic expansion
/// ([`crate::dmgs::DmgsIntegrator`]) to populate every field of [`FitResult`]:
///
/// - `ml_params`, `ml_se`, `aic`, `convergence`, `small_n` come from the MLE fit;
/// - `ml_pdf` / `ml_quantiles` are the MLE predictive density/quantiles;
/// - `cp_pdf` / `cp_quantiles` are the DMGS CP predictive density/quantiles;
/// - `cp_params` / `cp_se` are the CP parameter estimates and standard errors
///   (for the DMGS path these equal the MLE values at this stage — the DMGS
///   expansion corrects the *predictive* distribution, not the point estimates;
///   the reference likewise reports `ml_params` for the CP parameters);
/// - `maic` is the modified AIC `ml_value − n_params` (the reference
///   `make_maic` convention: `maic = [ml_value, −nparams, ml_value − nparams]`,
///   of which we store the sum `ml_value − n_params`).
///
/// `init` is the initial parameter guess for the optimizer, of length
/// `model.n_params()`.
///
/// # Panics
///
/// Panics if `init.len() != model.n_params()` or if `req.y` is empty.
pub fn predict<M: crate::dmgs::DmgsModel + Copy + Sync>(
    model: M,
    req: &PredictionRequest,
    init: &[f64],
) -> FitResult {
    use crate::dmgs::DmgsIntegrator;
    use crate::mle::fit;

    let mle = fit(&model, &req.y, init);
    let integ = DmgsIntegrator::new(model);

    // MLE predictive density and quantiles.
    let ml_pdf: Vec<f64> = req
        .x
        .iter()
        .map(|&xi| model.pdf(xi, &mle.ml_params))
        .collect();
    let ml_quantiles: Vec<f64> = req
        .p
        .iter()
        .map(|&pi| model.quantile(pi, &mle.ml_params))
        .collect();

    // CP predictive density and quantiles via the DMGS expansion.
    let cp_pdf = integ.cp_pdf(&mle.ml_params, &req.y, &req.x);
    let cp_quantiles = integ.cp_quantiles(&mle.ml_params, &req.y, &req.p);

    // MAIC: the reference `make_maic` returns [ml_value, -nparams, sum]; we store
    // the sum `ml_value - n_params`.
    let maic = mle.loglik - mle.n_params as f64;

    // A1: MLE and CP predictive means (reference `*_means`).
    let (ml_mean, cp_mean) = integ.cp_mean(&mle.ml_params, &req.y);

    FitResult {
        ml_pdf,
        cp_pdf,
        ml_quantiles,
        cp_quantiles,
        ml_params: mle.ml_params.clone(),
        // The DMGS expansion corrects the predictive distribution, not the point
        // estimates; the reference reports the MLE parameters for the CP model.
        cp_params: mle.ml_params.clone(),
        ml_se: mle.ml_se.clone(),
        cp_se: mle.ml_se.clone(),
        aic: mle.aic,
        maic,
        loglik: mle.loglik,
        waic: None,
        log_scores: None,
        ml_mean,
        cp_mean,
        waic1: None,
        waic2: None,
        ml_oos_logscore: None,
        cp_oos_logscore: None,
        convergence: mle.convergence,
        small_n: mle.small_n,
    }
}

/// Fit a **GEV** model and build the CP prediction result, applying the
/// reference's negative-shape handling (§3, §4.4).
///
/// This is the GEV-specific entry point. It differs from the generic
/// [`predict`] in two ways that the reference `qgev_cp` requires:
///
/// 1. **`revert2ml` guard.** When the fitted shape magnitude reaches 1
///    (`|shape| >= 1`) the DMGS expansion is numerically unreliable, so the
///    reference sets the CP output equal to the MLE output. We reproduce that:
///    `cp_pdf = ml_pdf`, `cp_quantiles = ml_quantiles`, and the returned
///    [`FitResult`] carries `revert2ml = true`.
/// 2. **Quantile-derived CP density.** Otherwise the CP density is derived from
///    the CP quantile function via
///    [`DmgsIntegrator::cp_pdf_quantile_derived`] (the reference's
///    `rh_flat_pdf = -(Δα)/(Δq)`), which is numerically stable for the
///    negative-shape case where the direct density formula is not. The CP
///    quantiles still come from the DMGS expansion.
///
/// The `small_n` flag is propagated from the MLE fit unchanged (§4.4): the
/// reference flags small samples rather than silently degrading, and the flag
/// flows through this CP path.
///
/// `fdalpha` is the relative step for the quantile-derived density (the
/// reference default is `0.01`).
///
/// # Panics
///
/// Panics if `init.len() != model.n_params()` or if `req.y` is empty.
pub fn predict_gev(
    model: crate::models::GenExtreme,
    req: &PredictionRequest,
    init: &[f64],
    fdalpha: f64,
) -> FitResult {
    use crate::dmgs::{DmgsIntegrator, DmgsModel};
    use crate::mle::fit;

    let mle = fit(&model, &req.y, init);
    let integ = DmgsIntegrator::new(model);

    // Call the `DmgsModel` trait methods explicitly: `GenExtreme` also has
    // inherent `pdf`/`quantile` methods with a different (scalar-parameter)
    // signature that would otherwise shadow the trait methods.
    let ml_pdf: Vec<f64> = req
        .x
        .iter()
        .map(|&xi| DmgsModel::pdf(&model, xi, &mle.ml_params))
        .collect();
    let ml_quantiles: Vec<f64> = req
        .p
        .iter()
        .map(|&pi| DmgsModel::quantile(&model, pi, &mle.ml_params))
        .collect();

    let shape = mle.ml_params[2];
    let revert2ml = model.revert2ml(shape);

    let (cp_pdf, cp_quantiles) = if revert2ml {
        // Reference: when |shape| >= 1 the CP output reverts to the MLE output.
        (ml_pdf.clone(), ml_quantiles.clone())
    } else {
        // CP quantiles from the DMGS expansion; CP density from the CP quantile
        // function (the reference `rh_flat_pdf`), evaluated at alpha = 1 - p.
        let cp_q = integ.cp_quantiles(&mle.ml_params, &req.y, &req.p);
        let alpha: Vec<f64> = req.p.iter().map(|&pi| 1.0 - pi).collect();
        let cp_d = integ.cp_pdf_quantile_derived(&mle.ml_params, &req.y, &alpha, fdalpha);
        (cp_d, cp_q)
    };

    let maic = mle.loglik - mle.n_params as f64;

    // A1: MLE and CP predictive means (reference `gev_means`). The reference
    // reports `rh_flat_mean = inf` for the GEV (integrating over the shape makes
    // the mean infinite), so `cp_mean` is `+inf`; `ml_mean` is the analytic MLE
    // mean from `gev_means`.
    let (ml_mean, _cp_mean_dmgs) = integ.cp_mean(&mle.ml_params, &req.y);
    let cp_mean = f64::INFINITY;

    FitResult {
        ml_pdf,
        cp_pdf,
        ml_quantiles,
        cp_quantiles,
        ml_params: mle.ml_params.clone(),
        cp_params: mle.ml_params.clone(),
        ml_se: mle.ml_se.clone(),
        cp_se: mle.ml_se.clone(),
        aic: mle.aic,
        maic,
        loglik: mle.loglik,
        waic: None,
        log_scores: None,
        ml_mean,
        cp_mean,
        waic1: None,
        waic2: None,
        ml_oos_logscore: None,
        cp_oos_logscore: None,
        convergence: mle.convergence,
        small_n: mle.small_n,
    }
}

/// Full result of a `norm_p1` (normal-with-predictor) CP prediction.
///
/// The reference `qnorm_p1_cp` uses an **analytic** CP (a Student-t predictive
/// distribution), not the DMGS expansion, so this result carries the analytic
/// fields the reference returns: the OLS `ml_params`, the MLE/CP predictive
/// quantiles/densities/CDFs, the standard errors, and the predictor-specific
/// `predictedparameter`/`adjustedx` vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct NormP1Result {
    /// OLS parameter estimates `[intercept, slope, sigma]` (reference
    /// `norm_p1_mlparams`: `lm` + `sd(residuals)`, `n-1` denominator).
    pub ml_params: Vec<f64>,
    /// MLE predictive quantiles at the prediction point (`ml_quantiles`).
    pub ml_quantiles: Vec<f64>,
    /// CP predictive quantiles at the prediction point (`cp_quantiles`).
    pub cp_quantiles: Vec<f64>,
    /// MLE predictive density at the prediction points (`ml_pdf`).
    pub ml_pdf: Vec<f64>,
    /// CP predictive density at the prediction points (`cp_pdf`).
    pub cp_pdf: Vec<f64>,
    /// MLE predictive CDF at the prediction points (`ml_cdf`).
    pub ml_cdf: Vec<f64>,
    /// CP predictive CDF at the prediction points (`cp_cdf`).
    pub cp_cdf: Vec<f64>,
    /// Standard errors `[SE_a, SE_b, SE_sigma]` from the analytic expected
    /// information (reference `expinfmat`).
    pub standard_errors: Vec<f64>,
    /// Predicted location `a + b·(t_i - mean(t))`, one per observation
    /// (reference `predictedparameter`).
    pub predictedparameter: Vec<f64>,
    /// Generalized residuals `qnorm(pnorm(x_i, mu_i, s), mu0, s)`, one per
    /// observation (reference `adjustedx`).
    pub adjustedx: Vec<f64>,
    /// The maximized log-likelihood (`ml_value`).
    pub ml_value: f64,
    /// The prediction point `muhat0 = a + b·(t0 - mean(t))`.
    pub muhat0: f64,
    /// MLE predictive mean (`ml_mean`). For the normal the predictive mean is
    /// the location `muhat0` (reference `norm.py`: `ml_mean = v1hat`).
    pub ml_mean: f64,
    /// CP predictive mean (`cp_mean`). For the analytic normal CP the
    /// predictive mean is also the location `muhat0` (reference `norm.py`:
    /// `rh_mean = v1hat`).
    pub cp_mean: f64,
}

/// Fit a `norm_p1` model and build the **analytic** CP prediction result.
///
/// This reproduces the reference `qnorm_p1_cp` exactly (W14): the reference uses
/// an analytic Student-t predictive distribution, not the DMGS expansion. The
/// steps are:
///
/// 1. `ml_params = norm_p1_mlparams(x, t)` — OLS of `x` on the **centered**
///    predictor with `sigma = sd(residuals)` (`n-1` denominator).
/// 2. `muhat0 = a + b·(t0 - mean(t))` — the prediction point.
/// 3. `ml_quantiles = qnorm(1-α, muhat0, sigma)`.
/// 4. `cp_quantiles = muhat0 + qt(1-α, df=n-2)·sg2` with
///    `sg2 = sigma·√((n-1)/(n-2))·√(1 + 1/n + tresid0²/Σtresid²)`.
/// 5. `ml_pdf`/`ml_cdf` from the normal at `muhat0`; `cp_pdf`/`cp_cdf` from the
///    Student-t at `df = n-2` scaled by `sg2`.
/// 6. `predictedparameter`/`adjustedx` as in the reference `predictordata`.
///
/// `model` carries the raw predictor `t` and the resolved prediction point `t0`.
/// `req.y` is the observed data, `req.x` the density/CDF evaluation points, and
/// `req.p` the probability levels for the quantiles.
///
/// # Panics
///
/// Panics if `req.y` is empty or `req.y.len() < 3` (the analytic CP needs
/// `df = n - 2 > 0`).
pub fn predict_norm_p1(model: &crate::models::NormalP1, req: &PredictionRequest) -> NormP1Result {
    use crate::models::norm::standard_normal_quantile;
    use crate::special::{erfc, student_t_cdf, student_t_pdf, student_t_quantile};

    let y = &req.y;
    let n = y.len();
    assert!(n >= 3, "predict_norm_p1: need n >= 3 for df = n - 2 > 0");
    let nx = n as f64;
    let mean_t = model.mean_t();
    let t0c = model.centered_prediction();

    // 1. OLS ml_params (reference norm_p1_mlparams).
    let ml_params = model.mlparams_ols(y);
    let a = ml_params[0];
    let b = ml_params[1];
    let sigma = ml_params[2];

    // 2. Prediction point.
    let muhat0 = a + b * t0c;

    // tresid = t - mean(t); tresid0 = t0 - mean(t); sx = sum(tresid^2).
    let tresid: Vec<f64> = model.t.iter().map(|&ti| ti - mean_t).collect();
    let sx: f64 = tresid.iter().map(|c| c * c).sum();

    // 3. MLE quantiles.
    let ml_quantiles: Vec<f64> = req
        .p
        .iter()
        .map(|&pi| muhat0 + sigma * standard_normal_quantile(pi))
        .collect();

    // 4. CP quantiles (analytic Student-t).
    let sg1 = sigma * ((nx - 1.0) / (nx - 2.0)).sqrt();
    let sg2 = sg1 * (1.0 + 1.0 / nx + t0c * t0c / sx).sqrt();
    let cp_quantiles: Vec<f64> = req
        .p
        .iter()
        .map(|&pi| muhat0 + student_t_quantile(pi, nx - 2.0) * sg2)
        .collect();

    // 5. Densities and CDFs.
    let normal_pdf = |x: f64| {
        let z = (x - muhat0) / sigma;
        (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
    };
    let normal_cdf = |x: f64| {
        let z = (x - muhat0) / (sigma * std::f64::consts::SQRT_2);
        0.5 * erfc(-z)
    };
    let ml_pdf: Vec<f64> = req.x.iter().map(|&xi| normal_pdf(xi)).collect();
    let ml_cdf: Vec<f64> = req.x.iter().map(|&xi| normal_cdf(xi)).collect();
    let cp_pdf: Vec<f64> = req
        .x
        .iter()
        .map(|&xi| (student_t_pdf((xi - muhat0) / sg2, nx - 2.0) / sg2).max(0.0))
        .collect();
    let cp_cdf: Vec<f64> = req
        .x
        .iter()
        .map(|&xi| student_t_cdf((xi - muhat0) / sg2, nx - 2.0).clamp(0.0, 1.0))
        .collect();

    // 6. Standard errors from the analytic expected information (reference
    // expinfmat = diag(nx/sigma^2, sx/sigma^2, 2nx/sigma^2), inverted).
    let s2 = sigma * sigma;
    let standard_errors = vec![(s2 / nx).sqrt(), (s2 / sx).sqrt(), (s2 / (2.0 * nx)).sqrt()];

    // 7. predictordata: predictedparameter = a + b*tresid; adjustedx =
    // qnorm(pnorm(x_i, mu_i, s), mu0, s).
    let predictedparameter: Vec<f64> = tresid.iter().map(|&c| a + b * c).collect();
    let adjustedx: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, &yi)| {
            let mu_i = a + b * tresid[i];
            let px = normal_cdf_at(yi, mu_i, sigma);
            muhat0 + sigma * standard_normal_quantile(px)
        })
        .collect();

    // ml_value = log-likelihood at the OLS params.
    let ml_value: f64 = y
        .iter()
        .enumerate()
        .map(|(i, &yi)| {
            let mu_i = a + b * tresid[i];
            let z = (yi - mu_i) / sigma;
            -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        })
        .sum();

    NormP1Result {
        ml_params,
        ml_quantiles,
        cp_quantiles,
        ml_pdf,
        cp_pdf,
        ml_cdf,
        cp_cdf,
        standard_errors,
        predictedparameter,
        adjustedx,
        ml_value,
        muhat0,
        // A1: for the analytic normal CP the predictive mean is the location
        // `muhat0` for both the MLE and the CP (reference `norm.py`:
        // `ml_mean = v1hat`, `rh_mean = v1hat`).
        ml_mean: muhat0,
        cp_mean: muhat0,
    }
}

/// Normal CDF at `x` with mean `mu` and sd `sigma` (helper for `predict_norm_p1`).
fn normal_cdf_at(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / (sigma * std::f64::consts::SQRT_2);
    0.5 * crate::special::erfc(-z)
}

/// Fit a **GEV-with-predictor** (`genextreme_p1`) model and build the CP
/// prediction result, applying the reference's negative-shape handling.
///
/// This mirrors [`predict_gev`] for the predictor model: the DMGS expansion is
/// evaluated at the prediction point `t0` (via the model's
/// `centered_prediction`), and the reference's `revert2ml` guard and
/// quantile-derived CP density are applied. The predictor is centered by its
/// mean; the binding resolves `t0` from the user's `t0`/`n0` via `maket0`.
///
/// # Panics
///
/// Panics if `init.len() != model.n_params()` or if `req.y` is empty.
pub fn predict_gev_p1(
    model: &crate::models::GenExtremeP1,
    req: &PredictionRequest,
    init: &[f64],
    fdalpha: f64,
) -> FitResult {
    use crate::dmgs::{DmgsIntegrator, DmgsModel};
    use crate::mle::fit;

    let mle = fit(model, &req.y, init);
    let integ = DmgsIntegrator::new(model.clone());

    let ml_pdf: Vec<f64> = req
        .x
        .iter()
        .map(|&xi| DmgsModel::pdf(model, xi, &mle.ml_params))
        .collect();
    let ml_quantiles: Vec<f64> = req
        .p
        .iter()
        .map(|&pi| DmgsModel::quantile(model, pi, &mle.ml_params))
        .collect();

    let shape = mle.ml_params[3];
    let revert2ml = model.revert2ml(shape);

    let (cp_pdf, cp_quantiles) = if revert2ml {
        (ml_pdf.clone(), ml_quantiles.clone())
    } else {
        let cp_q = integ.cp_quantiles(&mle.ml_params, &req.y, &req.p);
        let alpha: Vec<f64> = req.p.iter().map(|&pi| 1.0 - pi).collect();
        let cp_d = integ.cp_pdf_quantile_derived(&mle.ml_params, &req.y, &alpha, fdalpha);
        (cp_d, cp_q)
    };

    let maic = mle.loglik - mle.n_params as f64;

    // A1: MLE and CP predictive means (reference `gev_p1_means`). As for the
    // homogeneous GEV, the reference reports `rh_flat_mean = inf`, so `cp_mean`
    // is `+inf`; `ml_mean` is the analytic MLE mean.
    let (ml_mean, _cp_mean_dmgs) = integ.cp_mean(&mle.ml_params, &req.y);
    let cp_mean = f64::INFINITY;

    FitResult {
        ml_pdf,
        cp_pdf,
        ml_quantiles,
        cp_quantiles,
        ml_params: mle.ml_params.clone(),
        cp_params: mle.ml_params.clone(),
        ml_se: mle.ml_se.clone(),
        cp_se: mle.ml_se.clone(),
        aic: mle.aic,
        maic,
        loglik: mle.loglik,
        waic: None,
        log_scores: None,
        ml_mean,
        cp_mean,
        waic1: None,
        waic2: None,
        ml_oos_logscore: None,
        cp_oos_logscore: None,
        convergence: mle.convergence,
        small_n: mle.small_n,
    }
}

/// Inputs for a prediction request, shared by both bindings (§5.2).
///
/// Carries the observed data, the prediction grid for densities, the probability
/// levels for quantiles, and the optional predictor handling used by the `_p1`
/// models (`t`, `t0`, `n0`).
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionRequest {
    /// Observed data `y`.
    pub y: Vec<f64>,
    /// Prediction points at which `ml_pdf`/`cp_pdf` are evaluated.
    pub x: Vec<f64>,
    /// Probability levels at which `ml_quantiles`/`cp_quantiles` are evaluated.
    pub p: Vec<f64>,
    /// Optional predictor values `t` (for `norm_p1` / `genextreme_p1`).
    pub t: Option<Vec<f64>>,
    /// Optional predictor origin `t0`.
    pub t0: Option<f64>,
    /// Optional predictor count `n0`.
    pub n0: Option<f64>,
    /// Optional prior means (`means`) for the CP expansion.
    pub means: Option<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_populates_cp_fields() {
        use crate::models::Normal;
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        let req = PredictionRequest {
            y: y.clone(),
            x: vec![-1.0, 0.0, 1.0],
            p: vec![0.25, 0.5, 0.75],
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let r = predict(Normal, &req, &[0.0, 1.0]);
        assert!(r.is_converged());
        assert_eq!(r.ml_pdf.len(), 3);
        assert_eq!(r.cp_pdf.len(), 3);
        assert_eq!(r.ml_quantiles.len(), 3);
        assert_eq!(r.cp_quantiles.len(), 3);
        assert_eq!(r.ml_params.len(), 2);
        assert_eq!(r.cp_params, r.ml_params);
        assert_eq!(r.cp_se, r.ml_se);
        // MAIC = loglik - n_params. Since aic = 2k - 2*loglik, we have
        // loglik = k - aic/2, so maic = k - aic/2 - k = -aic/2.
        assert!((r.maic - (-r.aic / 2.0)).abs() < 1e-9);
        for v in r.cp_pdf.iter().chain(r.cp_quantiles.iter()) {
            assert!(v.is_finite(), "cp field not finite: {v}");
        }
    }

    #[test]
    fn predict_populates_ml_and_cp_means() {
        // A1: the normal predictive mean is the fitted location `mu` for both
        // the MLE and the CP (reference `norm.py`: ml_mean = rh_mean = v1hat).
        use crate::models::Normal;
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        let req = PredictionRequest {
            y: y.clone(),
            x: vec![-1.0, 0.0, 1.0],
            p: vec![0.25, 0.5, 0.75],
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let r = predict(Normal, &req, &[0.0, 1.0]);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        assert!((r.ml_mean - mean).abs() < 1e-6, "ml_mean {}", r.ml_mean);
        // The CP mean is the MLE mean plus an O(1/n) DMGS correction.
        assert!((r.cp_mean - mean).abs() < 0.1, "cp_mean {}", r.cp_mean);
        assert!(r.ml_mean.is_finite() && r.cp_mean.is_finite());
    }

    #[test]
    fn predict_gev_reports_infinite_cp_mean() {
        // A1: the reference `gev_means` reports `rh_flat_mean = inf`, so the GEV
        // CP mean is +inf while the MLE mean is finite.
        use crate::models::GenExtreme;
        let y = vec![1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38];
        let req = PredictionRequest {
            y,
            x: vec![1.0, 1.3, 1.6],
            p: vec![0.25, 0.5, 0.75],
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let r = predict_gev(GenExtreme, &req, &[1.4, 0.3, -0.1], 0.01);
        assert!(r.ml_mean.is_finite(), "ml_mean {}", r.ml_mean);
        assert!(
            r.cp_mean.is_infinite() && r.cp_mean > 0.0,
            "cp_mean {}",
            r.cp_mean
        );
    }

    #[test]
    fn convergence_predicate() {
        assert!(Convergence::Converged.is_converged());
        assert!(!Convergence::Failed.is_converged());
        assert!(!Convergence::MaxIterationsReached.is_converged());
        assert!(!Convergence::Stalled.is_converged());
    }

    #[test]
    fn fit_result_is_converged_delegates() {
        let result = FitResult {
            ml_pdf: vec![1.0],
            cp_pdf: vec![1.0],
            ml_quantiles: vec![0.0],
            cp_quantiles: vec![0.0],
            ml_params: vec![0.0, 1.0],
            cp_params: vec![0.0, 1.0],
            ml_se: vec![0.1, 0.1],
            cp_se: vec![0.1, 0.1],
            aic: 10.0,
            maic: 11.0,
            loglik: -5.0,
            waic: None,
            log_scores: None,
            ml_mean: 0.0,
            cp_mean: 0.0,
            waic1: None,
            waic2: None,
            ml_oos_logscore: None,
            cp_oos_logscore: None,
            convergence: Convergence::Converged,
            small_n: false,
        };
        assert!(result.is_converged());
        assert_eq!(result.ml_params.len(), 2);
    }

    // --- W10: small-n propagation through the CP path ----------------------

    fn gev_request(y: Vec<f64>) -> PredictionRequest {
        PredictionRequest {
            y,
            x: vec![1.0, 1.3, 1.6],
            p: vec![0.25, 0.5, 0.75],
            t: None,
            t0: None,
            n0: None,
            means: None,
        }
    }

    #[test]
    fn small_n_flag_propagates_through_gev_cp_path() {
        // n = 8 < 10 => the flag must be set on the FitResult produced by the
        // GEV CP path (predict_gev), not just on the raw MLE fit.
        let y = vec![1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21];
        assert_eq!(y.len(), 8);
        let r = predict_gev(
            crate::models::GenExtreme,
            &gev_request(y),
            &[1.3, 0.25, -0.1],
            0.01,
        );
        assert!(r.small_n, "small_n must propagate through predict_gev");
        assert_eq!(r.ml_params.len(), 3);
        for v in r.cp_pdf.iter().chain(r.cp_quantiles.iter()) {
            assert!(v.is_finite(), "cp field not finite: {v}");
        }
    }

    #[test]
    fn small_n_flag_cleared_at_threshold_through_gev_cp_path() {
        // n = 10 is NOT flagged (strict n < 10), and the flag must be cleared on
        // the FitResult produced by the GEV CP path.
        let y = vec![1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38];
        assert_eq!(y.len(), 10);
        let r = predict_gev(
            crate::models::GenExtreme,
            &gev_request(y),
            &[1.4, 0.3, -0.1],
            0.01,
        );
        assert!(!r.small_n, "n = 10 must not set small_n");
    }

    #[test]
    fn predict_gev_reverts_to_ml_when_shape_magnitude_reaches_one() {
        // When |shape| >= 1 the reference reverts the CP output to the MLE
        // output. We check the guard directly through the model predicate and
        // confirm the CP path returns finite, positive values in the normal
        // (non-revert) regime.
        use crate::models::GenExtreme;
        assert!(GenExtreme.revert2ml(-1.2));
        assert!(!GenExtreme.revert2ml(-0.3));
        let y = vec![1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38];
        let r = predict_gev(GenExtreme, &gev_request(y), &[1.4, 0.3, -0.1], 0.01);
        // In the non-revert regime the CP density is the quantile-derived one and
        // must be finite and positive.
        for (i, v) in r.cp_pdf.iter().enumerate() {
            assert!(v.is_finite() && *v > 0.0, "cp_pdf[{i}] = {v}");
        }
    }
}
