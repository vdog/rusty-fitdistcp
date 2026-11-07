//! Distribution models and their derivatives (§3, §4.1).
//!
//! Phase 1 targets ten models (see §3). This module defines the derivative
//! interface used by the DMGS expansion; concrete models are implemented in a
//! later work item.
//!
//! # Analytic-first / numerical-fallback design (§4.1)
//!
//! The DMGS expansion needs, for each model:
//!
//! - the log-likelihood itself,
//! - the score (first derivatives),
//! - the observed information (second derivatives / Hessian),
//! - third derivatives (for the bias correction),
//! - model-specific mixed derivatives `∂²ℓ/∂θ_i∂y_a` for the Haar-prior term.
//!
//! The [`Derivatives`] trait exposes all of these. Every derivative method has a
//! **default implementation** that computes the quantity by finite differences
//! from [`Derivatives::loglik`] (see [`numerical`]). A model therefore only has
//! to implement [`Derivatives::loglik`] and [`Derivatives::n_params`] to obtain a
//! complete, correct set of derivatives; it can then **override** any individual
//! method with an analytic implementation where the reference package has one.
//!
//! This makes the analytic-vs-numerical switch *per method and per model*, and
//! directly testable: the tests below check that the numerical defaults agree
//! with a known analytic Gaussian case, and that overriding a method changes the
//! result.
//!
//! [`AnalyticDerivatives`] additionally provides an explicit, per-model
//! selection mechanism ([`DerivativeMode`]) for callers that want to force one
//! path or the other (e.g. to benchmark, or to validate an analytic
//! implementation against the numerical fallback).

pub mod numerical;

// # W17 finding: Rayon does not help these reductions
//
// The per-observation work in `loglik`/`score`/`hessian` is a handful of flops
// (a few `powf`/`ln` at most), so the reduction is **memory-bound and
// low-arithmetic-intensity**. The W17 benchmark sweep measured a Rayon-parallel
// reduction at every sample size from 100 to 10 000 and found it **1–3 orders of
// magnitude slower** than the sequential loop (e.g. `norm/score` at n = 1 000:
// 974 ns sequential vs 1.10 ms parallel), because Rayon's per-chunk scheduling
// and synchronization cost dwarfs the tiny per-item work. The crossover is
// therefore above any realistic sample size, so the helpers below are
// **sequential**. The genuine W17 wins in this area are the **fused single-pass**
// reductions (computing `Σy` and `Σy²` together instead of in two passes), which
// are kept.

/// Sum of `f(y_i)` over `y` (W17).
///
/// Sequential: see the module note above for why Rayon is not used here.
pub(crate) fn par_sum<F>(y: &[f64], f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    y.iter().map(|&yi| f(yi)).sum()
}

/// Reduction of `f(y_i)` into a 2-tuple (W17). Sequential; see [`PARALLEL_MIN_N`].
pub(crate) fn par_reduce2<F>(y: &[f64], f: F) -> (f64, f64)
where
    F: Fn(f64) -> (f64, f64),
{
    y.iter()
        .map(|&yi| f(yi))
        .fold((0.0, 0.0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2))
}

/// Reduction of `f(y_i)` into a 3-tuple (W17). Sequential; see [`PARALLEL_MIN_N`].
pub(crate) fn par_reduce3<F>(y: &[f64], f: F) -> (f64, f64, f64)
where
    F: Fn(f64) -> (f64, f64, f64),
{
    y.iter()
        .map(|&yi| f(yi))
        .fold((0.0, 0.0, 0.0), |(a1, a2, a3), (b1, b2, b3)| {
            (a1 + b1, a2 + b2, a3 + b3)
        })
}

/// Reduction of `f(y_i)` into a 4-tuple (W17). Sequential; see [`PARALLEL_MIN_N`].
pub(crate) fn par_reduce4<F>(y: &[f64], f: F) -> (f64, f64, f64, f64)
where
    F: Fn(f64) -> (f64, f64, f64, f64),
{
    y.iter().map(|&yi| f(yi)).fold(
        (0.0, 0.0, 0.0, 0.0),
        |(a1, a2, a3, a4), (b1, b2, b3, b4)| (a1 + b1, a2 + b2, a3 + b3, a4 + b4),
    )
}

/// Fill `out[i] = f(y[i])` (W17). Sequential; see [`PARALLEL_MIN_N`].
pub(crate) fn par_fill<F>(out: &mut [f64], y: &[f64], f: F)
where
    F: Fn(f64) -> f64,
{
    for (o, &yi) in out.iter_mut().zip(y.iter()) {
        *o = f(yi);
    }
}

// --- Phase 1 distribution models (§3) -------------------------------------
//
// Ten models, split by integration path:
//
// | Module            | Model            | Path     |
// |-------------------|------------------|----------|
// | `norm`            | Normal           | Analytic |
// | `norm_p1`         | Normal + pred.   | Analytic |
// | `genextreme`      | GEV              | DMGS     |
// | `genextreme_p1`   | GEV + pred.      | DMGS     |
// | `genpareto`       | GPD              | DMGS     |
// | `gumbel`          | Gumbel           | DMGS     |
// | `lnorm`           | Log-normal       | Analytic |
// | `weibull`         | Weibull          | DMGS     |
// | `gamma`           | Gamma            | DMGS     |
// | `expon`           | Exponential      | Analytic |
pub mod expon;
pub mod gamma;
pub mod genextreme;
pub mod genextreme_p1;
pub mod genpareto;
pub mod gumbel;
pub mod lnorm;
pub mod norm;
pub mod norm_p1;
pub mod weibull;

pub use expon::Exponential;
pub use gamma::Gamma;
pub use genextreme::GenExtreme;
pub use genextreme_p1::GenExtremeP1;
pub use genpareto::GenPareto;
pub use gumbel::Gumbel;
pub use lnorm::LogNormal;
pub use norm::Normal;
pub use norm_p1::NormalP1;
pub use weibull::Weibull;

pub use numerical::{
    numerical_hessian, numerical_mixed_derivatives, numerical_score, numerical_third_derivatives,
    FiniteDiffConfig,
};

/// Analytic-first derivative provider for a model's log-likelihood (§4.1).
///
/// Implementors must provide [`Derivatives::loglik`] and
/// [`Derivatives::n_params`]. The derivative methods default to the numerical
/// finite-difference fallback in [`numerical`]; override any of them to supply
/// an analytic derivative.
///
/// # Contract
///
/// - `params` has length [`Derivatives::n_params`].
/// - `y` is the observed data.
/// - [`Derivatives::score`] returns `n_params` values.
/// - [`Derivatives::hessian`] returns a row-major `n_params × n_params` matrix.
/// - [`Derivatives::third_derivatives`] returns the flattened
///   `n_params × n_params × n_params` tensor in row-major order.
/// - [`Derivatives::mixed_derivatives`] returns a row-major
///   `n_params × y.len()` matrix.
pub trait Derivatives {
    /// The log-likelihood `ℓ(θ; y)` of the model.
    ///
    /// This is the single required primitive: the default derivative methods
    /// below are all computed from it by finite differences.
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64;

    /// Number of model parameters `p`.
    ///
    /// Used to size the derivative outputs and to validate inputs.
    fn n_params(&self) -> usize;

    /// Finite-difference configuration used by the numerical fallback.
    ///
    /// Defaults to [`FiniteDiffConfig::default`]; override to tune the step for
    /// a model with parameters on unusual scales.
    fn fd_config(&self) -> FiniteDiffConfig {
        FiniteDiffConfig::default()
    }

    /// Log-likelihood first derivatives (score) w.r.t. each parameter.
    ///
    /// Returns one value per parameter. Defaults to the central-difference
    /// fallback [`numerical_score`].
    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let cfg = self.fd_config();
        numerical_score(&|p: &[f64]| self.loglik(p, y), params, &cfg)
    }

    /// Log-likelihood second derivatives (observed information / Hessian).
    ///
    /// Returns a row-major `n_params × n_params` matrix flattened into a `Vec`.
    /// Defaults to the central-difference fallback [`numerical_hessian`].
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let cfg = self.fd_config();
        numerical_hessian(&|p: &[f64]| self.loglik(p, y), params, &cfg)
    }

    /// Log-likelihood third derivatives, used for the DMGS bias correction.
    ///
    /// Returns the flattened `n_params × n_params × n_params` tensor in
    /// row-major order. Defaults to the central-difference fallback
    /// [`numerical_third_derivatives`].
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64>
    where
        Self: Sync,
    {
        let cfg = self.fd_config();
        numerical_third_derivatives(&|p: &[f64]| self.loglik(p, y), params, &cfg)
    }

    /// Model-specific mixed derivatives `∂²ℓ/∂θ_i∂y_a` for the Haar-prior term.
    ///
    /// Returns a row-major `n_params × y.len()` matrix flattened into a `Vec`.
    /// Defaults to the central-difference fallback
    /// [`numerical_mixed_derivatives`].
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64>
    where
        Self: Sync,
    {
        let cfg = self.fd_config();
        numerical_mixed_derivatives(&|p: &[f64], yy: &[f64]| self.loglik(p, yy), params, y, &cfg)
    }
}

/// Which derivative path to use for a model (§4.1).
///
/// The default is [`DerivativeMode::Auto`], which uses the model's own
/// implementations (analytic where overridden, numerical otherwise). The other
/// variants force a single path and are primarily useful for testing and
/// benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivativeMode {
    /// Use the model's own methods: analytic where overridden, numerical
    /// fallback otherwise. This is the production default.
    #[default]
    Auto,
    /// Force the numerical finite-difference fallback for every derivative,
    /// ignoring any analytic overrides. Useful to validate an analytic
    /// implementation against the fallback.
    Numerical,
    /// Force the model's own (presumed analytic) methods for every derivative.
    /// Useful to benchmark the analytic path in isolation.
    Analytic,
}

/// Adapter that selects the analytic or numerical derivative path per model.
///
/// Wraps any [`Derivatives`] model and a [`DerivativeMode`], and exposes the
/// same derivative surface. This is the mechanism that makes the
/// analytic-vs-numerical switch explicit, per-model, and testable:
///
/// ```ignore
/// let model = MyModel::new();
/// let auto = AnalyticDerivatives::new(model, DerivativeMode::Auto);
/// let num  = AnalyticDerivatives::new(model, DerivativeMode::Numerical);
/// // `auto.score(..)` uses analytic overrides; `num.score(..)` always uses FD.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AnalyticDerivatives<M> {
    model: M,
    mode: DerivativeMode,
}

impl<M: Derivatives> AnalyticDerivatives<M> {
    /// Wrap `model` with the given derivative-selection `mode`.
    pub fn new(model: M, mode: DerivativeMode) -> Self {
        Self { model, mode }
    }

    /// The wrapped model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// The active derivative-selection mode.
    pub fn mode(&self) -> DerivativeMode {
        self.mode
    }

    /// Log-likelihood, delegated to the wrapped model (mode-independent).
    pub fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        self.model.loglik(params, y)
    }

    /// Number of parameters, delegated to the wrapped model.
    pub fn n_params(&self) -> usize {
        self.model.n_params()
    }

    /// Score under the selected mode.
    pub fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        match self.mode {
            DerivativeMode::Auto | DerivativeMode::Analytic => self.model.score(params, y),
            DerivativeMode::Numerical => {
                let cfg = self.model.fd_config();
                numerical_score(&|p: &[f64]| self.model.loglik(p, y), params, &cfg)
            }
        }
    }

    /// Hessian under the selected mode.
    pub fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        match self.mode {
            DerivativeMode::Auto | DerivativeMode::Analytic => self.model.hessian(params, y),
            DerivativeMode::Numerical => {
                let cfg = self.model.fd_config();
                numerical_hessian(&|p: &[f64]| self.model.loglik(p, y), params, &cfg)
            }
        }
    }

    /// Third derivatives under the selected mode.
    pub fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64>
    where
        M: Sync,
    {
        match self.mode {
            DerivativeMode::Auto | DerivativeMode::Analytic => {
                self.model.third_derivatives(params, y)
            }
            DerivativeMode::Numerical => {
                let cfg = self.model.fd_config();
                numerical_third_derivatives(&|p: &[f64]| self.model.loglik(p, y), params, &cfg)
            }
        }
    }

    /// Mixed derivatives under the selected mode.
    pub fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64>
    where
        M: Sync,
    {
        match self.mode {
            DerivativeMode::Auto | DerivativeMode::Analytic => {
                self.model.mixed_derivatives(params, y)
            }
            DerivativeMode::Numerical => {
                let cfg = self.model.fd_config();
                numerical_mixed_derivatives(
                    &|p: &[f64], yy: &[f64]| self.model.loglik(p, yy),
                    params,
                    y,
                    &cfg,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.2 golden tolerance: `|a - b| <= atol + rtol * |b|`.
    const RTOL: f64 = 1e-8;
    const ATOL: f64 = 1e-10;

    /// Finite differences are not exact; the fallback is validated against a
    /// documented looser tolerance than the §7.2 golden policy (see
    /// [`numerical`]).
    const FD_RTOL: f64 = 1e-6;
    const FD_ATOL: f64 = 1e-8;

    fn assert_close_fd(actual: f64, expected: f64, what: &str) {
        let tol = FD_ATOL + FD_RTOL * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual = {actual:.17e}, expected = {expected:.17e}, \
             |diff| = {:.3e} > tol = {:.3e}",
            (actual - expected).abs(),
            tol
        );
    }

    /// A Gaussian model that implements **only** `loglik` + `n_params`, so all
    /// derivatives come from the numerical fallback defaults.
    struct GaussianNumerical;

    impl Derivatives for GaussianNumerical {
        fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
            let mu = params[0];
            let sigma = params[1];
            let n = y.len() as f64;
            let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
            -n * sigma.ln()
                - ss / (2.0 * sigma * sigma)
                - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
        }

        fn n_params(&self) -> usize {
            2
        }
    }

    /// The same Gaussian model, but with **analytic** score and Hessian
    /// overrides (third derivatives and mixed derivatives still fall back).
    struct GaussianAnalytic;

    impl Derivatives for GaussianAnalytic {
        fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
            GaussianNumerical.loglik(params, y)
        }

        fn n_params(&self) -> usize {
            2
        }

        fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
            let mu = params[0];
            let sigma = params[1];
            let n = y.len() as f64;
            let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
            let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
            vec![
                s1 / (sigma * sigma),
                -n / sigma + ss / (sigma * sigma * sigma),
            ]
        }

        fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
            let mu = params[0];
            let sigma = params[1];
            let n = y.len() as f64;
            let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
            let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
            let s2 = sigma * sigma;
            let s3 = s2 * sigma;
            let s4 = s2 * s2;
            vec![
                -n / s2,
                -2.0 * s1 / s3,
                -2.0 * s1 / s3,
                n / s2 - 3.0 * ss / s4,
            ]
        }
    }

    fn sample() -> Vec<f64> {
        vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8]
    }

    fn analytic_score(mu: f64, sigma: f64, y: &[f64]) -> Vec<f64> {
        let n = y.len() as f64;
        let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
        let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
        vec![
            s1 / (sigma * sigma),
            -n / sigma + ss / (sigma * sigma * sigma),
        ]
    }

    fn analytic_hessian(mu: f64, sigma: f64, y: &[f64]) -> Vec<f64> {
        let n = y.len() as f64;
        let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
        let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let s4 = s2 * s2;
        vec![
            -n / s2,
            -2.0 * s1 / s3,
            -2.0 * s1 / s3,
            n / s2 - 3.0 * ss / s4,
        ]
    }

    #[test]
    fn default_score_matches_analytic_gaussian() {
        let y = sample();
        let params = [0.4, 1.3];
        let got = GaussianNumerical.score(&params, &y);
        let want = analytic_score(params[0], params[1], &y);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_close_fd(*g, *w, &format!("default score[{i}]"));
        }
    }

    #[test]
    fn default_hessian_matches_analytic_gaussian() {
        let y = sample();
        let params = [0.4, 1.3];
        let got = GaussianNumerical.hessian(&params, &y);
        let want = analytic_hessian(params[0], params[1], &y);
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_close_fd(*g, *w, &format!("default hessian[{idx}]"));
        }
    }

    #[test]
    fn default_third_derivatives_are_finite_and_symmetric() {
        let y = sample();
        let params = [0.4, 1.3];
        let got = GaussianNumerical.third_derivatives(&params, &y);
        assert_eq!(got.len(), 8);
        assert!(got.iter().all(|v| v.is_finite()));
        // Symmetry under permutation of indices for n = 2.
        assert_close_fd(got[1], got[2], "third symmetry (0,0,1)=(0,1,0)");
        assert_close_fd(got[1], got[4], "third symmetry (0,0,1)=(1,0,0)");
        assert_close_fd(got[3], got[5], "third symmetry (0,1,1)=(1,0,1)");
        assert_close_fd(got[3], got[6], "third symmetry (0,1,1)=(1,1,0)");
    }

    #[test]
    fn default_mixed_derivatives_have_expected_shape() {
        let y = sample();
        let params = [0.4, 1.3];
        let got = GaussianNumerical.mixed_derivatives(&params, &y);
        assert_eq!(got.len(), 2 * y.len());
        assert!(got.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn analytic_override_agrees_with_numerical_fallback() {
        // The analytic score/Hessian must agree with the numerical fallback
        // within the finite-difference tolerance — this is the core
        // analytic-vs-numerical consistency check.
        let y = sample();
        let params = [0.4, 1.3];
        let num = GaussianNumerical.score(&params, &y);
        let ana = GaussianAnalytic.score(&params, &y);
        for (i, (n, a)) in num.iter().zip(ana.iter()).enumerate() {
            assert_close_fd(*a, *n, &format!("score analytic vs numerical [{i}]"));
        }
        let num_h = GaussianNumerical.hessian(&params, &y);
        let ana_h = GaussianAnalytic.hessian(&params, &y);
        for (i, (n, a)) in num_h.iter().zip(ana_h.iter()).enumerate() {
            assert_close_fd(*a, *n, &format!("hessian analytic vs numerical [{i}]"));
        }
    }

    #[test]
    fn analytic_override_is_exact_under_golden_tolerance() {
        // The analytic path should hit the §7.2 golden tolerance exactly.
        let y = sample();
        let params = [0.4, 1.3];
        let got = GaussianAnalytic.score(&params, &y);
        let want = analytic_score(params[0], params[1], &y);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = ATOL + RTOL * w.abs();
            assert!(
                (g - w).abs() <= tol,
                "analytic score[{i}] not exact: {g} vs {w}"
            );
        }
    }

    #[test]
    fn mode_numerical_forces_fallback_even_with_analytic_overrides() {
        let y = sample();
        let params = [0.4, 1.3];
        let wrapped = AnalyticDerivatives::new(GaussianAnalytic, DerivativeMode::Numerical);
        let got = wrapped.score(&params, &y);
        let want = analytic_score(params[0], params[1], &y);
        // Forced numerical must still agree with the analytic reference within
        // the finite-difference tolerance.
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_close_fd(*g, *w, &format!("forced-numerical score[{i}]"));
        }
    }

    #[test]
    fn mode_analytic_uses_model_overrides() {
        let y = sample();
        let params = [0.4, 1.3];
        let wrapped = AnalyticDerivatives::new(GaussianAnalytic, DerivativeMode::Analytic);
        let got = wrapped.score(&params, &y);
        let want = analytic_score(params[0], params[1], &y);
        // Analytic mode must be exact under the golden tolerance.
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = ATOL + RTOL * w.abs();
            assert!(
                (g - w).abs() <= tol,
                "analytic-mode score[{i}] not exact: {g} vs {w}"
            );
        }
    }

    #[test]
    fn mode_auto_uses_overrides_where_present() {
        let y = sample();
        let params = [0.4, 1.3];
        let wrapped = AnalyticDerivatives::new(GaussianAnalytic, DerivativeMode::Auto);
        assert_eq!(wrapped.mode(), DerivativeMode::Auto);
        assert_eq!(wrapped.n_params(), 2);
        // Auto delegates to the model's own (analytic) score.
        let got = wrapped.score(&params, &y);
        let want = analytic_score(params[0], params[1], &y);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = ATOL + RTOL * w.abs();
            assert!(
                (g - w).abs() <= tol,
                "auto-mode score[{i}] not exact: {g} vs {w}"
            );
        }
        // loglik is mode-independent.
        assert_close_fd(
            wrapped.loglik(&params, &y),
            GaussianNumerical.loglik(&params, &y),
            "loglik delegation",
        );
    }

    #[test]
    fn mode_default_is_auto() {
        assert_eq!(DerivativeMode::default(), DerivativeMode::Auto);
    }
}

/// W9: analytic-vs-numerical consistency tests for all ten Phase-1 models.
///
/// Each model's analytic `score`, `hessian`, `third_derivatives`, and
/// `mixed_derivatives` are compared against the numerical finite-difference
/// fallback. The tolerances are documented per derivative order: central
/// differences are `O(h²)` accurate, so first/second derivatives agree to
/// ~`1e-6` relative and third derivatives (which carry `O(ε/h³)` round-off) to
/// ~`1e-3` relative. Models that keep the numerical fallback for a derivative
/// (GEV third/mixed, GEV-p1 all, gamma CDF hooks) trivially agree.
#[cfg(test)]
mod analytic_vs_numerical {
    use super::*;

    /// First/second-derivative tolerance (central differences, `O(h²)`).
    const RTOL12: f64 = 1e-5;
    const ATOL12: f64 = 1e-7;
    /// Third-derivative tolerance (central differences carry `O(ε/h³)`).
    const RTOL3: f64 = 1e-3;
    const ATOL3: f64 = 1e-3;

    fn assert_close(actual: f64, expected: f64, rtol: f64, atol: f64, what: &str) {
        let tol = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual = {actual:.17e}, expected = {expected:.17e}, \
             |diff| = {:.3e} > tol = {:.3e}",
            (actual - expected).abs(),
            tol
        );
    }

    fn check_model<M: Derivatives + Sync>(m: &M, params: &[f64], y: &[f64], name: &str) {
        let cfg = m.fd_config();
        let ana_s = m.score(params, y);
        let num_s = numerical_score(&|p: &[f64]| m.loglik(p, y), params, &cfg);
        for (i, (a, n)) in ana_s.iter().zip(num_s.iter()).enumerate() {
            assert_close(*a, *n, RTOL12, ATOL12, &format!("{name}.score[{i}]"));
        }
        let ana_h = m.hessian(params, y);
        let num_h = numerical_hessian(&|p: &[f64]| m.loglik(p, y), params, &cfg);
        for (i, (a, n)) in ana_h.iter().zip(num_h.iter()).enumerate() {
            assert_close(*a, *n, RTOL12, ATOL12, &format!("{name}.hessian[{i}]"));
        }
        let ana_t = m.third_derivatives(params, y);
        let num_t = numerical_third_derivatives(&|p: &[f64]| m.loglik(p, y), params, &cfg);
        for (i, (a, n)) in ana_t.iter().zip(num_t.iter()).enumerate() {
            assert_close(*a, *n, RTOL3, ATOL3, &format!("{name}.third[{i}]"));
        }
        let ana_m = m.mixed_derivatives(params, y);
        let num_m =
            numerical_mixed_derivatives(&|p: &[f64], yy: &[f64]| m.loglik(p, yy), params, y, &cfg);
        for (i, (a, n)) in ana_m.iter().zip(num_m.iter()).enumerate() {
            assert_close(*a, *n, RTOL12, ATOL12, &format!("{name}.mixed[{i}]"));
        }
    }

    fn general_sample() -> Vec<f64> {
        vec![
            9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41, 10.93, 8.21, 12.05,
            10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.90, 7.63, 10.81, 9.29, 12.44, 8.98, 10.17,
            11.53, 9.60, 10.72, 8.40,
        ]
    }

    fn positive_sample() -> Vec<f64> {
        vec![
            0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55, 0.33, 2.91, 1.02, 0.69,
            3.77, 1.34, 0.51, 2.18, 0.88, 1.96, 0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31,
            1.72, 0.93,
        ]
    }

    #[test]
    fn norm_analytic_matches_numerical() {
        check_model(&Normal, &[10.0, 2.0], &general_sample(), "norm");
    }

    #[test]
    fn expon_analytic_matches_numerical() {
        check_model(&Exponential, &[0.8], &positive_sample(), "expon");
    }

    #[test]
    fn lnorm_analytic_matches_numerical() {
        check_model(&LogNormal, &[0.0, 1.0], &positive_sample(), "lnorm");
    }

    #[test]
    fn gumbel_analytic_matches_numerical() {
        check_model(&Gumbel, &[10.0, 2.0], &general_sample(), "gumbel");
    }

    #[test]
    fn weibull_analytic_matches_numerical() {
        check_model(&Weibull, &[1.4, 1.1], &positive_sample(), "weibull");
    }

    #[test]
    fn gamma_analytic_matches_numerical() {
        check_model(&Gamma, &[2.0, 1.3], &positive_sample(), "gamma");
    }

    #[test]
    fn genextreme_analytic_matches_numerical() {
        // shape != 0 so the analytic score/Hessian are used.
        check_model(
            &GenExtreme,
            &[10.0, 2.0, 0.2],
            &general_sample(),
            "genextreme",
        );
    }

    #[test]
    fn genpareto_analytic_matches_numerical() {
        // shape != 0 so the analytic forms are used.
        check_model(&GenPareto, &[1.5, 0.3], &positive_sample(), "genpareto");
    }

    #[test]
    fn norm_p1_analytic_matches_numerical() {
        let t: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect();
        let t_mean = t.iter().sum::<f64>() / t.len() as f64;
        let m = NormalP1::new(t, t_mean, 1.0);
        check_model(&m, &[10.0, 0.5, 2.0], &general_sample(), "norm_p1");
    }

    #[test]
    fn genextreme_p1_uses_numerical_fallback() {
        // The reference has no complete analytic 4-parameter form, so this model
        // keeps the numerical fallback; the check is trivially satisfied but
        // documents the intent.
        let t: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect();
        let t_mean = t.iter().sum::<f64>() / t.len() as f64;
        let m = GenExtremeP1::new(t, t_mean, 1.0);
        check_model(
            &m,
            &[10.0, 0.5, 2.0, 0.1],
            &general_sample(),
            "genextreme_p1",
        );
    }
}
