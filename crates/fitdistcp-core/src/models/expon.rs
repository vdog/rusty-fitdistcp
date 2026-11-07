//! Exponential distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [rate]` with `rate > 0` (the reference `fitdistcp` convention uses
//! the **rate**, i.e. `dexp` in R):
//!
//! ```text
//! f(y; rate) = rate exp(-rate y),   y >= 0
//! ```
//!
//! # Integration path
//!
//! **Analytic** (§3): the exponential is the simplest DMGS sanity check and has
//! a closed-form CP.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9). With `S = Σ y_i`:
//!
//! - `∂ℓ/∂rate  = n/rate - S`
//! - `∂²ℓ/∂rate² = -n/rate²`
//! - `∂³ℓ/∂rate³ = 2n/rate³`
//! - `∂²ℓ/∂rate∂y_a = -1`
//!
//! The density/CDF derivative hooks used by the DMGS expansion are the
//! reference's `exp_fd`/`exp_fdd`/`exp_pd`/`exp_pdd` transcribed directly.

use crate::models::Derivatives;

/// The exponential distribution model (rate parameterization).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Exponential;

impl Exponential {
    /// Create an exponential model.
    pub fn new() -> Self {
        Exponential
    }

    /// Log-density `ln f(y; rate)` at a single point.
    pub fn logpdf(&self, y: f64, rate: f64) -> f64 {
        rate.ln() - rate * y
    }

    /// Density `f(y; rate)` at a single point.
    pub fn pdf(&self, y: f64, rate: f64) -> f64 {
        rate * (-rate * y).exp()
    }

    /// CDF `F(y; rate) = 1 - exp(-rate y)`.
    pub fn cdf(&self, y: f64, rate: f64) -> f64 {
        -(-rate * y).exp_m1()
    }

    /// Quantile `F^{-1}(p; rate) = -ln(1 - p) / rate`.
    pub fn quantile(&self, p: f64, rate: f64) -> f64 {
        -(1.0 - p).ln() / rate
    }
}

impl Derivatives for Exponential {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let rate = params[0];
        let n = y.len() as f64;
        // W17 (H4/H6): gated parallel reduction over observations.
        let s: f64 = crate::models::par_sum(y, |yi| yi);
        n * rate.ln() - rate * s
    }

    fn n_params(&self) -> usize {
        1
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let rate = params[0];
        let n = y.len() as f64;
        let s: f64 = crate::models::par_sum(y, |yi| yi);
        vec![n / rate - s]
    }

    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let rate = params[0];
        let n = y.len() as f64;
        vec![-n / (rate * rate)]
    }

    /// Analytic third derivative `∂³ℓ/∂rate³ = 2n/rate³` (W9).
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let rate = params[0];
        let n = y.len() as f64;
        vec![2.0 * n / (rate * rate * rate)]
    }

    /// Analytic mixed derivatives `∂²ℓ/∂rate∂y_a = -1` (W9).
    fn mixed_derivatives(&self, _params: &[f64], y: &[f64]) -> Vec<f64> {
        vec![-1.0; y.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::fit;

    const RTOL: f64 = 1e-8;
    const ATOL: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64, what: &str) {
        let tol = ATOL + RTOL * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual = {actual:.17e}, expected = {expected:.17e}, \
             |diff| = {:.3e} > tol = {:.3e}",
            (actual - expected).abs(),
            tol
        );
    }

    #[test]
    fn logpdf_matches_reference() {
        let m = Exponential;
        // ln f(1; 1) = -1
        assert_close(m.logpdf(1.0, 1.0), -1.0, "logpdf(1;1)");
        // ln f(2; 0.5) = ln 0.5 - 1
        assert_close(m.logpdf(2.0, 0.5), 0.5_f64.ln() - 1.0, "logpdf(2;0.5)");
    }

    #[test]
    fn cdf_matches_reference() {
        let m = Exponential;
        assert_close(m.cdf(0.0, 1.0), 0.0, "cdf(0)");
        // F(1; 1) = 1 - e^{-1} ≈ 0.6321205588285577
        assert_close(m.cdf(1.0, 1.0), 0.632_120_558_828_557_7, "cdf(1;1)");
    }

    #[test]
    fn quantile_matches_reference() {
        let m = Exponential;
        // q(0.5; 1) = ln 2 ≈ 0.6931471805599453
        assert_close(m.quantile(0.5, 1.0), 2.0_f64.ln(), "q(0.5;1)");
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = Exponential;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 2.5);
            assert_close(m.cdf(x, 2.5), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn fits_synthetic_data() {
        let m = Exponential;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let fit = fit(&m, &y, &[1.0]);
        assert!(fit.convergence.is_converged());
        // MLE rate = 1 / mean(y).
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        assert_close(fit.ml_params[0], 1.0 / mean, "rate");
    }
}
