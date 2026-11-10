//! Analytic calibrated-prior (CP) quantiles for the closed-form models
//! (Phase 1, Issue 1).
//!
//! # Why this module exists
//!
//! The reference `fitdistcp` package computes the CP predictive quantiles for
//! `norm`, `lnorm`, and `expon` with **exact analytic formulas** (a Student-t
//! predictive distribution), not the DMGS asymptotic expansion. The Rust core
//! previously routed these three models through the DMGS expansion, which is a
//! first-order approximation of the exact Bayesian prediction integral. The
//! resulting `O(1/n)` truncation error is small for `norm`/`expon` but large for
//! `lnorm` (up to ~48% at `n = 10`), because the log-normal CP quantile is an
//! exponential of a Student-t quantile and the truncation error is amplified by
//! the exponentiation.
//!
//! This module reproduces the reference's exact analytic CP quantiles so the
//! three models match the reference to near machine precision. The DMGS path
//! remains available (see [`crate::api::predict`] and the `use_dmgs` flag) for
//! the golden cross-check.
//!
//! # Reference formulas
//!
//! - **`norm`** (reference `qnorm_cp`, [`norm.py:70`] / `030a_norm.R:67-75`):
//!   ```text
//!   sgu = v2hat * sqrt(n / (n - 1))          # MLE sigma -> unbiased
//!   sg  = sgu  * sqrt((n + 1) / n)           # unbiased -> predictive
//!   cp_quantile(p) = mu + qt(p, df = n - 1) * sg
//!   ```
//!   where `mu = v1hat` is the MLE mean and `v2hat` the MLE standard deviation.
//!
//! - **`lnorm`** (reference `qlnorm_cp`, [`lnorm.py:77`] / `035a_lnorm.R`):
//!   ```text
//!   mu = mean(log x)
//!   s1 = sd(log x, ddof = 1)                 # unbiased log-scale sd
//!   cp_quantile(p) = exp(mu + qt(p, df = n - 1) * s1 * sqrt(1 + 1/n))
//!   ```
//!
//! - **`expon`** (reference `qexp_cp`, [`expon.py:76`] / `010a_exp.R`):
//!   ```text
//!   cp_quantile(p) = sumx * (alpha^(-1/n) - 1),   alpha = 1 - p
//!   ```
//!   where `sumx = Σ x_i`.
//!
//! The Student-t quantile `qt(p, df)` is provided by
//! [`crate::special::student_t_quantile`].
//!
//! [`norm.py:70`]: ../../../../tools/fitdistcp-py-ref/fitdistcp/norm.py
//! [`lnorm.py:77`]: ../../../../tools/fitdistcp-py-ref/fitdistcp/lnorm.py
//! [`expon.py:76`]: ../../../../tools/fitdistcp-py-ref/fitdistcp/expon.py

use crate::special::student_t_quantile;

/// Analytic CP quantiles for the **normal** model (reference `qnorm_cp`).
///
/// `params = [mu, sigma]` are the MLE mean and standard deviation (the Rust
/// `Normal` parameterization). `y` is the observed data (used only for `n`).
/// `p` are the probability levels; the returned quantiles are at probability
/// `p` (the reference evaluates `qt(1 - alpha)` with `alpha = 1 - p`, i.e. at
/// `p`).
///
/// The predictive distribution is a Student-t with `df = n - 1`, centered at
/// `mu`, scaled by `sg = sigma * sqrt(n/(n-1)) * sqrt((n+1)/n)`.
///
/// # Panics
///
/// Panics if `y` is empty or `y.len() < 2` (the analytic CP needs `df = n - 1 >
/// 0`).
pub fn cp_quantiles_norm(params: &[f64], y: &[f64], p: &[f64]) -> Vec<f64> {
    let n = y.len();
    assert!(n >= 2, "cp_quantiles_norm: need n >= 2 for df = n - 1 > 0");
    let nx = n as f64;
    let mu = params[0];
    let sigma = params[1];

    // MLE sigma -> unbiased -> predictive (reference qnorm_cp).
    let sgu = sigma * (nx / (nx - 1.0)).sqrt();
    let sg = sgu * ((nx + 1.0) / nx).sqrt();

    p.iter()
        .map(|&pi| mu + student_t_quantile(pi, nx - 1.0) * sg)
        .collect()
}

/// Analytic CP quantiles for the **log-normal** model (reference `qlnorm_cp`).
///
/// `params = [meanlog, sdlog]` are the MLE log-scale mean and sd (the Rust
/// `LogNormal` parameterization). `y` is the observed data; the reference uses
/// `mu = mean(log x)` and `s1 = sd(log x, ddof = 1)` (the unbiased log-scale
/// sd), which are recomputed here from `y` to match the reference exactly.
/// `p` are the probability levels.
///
/// The predictive quantile is
/// `exp(mu + qt(p, df = n - 1) * s1 * sqrt(1 + 1/n))`.
///
/// # Panics
///
/// Panics if `y` is empty or `y.len() < 2` (the analytic CP needs `df = n - 1 >
/// 0`).
pub fn cp_quantiles_lnorm(_params: &[f64], y: &[f64], p: &[f64]) -> Vec<f64> {
    let n = y.len();
    assert!(n >= 2, "cp_quantiles_lnorm: need n >= 2 for df = n - 1 > 0");
    let nx = n as f64;

    // mu = mean(log x); s1 = sd(log x, ddof = 1) (reference qlnorm_cp).
    let logs: Vec<f64> = y.iter().map(|&yi| yi.ln()).collect();
    let mu = logs.iter().sum::<f64>() / nx;
    let var = logs.iter().map(|&l| (l - mu) * (l - mu)).sum::<f64>() / (nx - 1.0);
    let s1 = var.sqrt();

    let scale = s1 * (1.0 + 1.0 / nx).sqrt();
    p.iter()
        .map(|&pi| (mu + student_t_quantile(pi, nx - 1.0) * scale).exp())
        .collect()
}

/// Analytic CP quantiles for the **exponential** model (reference `qexp_cp`).
///
/// `params = [rate]` is the MLE rate (the Rust `Exponential` parameterization).
/// `y` is the observed data; the reference uses `sumx = Σ x_i`. `p` are the
/// probability levels.
///
/// The predictive quantile is `sumx * (alpha^(-1/n) - 1)` with `alpha = 1 - p`.
///
/// # Panics
///
/// Panics if `y` is empty.
pub fn cp_quantiles_expon(_params: &[f64], y: &[f64], p: &[f64]) -> Vec<f64> {
    let n = y.len();
    assert!(n >= 1, "cp_quantiles_expon: need n >= 1");
    let nx = n as f64;
    let sumx: f64 = y.iter().sum();

    p.iter()
        .map(|&pi| {
            let alpha = 1.0 - pi;
            sumx * (alpha.powf(-1.0 / nx) - 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn norm_cp_quantile_at_median_is_mu() {
        // qt(0.5, df) = 0, so the CP median is the MLE mean.
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        let mu = y.iter().sum::<f64>() / y.len() as f64;
        let q = cp_quantiles_norm(&[mu, 1.0], &y, &[0.5]);
        assert_close(q[0], mu, "norm cp median");
    }

    #[test]
    fn norm_cp_quantiles_are_monotone() {
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        let q = cp_quantiles_norm(&[0.0, 1.0], &y, &[0.1, 0.25, 0.5, 0.75, 0.9]);
        for w in q.windows(2) {
            assert!(w[0] < w[1], "not monotone: {q:?}");
        }
    }

    #[test]
    fn lnorm_cp_quantile_at_median_is_exp_mu() {
        // qt(0.5, df) = 0, so the CP median is exp(mean(log x)).
        let y: Vec<f64> = vec![0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55];
        let mu = y.iter().map(|v| v.ln()).sum::<f64>() / y.len() as f64;
        let q = cp_quantiles_lnorm(&[mu, 1.0], &y, &[0.5]);
        assert_close(q[0], mu.exp(), "lnorm cp median");
    }

    #[test]
    fn expon_cp_quantile_at_median_matches_formula() {
        let y = vec![0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.1, 0.7, 1.8];
        let sumx: f64 = y.iter().sum();
        let n = y.len() as f64;
        let q = cp_quantiles_expon(&[n / sumx], &y, &[0.5]);
        let expected = sumx * (0.5_f64.powf(-1.0 / n) - 1.0);
        assert_close(q[0], expected, "expon cp median");
    }

    #[test]
    fn expon_cp_quantiles_are_monotone() {
        let y = vec![0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.1, 0.7, 1.8];
        let q = cp_quantiles_expon(&[1.0], &y, &[0.1, 0.25, 0.5, 0.75, 0.9]);
        for w in q.windows(2) {
            assert!(w[0] < w[1], "not monotone: {q:?}");
        }
    }
}
