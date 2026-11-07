//! Log-normal distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [meanlog, sdlog]` with `sdlog > 0` (the reference `fitdistcp`
//! convention, matching R's `dlnorm`): the parameters are the mean and standard
//! deviation of the **logarithm** of the variable.
//!
//! ```text
//! f(y; meanlog, sdlog) = 1/(y sdlog sqrt(2 pi))
//!                        exp(-(ln y - meanlog)^2 / (2 sdlog^2)),  y > 0
//! ```
//!
//! # Integration path
//!
//! **Analytic** (§3): the log-normal is the log-transform of the normal, so its
//! derivatives reduce to the normal's on `ln y`.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9): the score, Hessian, and third
//! derivatives are the normal's evaluated at `z_i = ln y_i`; the mixed
//! derivatives carry the extra `1/y_a` Jacobian factor. The density/CDF
//! derivative hooks used by the DMGS expansion are the reference's
//! `lnorm_fd`/`lnorm_fdd`/`lnorm_pd`/`lnorm_pdd` transcribed directly.

use crate::models::norm::standard_normal_quantile;
use crate::models::Derivatives;
use crate::special::erfc;

/// The log-normal distribution model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogNormal;

impl LogNormal {
    /// Create a log-normal model.
    pub fn new() -> Self {
        LogNormal
    }

    /// Log-density `ln f(y; meanlog, sdlog)` at a single point.
    pub fn logpdf(&self, y: f64, meanlog: f64, sdlog: f64) -> f64 {
        let z = (y.ln() - meanlog) / sdlog;
        -0.5 * z * z - y.ln() - sdlog.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Density `f(y; meanlog, sdlog)` at a single point.
    pub fn pdf(&self, y: f64, meanlog: f64, sdlog: f64) -> f64 {
        self.logpdf(y, meanlog, sdlog).exp()
    }

    /// CDF `F(y; meanlog, sdlog)` at a single point.
    pub fn cdf(&self, y: f64, meanlog: f64, sdlog: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        let z = (y.ln() - meanlog) / (sdlog * std::f64::consts::SQRT_2);
        0.5 * erfc(-z)
    }

    /// Quantile `F^{-1}(p; meanlog, sdlog) = exp(meanlog + sdlog Φ^{-1}(p))`.
    pub fn quantile(&self, p: f64, meanlog: f64, sdlog: f64) -> f64 {
        (meanlog + sdlog * standard_normal_quantile(p)).exp()
    }
}

impl Derivatives for LogNormal {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let meanlog = params[0];
        let sdlog = params[1];
        // W17 (H4/H6): gated parallel reduction over observations.
        crate::models::par_sum(y, |yi| self.logpdf(yi, meanlog, sdlog))
    }

    fn n_params(&self) -> usize {
        2
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let meanlog = params[0];
        let sdlog = params[1];
        let n = y.len() as f64;
        // W17 (H4/H6): single gated pass computing both sufficient statistics.
        let (s1, ss) = crate::models::par_reduce2(y, |yi| {
            let d = yi.ln() - meanlog;
            (d, d * d)
        });
        let s2 = sdlog * sdlog;
        vec![s1 / s2, -n / sdlog + ss / (s2 * sdlog)]
    }

    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let meanlog = params[0];
        let sdlog = params[1];
        let n = y.len() as f64;
        let (s1, ss) = crate::models::par_reduce2(y, |yi| {
            let d = yi.ln() - meanlog;
            (d, d * d)
        });
        let s2 = sdlog * sdlog;
        let s3 = s2 * sdlog;
        let s4 = s2 * s2;
        vec![
            -n / s2,
            -2.0 * s1 / s3,
            -2.0 * s1 / s3,
            n / s2 - 3.0 * ss / s4,
        ]
    }

    /// Analytic third derivatives of the log-likelihood (W9).
    ///
    /// Identical to the normal's third derivatives with `z_i = ln y_i` (the
    /// `-ln y_i` term in the log-density is linear in the data and contributes
    /// nothing). Row-major `2 × 2 × 2`, index `(i, j, k)` at `i*4 + j*2 + k`.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let meanlog = params[0];
        let sdlog = params[1];
        let n = y.len() as f64;
        let (s1, ss) = crate::models::par_reduce2(y, |yi| {
            let d = yi.ln() - meanlog;
            (d, d * d)
        });
        let s2 = sdlog * sdlog;
        let s3 = s2 * sdlog;
        let s4 = s2 * s2;
        let s5 = s4 * sdlog;
        let d_mmm = 0.0;
        let d_mms = 2.0 * n / s3;
        let d_mss = 6.0 * s1 / s4;
        let d_sss = -2.0 * n / s3 + 12.0 * ss / s5;
        vec![
            d_mmm, d_mms, // (0,0,0), (0,0,1)
            d_mms, d_mss, // (0,1,0), (0,1,1)
            d_mms, d_mss, // (1,0,0), (1,0,1)
            d_mss, d_sss, // (1,1,0), (1,1,1)
        ]
    }

    /// Analytic mixed derivatives `∂²ℓ/∂θ_i∂y_a` (W9).
    ///
    /// Row-major `2 × n`: `∂²ℓ/∂meanlog∂y_a = 1/(y_a sdlog²)` and
    /// `∂²ℓ/∂sdlog∂y_a = 2 (ln y_a - meanlog)/(y_a sdlog³)`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let meanlog = params[0];
        let sdlog = params[1];
        let ny = y.len();
        let s2 = sdlog * sdlog;
        let s3 = s2 * sdlog;
        let mut out = vec![0.0; 2 * ny];
        // W17 (H3/H6): gated parallel fill of the two row-major rows.
        let (row0, row1) = out.split_at_mut(ny);
        crate::models::par_fill(row0, y, |ya| 1.0 / (ya * s2));
        crate::models::par_fill(row1, y, |ya| 2.0 * (ya.ln() - meanlog) / (ya * s3));
        out
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
        let m = LogNormal;
        // ln f(1; 0, 1) = -0.5 ln(2 pi)
        assert_close(
            m.logpdf(1.0, 0.0, 1.0),
            -0.5 * (2.0 * std::f64::consts::PI).ln(),
            "logpdf(1;0,1)",
        );
        // ln f(e; 0, 1) = -1 - 0.5 - 0.5 ln(2 pi)
        assert_close(
            m.logpdf(std::f64::consts::E, 0.0, 1.0),
            -1.0 - 0.5 - 0.5 * (2.0 * std::f64::consts::PI).ln(),
            "logpdf(e;0,1)",
        );
    }

    #[test]
    fn cdf_matches_reference() {
        let m = LogNormal;
        assert_close(m.cdf(1.0, 0.0, 1.0), 0.5, "cdf(1;0,1)");
        assert_close(m.cdf(0.0, 0.0, 1.0), 0.0, "cdf(0)");
        // F(e; 0, 1) = Phi(1) ≈ 0.8413447460685429
        assert_close(
            m.cdf(std::f64::consts::E, 0.0, 1.0),
            0.841_344_746_068_542_9,
            "cdf(e;0,1)",
        );
    }

    #[test]
    fn quantile_matches_reference() {
        let m = LogNormal;
        // q(0.5; 0, 1) = exp(0) = 1
        assert_close(m.quantile(0.5, 0.0, 1.0), 1.0, "q(0.5)");
        // q(0.975; 0, 1) = exp(1.959963984540054)
        assert_close(
            m.quantile(0.975, 0.0, 1.0),
            1.959_963_984_540_054_f64.exp(),
            "q(0.975)",
        );
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = LogNormal;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 0.5, 0.8);
            assert_close(m.cdf(x, 0.5, 0.8), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn fits_synthetic_data() {
        let m = LogNormal;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let fit = fit(&m, &y, &[0.0, 1.0]);
        assert!(fit.convergence.is_converged());
        // MLE meanlog = mean(ln y), sdlog = sqrt(biased var(ln y)).
        let n = y.len() as f64;
        let logs: Vec<f64> = y.iter().map(|&yi| yi.ln()).collect();
        let mean = logs.iter().sum::<f64>() / n;
        let var = logs.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / n;
        assert_close(fit.ml_params[0], mean, "meanlog");
        assert_close(fit.ml_params[1], var.sqrt(), "sdlog");
    }
}
