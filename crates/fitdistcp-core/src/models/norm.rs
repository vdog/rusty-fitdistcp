//! Normal distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [mu, sigma]` with `sigma > 0`:
//!
//! ```text
//! f(y; mu, sigma) = 1/(sigma sqrt(2 pi)) exp(-(y - mu)^2 / (2 sigma^2))
//! ```
//!
//! This matches the reference `fitdistcp` convention (`norm` uses the mean and
//! standard deviation, not the variance).
//!
//! # Integration path
//!
//! **Analytic** (§3): the normal is the baseline model with a closed-form CP.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9). The score, Hessian, third derivatives,
//! and mixed derivatives are closed form; the density/CDF derivative hooks used
//! by the DMGS expansion (`f1`, `f2`, `mu1`, `mu2`) are the reference's
//! `norm_fd`/`norm_fdd`/`norm_pd`/`norm_pdd` transcribed directly. The analytic
//! forms are validated against the numerical fallback in the tests.
//!
//! With `s1 = Σ (y_i - mu)` and `ss = Σ (y_i - mu)^2`:
//!
//! - `∂ℓ/∂mu    = s1 / sigma^2`
//! - `∂ℓ/∂sigma = -n/sigma + ss/sigma^3`
//! - `∂²ℓ/∂mu²       = -n/sigma^2`
//! - `∂²ℓ/∂mu∂sigma  = -2 s1 / sigma^3`
//! - `∂²ℓ/∂sigma²    = n/sigma^2 - 3 ss/sigma^4`
//! - `∂³ℓ/∂mu³       = 0`
//! - `∂³ℓ/∂mu²∂sigma = 2n/sigma^3`
//! - `∂³ℓ/∂mu∂sigma² = 6 s1 / sigma^4`
//! - `∂³ℓ/∂sigma³    = -2n/sigma^3 + 12 ss/sigma^5`
//! - `∂²ℓ/∂mu∂y_a    = 1/sigma^2`
//! - `∂²ℓ/∂sigma∂y_a = 2 (y_a - mu)/sigma^3`

use crate::models::Derivatives;
use crate::special::erfc;

/// The normal distribution model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Normal;

impl Normal {
    /// Create a normal model.
    pub fn new() -> Self {
        Normal
    }

    /// Log-density `ln f(y; mu, sigma)` at a single point.
    pub fn logpdf(&self, y: f64, mu: f64, sigma: f64) -> f64 {
        let z = (y - mu) / sigma;
        -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Density `f(y; mu, sigma)` at a single point.
    pub fn pdf(&self, y: f64, mu: f64, sigma: f64) -> f64 {
        self.logpdf(y, mu, sigma).exp()
    }

    /// CDF `F(y; mu, sigma)` at a single point.
    pub fn cdf(&self, y: f64, mu: f64, sigma: f64) -> f64 {
        let z = (y - mu) / (sigma * std::f64::consts::SQRT_2);
        0.5 * erfc(-z)
    }

    /// Quantile (inverse CDF) `F^{-1}(p; mu, sigma)`.
    ///
    /// Uses the Acklam rational approximation refined by one Halley step, which
    /// is accurate to near machine precision over `p in (0, 1)`.
    pub fn quantile(&self, p: f64, mu: f64, sigma: f64) -> f64 {
        mu + sigma * standard_normal_quantile(p)
    }
}

/// Standard-normal quantile `Φ^{-1}(p)` (Acklam + one Halley refinement).
pub(crate) fn standard_normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Acklam's coefficients.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley refinement step: x <- x - (Phi(x) - p) / phi(x), with the
    // second-order correction. This lifts the approximation to ~1e-15.
    let e = 0.5 * erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

impl Derivatives for Normal {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let mu = params[0];
        let sigma = params[1];
        // W17 (H4/H6): gated parallel reduction over observations.
        crate::models::par_sum(y, |yi| self.logpdf(yi, mu, sigma))
    }

    fn n_params(&self) -> usize {
        2
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let mu = params[0];
        let sigma = params[1];
        let n = y.len() as f64;
        // W17 (H4/H6): single gated pass computing both sufficient statistics.
        let (s1, ss) = crate::models::par_reduce2(y, |yi| {
            let d = yi - mu;
            (d, d * d)
        });
        let s2 = sigma * sigma;
        vec![s1 / s2, -n / sigma + ss / (s2 * sigma)]
    }

    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let mu = params[0];
        let sigma = params[1];
        let n = y.len() as f64;
        let (s1, ss) = crate::models::par_reduce2(y, |yi| {
            let d = yi - mu;
            (d, d * d)
        });
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

    /// Analytic third derivatives of the log-likelihood (W9).
    ///
    /// Row-major `2 × 2 × 2` tensor, index `(i, j, k)` at `i*4 + j*2 + k`:
    /// `∂³ℓ/∂mu³ = 0`, `∂³ℓ/∂mu²∂sigma = 2n/sigma³`,
    /// `∂³ℓ/∂mu∂sigma² = 6 s1/sigma⁴`,
    /// `∂³ℓ/∂sigma³ = -2n/sigma³ + 12 ss/sigma⁵`.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let mu = params[0];
        let sigma = params[1];
        let n = y.len() as f64;
        let (s1, ss) = crate::models::par_reduce2(y, |yi| {
            let d = yi - mu;
            (d, d * d)
        });
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let s4 = s2 * s2;
        let s5 = s4 * sigma;
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
    /// Row-major `2 × n` matrix: `∂²ℓ/∂mu∂y_a = 1/sigma²` (constant in `a`) and
    /// `∂²ℓ/∂sigma∂y_a = 2 (y_a - mu)/sigma³`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let mu = params[0];
        let sigma = params[1];
        let ny = y.len();
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let mut out = vec![0.0; 2 * ny];
        // W17 (H3/H6): gated parallel fill of the two row-major rows.
        let (row0, row1) = out.split_at_mut(ny);
        crate::models::par_fill(row0, y, |_| 1.0 / s2);
        crate::models::par_fill(row1, y, |ya| 2.0 * (ya - mu) / s3);
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
    fn standard_normal_logpdf_matches_reference() {
        let m = Normal;
        // ln f(0; 0, 1) = -0.5 ln(2 pi)
        assert_close(
            m.logpdf(0.0, 0.0, 1.0),
            -0.5 * (2.0 * std::f64::consts::PI).ln(),
            "logpdf(0)",
        );
        // ln f(1; 0, 1) = -0.5 - 0.5 ln(2 pi)
        assert_close(
            m.logpdf(1.0, 0.0, 1.0),
            -0.5 - 0.5 * (2.0 * std::f64::consts::PI).ln(),
            "logpdf(1)",
        );
    }

    #[test]
    fn standard_normal_cdf_matches_reference() {
        let m = Normal;
        assert_close(m.cdf(0.0, 0.0, 1.0), 0.5, "cdf(0)");
        // Phi(1.96) ≈ 0.9750021048517795
        assert_close(m.cdf(1.96, 0.0, 1.0), 0.975_002_104_851_779_5, "cdf(1.96)");
        // Phi(-1) ≈ 0.15865525393145707
        assert_close(m.cdf(-1.0, 0.0, 1.0), 0.158_655_253_931_457_07, "cdf(-1)");
    }

    #[test]
    fn standard_normal_quantile_matches_reference() {
        let m = Normal;
        assert_close(m.quantile(0.5, 0.0, 1.0), 0.0, "q(0.5)");
        // Phi^{-1}(0.975) ≈ 1.959963984540054
        assert_close(
            m.quantile(0.975, 0.0, 1.0),
            1.959_963_984_540_054,
            "q(0.975)",
        );
        // Phi^{-1}(0.025) ≈ -1.959963984540054
        assert_close(
            m.quantile(0.025, 0.0, 1.0),
            -1.959_963_984_540_054,
            "q(0.025)",
        );
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = Normal;
        for &p in &[0.01_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = m.quantile(p, 2.0, 3.0);
            assert_close(m.cdf(x, 2.0, 3.0), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn analytic_score_matches_numerical_fallback() {
        let m = Normal;
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        let params = [0.4, 1.3];
        let ana = m.score(&params, &y);
        let num = crate::models::numerical_score(
            &|p: &[f64]| m.loglik(p, &y),
            &params,
            &crate::models::FiniteDiffConfig::default(),
        );
        for (i, (a, n)) in ana.iter().zip(num.iter()).enumerate() {
            let tol = 1e-6 + 1e-6 * n.abs();
            assert!((a - n).abs() <= tol, "score[{i}]: {a} vs {n}");
        }
    }

    #[test]
    fn fits_synthetic_data() {
        let m = Normal;
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        let fit = fit(&m, &y, &[0.0, 1.0]);
        assert!(fit.convergence.is_converged());
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        assert_close(fit.ml_params[0], mean, "mu");
        assert_close(fit.ml_params[1], var.sqrt(), "sigma");
    }
}
