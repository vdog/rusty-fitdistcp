//! Normal distribution with a predictor (`norm_p1`) model (§3).
//!
//! # Parameterization (reference `fitdistcp`)
//!
//! `norm_p1` is the normal model whose **location** is a linear function of a
//! predictor `t`:
//!
//! ```text
//! mu_i = beta0 + beta1 * (t_i - mean(t))
//! y_i  ~ Normal(mu_i, sigma)
//! ```
//!
//! `params = [beta0, beta1, sigma]` with `sigma > 0`. This is the reference
//! `fitdistcp` convention (W14): the predictor is **centered by its mean**
//! (`t = t - mean(t)`), and the model is fit on the centered predictor with
//! `mu = a + b * t_centered`. The reference's `t0`/`n0` arguments do **not**
//! enter the parameterization; they only select the **prediction point** at
//! which the predictive quantiles/densities are reported (see
//! [`NormalP1::centered_prediction`] and the binding-level `maket0`).
//!
//! The reference `maket0` resolves the prediction point as:
//!
//! - if `t0` is supplied, use it directly (`n0` must be absent);
//! - otherwise `t0 = t[n0]` (a **positional index** into `t`, 1-based in R).
//!
//! The prediction point is then centered the same way: `t0_centered =
//! t0 - mean(t)`, and `muhat0 = beta0 + beta1 * t0_centered`.
//!
//! # Integration path
//!
//! **Analytic** (§3): the normal is the baseline model with a closed-form CP.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9): the score, Hessian, third derivatives,
//! and mixed derivatives are the normal's with the predictor-centered residual
//! `r_i = y_i - (beta0 + beta1 * c_i)` where `c_i = t_i - mean(t)`. The
//! density/CDF derivative hooks are the reference's `norm_p1_fd`/`norm_p1_fdd`/
//! `norm_p1_pd`/`norm_p1_pdd`.

use crate::models::norm::standard_normal_quantile;
use crate::models::Derivatives;
use crate::special::erfc;

/// The normal-with-predictor (`norm_p1`) model.
///
/// Holds the predictor plumbing `t`, `t0`, `n0`. The predictor `t` must have the
/// same length as the data `y` when present.
///
/// # Field semantics (W14, reference convention)
///
/// - `t`: the **raw** predictor values, one per observation (empty when absent).
/// - `t0`: the **resolved prediction point** (a predictor value, not an index).
///   The binding resolves `t0` from the user's `t0`/`n0` via `maket0` before
///   constructing the model. When `t` is empty, `t0` is ignored.
/// - `n0`: retained for API compatibility; it does **not** enter the
///   parameterization (the reference uses it only to derive `t0`).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalP1 {
    /// Raw predictor values `t`, one per observation (empty when absent).
    pub t: Vec<f64>,
    /// Resolved prediction point `t0` (a predictor value).
    pub t0: f64,
    /// Predictor count `n0` (retained for API compatibility; unused in the fit).
    pub n0: f64,
}

impl Default for NormalP1 {
    fn default() -> Self {
        Self {
            t: Vec::new(),
            t0: 0.0,
            n0: 1.0,
        }
    }
}

impl NormalP1 {
    /// Create a `norm_p1` model with explicit predictor plumbing.
    ///
    /// `t0` is the **resolved prediction point** (a predictor value). `n0` is
    /// retained for API compatibility and does not affect the fit.
    pub fn new(t: Vec<f64>, t0: f64, n0: f64) -> Self {
        Self { t, t0, n0 }
    }

    /// Create a `norm_p1` model with no predictor (constant location).
    pub fn without_predictor() -> Self {
        Self::default()
    }

    /// The mean of the predictor `t` (the centering value).
    ///
    /// Returns `0.0` when no predictor is present.
    #[inline]
    pub fn mean_t(&self) -> f64 {
        if self.t.is_empty() {
            0.0
        } else {
            self.t.iter().sum::<f64>() / self.t.len() as f64
        }
    }

    /// The centered predictor value for observation `i`: `t_i - mean(t)`.
    ///
    /// Returns `0.0` when no predictor is present, so `mu_i = beta0`.
    #[inline]
    pub fn centered(&self, i: usize) -> f64 {
        if self.t.is_empty() {
            0.0
        } else {
            self.t[i] - self.mean_t()
        }
    }

    /// The centered **prediction point**: `t0 - mean(t)`.
    ///
    /// This is the predictor value at which the DMGS expansion evaluates the
    /// model's density/CDF/quantile (the reference `tresid0`). Returns `0.0`
    /// when no predictor is present.
    #[inline]
    pub fn centered_prediction(&self) -> f64 {
        if self.t.is_empty() {
            0.0
        } else {
            self.t0 - self.mean_t()
        }
    }

    /// Location `mu_i = beta0 + beta1 * centered(t_i)`.
    #[inline]
    pub fn location(&self, beta0: f64, beta1: f64, i: usize) -> f64 {
        beta0 + beta1 * self.centered(i)
    }

    /// Location at the prediction point: `muhat0 = beta0 + beta1 * centered(t0)`.
    #[inline]
    pub fn location_prediction(&self, beta0: f64, beta1: f64) -> f64 {
        beta0 + beta1 * self.centered_prediction()
    }

    /// Log-density `ln f(y_i; beta0, beta1, sigma)` at observation `i`.
    pub fn logpdf(&self, y: f64, beta0: f64, beta1: f64, sigma: f64, i: usize) -> f64 {
        let mu = self.location(beta0, beta1, i);
        let z = (y - mu) / sigma;
        -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Density `f(y_i; beta0, beta1, sigma)` at observation `i`.
    pub fn pdf(&self, y: f64, beta0: f64, beta1: f64, sigma: f64, i: usize) -> f64 {
        self.logpdf(y, beta0, beta1, sigma, i).exp()
    }

    /// CDF `F(y_i; beta0, beta1, sigma)` at observation `i`.
    pub fn cdf(&self, y: f64, beta0: f64, beta1: f64, sigma: f64, i: usize) -> f64 {
        let mu = self.location(beta0, beta1, i);
        let z = (y - mu) / (sigma * std::f64::consts::SQRT_2);
        0.5 * erfc(-z)
    }

    /// Quantile `F^{-1}(p; beta0, beta1, sigma)` at observation `i`.
    pub fn quantile(&self, p: f64, beta0: f64, beta1: f64, sigma: f64, i: usize) -> f64 {
        self.location(beta0, beta1, i) + sigma * standard_normal_quantile(p)
    }

    /// The reference `norm_p1_mlparams`: ordinary least squares of `y` on the
    /// **centered** predictor, with the residual standard deviation using the
    /// `n - 1` denominator (R's `sd(reg$residuals)`).
    ///
    /// The reference fits `lm(x ~ t)` on the centered predictor and returns
    /// `[intercept, slope, sd(residuals)]`. The `sd` uses the `n - 1`
    /// denominator, which differs from the MLE `sigma` (which uses `n`). This is
    /// the parameter vector the reference reports as `ml_params` and uses for the
    /// analytic CP quantiles/densities.
    ///
    /// Returns `[0, 0, 0]` when there is no predictor or fewer than 2
    /// observations.
    pub fn mlparams_ols(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len();
        if self.t.is_empty() || n < 2 {
            return vec![0.0, 0.0, 0.0];
        }
        let mean_t = self.mean_t();
        let mean_y = y.iter().sum::<f64>() / n as f64;
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let c = self.t[i] - mean_t;
            sxx += c * c;
            sxy += c * (yi - mean_y);
        }
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
        let intercept = mean_y - slope * 0.0; // centered predictor => intercept = mean(y)
                                              // Residual standard deviation with the n-1 denominator (R `sd`).
        let mut ss = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let c = self.t[i] - mean_t;
            let r = yi - (intercept + slope * c);
            ss += r * r;
        }
        let sigma = (ss / (n as f64 - 1.0)).sqrt();
        vec![intercept, slope, sigma]
    }
}

impl Derivatives for NormalP1 {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let beta0 = params[0];
        let beta1 = params[1];
        let sigma = params[2];
        y.iter()
            .enumerate()
            .map(|(i, &yi)| self.logpdf(yi, beta0, beta1, sigma, i))
            .sum()
    }

    fn n_params(&self) -> usize {
        3
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let beta0 = params[0];
        let beta1 = params[1];
        let sigma = params[2];
        let n = y.len() as f64;
        let s2 = sigma * sigma;
        let mut s_b0 = 0.0;
        let mut s_b1 = 0.0;
        let mut s_sig = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let c = self.centered(i);
            let r = yi - (beta0 + beta1 * c);
            s_b0 += r;
            s_b1 += r * c;
            s_sig += r * r;
        }
        vec![s_b0 / s2, s_b1 / s2, -n / sigma + s_sig / (s2 * sigma)]
    }

    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let beta0 = params[0];
        let beta1 = params[1];
        let sigma = params[2];
        let n = y.len() as f64;
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let s4 = s2 * s2;
        let mut s_c = 0.0;
        let mut s_cc = 0.0;
        let mut s_r = 0.0;
        let mut s_rc = 0.0;
        let mut s_rr = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let c = self.centered(i);
            let r = yi - (beta0 + beta1 * c);
            s_c += c;
            s_cc += c * c;
            s_r += r;
            s_rc += r * c;
            s_rr += r * r;
        }
        // Row-major 3x3: [b0, b1, sigma].
        vec![
            -n / s2,
            -s_c / s2,
            -2.0 * s_r / s3,
            -s_c / s2,
            -s_cc / s2,
            -2.0 * s_rc / s3,
            -2.0 * s_r / s3,
            -2.0 * s_rc / s3,
            n / s2 - 3.0 * s_rr / s4,
        ]
    }

    /// Analytic third derivatives (W9).
    ///
    /// Row-major `3 × 3 × 3`, index `(i, j, k)` at `i*9 + j*3 + k` with
    /// `0 = beta0`, `1 = beta1`, `2 = sigma`. With `r_i = y_i - mu_i`,
    /// `c_i = centered(i)`, `s1 = Σ r_i`, `s_c = Σ r_i c_i`, `s_cc = Σ c_i²`,
    /// `s_rr = Σ r_i²`:
    /// - `∂³ℓ/∂beta0²∂sigma = 2n/sigma³`
    /// - `∂³ℓ/∂beta0∂beta1∂sigma = 2 s_c/sigma³`
    /// - `∂³ℓ/∂beta0∂sigma² = 6 s1/sigma⁴`
    /// - `∂³ℓ/∂beta1²∂sigma = 2 s_cc/sigma³`
    /// - `∂³ℓ/∂beta1∂sigma² = 6 s_c/sigma⁴`
    /// - `∂³ℓ/∂sigma³ = -2n/sigma³ + 12 s_rr/sigma⁵`
    /// - all others zero.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let beta0 = params[0];
        let beta1 = params[1];
        let sigma = params[2];
        let n = y.len() as f64;
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let s4 = s2 * s2;
        let s5 = s4 * sigma;
        let mut s1 = 0.0;
        let mut s_c = 0.0;
        let mut s_cc = 0.0;
        let mut s_rr = 0.0;
        let mut s_c_only = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let c = self.centered(i);
            let r = yi - (beta0 + beta1 * c);
            s1 += r;
            s_c += r * c;
            s_cc += c * c;
            s_rr += r * r;
            s_c_only += c;
        }
        let d_b0b0s = 2.0 * n / s3;
        // ∂³ℓ/∂beta0∂beta1∂sigma = 2 Σ c_i / sigma³ (zero for a centered
        // predictor, but kept general).
        let d_b0b1s = 2.0 * s_c_only / s3;
        let d_b0ss = 6.0 * s1 / s4;
        let d_b1b1s = 2.0 * s_cc / s3;
        let d_b1ss = 6.0 * s_c / s4;
        let d_sss = -2.0 * n / s3 + 12.0 * s_rr / s5;
        // Build the symmetric tensor by index (i, j, k).
        let mut out = vec![0.0; 27];
        let mut set = |i: usize, j: usize, k: usize, v: f64| {
            for &(a, b, c) in &[
                (i, j, k),
                (i, k, j),
                (j, i, k),
                (j, k, i),
                (k, i, j),
                (k, j, i),
            ] {
                out[a * 9 + b * 3 + c] = v;
            }
        };
        set(0, 0, 2, d_b0b0s);
        set(0, 1, 2, d_b0b1s);
        set(0, 2, 2, d_b0ss);
        set(1, 1, 2, d_b1b1s);
        set(1, 2, 2, d_b1ss);
        set(2, 2, 2, d_sss);
        out
    }

    /// Analytic mixed derivatives `∂²ℓ/∂θ_i∂y_a` (W9).
    ///
    /// Row-major `3 × n`: `∂²ℓ/∂beta0∂y_a = 1/sigma²`,
    /// `∂²ℓ/∂beta1∂y_a = c_a/sigma²`, `∂²ℓ/∂sigma∂y_a = 2 r_a/sigma³`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let beta0 = params[0];
        let beta1 = params[1];
        let sigma = params[2];
        let ny = y.len();
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let mut out = vec![0.0; 3 * ny];
        for (a, &ya) in y.iter().enumerate() {
            let c = self.centered(a);
            let r = ya - (beta0 + beta1 * c);
            out[a] = 1.0 / s2;
            out[ny + a] = c / s2;
            out[2 * ny + a] = 2.0 * r / s3;
        }
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

    fn sample() -> (Vec<f64>, Vec<f64>) {
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = vec![0.1, 1.3, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1];
        (t, y)
    }

    #[test]
    fn without_predictor_matches_plain_normal() {
        let m = NormalP1::without_predictor();
        let n = crate::models::norm::Normal;
        let y = [0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8];
        // beta1 is unused when there is no predictor.
        for (i, &yi) in y.iter().enumerate() {
            assert_close(
                m.logpdf(yi, 0.4, 99.0, 1.3, i),
                n.logpdf(yi, 0.4, 1.3),
                "norm_p1(no t) = norm logpdf",
            );
        }
    }

    #[test]
    fn primitives_match_reference() {
        // t = [0, 2], mean(t) = 1, so centered = [-1, 1].
        let m = NormalP1::new(vec![0.0, 2.0], 1.0, 1.0);
        // At i = 0, centered = -1; with beta1 = 0, mu = beta0 = 0.
        assert_close(
            m.logpdf(0.0, 0.0, 0.0, 1.0, 0),
            -0.5 * (2.0 * std::f64::consts::PI).ln(),
            "logpdf",
        );
        assert_close(m.cdf(0.0, 0.0, 0.0, 1.0, 0), 0.5, "cdf");
        assert_close(m.quantile(0.5, 0.0, 0.0, 1.0, 0), 0.0, "quantile");
    }

    #[test]
    fn centering_uses_mean_of_t() {
        // t = [0, 2], mean = 1. centered(0) = -1, centered(1) = 1.
        let m = NormalP1::new(vec![0.0, 2.0], 1.0, 1.0);
        assert_close(m.centered(0), -1.0, "centered(0)");
        assert_close(m.centered(1), 1.0, "centered(1)");
        // Prediction point t0 = 1 => centered_prediction = 0.
        assert_close(m.centered_prediction(), 0.0, "centered_prediction");
        // t0 = 3 => centered_prediction = 2.
        let m2 = NormalP1::new(vec![0.0, 2.0], 3.0, 1.0);
        assert_close(m2.centered_prediction(), 2.0, "centered_prediction(3)");
    }

    #[test]
    fn ols_mlparams_matches_reference_convention() {
        // The reference `norm_p1_mlparams` uses lm() + sd(residuals) (n-1).
        let (t, y) = sample();
        let m = NormalP1::new(t, 4.5, 10.0);
        let p = m.mlparams_ols(&y);
        // Intercept = mean(y) (centered predictor).
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        assert_close(p[0], mean_y, "ols intercept = mean(y)");
        // sigma uses the n-1 denominator: compare to the MLE sigma * sqrt(n/(n-1)).
        let mle = fit(&m, &y, &[0.0, 1.0, 1.0]);
        let expected_sigma = mle.ml_params[2] * ((y.len() as f64) / (y.len() as f64 - 1.0)).sqrt();
        assert_close(
            p[2],
            expected_sigma,
            "ols sigma = mle sigma * sqrt(n/(n-1))",
        );
    }

    #[test]
    fn analytic_score_matches_numerical_fallback() {
        let (t, y) = sample();
        let m = NormalP1::new(t, 4.5, 10.0);
        let params = [0.2, 1.1, 1.0];
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
        let (t, y) = sample();
        let m = NormalP1::new(t, 4.5, 10.0);
        let fit = fit(&m, &y, &[0.0, 1.0, 1.0]);
        assert!(fit.convergence.is_converged());
        // The score must vanish at the optimum.
        let g = m.score(&fit.ml_params, &y);
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.abs() < 1e-6, "score[{i}] at MLE = {gi}");
        }
        // The fitted scale must be positive.
        assert!(fit.ml_params[2] > 0.0, "sigma must be positive");
    }
}
