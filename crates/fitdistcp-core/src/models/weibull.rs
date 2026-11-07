//! Weibull distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [shape, scale]` with `shape > 0`, `scale > 0` (the reference
//! `fitdistcp` convention, matching R's `dweibull`):
//!
//! ```text
//! f(y; shape, scale) = (shape/scale) (y/scale)^(shape-1)
//!                      exp(-(y/scale)^shape),   y >= 0
//! ```
//!
//! # Integration path
//!
//! **DMGS** (§3): the Weibull is integrated through the DMGS asymptotic
//! expansion (W8); this module supplies the model primitives and derivatives.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9). With `z_i = (y_i/scale)^shape`,
//! `L = Σ ln(y_i/scale)`, and `n` observations:
//!
//! - `∂ℓ/∂shape = n/shape + L - Σ z_i ln(y_i/scale)`
//! - `∂ℓ/∂scale = -n shape/scale + (shape/scale) Σ z_i`
//!
//! The Hessian and third derivatives are the reference's `weibull_logfdd` /
//! `weibull_logfddd` summed over observations; the density/CDF derivative hooks
//! are `weibull_fd`/`weibull_fdd`/`weibull_pd`/`weibull_pdd`.

use crate::models::Derivatives;

/// The Weibull distribution model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Weibull;

impl Weibull {
    /// Create a Weibull model.
    pub fn new() -> Self {
        Weibull
    }

    /// Log-density `ln f(y; shape, scale)` at a single point.
    pub fn logpdf(&self, y: f64, shape: f64, scale: f64) -> f64 {
        let z = y / scale;
        shape.ln() - scale.ln() + (shape - 1.0) * z.ln() - z.powf(shape)
    }

    /// Density `f(y; shape, scale)` at a single point.
    pub fn pdf(&self, y: f64, shape: f64, scale: f64) -> f64 {
        self.logpdf(y, shape, scale).exp()
    }

    /// CDF `F(y; shape, scale) = 1 - exp(-(y/scale)^shape)`.
    pub fn cdf(&self, y: f64, shape: f64, scale: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        -(-(y / scale).powf(shape)).exp_m1()
    }

    /// Quantile `F^{-1}(p; shape, scale) = scale (-ln(1 - p))^(1/shape)`.
    pub fn quantile(&self, p: f64, shape: f64, scale: f64) -> f64 {
        scale * (-(1.0 - p).ln()).powf(1.0 / shape)
    }
}

impl Derivatives for Weibull {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let shape = params[0];
        let scale = params[1];
        // W17 (H4/H6): gated parallel reduction over observations.
        crate::models::par_sum(y, |yi| self.logpdf(yi, shape, scale))
    }

    fn n_params(&self) -> usize {
        2
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let scale = params[1];
        let n = y.len() as f64;
        // W17 (H4/H6): single gated pass computing the three sums.
        let (l, sum_z, sum_z_ln) = crate::models::par_reduce3(y, |yi| {
            let ln_z = (yi / scale).ln();
            let z = (yi / scale).powf(shape);
            (ln_z, z, z * ln_z)
        });
        vec![
            n / shape + l - sum_z_ln,
            -n * shape / scale + (shape / scale) * sum_z,
        ]
    }

    /// Analytic Hessian (reference `weibull_logfdd`, summed over observations).
    ///
    /// Row-major `2 × 2`: `[shape, scale]`.
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        // W17 (H4/H6): gated parallel reduction of the three Hessian sums.
        let (h11, h12, h22) = crate::models::par_reduce3(y, |x| {
            let e1 = x / v2;
            let e2 = v1 - 1.0;
            let e3 = e1.powf(e2);
            let e6 = x.ln() - v2.ln();
            let e7 = e1.powf(v1);
            let a11 = -(e6 * e6 * e7 + 1.0 / (v1 * v1));
            let a12 = (x * (e3 + v1 * e6 * e3) / v2 - 1.0) / v2;
            let a22 =
                -(v1 * (x * (2.0 * e3 + x * e2 * e1.powf(v1 - 2.0) / v2) / v2 - 1.0) / (v2 * v2));
            (a11, a12, a22)
        });
        vec![h11, h12, h12, h22]
    }

    /// Analytic third derivatives (reference `weibull_logfddd`, summed).
    ///
    /// Row-major `2 × 2 × 2`, index `(i, j, k)` at `i*4 + j*2 + k`.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        // W17 (H4/H6): gated parallel reduction of the four third-derivative sums.
        let (d111, d112, d122, d222) = crate::models::par_reduce4(y, |x| {
            let e1 = x / v2;
            let e2 = v1 - 1.0;
            let e3 = e1.powf(e2);
            let e6 = x.ln() - v2.ln();
            let e7 = v1 - 2.0;
            let e8 = e1.powf(e7);
            let e9 = 2.0 * e3;
            let e12 = x * e2 * e8 / v2;
            let e13 = e1.powf(v1);
            let e14 = v2 * v2;
            let e15 = e9 + e12;
            let e19 = v1 * e6 * e3;
            let a111 = -(e6 * e6 * e6 * e13 - 2.0 / (v1 * v1 * v1));
            let a112 = x * (e9 + e19) * e6 / e14;
            let a122 = -((x * (e9 + v1 * (2.0 * (e6 * e3) + x * (e6 * e2 * e8 + e8) / v2) + e12)
                / v2
                - 1.0)
                / e14);
            let a222 = v1
                * (2.0 * (x * e15 / v2 - 1.0)
                    + x * (e9 + x * (4.0 * e8 + x * e7 * e1.powf(v1 - 3.0) / v2) * e2 / v2) / v2)
                / (v2 * v2 * v2);
            (a111, a112, a122, a222)
        });
        vec![
            d111, d112, // (0,0,0), (0,0,1)
            d112, d122, // (0,1,0), (0,1,1)
            d112, d122, // (1,0,0), (1,0,1)
            d122, d222, // (1,1,0), (1,1,1)
        ]
    }

    /// Analytic mixed derivatives `∂²ℓ/∂θ_i∂y_a` (W9).
    ///
    /// Row-major `2 × n`. With `z_a = y_a/scale`, `w_a = z_a^shape`,
    /// `L_a = ln(y_a/scale)`:
    /// `∂²ℓ/∂shape∂y_a = (1 - w_a)/y_a - shape w_a L_a / y_a` and
    /// `∂²ℓ/∂scale∂y_a = shape² w_a/(scale y_a)`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let scale = params[1];
        let ny = y.len();
        let mut out = vec![0.0; 2 * ny];
        // W17 (H3/H6): gated parallel fill of the two row-major rows.
        let (row0, row1) = out.split_at_mut(ny);
        crate::models::par_fill(row0, y, |ya| {
            let z = ya / scale;
            let w = z.powf(shape);
            let l = (ya / scale).ln();
            (1.0 - w) / ya - shape * w * l / ya
        });
        crate::models::par_fill(row1, y, |ya| {
            let z = ya / scale;
            let w = z.powf(shape);
            shape * shape * w / (scale * ya)
        });
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
    fn shape_one_is_exponential() {
        // Weibull(shape = 1, scale = 1/rate) == Exponential(rate).
        let m = Weibull;
        let rate = 2.0;
        let scale = 1.0 / rate;
        for &y in &[0.3_f64, 1.0, 2.5] {
            assert_close(
                m.logpdf(y, 1.0, scale),
                rate.ln() - rate * y,
                "weibull=exp logpdf",
            );
            assert_close(
                m.cdf(y, 1.0, scale),
                -(-rate * y).exp_m1(),
                "weibull=exp cdf",
            );
        }
    }

    #[test]
    fn cdf_matches_reference() {
        let m = Weibull;
        // F(1; 1, 1) = 1 - e^{-1}
        assert_close(m.cdf(1.0, 1.0, 1.0), 0.632_120_558_828_557_7, "cdf(1;1,1)");
        // F(1; 2, 1) = 1 - e^{-1}
        assert_close(m.cdf(1.0, 2.0, 1.0), 0.632_120_558_828_557_7, "cdf(1;2,1)");
    }

    #[test]
    fn quantile_matches_reference() {
        let m = Weibull;
        // q(0.5; 1, 1) = ln 2
        assert_close(m.quantile(0.5, 1.0, 1.0), 2.0_f64.ln(), "q(0.5;1,1)");
        // q(0.5; 2, 1) = sqrt(ln 2)
        assert_close(m.quantile(0.5, 2.0, 1.0), 2.0_f64.ln().sqrt(), "q(0.5;2,1)");
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = Weibull;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 1.7, 2.3);
            assert_close(m.cdf(x, 1.7, 2.3), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn analytic_score_matches_numerical_fallback() {
        let m = Weibull;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let params = [1.4, 1.1];
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
        let m = Weibull;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let fit = fit(&m, &y, &[1.0, 1.0]);
        assert!(fit.convergence.is_converged());
        // The score must vanish at the optimum.
        let g = m.score(&fit.ml_params, &y);
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.abs() < 1e-6, "score[{i}] at MLE = {gi}");
        }
    }
}
