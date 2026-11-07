//! Gumbel distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [location, scale]` with `scale > 0` (the reference `fitdistcp`
//! convention, matching R's `dgev` with `shape = 0`):
//!
//! ```text
//! f(y; loc, scale) = (1/scale) exp(-z - exp(-z)),  z = (y - loc)/scale
//! ```
//!
//! This is the GEV special case with shape `ξ = 0` (the Gumbel domain of
//! attraction).
//!
//! # Integration path
//!
//! **DMGS** (§3): the Gumbel is a GEV special case integrated through the DMGS
//! asymptotic expansion (W8); this module supplies the model primitives and
//! derivatives.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9). With `z_i = (y_i - loc)/scale` and
//! `e_i = exp(-z_i)`:
//!
//! - `∂ℓ/∂loc   = (1/scale) Σ (1 - e_i)`
//! - `∂ℓ/∂scale = (1/scale) Σ (-1 + z_i - z_i e_i)`
//!
//! The Hessian, third derivatives, and mixed derivatives are the reference's
//! `gumbel_logfdd`/`gumbel_logfddd` summed over observations; the density/CDF
//! derivative hooks are `gumbel_fd`/`gumbel_fdd`/`gumbel_pd`/`gumbel_pdd`.

use crate::models::Derivatives;

/// The Gumbel distribution model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gumbel;

impl Gumbel {
    /// Create a Gumbel model.
    pub fn new() -> Self {
        Gumbel
    }

    /// Log-density `ln f(y; loc, scale)` at a single point.
    pub fn logpdf(&self, y: f64, loc: f64, scale: f64) -> f64 {
        let z = (y - loc) / scale;
        -z - (-z).exp() - scale.ln()
    }

    /// Density `f(y; loc, scale)` at a single point.
    pub fn pdf(&self, y: f64, loc: f64, scale: f64) -> f64 {
        self.logpdf(y, loc, scale).exp()
    }

    /// CDF `F(y; loc, scale) = exp(-exp(-z))`.
    pub fn cdf(&self, y: f64, loc: f64, scale: f64) -> f64 {
        let z = (y - loc) / scale;
        (-(-z).exp()).exp()
    }

    /// Quantile `F^{-1}(p; loc, scale) = loc - scale ln(-ln p)`.
    pub fn quantile(&self, p: f64, loc: f64, scale: f64) -> f64 {
        loc - scale * (-p.ln()).ln()
    }
}

impl Derivatives for Gumbel {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let loc = params[0];
        let scale = params[1];
        // W17 (H4/H6): gated parallel reduction over observations.
        crate::models::par_sum(y, |yi| self.logpdf(yi, loc, scale))
    }

    fn n_params(&self) -> usize {
        2
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let loc = params[0];
        let scale = params[1];
        // W17 (H4/H6): single gated pass computing both sums.
        let (s_loc, s_scale) = crate::models::par_reduce2(y, |yi| {
            let z = (yi - loc) / scale;
            let e = (-z).exp();
            (1.0 - e, -1.0 + z - z * e)
        });
        vec![s_loc / scale, s_scale / scale]
    }

    /// Analytic Hessian (reference `gumbel_logfdd`, summed over observations).
    ///
    /// Row-major `2 × 2`: `[loc, scale]`.
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let loc = params[0];
        let scale = params[1];
        let s2 = scale * scale;
        // W17 (H4/H6): gated parallel reduction of the three Hessian sums.
        let (h11, h12, h22) = crate::models::par_reduce3(y, |yi| {
            let e1 = yi - loc;
            let e2 = e1 / scale;
            let e4 = (-e2).exp();
            let a11 = -e4 / s2;
            let a12 = -(((e2 - 1.0) * e4 + 1.0) / s2);
            let a22 = -((((e2 - 2.0) * e4 + 2.0) * e1 / scale - 1.0) / s2);
            (a11, a12, a22)
        });
        vec![h11, h12, h12, h22]
    }

    /// Analytic third derivatives (reference `gumbel_logfddd`, summed).
    ///
    /// Row-major `2 × 2 × 2`, index `(i, j, k)` at `i*4 + j*2 + k`.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let loc = params[0];
        let scale = params[1];
        let s3 = scale * scale * scale;
        // W17 (H4/H6): gated parallel reduction of the four third-derivative sums.
        let (d111, d112, d122, d222) = crate::models::par_reduce4(y, |yi| {
            let e1 = yi - loc;
            let e2 = e1 / scale;
            let e4 = (-e2).exp();
            let e6 = (e2 - 2.0) * e4;
            let a111 = -e4 / s3;
            let a112 = -(e6 / s3);
            let a122 = -((e6 * e1 / scale - 2.0 * ((e2 - 1.0) * e4 + 1.0)) / s3);
            let e10 = ((e2 - 4.0) * e1 / scale + 2.0) * e4 - 2.0;
            let a222 = -((e10 * e1 / scale - 2.0 * ((e6 + 2.0) * e1 / scale - 1.0)) / s3);
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
    /// Row-major `2 × n`: `∂²ℓ/∂loc∂y_a = e^{-z_a}/scale²` and
    /// `∂²ℓ/∂scale∂y_a = (1 - e^{-z_a} + z_a e^{-z_a})/scale²`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let loc = params[0];
        let scale = params[1];
        let ny = y.len();
        let s2 = scale * scale;
        let mut out = vec![0.0; 2 * ny];
        // W17 (H3/H6): gated parallel fill of the two row-major rows.
        let (row0, row1) = out.split_at_mut(ny);
        crate::models::par_fill(row0, y, |ya| {
            let z = (ya - loc) / scale;
            (-z).exp() / s2
        });
        crate::models::par_fill(row1, y, |ya| {
            let z = (ya - loc) / scale;
            let e = (-z).exp();
            (1.0 - e + z * e) / s2
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
    fn logpdf_matches_reference() {
        let m = Gumbel;
        // ln f(0; 0, 1) = -exp(0) = -1
        assert_close(m.logpdf(0.0, 0.0, 1.0), -1.0, "logpdf(0;0,1)");
    }

    #[test]
    fn cdf_matches_reference() {
        let m = Gumbel;
        // F(0; 0, 1) = exp(-1) ≈ 0.36787944117144233
        assert_close(m.cdf(0.0, 0.0, 1.0), 0.367_879_441_171_442_33, "cdf(0;0,1)");
    }

    #[test]
    fn quantile_matches_reference() {
        let m = Gumbel;
        // q(0.5; 0, 1) = -ln(ln 2) ≈ 0.36651292058166435
        assert_close(
            m.quantile(0.5, 0.0, 1.0),
            -(2.0_f64.ln()).ln(),
            "q(0.5;0,1)",
        );
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = Gumbel;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 1.5, 2.0);
            assert_close(m.cdf(x, 1.5, 2.0), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn analytic_score_matches_numerical_fallback() {
        let m = Gumbel;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let params = [0.8, 1.1];
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
        let m = Gumbel;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let fit = fit(&m, &y, &[0.5, 1.0]);
        assert!(fit.convergence.is_converged());
        // The score must vanish at the optimum.
        let g = m.score(&fit.ml_params, &y);
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.abs() < 1e-6, "score[{i}] at MLE = {gi}");
        }
    }
}
