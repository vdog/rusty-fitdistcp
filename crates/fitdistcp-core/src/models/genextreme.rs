//! Generalized extreme value (GEV) distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [location, scale, shape]` with `scale > 0` (the reference
//! `fitdistcp` convention, matching R's `evd::dgev` / `dgenextreme`):
//!
//! ```text
//! f(y; loc, scale, shape) = (1/scale) t^(-1/shape - 1) exp(-t^(-1/shape)),
//!     t = 1 + shape (y - loc)/scale
//! ```
//!
//! with support `1 + shape (y - loc)/scale > 0`. The `shape -> 0` limit is the
//! Gumbel distribution (see [`crate::models::gumbel`]).
//!
//! # Integration path
//!
//! **DMGS** (§3): the GEV is integrated through the DMGS asymptotic expansion
//! (W8). The **negative-shape** case, where DMGS cannot return values beyond the
//! MLE bound and the package derives the density from quantiles, is implemented
//! in W10:
//!
//! - [`GenExtreme::revert2ml`] reproduces the reference `qgev_cp` guard
//!   `abs(v3hat) >= 1`: when the fitted shape magnitude reaches 1 the CP output
//!   reverts to the MLE output.
//! - [`GenExtreme::ml_max`] returns the finite upper support bound
//!   `loc - scale/shape` for a negative shape (the reference `ml_max`).
//! - [`movexiawayfromzero`] reproduces the reference helper that nudges a shape
//!   near zero to `±1e-7` before evaluating the analytic derivative forms.
//! - The quantile-derived CP density itself lives in
//!   [`crate::dmgs::DmgsIntegrator::cp_pdf_quantile_derived`] (the reference
//!   `rh_flat_pdf = -(Δα)/(Δq)`), and the GEV-specific CP entry point is
//!   [`crate::api::predict_gev`].
//!
//! This module provides the standard density/CDF/quantile primitives, the
//! derivatives, and the negative-shape helpers.
//!
//! # Derivatives
//!
//! The score and Hessian are **analytic** (W9), transcribed from the reference
//! `gev_logfd`/`gev_logfdd`. The **third derivatives and mixed derivatives keep
//! the numerical fallback** from W5: the reference's `gev_logfddd` is a
//! ~180-line machine-generated expression that is impractical to transcribe
//! reliably by hand, and the GEV is not part of the DMGS golden comparison
//! (`cp_golden` covers gumbel/weibull/gamma). The numerical path is validated in
//! the tests against the analytic score/Hessian and against the score vanishing
//! at the MLE.
//!
//! The analytic forms are valid for `shape != 0`; at `shape ≈ 0` the model uses
//! the Gumbel limit for the density, and the analytic score/Hessian are
//! continuous in the limit.

use crate::models::Derivatives;

/// The minimum `|shape|` used by [`movexiawayfromzero`] (reference `minxi`).
///
/// The reference `movexiawayfromzero` (`005_utils.R`) nudges a shape parameter
/// that is too close to zero away from the `1/shape` singularity in the
/// analytic derivative helpers. The threshold is `10^-7`.
pub const MIN_XI: f64 = 1e-7;

/// Move a shape parameter away from zero (reference `movexiawayfromzero`).
///
/// If `|xi| < MIN_XI` the value is replaced by `±MIN_XI` with the **same sign**
/// (a non-negative `xi` becomes `+MIN_XI`, a negative one `-MIN_XI`). This is a
/// direct transcription of the reference helper in `005_utils.R` and is applied
/// before evaluating the analytic GEV/GPD derivative forms, which are singular
/// at `shape = 0`.
///
/// The reference pairs this with `fixgevrange`, which clamps the data to the
/// (slightly shifted) support bound so that nudging `xi` does not push an
/// observation outside the support. The Rust density/CDF already return the
/// correct boundary values, so only the derivative helpers need the nudge.
#[inline]
pub fn movexiawayfromzero(xi: f64) -> f64 {
    if xi.abs() < MIN_XI {
        if xi >= 0.0 {
            MIN_XI
        } else {
            -MIN_XI
        }
    } else {
        xi
    }
}

/// The generalized extreme value (GEV) distribution model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenExtreme;

impl GenExtreme {
    /// Create a GEV model.
    pub fn new() -> Self {
        GenExtreme
    }

    /// Log-density `ln f(y; loc, scale, shape)` at a single point.
    ///
    /// Returns `-inf` outside the support.
    pub fn logpdf(&self, y: f64, loc: f64, scale: f64, shape: f64) -> f64 {
        let z = (y - loc) / scale;
        if shape.abs() < 1e-12 {
            // Gumbel limit.
            return -z - (-z).exp() - scale.ln();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -scale.ln() - (1.0 / shape + 1.0) * t.ln() - t.powf(-1.0 / shape)
    }

    /// Density `f(y; loc, scale, shape)` at a single point.
    pub fn pdf(&self, y: f64, loc: f64, scale: f64, shape: f64) -> f64 {
        self.logpdf(y, loc, scale, shape).exp()
    }

    /// CDF `F(y; loc, scale, shape) = exp(-t^(-1/shape))`.
    pub fn cdf(&self, y: f64, loc: f64, scale: f64, shape: f64) -> f64 {
        let z = (y - loc) / scale;
        if shape.abs() < 1e-12 {
            return (-(-z).exp()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            // Below the lower support bound (shape > 0) or above the upper
            // bound (shape < 0).
            return if shape > 0.0 { 0.0 } else { 1.0 };
        }
        (-t.powf(-1.0 / shape)).exp()
    }

    /// Quantile `F^{-1}(p; loc, scale, shape)`.
    pub fn quantile(&self, p: f64, loc: f64, scale: f64, shape: f64) -> f64 {
        if shape.abs() < 1e-12 {
            return loc - scale * (-p.ln()).ln();
        }
        loc + (scale / shape) * ((-p.ln()).powf(-shape) - 1.0)
    }

    /// The upper support bound `ml_max` (reference `qgev_cp`).
    ///
    /// For a **negative** shape the GEV has a finite upper endpoint
    /// `loc - scale/shape`; for `shape >= 0` the support is unbounded above and
    /// this returns `+inf`. The reference reports this as `ml_max` and uses it to
    /// detect predictions that fall beyond the MLE bound.
    pub fn ml_max(&self, loc: f64, scale: f64, shape: f64) -> f64 {
        if shape < 0.0 {
            loc - scale / shape
        } else {
            f64::INFINITY
        }
    }

    /// Whether the DMGS expansion must revert to the MLE (reference `revert2ml`).
    ///
    /// The reference `qgev_cp` sets `revert2ml = abs(v3hat) >= 1`: when the
    /// fitted shape magnitude reaches 1 the DMGS expansion is numerically
    /// unreliable (the `1/shape` terms blow up and the CP quantiles can leave the
    /// support), so the CP output is set equal to the MLE output. This is the
    /// reference's guard for the **negative-shape** case (and the symmetric
    /// positive-shape case).
    pub fn revert2ml(&self, shape: f64) -> bool {
        shape.abs() >= 1.0
    }
}

impl Derivatives for GenExtreme {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let loc = params[0];
        let scale = params[1];
        let shape = params[2];
        // W17 (H4/H6): gated parallel reduction over observations.
        crate::models::par_sum(y, |yi| self.logpdf(yi, loc, scale, shape))
    }

    fn n_params(&self) -> usize {
        3
    }

    /// Analytic score (reference `gev_logfd`, summed over observations).
    ///
    /// Returns `[∂ℓ/∂loc, ∂ℓ/∂scale, ∂ℓ/∂shape]`.
    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = params[2];
        if v3.abs() < 1e-12 {
            // At shape ≈ 0 the analytic form is singular (1/shape). Fall back to
            // the numerical derivative, which is well-defined in the Gumbel
            // limit and keeps the shape component non-degenerate.
            let cfg = self.fd_config();
            return crate::models::numerical_score(&|p: &[f64]| self.loglik(p, y), params, &cfg);
        }
        // W17 (H4/H6): gated parallel reduction of the three per-observation sums.
        let (s1, s2, s3) = crate::models::par_reduce3(y, |x| {
            let e1 = x - v1;
            let e3 = v3 * e1 / v2;
            let e4 = 1.0 + e3;
            let e5 = 1.0 / v3;
            let e6 = 1.0 + e5;
            let e8 = 1.0 / e4.powf(e5);
            let e9 = v2 * e4;
            let e11 = v3 * e6 - e8;
            let a1 = e11 / e9;
            let a2 = (e11 * e1 / e9 - 1.0) / v2;
            let a3 =
                -(((e8 - 1.0) * e3.ln_1p() / v3 - e1 / (v2 * e4.powf(e6))) / v3 + e6 * e1 / e9);
            (a1, a2, a3)
        });
        vec![s1, s2, s3]
    }

    /// Analytic Hessian (reference `gev_logfdd`, summed over observations).
    ///
    /// Row-major `3 × 3`: `[loc, scale, shape]`.
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = params[2];
        if v3.abs() < 1e-12 {
            // At shape ≈ 0 the analytic form is singular (1/shape). Fall back to
            // the numerical Hessian, which is well-defined in the Gumbel limit
            // and keeps the shape row/column non-degenerate (so the observed
            // information stays invertible).
            let cfg = self.fd_config();
            return crate::models::numerical_hessian(&|p: &[f64]| self.loglik(p, y), params, &cfg);
        }
        // W17 (H4/H6): gated parallel reduction of the 9-entry per-observation
        // sum. The per-observation contribution is computed by a closure; the
        // reduction is sequential for small `n` (see `PARALLEL_MIN_N`).
        let contrib = |x: f64| -> [f64; 9] {
            let e1 = x - v1;
            let e2 = v3 * e1;
            let e3 = e2 / v2;
            let e4 = 1.0 + e3;
            let e5 = 1.0 / v3;
            let e6 = e4.powf(e5);
            let e7 = 1.0 + e5;
            let e8 = v2 * e4;
            let e9 = 1.0 / e6;
            let e10 = v3 * e7;
            let e11 = e8 * e8;
            let e12 = e3.ln_1p();
            let e13 = e10 - e9;
            let e14 = e4.powf(e7);
            let e15 = v2 * e14;
            let e19 = e4.powf(e5 - 1.0) * e1 / v2 - e6 * e12 / v3;
            let e20 = e4.powf(e5 + 2.0);
            let e21 = e15 * e15;
            let e22 = e13 * e1;
            let e23 = e9 - 1.0;
            let e24 = 2.0 / v3;
            let e26 = (e19 / (v3 * e4.powf(e24)) + 1.0) / e8 - e22 / e11;
            let e28 = e13 / e11 + e1 / (v2 * v2 * v2 * e20);
            let e30 = e12 / e6 - v3 * e23;
            let e31 = v2 * v2;
            let e33 = e10 * e6 * e1;
            let e34 = v3 * e13;
            let e35 = v3 * v3;
            let mut h = [0.0_f64; 9];
            h[0] = e34 / e11 - 1.0 / (e31 * e20);
            h[1] = (e34 * e1 / e11 - (e1 / e15 + e10 - e9) / e8) / v2;
            h[2] = -(((e30 / v3 + e9) / e8 - e33 / e21) / v3 + e7 * (e2 / e11 - 1.0 / e8));
            h[4] = -((e22 / e8 - 1.0) / v2 + e28 * e1) / v2;
            h[5] = -((((e14 - e33 / v2) / e21 + e30 / (e31 * v3 * e4)) / v3 - e7 / e11) * e1);
            h[8] = -(((((2.0 / e6 - 1.0) * e1 / v2 - e19 * e12 / (v3 * e4.powf(e24 - 1.0))) / e4
                - 2.0 * (e23 * e12 / v3))
                / v3
                + v2 * (e7 * e6 * e1 / v2 - e14 * e12 / e35) * e1 / e21)
                / v3
                - (e7 * e1 / e11 + 1.0 / (v2 * e35 * e4)) * e1);
            // Symmetric off-diagonal entries.
            h[3] = -e28;
            h[6] = e26;
            h[7] = e26 * e1 / v2;
            h
        };
        // W17: sequential (see `PARALLEL_MIN_N` for why Rayon is not used for
        // these low-arithmetic-intensity reductions).
        let h = y
            .iter()
            .map(|&x| contrib(x))
            .fold([0.0_f64; 9], |mut a, b| {
                for k in 0..9 {
                    a[k] += b[k];
                }
                a
            });
        vec![h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]]
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
    fn shape_zero_matches_gumbel() {
        let m = GenExtreme;
        let g = crate::models::gumbel::Gumbel;
        for &y in &[-0.5_f64, 0.0, 1.0, 2.5] {
            assert_close(
                m.logpdf(y, 0.5, 1.2, 0.0),
                g.logpdf(y, 0.5, 1.2),
                "gev(shape=0)=gumbel logpdf",
            );
            assert_close(
                m.cdf(y, 0.5, 1.2, 0.0),
                g.cdf(y, 0.5, 1.2),
                "gev(shape=0)=gumbel cdf",
            );
        }
    }

    #[test]
    fn cdf_matches_reference() {
        let m = GenExtreme;
        // F(0; 0, 1, 1) = exp(-1) ≈ 0.36787944117144233
        assert_close(
            m.cdf(0.0, 0.0, 1.0, 1.0),
            0.367_879_441_171_442_33,
            "cdf(0;0,1,1)",
        );
    }

    #[test]
    fn quantile_matches_reference() {
        let m = GenExtreme;
        // q(0.5; 0, 1, 0) = -ln(ln 2)
        assert_close(
            m.quantile(0.5, 0.0, 1.0, 0.0),
            -(2.0_f64.ln()).ln(),
            "q(0.5;0,1,0)",
        );
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = GenExtreme;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 0.5, 1.2, 0.2);
            assert_close(m.cdf(x, 0.5, 1.2, 0.2), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn fits_synthetic_data() {
        let m = GenExtreme;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let fit = fit(&m, &y, &[0.5, 1.0, 0.1]);
        assert!(fit.convergence.is_converged());
        // The score must vanish at the optimum.
        let g = m.score(&fit.ml_params, &y);
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.abs() < 1e-6, "score[{i}] at MLE = {gi}");
        }
    }

    // --- W10: negative-shape handling --------------------------------------

    #[test]
    fn movexiawayfromzero_matches_reference() {
        // Reference `movexiawayfromzero` (005_utils.R): |xi| < 1e-7 is nudged to
        // ±1e-7 with the same sign.
        assert_eq!(movexiawayfromzero(0.0), MIN_XI);
        assert_eq!(movexiawayfromzero(1e-9), MIN_XI);
        assert_eq!(movexiawayfromzero(-1e-9), -MIN_XI);
        // Values at or beyond the threshold are unchanged.
        assert_eq!(movexiawayfromzero(0.2), 0.2);
        assert_eq!(movexiawayfromzero(-0.3), -0.3);
        assert_eq!(movexiawayfromzero(MIN_XI), MIN_XI);
        assert_eq!(movexiawayfromzero(-MIN_XI), -MIN_XI);
    }

    #[test]
    fn revert2ml_matches_reference_threshold() {
        // Reference `qgev_cp`: revert2ml = abs(v3hat) >= 1.
        let m = GenExtreme;
        assert!(!m.revert2ml(0.0));
        assert!(!m.revert2ml(0.5));
        assert!(!m.revert2ml(-0.99));
        assert!(m.revert2ml(1.0));
        assert!(m.revert2ml(-1.0));
        assert!(m.revert2ml(1.5));
        assert!(m.revert2ml(-2.0));
    }

    #[test]
    fn ml_max_is_upper_endpoint_for_negative_shape() {
        // Reference `qgev_cp`: ml_max = loc - scale/shape for shape < 0, else Inf.
        let m = GenExtreme;
        // loc=1.4, scale=0.28, shape=-0.3 => 1.4 - 0.28/(-0.3) = 1.4 + 0.9333...
        let expected = 1.4 - 0.28 / (-0.3);
        assert_close(m.ml_max(1.4, 0.28, -0.3), expected, "ml_max(neg)");
        assert!(m.ml_max(1.4, 0.28, 0.0).is_infinite());
        assert!(m.ml_max(1.4, 0.28, 0.3).is_infinite());
    }

    #[test]
    fn quantile_derived_density_approaches_direct_density_for_large_n() {
        // The quantile-derived density (the reference `rh_flat_pdf`
        // construction) is the reciprocal slope of the CP quantile function. As
        // n grows the DMGS correction is O(1/n), so the CP quantile function
        // tends to the ML quantile function and the quantile-derived density
        // must tend to the direct ML density. This validates the construction
        // against the direct density formula in the regime where they coincide.
        use crate::dmgs::DmgsIntegrator;
        let m = GenExtreme;
        // A large, well-behaved sample (n = 2000).
        let y: Vec<f64> = (0..2000)
            .map(|i| {
                let t = i as f64 / 2000.0;
                // Deterministic pseudo-sample spanning the GEV support.
                0.9 + 0.6 * (t * 8.0 - 4.0) * 0.25
            })
            .collect();
        let params = [0.9, 0.6, 0.15];
        let integ = DmgsIntegrator::new(m);
        let p = [0.25, 0.5, 0.75];
        let alpha: Vec<f64> = p.iter().map(|&pi| 1.0 - pi).collect();
        let qd = integ.cp_pdf_quantile_derived(&params, &y, &alpha, 0.01);
        for (i, &pi) in p.iter().enumerate() {
            let q = m.quantile(pi, params[0], params[1], params[2]);
            let direct = m.pdf(q, params[0], params[1], params[2]);
            // The CP correction is O(1/n) ~ 5e-4; allow a generous relative bound.
            let tol = 1e-2 + 1e-2 * direct.abs();
            assert!(
                (qd[i] - direct).abs() <= tol,
                "quantile-derived density[{i}] = {} vs direct {} (tol {tol})",
                qd[i],
                direct
            );
        }
    }
}
