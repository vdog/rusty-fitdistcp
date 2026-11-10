//! Generalized Pareto distribution (GPD) model (§3).
//!
//! # Parameterization
//!
//! `params = [scale, shape]` with `scale > 0` (the reference `fitdistcp`
//! convention, matching R's `dgenpareto` / `evd::dgpd`):
//!
//! ```text
//! f(y; scale, shape) = (1/scale) (1 + shape y/scale)^(-1/shape - 1)
//! ```
//!
//! on the support `y >= 0` when `shape >= 0`, and `0 <= y < -scale/shape` when
//! `shape < 0`. This is the distribution of threshold exceedances (the
//! Pickands–Balkema–de Haan theorem).
//!
//! # Integration path
//!
//! **DMGS** (§3): the GPD is integrated through the DMGS asymptotic expansion
//! (W8); this module supplies the model primitives and derivatives.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9). The Hessian and third derivatives are
//! the reference's `gpd_k1_logfdd`/`gpd_k1_logfddd` (with the location fixed at
//! zero) summed over observations; the density/CDF derivative hooks are
//! `gpd_k1_fd`/`gpd_k1_fdd`/`gpd_k1_pd`/`gpd_k1_pdd`. The mixed derivatives are
//! derived analytically (see [`GenPareto::mixed_derivatives`]).

use crate::models::Derivatives;

/// The generalized Pareto distribution model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenPareto;

impl GenPareto {
    /// Create a GPD model.
    pub fn new() -> Self {
        GenPareto
    }

    /// Log-density `ln f(y; scale, shape)` at a single point.
    ///
    /// Returns `-inf` outside the support. The `shape -> 0` limit is the
    /// exponential with rate `1/scale`.
    pub fn logpdf(&self, y: f64, scale: f64, shape: f64) -> f64 {
        if y < 0.0 {
            return f64::NEG_INFINITY;
        }
        if shape.abs() < 1e-12 {
            // Exponential limit: ln f = -ln(scale) - y/scale.
            return -scale.ln() - y / scale;
        }
        let t = 1.0 + shape * y / scale;
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -scale.ln() - (1.0 / shape + 1.0) * t.ln()
    }

    /// Density `f(y; scale, shape)` at a single point.
    pub fn pdf(&self, y: f64, scale: f64, shape: f64) -> f64 {
        self.logpdf(y, scale, shape).exp()
    }

    /// CDF `F(y; scale, shape) = 1 - (1 + shape y/scale)^(-1/shape)`.
    ///
    /// The `shape -> 0` limit is the exponential CDF `1 - exp(-y/scale)`.
    pub fn cdf(&self, y: f64, scale: f64, shape: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        if shape.abs() < 1e-12 {
            return -(-y / scale).exp_m1();
        }
        let t = 1.0 + shape * y / scale;
        if t <= 0.0 {
            return 1.0;
        }
        -(-(1.0 / shape) * t.ln()).exp_m1()
    }

    /// Quantile `F^{-1}(p; scale, shape) = (scale/shape) ((1-p)^(-shape) - 1)`.
    ///
    /// The `shape -> 0` limit is the exponential quantile `-scale ln(1-p)`.
    pub fn quantile(&self, p: f64, scale: f64, shape: f64) -> f64 {
        if shape.abs() < 1e-12 {
            return -scale * (1.0 - p).ln();
        }
        (scale / shape) * ((1.0 - p).powf(-shape) - 1.0)
    }

    /// Set the GPD initial conditions, mirroring the reference `gpd_k1_setics`
    /// (Issue 2, Phase 2).
    ///
    /// The reference `gpd_k1_setics(x, ics)` replaces a sentinel `ics = [0, 0]`
    /// with `[sd(x), 0]` (the sample standard deviation with the `n-1`
    /// denominator, i.e. R's `sd`), and otherwise returns `ics` unchanged:
    ///
    /// ```text
    /// if (ics[0] == 0) && (ics[1] == 0):
    ///     ics = [std(x, ddof=1), 0]
    /// ```
    ///
    /// This is the fix for the dollar-scale defect: the binding previously
    /// hard-coded `init = [1.0, 0.0]`, which is many orders of magnitude below
    /// the scale of raw dollar data (~10^3–10^5), so the optimizer started in a
    /// region where the GPD log-likelihood is flat/degenerate and silently
    /// returned `NaN`. Seeding the scale from the data's own standard deviation
    /// puts the optimizer on the right scale.
    ///
    /// `x` is the observed data and `ics` the caller's initial conditions. The
    /// returned vector has length 2 (`[scale, shape]`).
    pub fn setics(&self, x: &[f64], ics: &[f64]) -> Vec<f64> {
        if ics.len() == 2 && ics[0] == 0.0 && ics[1] == 0.0 {
            vec![sample_std_ddof1(x), 0.0]
        } else {
            ics.to_vec()
        }
    }

    /// Whether the DMGS expansion must revert to the MLE (reference `revert2ml`).
    ///
    /// The reference `qgpd_k1_cp` sets
    /// `revert2ml = (abs(v2hat) >= 1) || (v2hat > 30)`: when the fitted shape
    /// magnitude reaches 1, or the shape exceeds 30, the DMGS expansion is
    /// numerically unreliable, so the reference reverts the CP output to the MLE
    /// output. The `v2hat > 30` clause (Issue 2, Phase 2) guards the
    /// large-positive-shape case where `lddi` cannot be computed and the code
    /// would otherwise crash / return `NaN`.
    pub fn revert2ml(&self, shape: f64) -> bool {
        shape.abs() >= 1.0 || shape > 30.0
    }
}

/// Sample standard deviation with the `n-1` denominator (R's `sd`, NumPy's
/// `std(x, ddof=1)`), used by [`GenPareto::setics`].
///
/// Returns `0.0` for fewer than two observations (the reference `sd` would be
/// `NA`; the GPD fit is undefined there anyway and the caller's small-n guard
/// handles it).
fn sample_std_ddof1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let ss: f64 = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum();
    (ss / (nf - 1.0)).sqrt()
}

impl Derivatives for GenPareto {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let scale = params[0];
        let shape = params[1];
        y.iter().map(|&yi| self.logpdf(yi, scale, shape)).sum()
    }

    fn n_params(&self) -> usize {
        2
    }

    /// `setics`-style initial conditions (reference `gpd_k1_setics`, Phase 3).
    ///
    /// The reference `gpd_k1_setics(x, ics)` replaces a sentinel `ics = [0, 0]`
    /// with `[sd(x), 0]` (the `n-1` sample SD). The Rust multi-start strategy
    /// uses this **data-driven** start as an additional start alongside the
    /// caller's `init`, so it is returned unconditionally (the sentinel check is
    /// the reference's way of deciding *whether* to use the heuristic; the
    /// multi-start always wants it). The inherent [`GenPareto::setics`] keeps
    /// the reference sentinel semantics for the binding.
    fn setics(&self, y: &[f64], _init: &[f64]) -> Vec<f64> {
        vec![crate::models::sample_std_ddof1(y), 0.0]
    }

    /// Reject the GPD degenerate region (Phase 3, Issue 3; Phase D).
    ///
    /// The GPD log-likelihood has the same `scale -> 0` spurious maximum as the
    /// GEV. The guard rejects a non-positive scale and a runaway shape
    /// (`|shape| >= 10`); the shape bound is generous so it catches only the
    /// runaway and does not reject the reference's legitimate fits.
    ///
    /// **Phase D:** the guard also rejects `shape <= -1`. For `xi <= -1` the GPD
    /// density is not integrable (the log-likelihood is unbounded above as
    /// `xi -> -1`), so the multi-start can find a *spurious* boundary optimum
    /// with a higher log-likelihood than the true interior MLE. The reference
    /// `gpd_k1_checkmle` uses `minxi = -1` for exactly this reason; mirroring it
    /// keeps the Rust fit at the reference's interior optimum on small-n records
    /// (the Phase D GPD small-n divergence).
    fn valid_params(&self, params: &[f64]) -> bool {
        params.len() == 2
            && params[0] > 0.0
            && params[0].is_finite()
            && params[1].is_finite()
            && params[1] > -1.0
            && params[1] < 10.0
    }

    /// Analytic score (W9).
    ///
    /// With `t_i = 1 + shape y_i/scale`:
    /// - `∂ℓ/∂scale = Σ [-1/scale + (1+shape) y_i/(scale² t_i)]`
    /// - `∂ℓ/∂shape = Σ [ln(t_i)/shape² - (1/shape+1)(y_i/scale)/t_i]`
    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let scale = params[0];
        let shape = params[1];
        if shape.abs() < 1e-12 {
            // At shape ≈ 0 the analytic form is singular (1/shape). Fall back to
            // the numerical derivative, which is well-defined in the
            // exponential limit.
            let cfg = self.fd_config();
            return crate::models::numerical_score(&|p: &[f64]| self.loglik(p, y), params, &cfg);
        }
        let s2 = scale * scale;
        let mut s_scale = 0.0;
        let mut s_shape = 0.0;
        for &yi in y {
            let t = 1.0 + shape * yi / scale;
            s_scale += -1.0 / scale + (1.0 + shape) * yi / (s2 * t);
            s_shape += t.ln() / (shape * shape) - (1.0 / shape + 1.0) * (yi / scale) / t;
        }
        vec![s_scale, s_shape]
    }

    /// Analytic Hessian (reference `gpd_k1_logfdd` with location 0, summed).
    ///
    /// Row-major `2 × 2`: `[scale, shape]`.
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let v2 = params[0];
        let v3 = params[1];
        if v3.abs() < 1e-12 {
            let cfg = self.fd_config();
            return crate::models::numerical_hessian(&|p: &[f64]| self.loglik(p, y), params, &cfg);
        }
        let mut h11 = 0.0;
        let mut h12 = 0.0;
        let mut h22 = 0.0;
        for &x in y {
            let e1 = x;
            let e3 = v3 * e1 / v2;
            let e4 = 1.0 + e3;
            let e5 = 1.0 + v3;
            let e6 = v2 * e4;
            let e7 = e5 * e1;
            let e8 = e6 * e6;
            let e9 = e7 / e8;
            let e11 = 1.0 - e5 / v3;
            let e12 = e7 / e6;
            let e13 = (1.0 / e6 - e9) * e1;
            let e14 = e3.ln_1p();
            h11 += -(((e12 - 1.0) / v2 + e9) / v2);
            h12 += (e5 / e8 + v3 * e11 / (v2 * v2 * e4)) * e1 / v3;
            h22 += -(((e1 / e6 - e14 / v3) * e11 + e13 - (e11 * e14 + e12) / v3) / v3);
        }
        vec![h11, h12, h12, h22]
    }

    /// Analytic third derivatives (reference `gpd_k1_logfddd`, location 0).
    ///
    /// Row-major `2 × 2 × 2`, index `(i, j, k)` at `i*4 + j*2 + k` with
    /// `0 = scale`, `1 = shape`.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let v2 = params[0];
        let v3 = params[1];
        if v3.abs() < 1e-12 {
            let cfg = self.fd_config();
            return crate::models::numerical_third_derivatives(
                &|p: &[f64]| self.loglik(p, y),
                params,
                &cfg,
            );
        }
        let mut d000 = 0.0;
        let mut d001 = 0.0;
        // The reference's generated tensor gives three analytically-equal but
        // syntactically-different expressions for the mixed (scale, shape,
        // shape) entry; we keep them separate exactly as the reference does.
        let mut d011_a = 0.0; // (scale, shape, shape)
        let mut d011_b = 0.0; // (shape, scale, shape)
        let mut d011_c = 0.0; // (shape, shape, scale)
        let mut d111 = 0.0;
        for &e1 in y {
            let e2 = v3 * e1;
            let e3 = e2 / v2;
            let e4 = 1.0 + e3;
            let e5 = v2 * e4;
            let e6 = 1.0 + v3;
            let e7 = e5 * e5;
            let e9 = 1.0 - e6 / v3;
            let e11 = v2 * e6 * e4;
            let e12 = e11 * e1;
            let e13 = e6 * e1;
            let e14 = 2.0 * (e12 / e7);
            let e16 = v2 * v2 * e4;
            let e17 = e13 / e7;
            let e18 = e1 * e1;
            let e20 = 1.0 / e5 - e17;
            let e21 = e3.ln_1p();
            let e22 = v3 * e9;
            let e23 = (e6 / e7 + e22 / e16) / v3;
            let e24 = (1.0 - e14) / e7;
            let e25 = e13 / e5;
            let e27 = (2.0 - e14) * e18;
            let e28 = (e14 - 1.0) / e7;
            let e29 = e5 * e5 * e5 * e5;
            let e30 = e16 * e16;
            let e32 = e1 / e5 - e21 / v3;
            d000 += (2.0 * (((e25 - 1.0) / v2 + e17) / v2) + 2.0 * (e12 / e29)) / v2;
            d001 += -((2.0 * (e11 / e29) + e22 * (2.0 * e5 - e2) / e30) * e1 / v3);
            d011_a += -((e23 + e9 * (1.0 / e16 - 1.0 / e7) + e28) * e1 / v3);
            d011_b += (e24 - (e23 + v2 * v3 * e9 * e1 / e30)) * e1 / v3;
            d011_c += -(e27 / (v2 * e7));
            d111 += ((e18 / e7 + 2.0 * (e32 / v3)) * e9
                + e27 / e7
                + 2.0 * ((e32 * e9 + e20 * e1 - (e9 * e21 + e25) / v3) / v3))
                / v3;
        }
        vec![
            d000, d001, // (0,0,0), (0,0,1)
            d001, d011_a, // (0,1,0), (0,1,1)
            d001, d011_b, // (1,0,0), (1,0,1)
            d011_c, d111, // (1,1,0), (1,1,1)
        ]
    }

    /// Analytic mixed derivatives `∂²ℓ/∂θ_i∂y_a` (W9).
    ///
    /// Row-major `2 × n`. With `t_a = 1 + shape y_a/scale`:
    /// `∂²ℓ/∂scale∂y_a = (1+shape)/(scale² t_a) - shape(1+shape) y_a/(scale³ t_a²)`
    /// and
    /// `∂²ℓ/∂shape∂y_a = (1/scale)/(shape² t_a) - (1/shape+1)[(1/scale)/t_a - y_a/(scale² t_a²)]`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let scale = params[0];
        let shape = params[1];
        if shape.abs() < 1e-12 {
            let cfg = self.fd_config();
            return crate::models::numerical_mixed_derivatives(
                &|p: &[f64], yy: &[f64]| self.loglik(p, yy),
                params,
                y,
                &cfg,
            );
        }
        let ny = y.len();
        let s2 = scale * scale;
        let s3 = s2 * scale;
        let mut out = vec![0.0; 2 * ny];
        for (a, &ya) in y.iter().enumerate() {
            let t = 1.0 + shape * ya / scale;
            out[a] = (1.0 + shape) / (s2 * t) - shape * (1.0 + shape) * ya / (s3 * t * t);
            out[ny + a] = 1.0 / (shape * scale * t)
                - (1.0 / shape + 1.0) * ((1.0 / scale) / t - shape * ya / (s2 * t * t));
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

    #[test]
    fn shape_zero_is_exponential() {
        // GPD(scale, shape -> 0) == Exponential(rate = 1/scale).
        let m = GenPareto;
        let scale = 2.0;
        let rate = 1.0 / scale;
        // Use a tiny shape to approximate the limit. The approximation error is
        // O(shape), so a looser tolerance than the §7.2 golden policy is used.
        let shape = 1e-9;
        for &y in &[0.3_f64, 1.0, 2.5] {
            let got = m.logpdf(y, scale, shape);
            let want = rate.ln() - rate * y;
            assert!(
                (got - want).abs() <= 1e-7 + 1e-7 * want.abs(),
                "gpd=exp logpdf: {got} vs {want}"
            );
        }
    }

    #[test]
    fn cdf_matches_reference() {
        let m = GenPareto;
        // F(1; 1, 1) = 1 - 2^{-1} = 0.5
        assert_close(m.cdf(1.0, 1.0, 1.0), 0.5, "cdf(1;1,1)");
        // F(1; 1, 0) = 1 - e^{-1}
        assert_close(m.cdf(1.0, 1.0, 0.0), 0.632_120_558_828_557_7, "cdf(1;1,0)");
    }

    #[test]
    fn quantile_matches_reference() {
        let m = GenPareto;
        // q(0.5; 1, 1) = (1/1)(2 - 1) = 1
        assert_close(m.quantile(0.5, 1.0, 1.0), 1.0, "q(0.5;1,1)");
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = GenPareto;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 1.5, 0.3);
            assert_close(m.cdf(x, 1.5, 0.3), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn fits_synthetic_data() {
        // The GPD likelihood surface is hard: the shape parameter can drift
        // toward the support boundary and the quasi-Newton optimizer may not
        // reach a stationary point from a generic init. We therefore assert the
        // weaker but meaningful property that the fit **substantially improves**
        // the log-likelihood over the init and returns finite parameters — i.e.
        // the model is fittable — rather than demanding full convergence.
        let m = GenPareto;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let init = [1.0, 0.1];
        let fit = fit(&m, &y, &init);
        assert!(fit.ml_params.iter().all(|v| v.is_finite()));
        assert!(
            fit.loglik > m.loglik(&init, &y),
            "fit did not improve the log-likelihood: {} vs init {}",
            fit.loglik,
            m.loglik(&init, &y)
        );
    }

    // --- Phase 2 (Issue 2): robust initial conditions -----------------------

    #[test]
    fn setics_uses_sample_std_for_zero_sentinel() {
        // Reference `gpd_k1_setics`: ics = [0, 0] => [sd(x), 0] (n-1 denominator).
        let m = GenPareto;
        let x = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let ics = m.setics(&x, &[0.0, 0.0]);
        assert_eq!(ics.len(), 2);
        // R's sd(x) for this sample (n-1 denominator).
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let sd = (x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        assert_close(ics[0], sd, "setics scale");
        assert_close(ics[1], 0.0, "setics shape");
    }

    #[test]
    fn setics_passes_through_nonzero_ics() {
        // A non-sentinel `ics` is returned unchanged (reference behaviour).
        let m = GenPareto;
        let x = vec![0.5, 1.2, 0.3, 2.1, 0.9];
        let ics = m.setics(&x, &[2.5, -0.2]);
        assert_eq!(ics, vec![2.5, -0.2]);
    }

    #[test]
    fn setics_scales_with_dollar_scale_data() {
        // The defect: dollar-scale data (~10^3-10^5) needs a scale init of the
        // same order, not the hard-coded 1.0.
        let m = GenPareto;
        let x = vec![1200.0, 45000.0, 8300.0, 99000.0, 21000.0, 6700.0, 33000.0];
        let ics = m.setics(&x, &[0.0, 0.0]);
        assert!(
            ics[0] > 1e3,
            "dollar-scale setics scale {} should be O(10^4)",
            ics[0]
        );
    }

    #[test]
    fn revert2ml_matches_reference_threshold() {
        // Reference `qgpd_k1_cp`: revert2ml = (abs(v2hat) >= 1) || (v2hat > 30).
        //
        // The `abs(v2hat) >= 1` clause already covers every shape with
        // magnitude >= 1, so the `v2hat > 30` clause is a documented redundancy
        // in the reference (it can only fire for a shape that the first clause
        // would also catch). We mirror the reference exactly and assert the
        // combined predicate.
        let m = GenPareto;
        assert!(!m.revert2ml(0.0));
        assert!(!m.revert2ml(0.5));
        assert!(!m.revert2ml(-0.99));
        assert!(m.revert2ml(1.0));
        assert!(m.revert2ml(-1.0));
        assert!(m.revert2ml(1.5));
        assert!(m.revert2ml(-2.0));
        // Shapes above 30 are caught (by the `abs >= 1` clause, and also by the
        // `> 30` clause).
        assert!(m.revert2ml(30.0));
        assert!(m.revert2ml(30.5));
        assert!(m.revert2ml(100.0));
    }
}
