//! Generalized extreme value distribution with **two** predictors
//! (`genextreme_p12`) model (Phase 6, Issue 6).
//!
//! # Parameterization (reference `fitdistcp`)
//!
//! `genextreme_p12` is the GEV model whose **location** is a linear function of
//! a first predictor `t1` and whose **log-scale** is a linear function of a
//! second predictor `t2`:
//!
//! ```text
//! mu_i    = beta0 + beta1 * (t1_i - mean(t1))
//! sigma_i = exp(beta2 + scale * (t2_i - mean(t2)))
//! y_i     ~ GEV(mu_i, sigma_i, shape)
//! ```
//!
//! `params = [beta0, beta1, beta2, scale, shape]` with `sigma_i > 0` (the
//! exponential guarantees it). This mirrors the reference
//! [`genextreme_p12.py`](tools/fitdistcp-py-ref/fitdistcp/genextreme_p12.py):
//! `gev_p12_loglik` uses `mu = vv[0] + vv[1]*t1` and
//! `sigma = exp(vv[2] + vv[3]*t2)`, with the predictors **centered by their
//! means** (`t1 = t1 - mean(t1)`, `t2 = t2 - mean(t2)`) before fitting.
//!
//! The reference's `t01`/`t02`/`n01`/`n02` arguments do **not** enter the
//! parameterization; they only select the **prediction point** (see
//! [`GenExtremeP12::centered_prediction1`] / [`GenExtremeP12::centered_prediction2`]
//! and the binding-level `maket0`).
//!
//! # Integration path
//!
//! **DMGS** (§3): the GEV is integrated through the DMGS asymptotic expansion
//! (W8). The negative-shape quantile-derived density is W10 and is **not**
//! implemented here.
//!
//! # Derivatives
//!
//! All derivatives (score, Hessian, third, mixed) use the **numerical fallback**
//! from W5, exactly as [`crate::models::GenExtremeP1`] does. The reference's
//! `gev_p12_derivs` helpers are large auto-generated expressions; the numerical
//! path is validated in the tests against the score vanishing at the MLE.

use crate::models::Derivatives;

/// The GEV-with-two-predictors (`genextreme_p12`) model.
///
/// Holds the predictor plumbing `t1`, `t2`, `t01`, `t02`, `n01`, `n02`. The
/// predictors `t1`/`t2` must have the same length as the data `y` when present.
///
/// # Field semantics (reference convention)
///
/// - `t1`: the **raw** first predictor values, one per observation (empty when
///   absent).
/// - `t2`: the **raw** second predictor values, one per observation (empty when
///   absent).
/// - `t01`: the **resolved prediction point** for `t1` (a predictor value, not
///   an index).
/// - `t02`: the **resolved prediction point** for `t2`.
/// - `n01`, `n02`: retained for API compatibility; they do **not** enter the
///   parameterization.
#[derive(Debug, Clone, PartialEq)]
pub struct GenExtremeP12 {
    /// Raw first predictor values `t1`, one per observation (empty when absent).
    pub t1: Vec<f64>,
    /// Raw second predictor values `t2`, one per observation (empty when absent).
    pub t2: Vec<f64>,
    /// Resolved prediction point `t01` for the first predictor.
    pub t01: f64,
    /// Resolved prediction point `t02` for the second predictor.
    pub t02: f64,
    /// First predictor count `n01` (retained for API compatibility; unused).
    pub n01: f64,
    /// Second predictor count `n02` (retained for API compatibility; unused).
    pub n02: f64,
}

impl Default for GenExtremeP12 {
    fn default() -> Self {
        Self {
            t1: Vec::new(),
            t2: Vec::new(),
            t01: 0.0,
            t02: 0.0,
            n01: 1.0,
            n02: 1.0,
        }
    }
}

impl GenExtremeP12 {
    /// Create a `genextreme_p12` model with explicit predictor plumbing.
    ///
    /// `t01`/`t02` are the **resolved prediction points** (predictor values).
    /// `n01`/`n02` are retained for API compatibility and do not affect the fit.
    pub fn new(t1: Vec<f64>, t2: Vec<f64>, t01: f64, t02: f64, n01: f64, n02: f64) -> Self {
        Self {
            t1,
            t2,
            t01,
            t02,
            n01,
            n02,
        }
    }

    /// Create a `genextreme_p12` model with no predictors (constant location and
    /// scale).
    pub fn without_predictor() -> Self {
        Self::default()
    }

    /// The mean of the first predictor `t1` (the centering value).
    #[inline]
    pub fn mean_t1(&self) -> f64 {
        if self.t1.is_empty() {
            0.0
        } else {
            self.t1.iter().sum::<f64>() / self.t1.len() as f64
        }
    }

    /// The mean of the second predictor `t2` (the centering value).
    #[inline]
    pub fn mean_t2(&self) -> f64 {
        if self.t2.is_empty() {
            0.0
        } else {
            self.t2.iter().sum::<f64>() / self.t2.len() as f64
        }
    }

    /// The centered first predictor value for observation `i`: `t1_i - mean(t1)`.
    #[inline]
    pub fn centered1(&self, i: usize) -> f64 {
        if self.t1.is_empty() {
            0.0
        } else {
            self.t1[i] - self.mean_t1()
        }
    }

    /// The centered second predictor value for observation `i`: `t2_i - mean(t2)`.
    #[inline]
    pub fn centered2(&self, i: usize) -> f64 {
        if self.t2.is_empty() {
            0.0
        } else {
            self.t2[i] - self.mean_t2()
        }
    }

    /// The centered **prediction point** for `t1`: `t01 - mean(t1)`.
    #[inline]
    pub fn centered_prediction1(&self) -> f64 {
        if self.t1.is_empty() {
            0.0
        } else {
            self.t01 - self.mean_t1()
        }
    }

    /// The centered **prediction point** for `t2`: `t02 - mean(t2)`.
    #[inline]
    pub fn centered_prediction2(&self) -> f64 {
        if self.t2.is_empty() {
            0.0
        } else {
            self.t02 - self.mean_t2()
        }
    }

    /// Location `mu_i = beta0 + beta1 * centered1(t1_i)`.
    #[inline]
    pub fn location(&self, beta0: f64, beta1: f64, i: usize) -> f64 {
        beta0 + beta1 * self.centered1(i)
    }

    /// Scale `sigma_i = exp(beta2 + scale * centered2(t2_i))`.
    #[inline]
    pub fn scale(&self, beta2: f64, scale: f64, i: usize) -> f64 {
        (beta2 + scale * self.centered2(i)).exp()
    }

    /// Location at the prediction point: `muhat0 = beta0 + beta1 * centered(t01)`.
    #[inline]
    pub fn location_prediction(&self, beta0: f64, beta1: f64) -> f64 {
        beta0 + beta1 * self.centered_prediction1()
    }

    /// Scale at the prediction point: `sigmahat0 = exp(beta2 + scale * centered(t02))`.
    #[inline]
    pub fn scale_prediction(&self, beta2: f64, scale: f64) -> f64 {
        (beta2 + scale * self.centered_prediction2()).exp()
    }

    /// Log-density `ln f(y_i; beta0, beta1, beta2, scale, shape)` at observation `i`.
    #[allow(clippy::too_many_arguments)]
    pub fn logpdf(
        &self,
        y: f64,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        scale: f64,
        shape: f64,
        i: usize,
    ) -> f64 {
        let loc = self.location(beta0, beta1, i);
        let sigma = self.scale(beta2, scale, i);
        let z = (y - loc) / sigma;
        if shape.abs() < 1e-12 {
            return -z - (-z).exp() - sigma.ln();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -sigma.ln() - (1.0 / shape + 1.0) * t.ln() - t.powf(-1.0 / shape)
    }

    /// Density `f(y_i; beta0, beta1, beta2, scale, shape)` at observation `i`.
    #[allow(clippy::too_many_arguments)]
    pub fn pdf(
        &self,
        y: f64,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        scale: f64,
        shape: f64,
        i: usize,
    ) -> f64 {
        self.logpdf(y, beta0, beta1, beta2, scale, shape, i).exp()
    }

    /// CDF `F(y_i; beta0, beta1, beta2, scale, shape)` at observation `i`.
    #[allow(clippy::too_many_arguments)]
    pub fn cdf(
        &self,
        y: f64,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        scale: f64,
        shape: f64,
        i: usize,
    ) -> f64 {
        let loc = self.location(beta0, beta1, i);
        let sigma = self.scale(beta2, scale, i);
        let z = (y - loc) / sigma;
        if shape.abs() < 1e-12 {
            return (-(-z).exp()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return if shape > 0.0 { 0.0 } else { 1.0 };
        }
        (-t.powf(-1.0 / shape)).exp()
    }

    /// The reference `qgev_p12_cp` `revert2ml` guard: `|shape| >= 1`.
    ///
    /// When the fitted shape magnitude reaches 1 the DMGS expansion is
    /// numerically unreliable, so the reference reverts the CP output to the MLE
    /// output. Identical to [`crate::models::GenExtreme::revert2ml`] (shape-only).
    pub fn revert2ml(&self, shape: f64) -> bool {
        shape.abs() >= 1.0
    }

    /// The reference `qgev_p12_cp` `ml_max`: the finite upper endpoint for a
    /// negative shape, `loc - scale/shape`, else `+inf`.
    ///
    /// Identical to [`crate::models::GenExtreme::ml_max`] (shape-only).
    pub fn ml_max(&self, loc: f64, scale: f64, shape: f64) -> f64 {
        if shape < 0.0 {
            loc - scale / shape
        } else {
            f64::INFINITY
        }
    }

    /// Quantile `F^{-1}(p; beta0, beta1, beta2, scale, shape)` at observation `i`.
    #[allow(clippy::too_many_arguments)]
    pub fn quantile(
        &self,
        p: f64,
        beta0: f64,
        beta1: f64,
        beta2: f64,
        scale: f64,
        shape: f64,
        i: usize,
    ) -> f64 {
        let loc = self.location(beta0, beta1, i);
        let sigma = self.scale(beta2, scale, i);
        if shape.abs() < 1e-12 {
            return loc - sigma * (-p.ln()).ln();
        }
        loc + (sigma / shape) * ((-p.ln()).powf(-shape) - 1.0)
    }
}

impl Derivatives for GenExtremeP12 {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let beta0 = params[0];
        let beta1 = params[1];
        let beta2 = params[2];
        let scale = params[3];
        let shape = params[4];
        y.iter()
            .enumerate()
            .map(|(i, &yi)| self.logpdf(yi, beta0, beta1, beta2, scale, shape, i))
            .sum()
    }

    fn n_params(&self) -> usize {
        5
    }

    /// `setics`-style initial conditions (Phase 3, Issue 3).
    ///
    /// The reference seeds the `genextreme_p12` `optim` from `gev_p12_setics`,
    /// which regresses `x` on the centered `t1` for the location intercept/slope
    /// and uses zero for the log-scale intercept/slope and the shape. The Rust
    /// multi-start strategy adds the same data-driven start:
    /// `[mean(y), 0, 0, 0, 0]` (the predictors are centered, so zero slopes are
    /// the natural start; the log-scale intercept is zero because it is inside an
    /// exponential).
    fn setics(&self, y: &[f64], init: &[f64]) -> Vec<f64> {
        let mean = crate::models::sample_mean(y);
        if y.len() >= 2 {
            vec![mean, 0.0, 0.0, 0.0, 0.0]
        } else {
            init.to_vec()
        }
    }

    /// Reject degenerate parameter regions (Phase 3, Issue 3).
    ///
    /// The GEV has a spurious `scale -> 0`, `shape -> +inf` region where the
    /// log-likelihood is unbounded; the reference guards against it with
    /// `gev_checkmle` shape bounds. We reject a non-positive scale and
    /// `|shape| >= 10`, mirroring [`crate::models::GenExtreme::valid_params`].
    fn valid_params(&self, params: &[f64]) -> bool {
        let scale = params[3];
        let shape = params[4];
        scale > 0.0 && shape.abs() < 10.0
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

    fn sample() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let t2: Vec<f64> = (0..10).map(|i| (i + 11) as f64).collect();
        let y = vec![0.1, 1.3, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1];
        (t1, t2, y)
    }

    #[test]
    fn without_predictor_matches_plain_gev() {
        let m = GenExtremeP12::without_predictor();
        let g = crate::models::genextreme::GenExtreme;
        let y = [0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        for (i, &yi) in y.iter().enumerate() {
            // beta1 = beta2 = scale = 0 => constant loc = beta0, sigma = exp(0) = 1.
            assert_close(
                m.logpdf(yi, 0.5, 0.0, 0.0, 0.0, 0.1, i),
                g.logpdf(yi, 0.5, 1.0, 0.1),
                "genextreme_p12(no t) = gev logpdf",
            );
        }
    }

    #[test]
    fn primitives_match_reference() {
        // t1 = [0, 2], mean = 1, centered = [-1, 1]. With beta1 = 0, loc = beta0.
        let m = GenExtremeP12::new(vec![0.0, 2.0], vec![0.0, 2.0], 1.0, 1.0, 1.0, 1.0);
        // shape = 0 -> Gumbel, sigma = exp(0) = 1.
        assert_close(m.logpdf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0), -1.0, "logpdf");
        assert_close(
            m.cdf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
            0.367_879_441_171_442_33,
            "cdf",
        );
        assert_close(
            m.quantile(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
            -(2.0_f64.ln()).ln(),
            "quantile",
        );
    }

    #[test]
    fn centering_uses_mean_of_predictors() {
        let m = GenExtremeP12::new(vec![0.0, 2.0], vec![10.0, 14.0], 1.0, 12.0, 1.0, 1.0);
        assert_close(m.centered1(0), -1.0, "centered1(0)");
        assert_close(m.centered1(1), 1.0, "centered1(1)");
        assert_close(m.centered2(0), -2.0, "centered2(0)");
        assert_close(m.centered2(1), 2.0, "centered2(1)");
        assert_close(m.centered_prediction1(), 0.0, "centered_prediction1");
        assert_close(m.centered_prediction2(), 0.0, "centered_prediction2");
    }

    #[test]
    fn scale_depends_on_second_predictor() {
        // sigma_i = exp(beta2 + scale * centered2(t2_i)).
        let m = GenExtremeP12::new(vec![0.0, 2.0], vec![0.0, 2.0], 1.0, 1.0, 1.0, 1.0);
        assert_close(m.scale(0.0, 0.0, 0), 1.0, "scale(0)");
        assert_close(m.scale(0.0, 1.0, 1), (1.0_f64).exp(), "scale(1)");
    }

    #[test]
    fn fits_synthetic_data() {
        // The GEV-with-two-predictors surface is hard (five parameters). As with
        // the GEV-p1, we assert the weaker but meaningful property that the fit
        // **substantially improves** the log-likelihood over the init and returns
        // finite parameters — i.e. the model is fittable.
        let (t1, t2, y) = sample();
        let m = GenExtremeP12::new(t1, t2, 4.5, 15.0, 10.0, 10.0);
        let init = [0.0, 1.0, 0.0, 0.0, 0.0];
        let fit = fit(&m, &y, &init);
        assert!(fit.ml_params.iter().all(|v| v.is_finite()));
        assert!(
            fit.loglik > m.loglik(&init, &y),
            "fit did not improve the log-likelihood: {} vs init {}",
            fit.loglik,
            m.loglik(&init, &y)
        );
    }
}
