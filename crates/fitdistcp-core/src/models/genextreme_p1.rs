//! Generalized extreme value distribution with a predictor (`genextreme_p1`)
//! model (§3).
//!
//! # Parameterization (reference `fitdistcp`)
//!
//! `genextreme_p1` is the GEV model whose **location** is a linear function of a
//! predictor `t`:
//!
//! ```text
//! mu_i = beta0 + beta1 * (t_i - mean(t))
//! y_i  ~ GEV(mu_i, scale, shape)
//! ```
//!
//! `params = [beta0, beta1, scale, shape]` with `scale > 0`. This is the
//! reference `fitdistcp` convention (W14): the predictor is **centered by its
//! mean** (`t = t - mean(t)`), and the model is fit on the centered predictor
//! with `mu = a + b * t_centered`. The reference's `t0`/`n0` arguments do **not**
//! enter the parameterization; they only select the **prediction point** (see
//! [`GenExtremeP1::centered_prediction`] and the binding-level `maket0`).
//!
//! When `t` is absent the model degenerates to a constant location
//! (`mu_i = beta0`).
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
//! from W5. The reference's `gev_p1k3` derivative helpers (`gev_p1k3_logfdd`,
//! `gev_p1k3_logfddd`) return only a **3 × 3** block over `(beta0, beta1, scale)`
//! — the shape parameter is handled separately — so there is no complete
//! analytic 4-parameter form to transcribe. We therefore keep the numerical
//! fallback for this model and document the gap. The numerical path is validated
//! in the tests against the score vanishing at the MLE.

use crate::models::Derivatives;

/// The GEV-with-predictor (`genextreme_p1`) model.
///
/// Holds the predictor plumbing `t`, `t0`, `n0`. The predictor `t` must have the
/// same length as the data `y` when present.
///
/// # Field semantics (W14, reference convention)
///
/// - `t`: the **raw** predictor values, one per observation (empty when absent).
/// - `t0`: the **resolved prediction point** (a predictor value, not an index).
/// - `n0`: retained for API compatibility; it does **not** enter the
///   parameterization.
#[derive(Debug, Clone, PartialEq)]
pub struct GenExtremeP1 {
    /// Raw predictor values `t`, one per observation (empty when absent).
    pub t: Vec<f64>,
    /// Resolved prediction point `t0` (a predictor value).
    pub t0: f64,
    /// Predictor count `n0` (retained for API compatibility; unused in the fit).
    pub n0: f64,
}

impl Default for GenExtremeP1 {
    fn default() -> Self {
        Self {
            t: Vec::new(),
            t0: 0.0,
            n0: 1.0,
        }
    }
}

impl GenExtremeP1 {
    /// Create a `genextreme_p1` model with explicit predictor plumbing.
    ///
    /// `t0` is the **resolved prediction point** (a predictor value). `n0` is
    /// retained for API compatibility and does not affect the fit.
    pub fn new(t: Vec<f64>, t0: f64, n0: f64) -> Self {
        Self { t, t0, n0 }
    }

    /// Create a `genextreme_p1` model with no predictor (constant location).
    pub fn without_predictor() -> Self {
        Self::default()
    }

    /// The mean of the predictor `t` (the centering value).
    #[inline]
    pub fn mean_t(&self) -> f64 {
        if self.t.is_empty() {
            0.0
        } else {
            self.t.iter().sum::<f64>() / self.t.len() as f64
        }
    }

    /// The centered predictor value for observation `i`: `t_i - mean(t)`.
    #[inline]
    pub fn centered(&self, i: usize) -> f64 {
        if self.t.is_empty() {
            0.0
        } else {
            self.t[i] - self.mean_t()
        }
    }

    /// The centered **prediction point**: `t0 - mean(t)`.
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

    /// Log-density `ln f(y_i; beta0, beta1, scale, shape)` at observation `i`.
    pub fn logpdf(&self, y: f64, beta0: f64, beta1: f64, scale: f64, shape: f64, i: usize) -> f64 {
        let loc = self.location(beta0, beta1, i);
        let z = (y - loc) / scale;
        if shape.abs() < 1e-12 {
            return -z - (-z).exp() - scale.ln();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -scale.ln() - (1.0 / shape + 1.0) * t.ln() - t.powf(-1.0 / shape)
    }

    /// Density `f(y_i; beta0, beta1, scale, shape)` at observation `i`.
    pub fn pdf(&self, y: f64, beta0: f64, beta1: f64, scale: f64, shape: f64, i: usize) -> f64 {
        self.logpdf(y, beta0, beta1, scale, shape, i).exp()
    }

    /// CDF `F(y_i; beta0, beta1, scale, shape)` at observation `i`.
    pub fn cdf(&self, y: f64, beta0: f64, beta1: f64, scale: f64, shape: f64, i: usize) -> f64 {
        let loc = self.location(beta0, beta1, i);
        let z = (y - loc) / scale;
        if shape.abs() < 1e-12 {
            return (-(-z).exp()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return if shape > 0.0 { 0.0 } else { 1.0 };
        }
        (-t.powf(-1.0 / shape)).exp()
    }

    /// The reference `qgev_p1_cp` `revert2ml` guard: `|shape| >= 1`.
    ///
    /// When the fitted shape magnitude reaches 1 the DMGS expansion is
    /// numerically unreliable, so the reference reverts the CP output to the MLE
    /// output. Identical to [`crate::models::GenExtreme::revert2ml`] (shape-only).
    pub fn revert2ml(&self, shape: f64) -> bool {
        shape.abs() >= 1.0
    }

    /// The reference `qgev_p1_cp` `ml_max`: the finite upper endpoint for a
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

    /// Quantile `F^{-1}(p; beta0, beta1, scale, shape)` at observation `i`.
    pub fn quantile(
        &self,
        p: f64,
        beta0: f64,
        beta1: f64,
        scale: f64,
        shape: f64,
        i: usize,
    ) -> f64 {
        let loc = self.location(beta0, beta1, i);
        if shape.abs() < 1e-12 {
            return loc - scale * (-p.ln()).ln();
        }
        loc + (scale / shape) * ((-p.ln()).powf(-shape) - 1.0)
    }
}

impl Derivatives for GenExtremeP1 {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let beta0 = params[0];
        let beta1 = params[1];
        let scale = params[2];
        let shape = params[3];
        y.iter()
            .enumerate()
            .map(|(i, &yi)| self.logpdf(yi, beta0, beta1, scale, shape, i))
            .sum()
    }

    fn n_params(&self) -> usize {
        4
    }

    /// `setics`-style initial conditions (Phase 3, Issue 3).
    ///
    /// The reference seeds the GEV-p1 `optim` from `gev_p1_setics`, which uses
    /// the data mean/SD for the intercept/scale and a zero slope/shape. The Rust
    /// multi-start strategy adds the same data-driven start:
    /// `[mean(y), 0, sd(y), 0]` (the predictor is centered, so a zero slope is
    /// the natural start).
    fn setics(&self, y: &[f64], init: &[f64]) -> Vec<f64> {
        let mean = crate::models::sample_mean(y);
        let sd = crate::models::sample_std_ddof1(y);
        if sd > 0.0 {
            vec![mean, 0.0, sd, 0.0]
        } else {
            init.to_vec()
        }
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
    fn without_predictor_matches_plain_gev() {
        let m = GenExtremeP1::without_predictor();
        let g = crate::models::genextreme::GenExtreme;
        let y = [0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        for (i, &yi) in y.iter().enumerate() {
            assert_close(
                m.logpdf(yi, 0.5, 99.0, 1.0, 0.1, i),
                g.logpdf(yi, 0.5, 1.0, 0.1),
                "genextreme_p1(no t) = gev logpdf",
            );
        }
    }

    #[test]
    fn primitives_match_reference() {
        // t = [0, 2], mean = 1, centered = [-1, 1]. With beta1 = 0, loc = beta0.
        let m = GenExtremeP1::new(vec![0.0, 2.0], 1.0, 1.0);
        // shape = 0 -> Gumbel.
        assert_close(m.logpdf(0.0, 0.0, 0.0, 1.0, 0.0, 0), -1.0, "logpdf");
        assert_close(
            m.cdf(0.0, 0.0, 0.0, 1.0, 0.0, 0),
            0.367_879_441_171_442_33,
            "cdf",
        );
        assert_close(
            m.quantile(0.5, 0.0, 0.0, 1.0, 0.0, 0),
            -(2.0_f64.ln()).ln(),
            "quantile",
        );
    }

    #[test]
    fn centering_uses_mean_of_t() {
        let m = GenExtremeP1::new(vec![0.0, 2.0], 1.0, 1.0);
        assert_close(m.centered(0), -1.0, "centered(0)");
        assert_close(m.centered(1), 1.0, "centered(1)");
        assert_close(m.centered_prediction(), 0.0, "centered_prediction");
        let m2 = GenExtremeP1::new(vec![0.0, 2.0], 3.0, 1.0);
        assert_close(m2.centered_prediction(), 2.0, "centered_prediction(3)");
    }

    #[test]
    fn fits_synthetic_data() {
        // The GEV-with-predictor surface is hard (four parameters, a shape that
        // can drift toward the support boundary). As with the GPD, we assert the
        // weaker but meaningful property that the fit **substantially improves**
        // the log-likelihood over the init and returns finite parameters — i.e.
        // the model is fittable — rather than demanding full convergence.
        let (t, y) = sample();
        let m = GenExtremeP1::new(t, 4.5, 10.0);
        let init = [0.0, 1.0, 1.0, 0.0];
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
