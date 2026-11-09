//! Gamma distribution model (§3).
//!
//! # Parameterization
//!
//! `params = [shape, rate]` with `shape > 0`, `rate > 0` (the reference
//! `fitdistcp` convention, matching R's `dgamma` with `rate`):
//!
//! ```text
//! f(y; shape, rate) = rate^shape / Γ(shape) y^(shape-1) exp(-rate y),  y > 0
//! ```
//!
//! # Integration path
//!
//! **DMGS** (§3): the gamma is integrated through the DMGS asymptotic expansion
//! (W8); this module supplies the model primitives and derivatives.
//!
//! # Derivatives
//!
//! All derivatives are **analytic** (W9). With `S = Σ y_i`, `L = Σ ln y_i`,
//! `ψ = digamma(shape)`, `ψ₁ = trigamma(shape)`, and `ψ₂ = tetragamma(shape)`:
//!
//! - `∂ℓ/∂shape = n ln(rate) - n ψ + L`
//! - `∂ℓ/∂rate  = n shape/rate - S`
//! - `∂²ℓ/∂shape² = -n ψ₁`
//! - `∂²ℓ/∂shape∂rate = n/rate`
//! - `∂²ℓ/∂rate² = -n shape/rate²`
//! - `∂³ℓ/∂shape³ = -n ψ₂`
//! - `∂³ℓ/∂shape²∂rate = 0`
//! - `∂³ℓ/∂shape∂rate² = n/rate²`
//! - `∂³ℓ/∂rate³ = 2n shape/rate³`
//! - `∂²ℓ/∂shape∂y_a = 1/y_a`
//! - `∂²ℓ/∂rate∂y_a = -1`
//!
//! The density derivative hooks are the reference's `gamma_fd`/`gamma_fdd`. The
//! reference has **no** analytic CDF derivatives for the gamma (it uses the
//! numerical `gamma_mu1f`/`gamma_mu2f`), so the CDF hooks keep the numerical
//! default.

use crate::models::Derivatives;
use crate::special::{digamma, ln_gamma, tetragamma, trigamma};

/// The gamma distribution model (shape/rate parameterization).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gamma;

impl Gamma {
    /// Create a gamma model.
    pub fn new() -> Self {
        Gamma
    }

    /// Log-density `ln f(y; shape, rate)` at a single point.
    pub fn logpdf(&self, y: f64, shape: f64, rate: f64) -> f64 {
        shape * rate.ln() - ln_gamma(shape) + (shape - 1.0) * y.ln() - rate * y
    }

    /// Density `f(y; shape, rate)` at a single point.
    pub fn pdf(&self, y: f64, shape: f64, rate: f64) -> f64 {
        self.logpdf(y, shape, rate).exp()
    }

    /// CDF `F(y; shape, rate)` via the regularized lower incomplete gamma
    /// function `P(shape, rate y)`.
    pub fn cdf(&self, y: f64, shape: f64, rate: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        // `special::Gamma::inc_gamma(self, p)` computes the **regularized**
        // lower incomplete gamma `P(x, p) = γ(p, x)/Γ(p)` where `self` is the
        // argument `x` and the method argument is the shape `p`. So the call is
        // `inc_gamma(rate*y, shape)` — argument first, shape second — and the
        // result is already regularized (no division by Γ(shape)).
        let p = special::Gamma::inc_gamma(rate * y, shape);
        p.clamp(0.0, 1.0)
    }

    /// Quantile `F^{-1}(p; shape, rate)`.
    ///
    /// W17 (H2): computed by a **Newton iteration on the regularized lower
    /// incomplete gamma** `P(shape, x)` (with `x = rate·y`), replacing the
    /// previous bisection that needed up to 200 CDF evaluations. Newton
    /// converges in a handful of iterations because `P` is smooth and monotone
    /// and its derivative is available in closed form:
    ///
    /// ```text
    /// dP/dx = x^(shape-1) e^{-x} / Γ(shape)
    /// ```
    ///
    /// The iteration is safeguarded by a bisection bracket so it cannot diverge:
    /// if a Newton step leaves the bracket `[lo, hi]` (or fails to reduce the
    /// residual), we fall back to a bisection step. The bracket is maintained
    /// from the Wilson–Hilferty initial guess, so the result is accurate to near
    /// machine precision and matches the previous bisection to well within the
    /// §7.2 tolerance.
    pub fn quantile(&self, p: f64, shape: f64, rate: f64) -> f64 {
        if p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        // Solve P(shape, x) = p for x, then y = x / rate.
        let x = inverse_regularized_gamma(shape, p);
        x / rate
    }

    // --- (shape, scale) parameterization (A5) ------------------------------
    //
    // The reference `fitdistcp` gamma module uses the `(shape, scale)`
    // parameterization (`gamma.py` / `gamma_libs.py`), where `scale = 1/rate`.
    // The core model above uses `(shape, rate)` (matching R's `dgamma` with
    // `rate`). These entry points expose the reference parameterization at the
    // boundary so the Python binding can honor it without duplicating the
    // numerics: each simply converts `rate = 1/scale` and delegates.

    /// Convert a `(shape, scale)` pair to the core `(shape, rate)` pair.
    ///
    /// `rate = 1/scale`. This is the boundary conversion the reference
    /// parameterization requires.
    #[inline]
    pub fn scale_to_rate(scale: f64) -> f64 {
        1.0 / scale
    }

    /// Log-density `ln f(y; shape, scale)` in the reference `(shape, scale)`
    /// parameterization.
    pub fn logpdf_scale(&self, y: f64, shape: f64, scale: f64) -> f64 {
        self.logpdf(y, shape, Self::scale_to_rate(scale))
    }

    /// Density `f(y; shape, scale)` in the reference `(shape, scale)`
    /// parameterization.
    pub fn pdf_scale(&self, y: f64, shape: f64, scale: f64) -> f64 {
        self.pdf(y, shape, Self::scale_to_rate(scale))
    }

    /// CDF `F(y; shape, scale)` in the reference `(shape, scale)`
    /// parameterization.
    pub fn cdf_scale(&self, y: f64, shape: f64, scale: f64) -> f64 {
        self.cdf(y, shape, Self::scale_to_rate(scale))
    }

    /// Quantile `F^{-1}(p; shape, scale)` in the reference `(shape, scale)`
    /// parameterization.
    pub fn quantile_scale(&self, p: f64, shape: f64, scale: f64) -> f64 {
        self.quantile(p, shape, Self::scale_to_rate(scale))
    }

    /// The reference `gamma_means` MLE mean `shape · scale`.
    ///
    /// In the `(shape, scale)` parameterization the gamma mean is
    /// `E[Y] = shape · scale` (reference `gamma_libs.py::gamma_means`).
    #[inline]
    pub fn mean_scale(&self, shape: f64, scale: f64) -> f64 {
        shape * scale
    }
}

/// The gamma distribution model in the **reference `(shape, scale)`
/// parameterization** (Phase 5, Issue 5).
///
/// The reference `fitdistcp` gamma module (`100a_gamma.R` / `100c_gamma_derivs.R`)
/// parameterizes by `(shape, scale)` with `scale = 1/rate`, and computes the DMGS
/// expansion in that parameterization. The DMGS first-order truncation is **not
/// invariant under nonlinear reparameterization**, so computing the expansion in
/// `(shape, rate)` (as the core [`Gamma`] model does) diverges from the reference
/// by ~9% at `n = 30` (the documented Issue 5 root cause).
///
/// This wrapper presents the gamma in `(shape, scale)` so the existing
/// [`crate::dmgs::DmgsIntegrator`] machinery computes the expansion in the
/// reference's parameterization. It supplies the reference's analytic
/// `(shape, scale)` derivatives (`gamma_logfdd` / `gamma_logfddd` /
/// `gamma_fd` / `gamma_fdd` / `gamma_pd` / `gamma_pdd`), so the expansion matches
/// the reference to the DMGS tolerance.
///
/// The binding still **fits** in `(shape, rate)` (the core [`Gamma`] model); it
/// converts `scale = 1/rate` at the boundary and runs the CP expansion through
/// this model, exactly as it already does for the density.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GammaScale;

impl GammaScale {
    /// Create a `(shape, scale)` gamma model.
    pub fn new() -> Self {
        GammaScale
    }

    /// Log-density `ln f(y; shape, scale)` in the reference parameterization.
    pub fn logpdf(&self, y: f64, shape: f64, scale: f64) -> f64 {
        Gamma.logpdf(y, shape, 1.0 / scale)
    }

    /// Density `f(y; shape, scale)` in the reference parameterization.
    pub fn pdf(&self, y: f64, shape: f64, scale: f64) -> f64 {
        Gamma.pdf(y, shape, 1.0 / scale)
    }

    /// CDF `F(y; shape, scale)` in the reference parameterization.
    pub fn cdf(&self, y: f64, shape: f64, scale: f64) -> f64 {
        Gamma.cdf(y, shape, 1.0 / scale)
    }

    /// Quantile `F^{-1}(p; shape, scale)` in the reference parameterization.
    pub fn quantile(&self, p: f64, shape: f64, scale: f64) -> f64 {
        Gamma.quantile(p, shape, 1.0 / scale)
    }
}

impl Derivatives for GammaScale {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let shape = params[0];
        let scale = params[1];
        // Guard the valid region (mirrors the core `Gamma` guard): the reference
        // `gamma_loglik` uses R's `dgamma`, which is `NaN`/`-Inf` outside
        // `shape > 0, scale > 0`.
        if !(shape.is_finite() && shape > 0.0 && scale.is_finite() && scale > 0.0) {
            return f64::NEG_INFINITY;
        }
        crate::models::par_sum(y, |yi| self.logpdf(yi, shape, scale))
    }

    fn n_params(&self) -> usize {
        2
    }

    /// `setics`-style initial conditions in `(shape, scale)`.
    ///
    /// Method-of-moments: `shape = (mean/sd)^2`, `scale = sd^2/mean`.
    fn setics(&self, y: &[f64], init: &[f64]) -> Vec<f64> {
        let mean = crate::models::sample_mean(y);
        let sd = crate::models::sample_std_ddof1(y);
        if mean > 0.0 && sd > 0.0 {
            let shape = (mean / sd).powi(2);
            let scale = (sd * sd) / mean;
            vec![shape, scale]
        } else {
            init.to_vec()
        }
    }

    /// Analytic score in `(shape, scale)` (reference `gamma_logfdd` summed).
    ///
    /// With `ln f = (shape-1)·ln y - y/scale - shape·ln(scale) - lnΓ(shape)`:
    /// `∂ℓ/∂shape = Σ ln y - n ln(scale) - n ψ(shape)`,
    /// `∂ℓ/∂scale = Σ y/scale² - n shape/scale`.
    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let scale = params[1];
        let n = y.len() as f64;
        let (s, l) = crate::models::par_reduce2(y, |yi| (yi, yi.ln()));
        vec![
            l - n * scale.ln() - n * digamma(shape),
            s / (scale * scale) - n * shape / scale,
        ]
    }

    /// Analytic Hessian in `(shape, scale)` (reference `gamma_logfdd` summed).
    ///
    /// `∂²ℓ/∂shape² = -n ψ₁(shape)`, `∂²ℓ/∂shape∂scale = -n/scale`,
    /// `∂²ℓ/∂scale² = -2 Σ y/scale³ + n shape/scale²`.
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let scale = params[1];
        let n = y.len() as f64;
        let s: f64 = y.iter().sum();
        let s2 = scale * scale;
        let s3 = s2 * scale;
        vec![
            -n * trigamma(shape),
            -n / scale,
            -n / scale,
            -2.0 * s / s3 + n * shape / s2,
        ]
    }

    /// Analytic third derivatives in `(shape, scale)` (reference
    /// `gamma_logfddd` summed).
    ///
    /// Row-major `2 × 2 × 2`, index `(i, j, k)` at `i*4 + j*2 + k`:
    /// `∂³ℓ/∂shape³ = -n ψ₂(shape)`, `∂³ℓ/∂shape²∂scale = 0`,
    /// `∂³ℓ/∂shape∂scale² = n/scale²`,
    /// `∂³ℓ/∂scale³ = 6 Σ y/scale⁴ - 2 n shape/scale³`.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let scale = params[1];
        let n = y.len() as f64;
        let s: f64 = y.iter().sum();
        let s2 = scale * scale;
        let s3 = s2 * scale;
        let s4 = s2 * s2;
        let d111 = -n * tetragamma(shape);
        let d112 = 0.0;
        let d122 = n / s2;
        let d222 = 6.0 * s / s4 - 2.0 * n * shape / s3;
        vec![
            d111, d112, // (0,0,0), (0,0,1)
            d112, d122, // (0,1,0), (0,1,1)
            d112, d122, // (1,0,0), (1,0,1)
            d122, d222, // (1,1,0), (1,1,1)
        ]
    }

    /// Analytic mixed derivatives `∂²ℓ/∂θ_i∂y_a` in `(shape, scale)`.
    ///
    /// `∂²ℓ/∂shape∂y_a = 1/y_a`, `∂²ℓ/∂scale∂y_a = 1/scale²`.
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let scale = params[1];
        let ny = y.len();
        let mut out = vec![0.0; 2 * ny];
        let (row0, row1) = out.split_at_mut(ny);
        crate::models::par_fill(row0, y, |ya| 1.0 / ya);
        crate::models::par_fill(row1, y, |_| 1.0 / (scale * scale));
        out
    }
}

/// Invert the regularized lower incomplete gamma `P(a, x) = p` for `x > 0`.
///
/// Uses a safeguarded Newton iteration (Newton with a bisection fallback) on
/// `f(x) = P(a, x) - p`, whose derivative is the gamma density
/// `f'(x) = x^(a-1) e^{-x} / Γ(a)`. The initial guess is the Wilson–Hilferty
/// normal approximation, which is accurate to a few percent and puts Newton in
/// its quadratic-convergence basin. The bracket `[lo, hi]` is widened until it
/// contains the root, and every Newton step is clamped to it; if a step does not
/// reduce `|f|`, a bisection step is taken instead. This is robust for all
/// `a > 0` and `p ∈ (0, 1)` and converges to near machine precision.
///
/// `a` is the shape and `p` the target probability. Returns `x ≥ 0`.
fn inverse_regularized_gamma(a: f64, p: f64) -> f64 {
    // Regularized lower incomplete gamma P(a, x) (argument first, shape second).
    let cdf = |x: f64| special::Gamma::inc_gamma(x, a).clamp(0.0, 1.0);

    // --- Initial guess: Wilson–Hilferty (normal approximation). ---
    // For a > 1 this is accurate; for a <= 1 we fall back to a small positive
    // seed and let the bracket/Newton refine it.
    let mut x0 = if a > 1.0 {
        // z = standard normal quantile of p (Acklam-style rational approx via
        // the inverse erf is overkill here; use the Wilson–Hilferty form with a
        // normal quantile from the model's own helper).
        let z = crate::models::norm::standard_normal_quantile(p);
        let t = 1.0 - 1.0 / (9.0 * a) + z / (3.0 * a.sqrt());
        if t > 0.0 {
            a * t * t * t
        } else {
            a
        }
    } else {
        // Small shape: P(a, x) ≈ x^a / Γ(a+1) for small x, so x ≈ (p Γ(a+1))^(1/a).
        let g = crate::special::gamma(a + 1.0);
        (p * g).powf(1.0 / a).max(1e-12)
    };
    if !x0.is_finite() || x0 <= 0.0 {
        x0 = a.max(1e-12);
    }

    // --- Bracket [lo, hi] containing the root. ---
    //
    // `P(0) = 0 < p`, so `lo = 0` is always a valid lower bound. We only need to
    // expand `hi` upward until `P(hi) >= p`.
    let mut lo = 0.0_f64;
    let mut hi = x0.max(1e-12);
    let mut guard = 0;
    while cdf(hi) < p {
        hi *= 2.0;
        guard += 1;
        if guard > 200 || hi.is_infinite() {
            return f64::INFINITY;
        }
    }

    // --- Safeguarded Newton. ---
    //
    // Invariant: `P(lo) < p <= P(hi)`. Each iteration tries a Newton step; if it
    // is invalid or leaves the bracket, or fails to reduce the residual, we take
    // a bisection step instead. This is globally convergent and quadratically
    // fast once inside the Newton basin.
    let mut x = x0.clamp(lo, hi);
    let mut fx = cdf(x) - p;
    for _ in 0..100 {
        if fx == 0.0 || (hi - lo) <= 1e-15 * (1.0 + hi.abs()) {
            break;
        }
        // Newton step using the gamma density as the derivative.
        let ln_d = (a - 1.0) * x.ln() - x - crate::special::ln_gamma(a);
        let d = ln_d.exp();
        let mut x_new = if d.is_finite() && d > 0.0 {
            x - fx / d
        } else {
            f64::NAN
        };
        // Safeguard: if the Newton step is invalid or leaves the bracket, bisect.
        if !x_new.is_finite() || x_new <= lo || x_new >= hi {
            x_new = 0.5 * (lo + hi);
        }
        let f_new = cdf(x_new) - p;
        // If the Newton step did not reduce the residual, fall back to bisection.
        if f_new.abs() >= fx.abs() {
            x_new = 0.5 * (lo + hi);
            let f_bis = cdf(x_new) - p;
            if f_bis < 0.0 {
                lo = x_new;
            } else {
                hi = x_new;
            }
            x = x_new;
            fx = f_bis;
        } else {
            if f_new < 0.0 {
                lo = x_new;
            } else {
                hi = x_new;
            }
            x = x_new;
            fx = f_new;
        }
    }
    // Return the last accepted iterate `x` (which satisfies `P(x) ≈ p`), not the
    // bracket midpoint: when the root is approached from above, `lo` can remain
    // at its initial `0` and the midpoint would be far from the root.
    x
}

impl Derivatives for Gamma {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        let shape = params[0];
        let rate = params[1];
        // Guard the valid parameter region (Phase 3, Issue 3). The reference
        // `gamma_loglik` uses R's `dgamma`, which returns `NaN`/`-Inf` for
        // `shape <= 0` or `rate <= 0`. The Rust `ln_gamma` is defined for
        // negative non-integer arguments via reflection, so without this guard a
        // degenerate negative-shape region can carry a *finite* log-likelihood
        // and win the multi-start selection, corrupting downstream CP quantiles
        // (the `special` incomplete-gamma routine asserts `shape > 0`). Returning
        // `-inf` here keeps the optimizer inside the valid region.
        if !(shape.is_finite() && shape > 0.0 && rate.is_finite() && rate > 0.0) {
            return f64::NEG_INFINITY;
        }
        // W17 (H4/H6): gated parallel reduction over observations.
        crate::models::par_sum(y, |yi| self.logpdf(yi, shape, rate))
    }

    fn n_params(&self) -> usize {
        2
    }

    /// `setics`-style initial conditions (Phase 3, Issue 3).
    ///
    /// The reference seeds the gamma `optim` from `c(1, 1)`; the Rust
    /// multi-start strategy adds a **moment-based** start: for the
    /// `(shape, rate)` parameterization the method-of-moments estimates are
    /// `shape = (mean/sd)^2` and `rate = mean/sd^2`. This puts the optimizer on
    /// the data's scale instead of the unit scale.
    fn setics(&self, y: &[f64], init: &[f64]) -> Vec<f64> {
        let mean = crate::models::sample_mean(y);
        let sd = crate::models::sample_std_ddof1(y);
        if mean > 0.0 && sd > 0.0 {
            let shape = (mean / sd).powi(2);
            let rate = mean / (sd * sd);
            vec![shape, rate]
        } else {
            init.to_vec()
        }
    }

    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let rate = params[1];
        let n = y.len() as f64;
        // W17 (H4/H6): single gated pass computing Σy and Σln y.
        let (s, l) = crate::models::par_reduce2(y, |yi| (yi, yi.ln()));
        vec![n * rate.ln() - n * digamma(shape) + l, n * shape / rate - s]
    }

    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let rate = params[1];
        let n = y.len() as f64;
        vec![
            -n * trigamma(shape),
            n / rate,
            n / rate,
            -n * shape / (rate * rate),
        ]
    }

    /// Analytic third derivatives (W9).
    ///
    /// Row-major `2 × 2 × 2`, index `(i, j, k)` at `i*4 + j*2 + k`:
    /// `∂³ℓ/∂shape³ = -n ψ₂(shape)`, `∂³ℓ/∂shape²∂rate = 0`,
    /// `∂³ℓ/∂shape∂rate² = -n/rate²`, `∂³ℓ/∂rate³ = 2n shape/rate³`.
    ///
    /// NOTE: the reference `gamma_logfddd` is expressed in the `(shape, scale)`
    /// parameterization, where `∂³ℓ/∂shape∂scale² = +1/scale²`. In the Rust
    /// `(shape, rate)` parameterization the sign flips (the chain rule through
    /// `scale = 1/rate`), giving `∂³ℓ/∂shape∂rate² = -n/rate²`. We follow the
    /// Rust parameterization.
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let shape = params[0];
        let rate = params[1];
        let n = y.len() as f64;
        let r2 = rate * rate;
        let r3 = r2 * rate;
        let d111 = -n * tetragamma(shape);
        let d112 = 0.0;
        let d122 = -n / r2;
        let d222 = 2.0 * n * shape / r3;
        vec![
            d111, d112, // (0,0,0), (0,0,1)
            d112, d122, // (0,1,0), (0,1,1)
            d112, d122, // (1,0,0), (1,0,1)
            d122, d222, // (1,1,0), (1,1,1)
        ]
    }

    /// Analytic mixed derivatives `∂²ℓ/∂θ_i∂y_a` (W9).
    ///
    /// Row-major `2 × n`: `∂²ℓ/∂shape∂y_a = 1/y_a` and
    /// `∂²ℓ/∂rate∂y_a = -1`.
    fn mixed_derivatives(&self, _params: &[f64], y: &[f64]) -> Vec<f64> {
        let ny = y.len();
        let mut out = vec![0.0; 2 * ny];
        // W17 (H3/H6): gated parallel fill of the two row-major rows.
        let (row0, row1) = out.split_at_mut(ny);
        crate::models::par_fill(row0, y, |ya| 1.0 / ya);
        crate::models::par_fill(row1, y, |_| -1.0);
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
        // Gamma(shape = 1, rate) == Exponential(rate).
        let m = Gamma;
        let rate = 2.0;
        for &y in &[0.3_f64, 1.0, 2.5] {
            assert_close(
                m.logpdf(y, 1.0, rate),
                rate.ln() - rate * y,
                "gamma=exp logpdf",
            );
            assert_close(m.cdf(y, 1.0, rate), -(-rate * y).exp_m1(), "gamma=exp cdf");
        }
    }

    #[test]
    fn cdf_matches_reference() {
        let m = Gamma;
        // F(1; 1, 1) = 1 - e^{-1}
        assert_close(m.cdf(1.0, 1.0, 1.0), 0.632_120_558_828_557_7, "cdf(1;1,1)");
        // F(2; 2, 1) = 1 - 3 e^{-2} ≈ 0.5939941502901616
        assert_close(m.cdf(2.0, 2.0, 1.0), 0.593_994_150_290_161_6, "cdf(2;2,1)");
    }

    #[test]
    fn quantile_matches_reference() {
        let m = Gamma;
        // q(0.5; 1, 1) = ln 2
        assert_close(m.quantile(0.5, 1.0, 1.0), 2.0_f64.ln(), "q(0.5;1,1)");
    }

    #[test]
    fn cdf_quantile_roundtrip() {
        let m = Gamma;
        for &p in &[0.01_f64, 0.25, 0.5, 0.75, 0.99] {
            let x = m.quantile(p, 2.5, 1.3);
            assert_close(m.cdf(x, 2.5, 1.3), p, "cdf(quantile(p))");
        }
    }

    #[test]
    fn analytic_score_matches_numerical_fallback() {
        let m = Gamma;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let params = [2.0, 1.3];
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
        let m = Gamma;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let fit = fit(&m, &y, &[1.0, 1.0]);
        assert!(fit.convergence.is_converged());
        let g = m.score(&fit.ml_params, &y);
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.abs() < 1e-6, "score[{i}] at MLE = {gi}");
        }
    }

    // --- A5: (shape, scale) parameterization --------------------------------

    #[test]
    fn scale_entry_points_match_rate_parameterization() {
        // The reference uses (shape, scale); the core uses (shape, rate) with
        // rate = 1/scale. The scale entry points must agree with the rate ones
        // evaluated at rate = 1/scale.
        let m = Gamma;
        let (shape, scale) = (2.5, 1.3);
        let rate = 1.0 / scale;
        for &y in &[0.3_f64, 1.0, 2.5] {
            assert_close(
                m.logpdf_scale(y, shape, scale),
                m.logpdf(y, shape, rate),
                "logpdf_scale",
            );
            assert_close(
                m.pdf_scale(y, shape, scale),
                m.pdf(y, shape, rate),
                "pdf_scale",
            );
            assert_close(
                m.cdf_scale(y, shape, scale),
                m.cdf(y, shape, rate),
                "cdf_scale",
            );
        }
        for &p in &[0.1_f64, 0.5, 0.9] {
            assert_close(
                m.quantile_scale(p, shape, scale),
                m.quantile(p, shape, rate),
                "quantile_scale",
            );
        }
    }

    #[test]
    fn scale_to_rate_is_reciprocal() {
        assert_close(Gamma::scale_to_rate(2.0), 0.5, "scale_to_rate(2)");
        assert_close(Gamma::scale_to_rate(0.25), 4.0, "scale_to_rate(0.25)");
    }

    #[test]
    fn mean_scale_matches_reference() {
        // Reference gamma_means: ml_mean = shape * scale.
        let m = Gamma;
        assert_close(m.mean_scale(2.5, 1.3), 3.25, "mean_scale");
    }

    // --- Phase 5 (Issue 5): (shape, scale) DMGS model ------------------------

    #[test]
    fn gamma_scale_matches_rate_parameterization() {
        // The (shape, scale) model must describe the same distribution as the
        // (shape, rate) model with rate = 1/scale.
        let m = GammaScale;
        let (shape, scale) = (2.5, 1.3);
        let rate = 1.0 / scale;
        for &y in &[0.3_f64, 1.0, 2.5] {
            assert_close(
                m.logpdf(y, shape, scale),
                Gamma.logpdf(y, shape, rate),
                "gamma_scale logpdf",
            );
            assert_close(
                m.pdf(y, shape, scale),
                Gamma.pdf(y, shape, rate),
                "gamma_scale pdf",
            );
            assert_close(
                m.cdf(y, shape, scale),
                Gamma.cdf(y, shape, rate),
                "gamma_scale cdf",
            );
        }
        for &p in &[0.1_f64, 0.5, 0.9] {
            assert_close(
                m.quantile(p, shape, scale),
                Gamma.quantile(p, shape, rate),
                "gamma_scale quantile",
            );
        }
    }

    #[test]
    fn gamma_scale_score_matches_numerical_fallback() {
        let m = GammaScale;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let params = [2.0, 1.3];
        let ana = m.score(&params, &y);
        let num = crate::models::numerical_score(
            &|p: &[f64]| m.loglik(p, &y),
            &params,
            &crate::models::FiniteDiffConfig::default(),
        );
        for (i, (a, n)) in ana.iter().zip(num.iter()).enumerate() {
            let tol = 1e-6 + 1e-6 * n.abs();
            assert!((a - n).abs() <= tol, "gamma_scale score[{i}]: {a} vs {n}");
        }
    }

    #[test]
    fn gamma_scale_hessian_matches_numerical_fallback() {
        let m = GammaScale;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let params = [2.0, 1.3];
        let ana = m.hessian(&params, &y);
        let num = crate::models::numerical_hessian(
            &|p: &[f64]| m.loglik(p, &y),
            &params,
            &crate::models::FiniteDiffConfig::default(),
        );
        for (i, (a, n)) in ana.iter().zip(num.iter()).enumerate() {
            let tol = 1e-5 + 1e-5 * n.abs();
            assert!((a - n).abs() <= tol, "gamma_scale hessian[{i}]: {a} vs {n}");
        }
    }

    #[test]
    fn gamma_scale_third_derivatives_match_numerical_fallback() {
        let m = GammaScale;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let params = [2.0, 1.3];
        let ana = m.third_derivatives(&params, &y);
        let num = crate::models::numerical_third_derivatives(
            &|p: &[f64]| m.loglik(p, &y),
            &params,
            &crate::models::FiniteDiffConfig::default(),
        );
        for (i, (a, n)) in ana.iter().zip(num.iter()).enumerate() {
            let tol = 1e-3 + 1e-3 * n.abs();
            assert!((a - n).abs() <= tol, "gamma_scale third[{i}]: {a} vs {n}");
        }
    }

    #[test]
    fn gamma_scale_fits_synthetic_data() {
        // The (shape, scale) model is used for the DMGS expansion, not for
        // fitting (the binding fits with the core `Gamma` model). The meaningful
        // check is that the optimizer reaches the same stationary point: the
        // score vanishes at the returned parameters, and the (shape, scale)
        // optimum matches the (shape, rate) optimum converted.
        let m = GammaScale;
        let y = vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5];
        let scale_fit = fit(&m, &y, &[1.0, 1.0]);
        let g = m.score(&scale_fit.ml_params, &y);
        for (i, gi) in g.iter().enumerate() {
            assert!(gi.abs() < 1e-6, "gamma_scale score[{i}] at MLE = {gi}");
        }
        // The (shape, rate) fit converted to (shape, scale) must agree to the
        // optimizer tolerance (both reach the same stationary point; the
        // (shape, scale) surface is flatter so the optimizer stops slightly
        // earlier).
        let rate_fit = fit(&Gamma, &y, &[1.0, 1.0]);
        let shape_rel = (scale_fit.ml_params[0] - rate_fit.ml_params[0]).abs()
            / rate_fit.ml_params[0].abs();
        let scale_rel = (scale_fit.ml_params[1] - 1.0 / rate_fit.ml_params[1]).abs()
            / (1.0 / rate_fit.ml_params[1]).abs();
        assert!(
            shape_rel < 1e-4,
            "gamma_scale shape vs rate fit: rel = {shape_rel:.3e}"
        );
        assert!(
            scale_rel < 1e-4,
            "gamma_scale scale vs rate fit: rel = {scale_rel:.3e}"
        );
    }
}
