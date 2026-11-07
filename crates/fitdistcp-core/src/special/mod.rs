//! Special-function wrappers (§4.3).
//!
//! This module provides a small, stable, binding-agnostic surface over the
//! special functions required by the DMGS expansion and the model likelihoods.
//! The wrappers are intentionally thin: they normalise the underlying crates'
//! APIs (which differ in shape — free functions vs. traits, plain `f64` vs.
//! `(value, sign)` tuples) into a single consistent set of free functions that
//! both the R and Python bindings can call identically.
//!
//! # Delegation map
//!
//! | Wrapper            | Delegates to                          | Rationale |
//! |--------------------|---------------------------------------|-----------|
//! | [`gamma`]          | [`statrs::function::gamma::gamma`]    | Plain `f64 -> f64` API |
//! | [`ln_gamma`]       | [`statrs::function::gamma::ln_gamma`] | Plain `f64 -> f64` API (the `special` crate returns a `(value, sign)` tuple) |
//! | [`digamma`]        | [`statrs::function::gamma::digamma`]  | Plain `f64 -> f64` API |
//! | [`trigamma`]       | [`special::Gamma::trigamma`]          | `statrs` does not expose trigamma |
//! | [`ln_beta`]        | [`special::Beta::ln_beta`]            | `special` exposes the log-beta directly |
//! | [`erf`] / [`erfc`] | [`statrs::function::erf`]             | Needed for the normal CDF/quantile path |
//!
//! # Domain and edge-case behaviour
//!
//! The wrappers do **not** add their own domain checks; they inherit the
//! behaviour of the underlying implementation, which matches the conventions of
//! R's `gamma`/`lgamma`/`digamma`/`trigamma` and SciPy's `scipy.special`:
//!
//! - **Poles at non-positive integers.** `gamma`, `digamma`, and `trigamma`
//!   have poles at `x = 0, -1, -2, …`. At an exact pole the result is `±inf`
//!   (or `NaN` where the sign is indeterminate); callers must guard against
//!   non-positive-integer arguments when a finite value is required.
//! - **`ln_gamma` for negative non-integers.** The true log-gamma is complex
//!   for `x < 0`; the real-valued convention (R/SciPy) returns
//!   `ln|Γ(x)|`. `statrs::ln_gamma` follows this convention.
//! - **Large `x`.** All four functions use asymptotic (Stirling-type)
//!   expansions for large arguments, so they remain accurate and overflow-free
//!   well beyond the point where `gamma(x)` itself overflows to `+inf`
//!   (`x ≳ 171.6`). Prefer [`ln_gamma`] over `ln(gamma(x))` for large `x`.
//! - **`NaN`/`±inf` inputs** propagate to `NaN`/`±inf` per IEEE-754.
//!
//! # Tolerance policy
//!
//! Reference values in the tests below are compared using the golden-test
//! tolerance from design §7.2: `|a - b| <= atol + rtol * |b|` with
//! `rtol = 1e-8` and `atol = 1e-10`.

// Re-export the underlying traits so downstream modules can call the trait
// methods directly if they need the raw `special` API (e.g. `inc_gamma`).
pub use special::{Beta, Gamma};

/// The natural logarithm of the absolute value of the gamma function,
/// `ln|Γ(x)|`.
///
/// Delegates to [`statrs::function::gamma::ln_gamma`].
///
/// # Domain / edge cases
/// - `x > 0`: the usual log-gamma.
/// - `x < 0` (non-integer): returns `ln|Γ(x)|` (real-valued convention).
/// - `x` a non-positive integer (pole): returns `+inf`.
/// - `x = +inf`: returns `+inf`; `x = NaN`: returns `NaN`.
///
/// Prefer this over `gamma(x).ln()` for large `x`, where `gamma(x)` overflows.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

/// The gamma function `Γ(x)`.
///
/// Delegates to [`statrs::function::gamma::gamma`].
///
/// # Domain / edge cases
/// - `x` a non-positive integer (pole): returns `±inf` (sign per the
///   underlying implementation).
/// - Large `x` (`x ≳ 171.6`): overflows to `+inf`; use [`ln_gamma`] instead.
/// - `x = +inf`: returns `+inf`; `x = NaN`: returns `NaN`.
#[inline]
pub fn gamma(x: f64) -> f64 {
    statrs::function::gamma::gamma(x)
}

/// The digamma function `ψ(x) = d/dx ln Γ(x) = Γ'(x) / Γ(x)`.
///
/// Delegates to [`statrs::function::gamma::digamma`].
///
/// # Domain / edge cases
/// - `x` a non-positive integer (pole): returns `±inf` / `NaN`.
/// - Large `x`: `ψ(x) ≈ ln(x) - 1/(2x) - …`.
/// - `x = +inf`: returns `+inf`; `x = NaN`: returns `NaN`.
#[inline]
pub fn digamma(x: f64) -> f64 {
    statrs::function::gamma::digamma(x)
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x) = d/dx ψ(x)`.
///
/// Delegates to [`special::Gamma::trigamma`] (the `statrs` crate does not
/// expose trigamma).
///
/// # Domain / edge cases
/// - `x` a non-positive integer (pole): returns `+inf`.
/// - Large `x`: `ψ₁(x) ≈ 1/x + 1/(2x²) + …`.
/// - `x = +inf`: returns `0`; `x = NaN`: returns `NaN`.
#[inline]
pub fn trigamma(x: f64) -> f64 {
    special::Gamma::trigamma(&x)
}

/// The tetragamma function `ψ₂(x) = d³/dx³ ln Γ(x) = d/dx ψ₁(x)`.
///
/// This is the reference's `psigamma(x, 2)`. Neither `statrs` nor `special`
/// exposes it, so it is implemented here via the recurrence
/// `ψ₂(x) = ψ₂(x + 1) + 2/x³` (used to shift the argument up to `x ≥ 8`) plus
/// the asymptotic expansion
/// `ψ₂(x) ≈ -1/x² - 1/x³ - 1/(2x⁴) + 1/(6x⁶) - 1/(6x⁸) + 1/(10x¹⁰) - …`.
///
/// # Domain / edge cases
/// - `x` a non-positive integer (pole): returns `-inf` (the sign is
///   indeterminate at the pole; callers must guard).
/// - Large `x`: `ψ₂(x) ≈ -1/x² - 1/x³ - …`.
/// - `x = +inf`: returns `0`; `x = NaN`: returns `NaN`.
pub fn tetragamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    // Shift the argument up to x >= 8 using psi2(x) = psi2(x+1) - 2/x^3
    // (the derivative of the trigamma recurrence psi1(x) = psi1(x+1) + 1/x^2).
    let mut xx = x;
    let mut acc = 0.0;
    while xx < 8.0 {
        acc -= 2.0 / (xx * xx * xx);
        xx += 1.0;
    }
    // Asymptotic expansion for large xx.
    let inv = 1.0 / xx;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    let inv8 = inv4 * inv4;
    let inv10 = inv8 * inv2;
    let asym = -inv2 - inv3 - 0.5 * inv4 + inv6 / 6.0 - inv8 / 6.0 + inv10 / 10.0;
    asym + acc
}

/// The natural logarithm of the beta function,
/// `ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b)`.
///
/// Delegates to [`special::Beta::ln_beta`].
///
/// # Domain / edge cases
/// - `a > 0` and `b > 0`: the usual log-beta.
/// - Non-positive integer arguments (poles): returns `+inf`.
/// - `NaN` inputs propagate to `NaN`.
#[inline]
pub fn ln_beta(a: f64, b: f64) -> f64 {
    special::Beta::ln_beta(a, b)
}

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
///
/// Delegates to [`statrs::function::erf::erf`]. Used by the normal
/// CDF/quantile path.
///
/// # Domain / edge cases
/// - `erf(0) = 0`, `erf(±inf) = ±1`, `erf(NaN) = NaN`.
#[inline]
pub fn erf(x: f64) -> f64 {
    statrs::function::erf::erf(x)
}

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Delegates to [`statrs::function::erf::erfc`]. Preferred over `1 - erf(x)`
/// in the tails, where it avoids catastrophic cancellation.
///
/// # Domain / edge cases
/// - `erfc(0) = 1`, `erfc(+inf) = 0`, `erfc(-inf) = 2`, `erfc(NaN) = NaN`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    statrs::function::erf::erfc(x)
}

/// The regularized incomplete beta function `I_x(a, b)`.
///
/// Delegates to [`special::Beta::inc_beta`]. Used by the Student-t CDF, which
/// the analytic `norm_p1` CP path needs (W14).
#[inline]
pub fn inc_beta(x: f64, a: f64, b: f64) -> f64 {
    special::Beta::inc_beta(x, a, b, ln_beta(a, b))
}

/// The Student-t density `f(t; ν)` with `ν` degrees of freedom.
///
/// `f(t) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + t²/ν)^{-(ν+1)/2}`.
///
/// Used by the analytic `norm_p1` CP density (the reference `dt`).
pub fn student_t_pdf(t: f64, nu: f64) -> f64 {
    let ln_c =
        ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln();
    (ln_c - 0.5 * (nu + 1.0) * (1.0 + t * t / nu).ln()).exp()
}

/// The Student-t CDF `F(t; ν)` with `ν` degrees of freedom.
///
/// For `t ≥ 0`: `F(t) = 1 - ½ I_{ν/(ν+t²)}(ν/2, ½)`;
/// for `t < 0`: `F(t) = ½ I_{ν/(ν+t²)}(ν/2, ½)`.
///
/// Used by the analytic `norm_p1` CP CDF (the reference `pt`).
pub fn student_t_cdf(t: f64, nu: f64) -> f64 {
    if t.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if t == f64::INFINITY {
        return 1.0;
    }
    if t == f64::NEG_INFINITY {
        return 0.0;
    }
    let x = nu / (nu + t * t);
    let ib = inc_beta(x, 0.5 * nu, 0.5);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// The Student-t quantile `F^{-1}(p; ν)` with `ν` degrees of freedom.
///
/// Computed by bisection on [`student_t_cdf`] over a bracket that widens until
/// it contains the root. The reference uses R's `qt`; bisection to a relative
/// tolerance of `1e-12` reproduces it to well within the §7.2 tolerance for the
/// small `ν = nx - 2` values the `norm_p1` CP path uses.
///
/// Used by the analytic `norm_p1` CP quantiles (the reference `qt`).
pub fn student_t_quantile(p: f64, nu: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    // Symmetry: F(-t) = 1 - F(t). Solve for the **negative** root `s` with
    // F(s) = min(p, 1-p) < 0.5, then reflect for the upper tail.
    let target = p.min(1.0 - p);
    // Bracket [lo, hi] with lo < s < 0; widen lo until F(lo) <= target.
    let mut lo = -1.0;
    while student_t_cdf(lo, nu) > target {
        lo *= 2.0;
        if lo < -1e12 {
            break;
        }
    }
    let mut hi = 0.0;
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, nu) < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= 1e-16 * (1.0 + hi.abs()) {
            break;
        }
    }
    let s = 0.5 * (lo + hi);
    if p > 0.5 {
        -s
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden-test tolerance from design §7.2: `|a - b| <= atol + rtol * |b|`.
    const RTOL: f64 = 1e-8;
    const ATOL: f64 = 1e-10;

    /// Assert `actual ≈ expected` under the §7.2 tolerance policy.
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

    // --- ln_gamma -----------------------------------------------------------

    #[test]
    fn ln_gamma_half_is_ln_sqrt_pi() {
        // lgamma(0.5) = ln(√π) ≈ 0.5723649429247001
        assert_close(ln_gamma(0.5), 0.572_364_942_924_700_1, "ln_gamma(0.5)");
    }

    #[test]
    fn ln_gamma_one_is_zero() {
        // lgamma(1) = ln(0!) = 0
        assert_close(ln_gamma(1.0), 0.0, "ln_gamma(1)");
    }

    #[test]
    fn ln_gamma_two_is_zero() {
        // lgamma(2) = ln(1!) = 0
        assert_close(ln_gamma(2.0), 0.0, "ln_gamma(2)");
    }

    #[test]
    fn ln_gamma_five_is_ln_24() {
        // lgamma(5) = ln(4!) = ln(24) ≈ 3.1780538303479458
        assert_close(ln_gamma(5.0), 24.0_f64.ln(), "ln_gamma(5)");
    }

    #[test]
    fn ln_gamma_large_x_matches_stirling_leading_term() {
        // For large x, lgamma(x) ≈ (x - 0.5) ln x - x + 0.5 ln(2π).
        let x = 1.0e6_f64;
        let stirling = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln();
        // Leading-order Stirling is only ~1e-6 accurate here, so use a looser
        // relative check than the §7.2 golden tolerance.
        let rel = ((ln_gamma(x) - stirling) / stirling).abs();
        assert!(rel < 1e-6, "ln_gamma large-x Stirling rel err = {rel:.3e}");
    }

    // --- gamma --------------------------------------------------------------

    #[test]
    fn gamma_half_is_sqrt_pi() {
        // Γ(0.5) = √π ≈ 1.7724538509055159
        assert_close(gamma(0.5), std::f64::consts::PI.sqrt(), "gamma(0.5)");
    }

    #[test]
    fn gamma_integers_are_factorials() {
        // Γ(n) = (n-1)!
        assert_close(gamma(1.0), 1.0, "gamma(1)");
        assert_close(gamma(2.0), 1.0, "gamma(2)");
        assert_close(gamma(5.0), 24.0, "gamma(5)");
        assert_close(gamma(6.0), 120.0, "gamma(6)");
    }

    #[test]
    fn gamma_consistent_with_ln_gamma() {
        // gamma(x) == exp(ln_gamma(x)) for moderate x.
        for &x in &[0.3_f64, 1.7, 3.2, 10.5, 50.0] {
            assert_close(gamma(x), ln_gamma(x).exp(), "gamma vs exp(ln_gamma)");
        }
    }

    #[test]
    fn gamma_pole_at_zero_is_infinite() {
        assert!(gamma(0.0).is_infinite(), "gamma(0) should be a pole");
    }

    // --- digamma ------------------------------------------------------------

    #[test]
    fn digamma_one_is_negative_euler_mascheroni() {
        // digamma(1) = -γ ≈ -0.5772156649015329
        assert_close(digamma(1.0), -0.577_215_664_901_532_9, "digamma(1)");
    }

    #[test]
    fn digamma_two_is_one_minus_euler_mascheroni() {
        // digamma(2) = 1 - γ ≈ 0.42278433509846713
        assert_close(digamma(2.0), 0.422_784_335_098_467_13, "digamma(2)");
    }

    #[test]
    fn digamma_half_is_minus_gamma_minus_2ln2() {
        // digamma(0.5) = -γ - 2 ln 2 ≈ -1.9635100260214235
        assert_close(digamma(0.5), -1.963_510_026_021_423_5, "digamma(0.5)");
    }

    #[test]
    fn digamma_large_x_approaches_ln_x() {
        // digamma(x) ≈ ln(x) - 1/(2x) for large x.
        let x = 1.0e8_f64;
        let approx = x.ln() - 1.0 / (2.0 * x);
        assert_close(digamma(x), approx, "digamma large-x asymptotic");
    }

    // --- trigamma -----------------------------------------------------------

    #[test]
    fn trigamma_one_is_pi_squared_over_six() {
        // trigamma(1) = π²/6 ≈ 1.6449340668482264
        assert_close(
            trigamma(1.0),
            std::f64::consts::PI.powi(2) / 6.0,
            "trigamma(1)",
        );
    }

    #[test]
    fn trigamma_two_is_pi_squared_over_six_minus_one() {
        // trigamma(2) = π²/6 - 1 ≈ 0.6449340668482264
        assert_close(
            trigamma(2.0),
            std::f64::consts::PI.powi(2) / 6.0 - 1.0,
            "trigamma(2)",
        );
    }

    #[test]
    fn trigamma_half_is_pi_squared_over_two() {
        // trigamma(0.5) = π²/2 ≈ 4.934802200544679
        assert_close(
            trigamma(0.5),
            std::f64::consts::PI.powi(2) / 2.0,
            "trigamma(0.5)",
        );
    }

    #[test]
    fn trigamma_large_x_approaches_reciprocal() {
        // trigamma(x) ≈ 1/x + 1/(2x²) for large x.
        let x = 1.0e8_f64;
        let approx = 1.0 / x + 1.0 / (2.0 * x * x);
        assert_close(trigamma(x), approx, "trigamma large-x asymptotic");
    }

    #[test]
    fn trigamma_is_derivative_of_digamma() {
        // Central-difference check: ψ₁(x) ≈ (ψ(x+h) - ψ(x-h)) / (2h).
        let h = 1e-5_f64;
        for &x in &[0.7_f64, 1.3, 2.5, 5.0] {
            let fd = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
            // Finite-difference error is O(h²) ≈ 1e-10; use a slightly looser
            // absolute tolerance than the §7.2 golden policy.
            assert!(
                (trigamma(x) - fd).abs() < 1e-7,
                "trigamma vs finite-difference at x = {x}: {} vs {}",
                trigamma(x),
                fd
            );
        }
    }

    // --- tetragamma ---------------------------------------------------------

    #[test]
    fn tetragamma_one_is_minus_twice_apery() {
        // ψ₂(1) = -2 ζ(3) ≈ -2.4041138063191885 (Apéry's constant ζ(3)).
        assert_close(tetragamma(1.0), -2.404_113_806_319_188_5, "tetragamma(1)");
    }

    #[test]
    fn tetragamma_two_is_minus_twice_apery_plus_two() {
        // ψ₂(2) = ψ₂(1) + 2 = -2 ζ(3) + 2 ≈ -0.4041138063191885.
        assert_close(tetragamma(2.0), -0.404_113_806_319_188_5, "tetragamma(2)");
    }

    #[test]
    fn tetragamma_is_derivative_of_trigamma() {
        // Central-difference check: ψ₂(x) ≈ (ψ₁(x+h) - ψ₁(x-h)) / (2h).
        let h = 1e-5_f64;
        for &x in &[0.7_f64, 1.3, 2.5, 5.0] {
            let fd = (trigamma(x + h) - trigamma(x - h)) / (2.0 * h);
            assert!(
                (tetragamma(x) - fd).abs() < 1e-6,
                "tetragamma vs finite-difference at x = {x}: {} vs {}",
                tetragamma(x),
                fd
            );
        }
    }

    #[test]
    fn tetragamma_large_x_approaches_asymptotic() {
        // ψ₂(x) ≈ -1/x² - 1/x³ - 1/(2x⁴) for large x.
        let x = 1.0e6_f64;
        let approx = -1.0 / (x * x) - 1.0 / (x * x * x) - 0.5 / (x * x * x * x);
        assert_close(tetragamma(x), approx, "tetragamma large-x asymptotic");
    }

    // --- ln_beta ------------------------------------------------------------

    #[test]
    fn ln_beta_symmetric_and_known_value() {
        // B(1,1) = 1  =>  ln B(1,1) = 0
        assert_close(ln_beta(1.0, 1.0), 0.0, "ln_beta(1,1)");
        // B(2,3) = Γ(2)Γ(3)/Γ(5) = 1*2/24 = 1/12
        assert_close(ln_beta(2.0, 3.0), (1.0_f64 / 12.0).ln(), "ln_beta(2,3)");
        // Symmetry: ln B(a,b) == ln B(b,a)
        assert_close(ln_beta(2.0, 3.0), ln_beta(3.0, 2.0), "ln_beta symmetry");
    }

    // --- erf / erfc ---------------------------------------------------------

    #[test]
    fn erf_known_values() {
        assert_close(erf(0.0), 0.0, "erf(0)");
        // erf(1) ≈ 0.8427007929497149
        assert_close(erf(1.0), 0.842_700_792_949_714_9, "erf(1)");
        // erf(-x) = -erf(x)
        assert_close(erf(-1.0), -0.842_700_792_949_714_9, "erf(-1)");
    }

    #[test]
    fn erfc_complements_erf() {
        for &x in &[-2.0_f64, -0.5, 0.0, 0.5, 2.0] {
            assert_close(erfc(x), 1.0 - erf(x), "erfc = 1 - erf");
        }
    }
}
