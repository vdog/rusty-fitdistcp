//! Reference utility helpers ported from the Python `fitdistcp` package.
//!
//! This module collects the small, self-contained helpers that the reference
//! exposes in `utils.py` and that the bindings need at the API boundary. They
//! are kept in the core (rather than in a binding) so that R and Python share
//! identical semantics.
//!
//! # Ported helpers
//!
//! - [`maket0`] — resolve the prediction point `t0` from either an explicit
//!   `t0` or a positional `n0` (reference `utils.py::maket0`).
//! - [`make_maic`] — the 3-vector MAIC `[ml_value, -nparams, ml_value - nparams]`
//!   (reference `utils.py::make_maic`).
//! - [`movexiawayfromzero`] — re-exported from [`crate::models::genextreme`] for
//!   convenience (reference `utils.py::movexiawayfromzero`).
//!
//! # Error semantics
//!
//! The reference `maket0` **raises** when both or neither of `t0`/`n0` are
//! supplied. The core helper mirrors that with a [`Result`] so the binding can
//! surface a proper error instead of silently returning `0.0`.

use crate::models::genextreme::movexiawayfromzero;

/// Error returned by [`maket0`] when the `t0`/`n0` arguments are inconsistent.
///
/// The reference `utils.py::maket0` raises a Python `Exception` with one of two
/// messages; we reproduce both as distinct variants so the binding can map them
/// to the reference text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maket0Error {
    /// Neither `t0` nor `n0` was supplied (reference: *"Either t0 or n0 must be
    /// specified"*).
    NeitherSpecified,
    /// Both `t0` and `n0` were supplied (reference: *"Only one of t0 or n0 must
    /// be specified"*).
    BothSpecified,
    /// `n0` was supplied but is out of range for `t` (reference would index out
    /// of bounds; we report it explicitly).
    N0OutOfRange,
}

impl std::fmt::Display for Maket0Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Maket0Error::NeitherSpecified => {
                write!(f, "Either t0 or n0 must be specified")
            }
            Maket0Error::BothSpecified => {
                write!(f, "Only one of t0 or n0 must be specified")
            }
            Maket0Error::N0OutOfRange => {
                write!(f, "n0 is out of range for t")
            }
        }
    }
}

impl std::error::Error for Maket0Error {}

/// Resolve the prediction point `t0` from either an explicit `t0` or a
/// positional `n0` (reference `utils.py::maket0`).
///
/// # Reference semantics
///
/// ```text
/// if (t0 is None or isnan(t0)) and (n0 is None or isnan(n0)):
///     raise "Either t0 or n0 must be specified"
/// if (t0 is not None and not isnan(t0)) and (n0 is not None and not isnan(n0)):
///     raise "Only one of t0 or n0 must be specified"
/// if t0 is None or isnan(t0):
///     t0 = t[n0 - 1]   # 0-based indexing
/// return t0
/// ```
///
/// In Rust the "absent" sentinel is `None`; a supplied `NaN` is treated as
/// absent, matching the reference's `np.isnan` checks.
///
/// # Arguments
///
/// - `t0`: the explicit prediction point, or `None`.
/// - `n0`: the positional index (1-based, as in the reference), or `None`.
/// - `t`: the predictor vector; indexed as `t[n0 - 1]` (0-based).
///
/// # Errors
///
/// - [`Maket0Error::NeitherSpecified`] if both `t0` and `n0` are absent.
/// - [`Maket0Error::BothSpecified`] if both are present.
/// - [`Maket0Error::N0OutOfRange`] if `n0` is present but `t[n0 - 1]` is out of
///   bounds (the reference would raise an `IndexError`).
pub fn maket0(t0: Option<f64>, n0: Option<f64>, t: &[f64]) -> Result<f64, Maket0Error> {
    // Treat a supplied NaN as absent (reference `np.isnan`).
    let t0_present = matches!(t0, Some(v) if !v.is_nan());
    let n0_present = matches!(n0, Some(v) if !v.is_nan());

    if !t0_present && !n0_present {
        return Err(Maket0Error::NeitherSpecified);
    }
    if t0_present && n0_present {
        return Err(Maket0Error::BothSpecified);
    }
    if t0_present {
        return Ok(t0.unwrap());
    }
    // n0 present: t0 = t[n0 - 1] (0-based).
    let n0v = n0.unwrap();
    if !n0v.is_finite() || n0v < 1.0 {
        return Err(Maket0Error::N0OutOfRange);
    }
    let idx = (n0v as usize) - 1;
    t.get(idx).copied().ok_or(Maket0Error::N0OutOfRange)
}

/// The 3-vector MAIC `[ml_value, -nparams, ml_value - nparams]`
/// (reference `utils.py::make_maic`).
///
/// The reference returns a length-3 vector whose third entry is the sum of the
/// first two. The core [`crate::api::FitResult`] stores only the sum
/// (`ml_value - nparams`), but this helper is provided so the binding can
/// reproduce the full reference vector.
pub fn make_maic(ml_value: f64, nparams: f64) -> [f64; 3] {
    [ml_value, -nparams, ml_value - nparams]
}

/// Re-export of [`movexiawayfromzero`] (reference `utils.py::movexiawayfromzero`).
///
/// Nudges a shape parameter that is too close to zero away from the `1/shape`
/// singularity, preserving its sign.
pub fn movexiawayfromzero_util(xi: f64) -> f64 {
    movexiawayfromzero(xi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maket0_explicit_t0_is_returned() {
        let t = vec![10.0, 20.0, 30.0];
        assert_eq!(maket0(Some(42.0), None, &t), Ok(42.0));
    }

    #[test]
    fn maket0_from_n0_uses_zero_based_index() {
        // Reference: t0 = t[n0 - 1] (0-based). n0 = 2 -> t[1] = 20.
        let t = vec![10.0, 20.0, 30.0];
        assert_eq!(maket0(None, Some(2.0), &t), Ok(20.0));
        assert_eq!(maket0(None, Some(1.0), &t), Ok(10.0));
        assert_eq!(maket0(None, Some(3.0), &t), Ok(30.0));
    }

    #[test]
    fn maket0_neither_specified_errors() {
        let t = vec![1.0, 2.0];
        assert_eq!(maket0(None, None, &t), Err(Maket0Error::NeitherSpecified));
        // NaN is treated as absent.
        assert_eq!(
            maket0(Some(f64::NAN), Some(f64::NAN), &t),
            Err(Maket0Error::NeitherSpecified)
        );
    }

    #[test]
    fn maket0_both_specified_errors() {
        let t = vec![1.0, 2.0];
        assert_eq!(
            maket0(Some(5.0), Some(1.0), &t),
            Err(Maket0Error::BothSpecified)
        );
    }

    #[test]
    fn maket0_n0_out_of_range_errors() {
        let t = vec![1.0, 2.0];
        assert_eq!(maket0(None, Some(5.0), &t), Err(Maket0Error::N0OutOfRange));
        assert_eq!(maket0(None, Some(0.0), &t), Err(Maket0Error::N0OutOfRange));
    }

    #[test]
    fn maket0_error_messages_match_reference() {
        assert_eq!(
            Maket0Error::NeitherSpecified.to_string(),
            "Either t0 or n0 must be specified"
        );
        assert_eq!(
            Maket0Error::BothSpecified.to_string(),
            "Only one of t0 or n0 must be specified"
        );
    }

    #[test]
    fn make_maic_matches_reference() {
        let m = make_maic(10.0, 2.0);
        assert_eq!(m, [10.0, -2.0, 8.0]);
    }

    #[test]
    fn movexiawayfromzero_util_delegates() {
        assert_eq!(movexiawayfromzero_util(0.0), 1e-7);
        assert_eq!(movexiawayfromzero_util(-0.5), -0.5);
    }
}
