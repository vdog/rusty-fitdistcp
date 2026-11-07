//! `fitdistcp-core` — binding-agnostic numerics core.
//!
//! This crate must NOT depend on `extendr-api` or `pyo3`. Both binding crates
//! (`fitdistcp-r`, `fitdistcp-py`) are thin adapters over the API defined here so
//! that numerics are identical across R and Python.
//!
//! Phase 0 provides a trivial entry point used to prove that both bindings can
//! call into the shared core and return identical results. Phase 1a adds the
//! shared API structs, the module skeleton, and the integration/derivative
//! traits (interfaces only).

pub mod api;
pub mod dmgs;
pub mod mle;
pub mod models;
pub mod special;

// Re-export the key shared API types so bindings can `use fitdistcp_core::...`.
pub use api::{predict, predict_gev, Convergence, FitResult, MleFit, PredictionRequest};
// Re-export the interface traits defined in the module skeleton.
pub use dmgs::{dmgs_cp_pdf, dmgs_cp_quantiles, DmgsIntegrator, DmgsModel, Integrator};
// Derivative interface + analytic-first / numerical-fallback machinery (§4.1).
pub use models::{
    numerical_hessian, numerical_mixed_derivatives, numerical_score, numerical_third_derivatives,
    AnalyticDerivatives, DerivativeMode, Derivatives, FiniteDiffConfig,
};
// Phase 1 distribution models (§3).
pub use models::{
    Exponential, Gamma, GenExtreme, GenExtremeP1, GenPareto, Gumbel, LogNormal, Normal, NormalP1,
    Weibull,
};
// W10: negative-shape GEV helper (reference `movexiawayfromzero`).
pub use models::genextreme::movexiawayfromzero;

/// Version string of the core crate, surfaced through both bindings.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Trivial Phase 0 smoke-test function.
///
/// Returns the element-wise sum of two equal-length slices. This exists purely
/// to validate the FFI path (R and Python must both return the same values).
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
pub fn add_vectors(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(
        a.len(),
        b.len(),
        "add_vectors: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Trivial Phase 0 smoke-test function returning a scalar.
///
/// Computes the sum of a slice, used to confirm scalar return marshalling.
pub fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_vectors_works() {
        assert_eq!(
            add_vectors(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            vec![5.0, 7.0, 9.0]
        );
    }

    #[test]
    fn sum_works() {
        assert_eq!(sum(&[1.0, 2.0, 3.0, 4.0]), 10.0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn add_vectors_length_mismatch_panics() {
        let _ = add_vectors(&[1.0], &[1.0, 2.0]);
    }
}
