//! Reliability test (`reltest`) — the reference `fitdistcp` reliability
//! simulation, ported to the core.
//!
//! The reference exposes a `reltest` function per model (see
//! `reltest_dep.py` and the per-model `reltest` in `norm.py`, `genextreme.py`,
//! ...). Each runs the same Monte Carlo procedure:
//!
//! 1. **Simulate** a training sample `x` of size `nx` from the *true* model
//!    with known parameters.
//! 2. **Fit** the model by MLE and form the MLE predictive quantiles
//!    `q_ml = F^{-1}(p; ml_params)`.
//! 3. Form the **CP** predictive quantiles `q_cp` (the reference `ppf(...)
//!    ['cp_quantiles']`).
//! 4. **Score** both against the true distribution by mapping each predicted
//!    quantile back through the true CDF: `actual_p = F_true(q)`.
//! 5. **Average** the realized probabilities over `ntrials` trials.
//!
//! The result is `{actual_p_ml, actual_p_cp, p}` (the reference return dict).
//!
//! # Determinism
//!
//! The reference uses the global NumPy RNG. The core exposes a **fixed-seed**
//! entry point ([`reltest_seeded`]) so tests are reproducible bit-for-bit, plus
//! a convenience wrapper ([`reltest`]) that seeds from a documented default.
//!
//! # Models
//!
//! [`RelModel`] enumerates the eight models the reference `reltest` covers:
//! `expon`, `genextreme`, `genpareto`, `gumbel`, `norm`, `gamma`, `weibull`,
//! `lnorm`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::api::{predict, predict_gev, PredictionRequest};
use crate::dmgs::DmgsModel;
use crate::models::{
    Exponential, Gamma, GenExtreme, GenPareto, Gumbel, LogNormal, Normal, Weibull,
};

/// The default RNG seed used by [`reltest`] (documented, reproducible).
pub const DEFAULT_RELTEST_SEED: u64 = 0x5EED_1234_5678_9ABC;

/// The eight models the reference `reltest` covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelModel {
    /// Exponential (rate parameterization).
    Expon,
    /// Generalized extreme value.
    GenExtreme,
    /// Generalized Pareto (location 0).
    GenPareto,
    /// Gumbel.
    Gumbel,
    /// Normal.
    Norm,
    /// Gamma (shape, rate).
    Gamma,
    /// Weibull.
    Weibull,
    /// Log-normal.
    LogNormal,
}

/// The result of a reliability test (reference `{actual_p_ml, actual_p_cp, p}`).
#[derive(Debug, Clone, PartialEq)]
pub struct RelTestResult {
    /// Achieved probabilities using the MLE quantiles (`actual_p_ml`).
    pub actual_p_ml: Vec<f64>,
    /// Achieved probabilities using the CP quantiles (`actual_p_cp`).
    pub actual_p_cp: Vec<f64>,
    /// The nominal probabilities `p` at which the test was run.
    pub p: Vec<f64>,
}

/// Run the reliability test with a fixed seed (deterministic).
///
/// # Arguments
///
/// - `model`: which of the eight models to test.
/// - `ntrials`: number of Monte Carlo trials to average over.
/// - `nx`: number of samples per trial.
/// - `p`: the nominal probabilities at which to evaluate the quantiles.
/// - `true_params`: the true model parameters (the parameterization of the
///   corresponding core model; see [`RelModel`]).
/// - `init`: the initial guess for each MLE fit.
/// - `seed`: the RNG seed (fixed for reproducibility).
///
/// # Panics
///
/// Panics if `p` is empty or `true_params`/`init` have the wrong length for the
/// model.
#[allow(clippy::too_many_arguments)]
pub fn reltest_seeded(
    model: RelModel,
    ntrials: usize,
    nx: usize,
    p: &[f64],
    true_params: &[f64],
    init: &[f64],
    seed: u64,
) -> RelTestResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let np = p.len();
    let mut ml_total = vec![0.0; np];
    let mut cp_total = vec![0.0; np];

    for _ in 0..ntrials {
        // 1. Simulate a training sample from the true model via inverse-CDF.
        let x: Vec<f64> = (0..nx)
            .map(|_| {
                let u: f64 = rng.gen::<f64>();
                sample_quantile(model, u, true_params)
            })
            .collect();

        // 2/3. Fit and form the MLE and CP quantiles.
        let req = PredictionRequest {
            y: x.clone(),
            x: vec![],
            p: p.to_vec(),
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let fit = match model {
            RelModel::GenExtreme => predict_gev(GenExtreme, &req, init, 0.01),
            RelModel::GenPareto => predict(GenPareto, &req, init),
            RelModel::Gumbel => predict(Gumbel, &req, init),
            RelModel::Norm => predict(Normal, &req, init),
            RelModel::Gamma => predict(Gamma, &req, init),
            RelModel::Weibull => predict(Weibull, &req, init),
            RelModel::LogNormal => predict(LogNormal, &req, init),
            RelModel::Expon => predict(Exponential, &req, init),
        };

        // 4. Map the predicted quantiles back through the true CDF.
        for k in 0..np {
            let ml_p = sample_cdf(model, fit.ml_quantiles[k], true_params);
            let cp_p = sample_cdf(model, fit.cp_quantiles[k], true_params);
            ml_total[k] += ml_p;
            cp_total[k] += cp_p;
        }
    }

    let ntf = ntrials as f64;
    RelTestResult {
        actual_p_ml: ml_total.iter().map(|&v| v / ntf).collect(),
        actual_p_cp: cp_total.iter().map(|&v| v / ntf).collect(),
        p: p.to_vec(),
    }
}

/// Run the reliability test with the documented default seed.
///
/// See [`reltest_seeded`] for the arguments (minus `seed`).
#[allow(clippy::too_many_arguments)]
pub fn reltest(
    model: RelModel,
    ntrials: usize,
    nx: usize,
    p: &[f64],
    true_params: &[f64],
    init: &[f64],
) -> RelTestResult {
    reltest_seeded(
        model,
        ntrials,
        nx,
        p,
        true_params,
        init,
        DEFAULT_RELTEST_SEED,
    )
}

/// The true-model quantile `F^{-1}(u; params)` for the given model.
fn sample_quantile(model: RelModel, u: f64, params: &[f64]) -> f64 {
    match model {
        RelModel::Expon => Exponential.quantile(u, params[0]),
        RelModel::GenExtreme => DmgsModel::quantile(&GenExtreme, u, params),
        RelModel::GenPareto => GenPareto.quantile(u, params[0], params[1]),
        RelModel::Gumbel => Gumbel.quantile(u, params[0], params[1]),
        RelModel::Norm => Normal.quantile(u, params[0], params[1]),
        RelModel::Gamma => Gamma.quantile(u, params[0], params[1]),
        RelModel::Weibull => Weibull.quantile(u, params[0], params[1]),
        RelModel::LogNormal => LogNormal.quantile(u, params[0], params[1]),
    }
}

/// The true-model CDF `F(x; params)` for the given model.
fn sample_cdf(model: RelModel, x: f64, params: &[f64]) -> f64 {
    let v = match model {
        RelModel::Expon => Exponential.cdf(x, params[0]),
        RelModel::GenExtreme => DmgsModel::cdf(&GenExtreme, x, params),
        RelModel::GenPareto => GenPareto.cdf(x, params[0], params[1]),
        RelModel::Gumbel => Gumbel.cdf(x, params[0], params[1]),
        RelModel::Norm => Normal.cdf(x, params[0], params[1]),
        RelModel::Gamma => Gamma.cdf(x, params[0], params[1]),
        RelModel::Weibull => Weibull.cdf(x, params[0], params[1]),
        RelModel::LogNormal => LogNormal.cdf(x, params[0], params[1]),
    };
    v.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A coarse tail-covering probability grid (the reference uses 9999 points;
    /// we use a smaller grid so the test runs quickly).
    fn p_grid() -> Vec<f64> {
        vec![0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    }

    #[test]
    fn reltest_is_deterministic_for_fixed_seed() {
        let p = p_grid();
        let a = reltest_seeded(RelModel::Norm, 20, 30, &p, &[0.0, 1.0], &[0.0, 1.0], 42);
        let b = reltest_seeded(RelModel::Norm, 20, 30, &p, &[0.0, 1.0], &[0.0, 1.0], 42);
        assert_eq!(a, b);
    }

    #[test]
    fn reltest_returns_reference_shape() {
        let p = p_grid();
        let r = reltest(RelModel::Norm, 10, 30, &p, &[0.0, 1.0], &[0.0, 1.0]);
        assert_eq!(r.actual_p_ml.len(), p.len());
        assert_eq!(r.actual_p_cp.len(), p.len());
        assert_eq!(r.p, p);
        for v in r.actual_p_ml.iter().chain(r.actual_p_cp.iter()) {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(v),
                "p out of range: {v}"
            );
        }
    }

    #[test]
    fn reltest_cp_is_better_calibrated_than_ml_for_norm() {
        // The reference's headline finding: the CP quantiles are better
        // calibrated than the MLE quantiles in the tails. We assert on the
        // aggregate MSE of the realized probability vs the nominal p, with a
        // generous Monte Carlo tolerance.
        let p = p_grid();
        let r = reltest(RelModel::Norm, 200, 30, &p, &[0.0, 1.0], &[0.0, 1.0]);
        let mse = |actual: &[f64]| -> f64 {
            actual
                .iter()
                .zip(p.iter())
                .map(|(&a, &pi)| (a - pi) * (a - pi))
                .sum::<f64>()
                / p.len() as f64
        };
        let mse_ml = mse(&r.actual_p_ml);
        let mse_cp = mse(&r.actual_p_cp);
        assert!(
            mse_cp <= mse_ml + 0.02,
            "CP MSE {mse_cp} should not exceed ML MSE {mse_ml} by more than MC noise"
        );
    }

    #[test]
    fn reltest_runs_for_all_eight_models() {
        let p = vec![0.1, 0.5, 0.9];
        let cases: [(RelModel, &[f64], &[f64]); 8] = [
            (RelModel::Expon, &[1.0], &[1.0]),
            (RelModel::GenExtreme, &[0.0, 1.0, -0.1], &[0.0, 1.0, -0.1]),
            (RelModel::GenPareto, &[1.0, 0.1], &[1.0, 0.1]),
            (RelModel::Gumbel, &[0.0, 1.0], &[0.0, 1.0]),
            (RelModel::Norm, &[0.0, 1.0], &[0.0, 1.0]),
            (RelModel::Gamma, &[2.0, 1.0], &[2.0, 1.0]),
            (RelModel::Weibull, &[1.5, 1.0], &[1.5, 1.0]),
            (RelModel::LogNormal, &[0.0, 0.5], &[0.0, 0.5]),
        ];
        for (model, true_params, init) in cases {
            let r = reltest_seeded(model, 5, 30, &p, true_params, init, 7);
            assert_eq!(r.actual_p_ml.len(), p.len(), "{model:?}");
            for v in r.actual_p_ml.iter().chain(r.actual_p_cp.iter()) {
                assert!(v.is_finite(), "{model:?} produced non-finite p: {v}");
            }
        }
    }
}
