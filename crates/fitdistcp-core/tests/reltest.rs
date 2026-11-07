//! W18 — `reltest` reliability simulation.
//!
//! This is the Rust port of the reference `fitdistcp` reliability simulation
//! (`tools/fitdistcp-ref/R/003a_reltest2.R`, `003b_reltest2_libs.R`,
//! `003c_reltest2_plot_libs.R`). The reference repeatedly:
//!
//! 1. **simulates** a training sample `xx` from a *known* model with known
//!    parameters (`reltest2_simulate`),
//! 2. **fits** the model and forms the MLE and CP (calibrating-prior)
//!    predictive quantiles (`reltest2_predict` → `qgev_cp` / `qgpd_k1_cp`),
//! 3. **scores** the predictions against the *true* distribution by mapping each
//!    predicted quantile back through the true CDF to an exceedance probability
//!    `ep = 1 - F_true(pred)` (`reltest2_makeep`), and
//! 4. **aggregates** the exceedance probabilities over many trials, comparing
//!    them to the nominal `alpha` (`reltest2_plot`).
//!
//! The reference's headline finding is that the **CP predictive quantiles are
//! better calibrated than the MLE quantiles**: the MLE quantiles systematically
//! under-cover the tails, so the mean squared deviation of the realized
//! exceedance probability from the nominal `alpha` is larger for the MLE than
//! for the CP.
//!
//! # What this test asserts
//!
//! Per the design (§7.2, "Reliability simulation (`reltest`)"): *"CP mean
//! squared error must be ≤ MLE MSE within Monte Carlo error over a fixed seed;
//! assert on the aggregate, not per-replicate."*
//!
//! We therefore run a fixed-seed Monte Carlo and assert on the **aggregate**
//! MSE of the realized exceedance probability against the nominal `alpha`:
//!
//! ```text
//! MSE_method = (1 / (ntrials · nalpha)) · Σ_trials Σ_alpha (ep_method - alpha)²
//! ```
//!
//! and require `MSE_cp <= MSE_mle + mc_tolerance`, where `mc_tolerance` is the
//! Monte Carlo standard error of the *difference* of the two MSEs (a paired
//! estimate, since both methods see the same simulated samples). This is the
//! "within Monte Carlo error" clause: the assertion is not that CP wins on every
//! replicate, but that the aggregate CP error does not exceed the MLE error by
//! more than sampling noise.
//!
//! # Seed and replicate count
//!
//! - **Seed:** `0x5EED_1234_5678_9ABC` (fixed, documented below) — the whole
//!   simulation is reproducible bit-for-bit.
//! - **Replicates:** `N_TRIALS = 400` training samples per model, each of size
//!   `NX = 50` (the reference's default `nx`). This is deliberately smaller than
//!   the reference's "5000 typically gives good results" so the test runs in a
//!   few seconds in CI, while still being large enough for a meaningful
//!   aggregate assertion (the paired MC standard error is reported in the test
//!   output).
//!
//! # Models
//!
//! The reference `reltest2` covers `gev`, `gpd_k1` and `gev_p1`. We port the two
//! homogeneous cases that the core exposes through the DMGS CP path:
//! **GEV** (`GenExtreme`) and **GPD** (`GenPareto`, the `gpd_k1` case with a
//! fixed location of 0). The predictor case `gev_p1` is exercised by the W14
//! predictor tests and is out of scope for this aggregate reliability check.

use fitdistcp_core::api::{predict, predict_gev, PredictionRequest};
use fitdistcp_core::dmgs::DmgsModel;
use fitdistcp_core::models::{GenExtreme, GenPareto};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma, StandardNormal};

/// Fixed RNG seed for the whole simulation (documented, reproducible).
const SEED: u64 = 0x5EED_1234_5678_9ABC;

/// Number of Monte Carlo training samples per model.
const N_TRIALS: usize = 400;

/// Training-sample size (the reference default `nx`).
const NX: usize = 50;

/// Nominal exceedance probabilities `alpha` at which reliability is measured.
///
/// The reference uses `alpha = seq(0.005, 0.995, 0.005)` (199 values). We use a
/// coarser but still tail-covering grid so the aggregate MSE is dominated by the
/// tails (where the MLE under-covers) without paying for 199 quantile
/// evaluations per trial.
const ALPHAS: [f64; 9] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

/// Aggregate reliability result for one method (MLE or CP).
#[derive(Debug, Clone)]
struct Reliability {
    /// Mean squared deviation of the realized exceedance probability from the
    /// nominal `alpha`, averaged over trials and `alpha` levels.
    mse: f64,
    /// Per-trial MSE values (used to form the paired MC standard error).
    per_trial: Vec<f64>,
}

/// Compute the aggregate reliability of a set of per-trial exceedance
/// probabilities.
///
/// `eps[trial][k]` is the realized exceedance probability for trial `trial` at
/// nominal level `ALPHAS[k]`. The per-trial MSE is the mean over `alpha` of
/// `(ep - alpha)²`; the aggregate MSE is the mean of the per-trial MSEs.
fn reliability(eps: &[Vec<f64>]) -> Reliability {
    let per_trial: Vec<f64> = eps
        .iter()
        .map(|row| {
            let s: f64 = row
                .iter()
                .zip(ALPHAS.iter())
                .map(|(&ep, &a)| (ep - a) * (ep - a))
                .sum();
            s / ALPHAS.len() as f64
        })
        .collect();
    let mse = per_trial.iter().sum::<f64>() / per_trial.len() as f64;
    Reliability { mse, per_trial }
}

/// Paired Monte Carlo standard error of the difference `MSE_cp - MSE_mle`.
///
/// Because both methods are evaluated on the *same* simulated samples, the
/// difference is paired and its standard error is `sd(d_t) / sqrt(T)` where
/// `d_t = mse_cp_t - mse_mle_t` is the per-trial difference.
fn paired_mc_se(mle: &Reliability, cp: &Reliability) -> f64 {
    let t = mle.per_trial.len() as f64;
    let diffs: Vec<f64> = mle
        .per_trial
        .iter()
        .zip(cp.per_trial.iter())
        .map(|(&m, &c)| c - m)
        .collect();
    let mean = diffs.iter().sum::<f64>() / t;
    let var = diffs.iter().map(|&d| (d - mean) * (d - mean)).sum::<f64>() / (t - 1.0);
    (var / t).sqrt()
}

/// Map a predicted quantile back through the true CDF to an exceedance
/// probability `ep = 1 - F_true(pred)` (the reference `reltest2_makeep`).
fn exceedance<F: Fn(f64) -> f64>(pred: f64, true_cdf: F) -> f64 {
    (1.0 - true_cdf(pred)).clamp(0.0, 1.0)
}

// --- GEV reliability --------------------------------------------------------

/// Run the GEV reliability simulation and return `(mle, cp)` aggregates.
///
/// True model: `GEV(loc = 0, scale = 1, shape = -0.1)` — a mildly heavy-tailed
/// negative-shape GEV, the regime the reference highlights (the MLE under-covers
/// the upper tail; the CP corrects it). The reference `reltest2_simulate`
/// resamples until the fitted shape is within `(-0.45, 0.45)`; we instead fix a
/// well-behaved true shape and skip any trial whose fit fails to converge, which
/// keeps the simulation deterministic and avoids the reference's rejection loop.
fn gev_reliability() -> (Reliability, Reliability) {
    let model = GenExtreme;
    let (loc, scale, shape) = (0.0_f64, 1.0_f64, -0.1_f64);
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut mle_eps: Vec<Vec<f64>> = Vec::with_capacity(N_TRIALS);
    let mut cp_eps: Vec<Vec<f64>> = Vec::with_capacity(N_TRIALS);

    let mut accepted = 0usize;
    while accepted < N_TRIALS {
        // Simulate a training sample from the true GEV via inverse-CDF sampling.
        // Call the `DmgsModel` trait method explicitly: `GenExtreme` also has an
        // inherent `quantile` with a scalar-parameter signature that would
        // otherwise shadow it.
        let y: Vec<f64> = (0..NX)
            .map(|_| {
                let u: f64 = rng.gen::<f64>();
                DmgsModel::quantile(&model, u, &[loc, scale, shape])
            })
            .collect();

        // Fit MLE + CP predictive quantiles at the nominal levels.
        let p: Vec<f64> = ALPHAS.iter().map(|&a| 1.0 - a).collect();
        let req = PredictionRequest {
            y: y.clone(),
            x: vec![],
            p,
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let fit = predict_gev(model, &req, &[loc, scale, shape], 0.01);
        if !fit.is_converged() {
            continue;
        }
        // Guard against a degenerate fit (the reference's shape rejection).
        if fit.ml_params[2].abs() >= 0.45 {
            continue;
        }

        let true_cdf = |x: f64| DmgsModel::cdf(&model, x, &[loc, scale, shape]);
        let mle_row: Vec<f64> = fit
            .ml_quantiles
            .iter()
            .map(|&q| exceedance(q, true_cdf))
            .collect();
        let cp_row: Vec<f64> = fit
            .cp_quantiles
            .iter()
            .map(|&q| exceedance(q, true_cdf))
            .collect();
        mle_eps.push(mle_row);
        cp_eps.push(cp_row);
        accepted += 1;
    }

    (reliability(&mle_eps), reliability(&cp_eps))
}

// --- GPD (gpd_k1) reliability ----------------------------------------------

/// Run the GPD reliability simulation and return `(mle, cp)` aggregates.
///
/// True model: `GPD(scale = 1, shape = 0.1)` with a fixed location of 0 (the
/// reference `gpd_k1` case). The core's `GenPareto` is the two-parameter
/// `(scale, shape)` model with location 0, matching `gpd_k1`.
fn gpd_reliability() -> (Reliability, Reliability) {
    let model = GenPareto;
    let (scale, shape) = (1.0_f64, 0.1_f64);
    let mut rng = StdRng::seed_from_u64(SEED ^ 0xABCD);

    let mut mle_eps: Vec<Vec<f64>> = Vec::with_capacity(N_TRIALS);
    let mut cp_eps: Vec<Vec<f64>> = Vec::with_capacity(N_TRIALS);

    let mut accepted = 0usize;
    while accepted < N_TRIALS {
        // Simulate from the true GPD via inverse-CDF sampling. Call the
        // `DmgsModel` trait method explicitly (see the GEV note above).
        let y: Vec<f64> = (0..NX)
            .map(|_| {
                let u: f64 = rng.gen::<f64>();
                DmgsModel::quantile(&model, u, &[scale, shape])
            })
            .collect();

        let p: Vec<f64> = ALPHAS.iter().map(|&a| 1.0 - a).collect();
        let req = PredictionRequest {
            y: y.clone(),
            x: vec![],
            p,
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let fit = predict(model, &req, &[scale, shape]);
        if !fit.is_converged() {
            continue;
        }
        if fit.ml_params[1].abs() >= 0.45 {
            continue;
        }

        let true_cdf = |x: f64| DmgsModel::cdf(&model, x, &[scale, shape]);
        let mle_row: Vec<f64> = fit
            .ml_quantiles
            .iter()
            .map(|&q| exceedance(q, true_cdf))
            .collect();
        let cp_row: Vec<f64> = fit
            .cp_quantiles
            .iter()
            .map(|&q| exceedance(q, true_cdf))
            .collect();
        mle_eps.push(mle_row);
        cp_eps.push(cp_row);
        accepted += 1;
    }

    (reliability(&mle_eps), reliability(&cp_eps))
}

// --- Aggregate assertion helper --------------------------------------------

/// Assert the design's aggregate reliability claim for one model:
/// `MSE_cp <= MSE_mle + mc_tolerance`, where `mc_tolerance` is the paired
/// Monte Carlo standard error of the difference.
fn assert_cp_beats_mle(model_name: &str, mle: &Reliability, cp: &Reliability) {
    let se = paired_mc_se(mle, cp);
    let diff = cp.mse - mle.mse;
    // "Within Monte Carlo error": allow the CP to be worse by up to one paired
    // standard error (a one-sided 1-sigma band). The reference's finding is that
    // the CP is *better* (diff < 0); the tolerance only guards against sampling
    // noise on a modest replicate count.
    let tol = se.max(1e-12);
    println!(
        "[reltest/{model_name}] MSE_mle = {:.6e}, MSE_cp = {:.6e}, \
         diff = {:.6e}, paired_MC_SE = {:.6e}, n_trials = {}",
        mle.mse,
        cp.mse,
        diff,
        se,
        mle.per_trial.len()
    );
    assert!(
        diff <= tol,
        "{model_name}: CP MSE ({:.6e}) exceeds MLE MSE ({:.6e}) by {:.6e}, \
         more than the paired Monte Carlo SE ({:.6e})",
        cp.mse,
        mle.mse,
        diff,
        se
    );
}

// --- Tests ------------------------------------------------------------------

/// GEV reliability: the CP predictive quantiles must be at least as well
/// calibrated as the MLE quantiles, in aggregate, within Monte Carlo error.
#[test]
fn reltest_gev_cp_beats_mle() {
    let (mle, cp) = gev_reliability();
    assert_cp_beats_mle("gev", &mle, &cp);
}

/// GPD (`gpd_k1`) reliability: same aggregate claim as the GEV case.
#[test]
fn reltest_gpd_cp_beats_mle() {
    let (mle, cp) = gpd_reliability();
    assert_cp_beats_mle("gpd_k1", &mle, &cp);
}

/// Sanity check on the simulation harness itself: the true-model quantiles,
/// mapped back through the true CDF, must reproduce the nominal `alpha` exactly
/// (zero exceedance error). This validates the `exceedance` mapping and the
/// inverse-CDF sampler independently of any fit.
#[test]
fn reltest_true_model_is_perfectly_calibrated() {
    let model = GenExtreme;
    let params = [0.0_f64, 1.0_f64, -0.1_f64];
    for &a in ALPHAS.iter() {
        let q = DmgsModel::quantile(&model, 1.0 - a, &params);
        let ep = exceedance(q, |x| DmgsModel::cdf(&model, x, &params));
        assert!(
            (ep - a).abs() < 1e-9,
            "true-model exceedance at alpha = {a}: got {ep}"
        );
    }
}

/// Determinism check: the fixed seed must make the simulation reproducible.
/// Running the GEV simulation twice must give bit-identical aggregate MSEs.
#[test]
fn reltest_is_deterministic_under_fixed_seed() {
    let (mle1, cp1) = gev_reliability();
    let (mle2, cp2) = gev_reliability();
    assert_eq!(mle1.mse.to_bits(), mle2.mse.to_bits());
    assert_eq!(cp1.mse.to_bits(), cp2.mse.to_bits());
}

/// The simulation must actually exercise the RNG and the sampler: the standard
/// normal and gamma samplers used by the harness are available and produce
/// finite draws. This guards against a silently broken dev-dependency wiring.
#[test]
fn reltest_rng_harness_is_wired() {
    let mut rng = StdRng::seed_from_u64(SEED);
    let z: f64 = StandardNormal.sample(&mut rng);
    assert!(z.is_finite());
    let g = Gamma::new(2.0, 1.0).expect("valid gamma");
    let v: f64 = g.sample(&mut rng);
    assert!(v.is_finite() && v > 0.0);
}
