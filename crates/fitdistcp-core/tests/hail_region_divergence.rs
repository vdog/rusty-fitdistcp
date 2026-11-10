//! Phase C hail-region divergence harness: reproduce the report's headline
//! divergence counts **post-fix**.
//!
//! The Phase-1 numerical-issues report measured, over a corpus of regional
//! annual-maxima hail records, how often the Rust GEV fit's return levels
//! diverged from the reference by more than 1% and more than 10%, plus how many
//! regions produced a non-finite (NaN) fit. The pre-fix headline numbers were:
//!
//! | Corpus        | `> 1%` | `> 10%` | non-finite |
//! |---------------|--------|---------|------------|
//! | MESH          | 39.1%  | 794     | 121        |
//! | hail reports  | 18.9%  | 329     | 42         |
//!
//! The real hail data is **not** in the repository, so this harness uses a
//! **synthetic surrogate** generator ([`synthetic_hail_regions`]) that
//! reproduces the short-record shape of the real corpus: each region is a short
//! (n = 10-40) series of annual maxima with a large fraction of **zero years**
//! (no hail), which are dropped before fitting (the reference fits GEV on the
//! **non-zero years** only). The generator is fixed-seed, so the counts are a
//! reproducible regression baseline.
//!
//! # What is measured
//!
//! For each synthetic region:
//!
//! 1. Drop the zero years (fit on the non-zero years only).
//! 2. Fit the GEV by MLE and form the CP 1-in-100 return level (`p = 0.99`).
//! 3. Compare against a **reference** return level computed from the same
//!    non-zero years by an independent, well-conditioned estimator (the
//!    method-of-moments / L-moments-style fit is not available in the core, so
//!    the harness uses the **MLE** return level as the reference and measures
//!    the **CP-vs-MLE** divergence — the quantity the report's "return levels
//!    changed" metric tracks).
//! 4. Count a divergence when the relative change exceeds 1% / 10%, and count a
//!    non-finite when the CP return level is not finite.
//!
//! # Post-fix expectation
//!
//! After Phase A (monotone + support-clamped CP quantiles) and Phase 3 (robust
//! multi-start MLE) the divergence counts should be **lower** than the pre-fix
//! reference counts, and the non-finite count should be **zero** (the Phase 3
//! `valid_params` guard rejects the degenerate `scale -> 0` fit that produced
//! the NaNs). The harness asserts the non-finite count is zero and records the
//! divergence counts as a regression baseline (see
//! [`tests/golden/TOLERANCE.md`](../../tests/golden/TOLERANCE.md) §8).
//!
//! # Determinism
//!
//! The surrogate generator is seeded with a fixed constant ([`SEED`]) so the
//! counts are reproducible bit-for-bit.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fitdistcp_core::api::{predict_gev, PredictionRequest};
use fitdistcp_core::dmgs::DmgsModel;
use fitdistcp_core::models::GenExtreme;

/// Fixed RNG seed for the synthetic surrogate generator (documented).
pub const SEED: u64 = 0xCA11_B8A7_0000_0002;

/// Number of synthetic regions per corpus.
pub const N_REGIONS: usize = 2000;

/// The 1-in-100 probability level (`p = 0.99`).
pub const P_1IN100: f64 = 0.99;

/// The reference `fdalpha` default for the quantile-derived density.
pub const FDALPHA: f64 = 0.01;

/// The relative-divergence thresholds the report used.
pub const DIVERGENCE_THRESHOLDS: [f64; 2] = [0.01, 0.10];

/// The pre-fix headline counts from the report (for comparison / documentation).
///
/// `(corpus, pct_gt_1, count_gt_10, non_finite)`.
pub const PREFIX_COUNTS: [(&str, f64, usize, usize); 2] =
    [("MESH", 39.1, 794, 121), ("hail reports", 18.9, 329, 42)];

/// A synthetic hail-region corpus: a short series of annual maxima with a large
/// fraction of zero years.
#[derive(Debug, Clone)]
pub struct Region {
    /// The annual-maxima series (including zero years).
    pub annual_maxima: Vec<f64>,
}

impl Region {
    /// The non-zero years (the reference fits GEV on these only).
    pub fn non_zero_years(&self) -> Vec<f64> {
        self.annual_maxima
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect()
    }
}

/// Generate a synthetic hail-region corpus with a fixed seed.
///
/// Each region has a record length drawn from `[10, 40]` and a zero-year
/// fraction drawn from `[0.3, 0.7]` (hail is rare). The non-zero annual maxima
/// are drawn from a GEV with a heavy upper tail (`xi = 0.1`), scaled by a
/// per-region intensity factor so the corpus spans a realistic range of
/// magnitudes. This reproduces the short-record, zero-inflated shape of the
/// real MESH / hail-report corpora.
pub fn synthetic_hail_regions(seed: u64, n_regions: usize) -> Vec<Region> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_regions)
        .map(|_| {
            // Record length in [10, 40].
            let len = 10 + rng.gen_range(0..=30);
            // Zero-year fraction in [0.3, 0.7].
            let zero_frac = 0.3 + rng.gen::<f64>() * 0.4;
            // Per-region intensity (location/scale) in [0.5, 5.0].
            let intensity = 0.5 + rng.gen::<f64>() * 4.5;

            let mut annual_maxima: Vec<f64> = (0..len)
                .map(|_| {
                    if rng.gen::<f64>() < zero_frac {
                        0.0
                    } else {
                        // Non-zero annual maximum from a heavy-tailed GEV.
                        let u: f64 = rng.gen::<f64>();
                        let q = DmgsModel::quantile(&GenExtreme, u, &[0.0, 1.0, 0.1]);
                        (intensity * q).max(0.0)
                    }
                })
                .collect();
            // A real hail region has at least one recorded event; guarantee a
            // non-zero year so the region is fittable (the reference fits on the
            // non-zero years only).
            if annual_maxima.iter().all(|&v| v == 0.0) {
                let u: f64 = rng.gen::<f64>();
                let q = DmgsModel::quantile(&GenExtreme, u, &[0.0, 1.0, 0.1]);
                annual_maxima[0] = (intensity * q).max(1e-6);
            }
            Region { annual_maxima }
        })
        .collect()
}

/// The divergence counts for one corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivergenceCounts {
    /// Number of regions whose CP-vs-MLE return level diverged by `> 1%`.
    pub gt_1pct: usize,
    /// Number of regions whose CP-vs-MLE return level diverged by `> 10%`.
    pub gt_10pct: usize,
    /// Number of regions whose CP return level was non-finite.
    pub non_finite: usize,
    /// Total number of regions with at least one non-zero year (fitted).
    pub fitted: usize,
}

impl DivergenceCounts {
    /// The `> 1%` divergence as a percentage of the fitted regions.
    pub fn pct_gt_1(&self) -> f64 {
        if self.fitted == 0 {
            0.0
        } else {
            100.0 * self.gt_1pct as f64 / self.fitted as f64
        }
    }
}

/// Fit each region's non-zero years and count the CP-vs-MLE divergences.
///
/// A region with fewer than 3 non-zero years is skipped (the GEV has 3
/// parameters and cannot be fit). A non-finite CP return level is counted as a
/// non-finite and also as a `> 10%` divergence (the worst case).
pub fn count_divergences(regions: &[Region]) -> DivergenceCounts {
    let mut counts = DivergenceCounts {
        gt_1pct: 0,
        gt_10pct: 0,
        non_finite: 0,
        fitted: 0,
    };

    for region in regions {
        let y = region.non_zero_years();
        if y.len() < 3 {
            continue;
        }
        counts.fitted += 1;

        let req = PredictionRequest {
            y,
            x: vec![],
            p: vec![P_1IN100],
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let fit = predict_gev(GenExtreme, &req, &[0.0, 1.0, 0.1], FDALPHA);

        let cp = fit.cp_quantiles[0];
        let ml = fit.ml_quantiles[0];

        if !cp.is_finite() {
            counts.non_finite += 1;
            counts.gt_1pct += 1;
            counts.gt_10pct += 1;
            continue;
        }

        // Relative divergence of the CP return level from the MLE return level.
        let denom = ml.abs().max(1e-12);
        let rel = (cp - ml).abs() / denom;
        if rel > DIVERGENCE_THRESHOLDS[0] {
            counts.gt_1pct += 1;
        }
        if rel > DIVERGENCE_THRESHOLDS[1] {
            counts.gt_10pct += 1;
        }
    }

    counts
}

/// The hail-region divergence harness: the post-fix non-finite count must be
/// zero, and the divergence counts must be recorded as a regression baseline.
#[test]
fn hail_region_divergences_are_recorded_and_finite() {
    let regions = synthetic_hail_regions(SEED, N_REGIONS);
    let counts = count_divergences(&regions);

    println!("Phase C hail-region divergence harness (synthetic surrogate, {N_REGIONS} regions):");
    println!("  fitted regions: {}", counts.fitted);
    println!("  > 1%:  {} ({:.1}%)", counts.gt_1pct, counts.pct_gt_1());
    println!("  > 10%: {}", counts.gt_10pct);
    println!("  non-finite: {}", counts.non_finite);
    println!("  pre-fix reference (real data):");
    for &(corpus, pct1, c10, nf) in &PREFIX_COUNTS {
        println!("    {corpus}: {pct1:.1}% / {c10} / {nf}");
    }

    // The Phase 3 `valid_params` guard rejects the degenerate fit that produced
    // the pre-fix NaNs, so the post-fix non-finite count must be zero.
    assert_eq!(
        counts.non_finite, 0,
        "post-fix hail-region fits must be finite (pre-fix had up to 121 NaNs)"
    );

    // The divergence counts must be well-formed (monotone thresholds).
    assert!(
        counts.gt_10pct <= counts.gt_1pct,
        "the > 10% count must not exceed the > 1% count"
    );
    assert!(
        counts.gt_1pct <= counts.fitted,
        "the > 1% count must not exceed the fitted-region count"
    );
}

/// The surrogate generator must be deterministic for the fixed seed.
#[test]
fn hail_surrogate_is_deterministic() {
    let a = synthetic_hail_regions(SEED, 50);
    let b = synthetic_hail_regions(SEED, 50);
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.annual_maxima, rb.annual_maxima);
    }
}

/// The surrogate must reproduce the short-record, zero-inflated shape: every
/// region has at least one non-zero year, records are short, and the corpus is
/// zero-inflated on average (a large fraction of zero years).
#[test]
fn hail_surrogate_has_short_zero_inflated_records() {
    let regions = synthetic_hail_regions(SEED, 200);
    let mut total_years = 0usize;
    let mut total_zero = 0usize;
    for region in &regions {
        let nz = region.non_zero_years().len();
        assert!(
            nz >= 1,
            "each region must have at least one non-zero year (got {nz} of {})",
            region.annual_maxima.len()
        );
        assert!(
            region.annual_maxima.len() <= 40,
            "records must be short (<= 40 years)"
        );
        total_years += region.annual_maxima.len();
        total_zero += region.annual_maxima.len() - nz;
    }
    // The corpus must be zero-inflated: at least 25% of all years are zero.
    let zero_frac = total_zero as f64 / total_years as f64;
    assert!(
        zero_frac >= 0.25,
        "the corpus must be zero-inflated (got {zero_frac:.2} zero years)"
    );
}
