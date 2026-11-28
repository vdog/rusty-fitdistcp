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
//! # Issue #5: the `n >= 8` filter
//!
//! The maintainer's real-data evidence is restricted to regions with `>= 8`
//! non-zero years ("records of 2-7 points aren't fitted in practice"). The
//! harness therefore skips regions with fewer than [`MIN_NON_ZERO_YEARS`] (`8`)
//! non-zero years, so the pinned counts reflect the records actually fitted.
//! The **unfiltered** (n >= 3) huge-finite count (192) is still reproduced by
//! [`hail_region_unfiltered_huge_finite_count_is_192`] for the historical
//! record.
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
//! reference counts. The non-finite count is **zero**, but for two distinct
//! reasons:
//!
//! - the Phase 3 `valid_params` guard rejects the degenerate `scale -> 0` fit
//!   (item B), and
//! - the Issue #4 near-−1 `revert2ml` widening (`shape <= -0.96`) reverts the
//!   DMGS `1/fhat` blow-up at a shape just above −1 to the finite ML output
//!   (item C's GEV analogue). Before Issue #4 these 9 regions produced `NaN`.
//!
//! The harness asserts the non-finite count is zero and records the divergence
//! counts as a regression baseline (see
//! [`tests/golden/TOLERANCE.md`](../../tests/golden/TOLERANCE.md) §8).
//!
//! # Issue #5: the huge-finite count is reference-matching, not a defect
//!
//! The remaining huge-finite CP return levels (`|cp| > HUGE_CP_BOUND`) are
//! **not** a Rust defect. The maintainer verified that they **match the
//! reference**: every huge value has ML `xi >= 3` (median ~7.5), and the
//! reference also reverts to the ML quantile when `|xi| >= 1`, so the CP level
//! **is** the ML quantile — huge for both implementations (a GEV with `xi > 1`
//! has an extremely heavy tail). The count is therefore a **regression
//! baseline**, pinned to the measured value on the `n >= 8` regime. See
//! [`tests/golden/TOLERANCE.md`](../../tests/golden/TOLERANCE.md) §3.14b/§8.2
//! and [`crate::gev_huge_finite_golden`] for the reference-CP golden check.
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

/// The minimum number of **non-zero** years a region must have to be fitted.
///
/// Issue #5: the maintainer's real-data evidence is restricted to regions with
/// `>= 8` non-zero years ("records of 2-7 points aren't fitted in practice").
/// The surrogate is filtered to the same regime so the pinned huge-finite count
/// reflects the records that are actually fitted. The pre-filter (n >= 3) count
/// is still reported for the historical record.
pub const MIN_NON_ZERO_YEARS: usize = 8;

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
    /// Number of regions whose fit was flagged **degenerate** (item B2:
    /// `degenerate_fit == true`, i.e. every multi-start candidate failed the
    /// relative scale/location guard). Item B1 must drive this to zero on the
    /// corpus (the multi-start finds a valid interior candidate).
    pub degenerate: usize,
    /// Number of regions whose CP return level was a **huge finite** value
    /// (`|cp| > HUGE_CP_BOUND`). These are the DMGS small-`n` blow-up at a
    /// large **positive** shape (the GEV analogue of item C), **not** the
    /// item-B `scale -> 0` degeneracy (which is counted by `degenerate`).
    pub huge_finite: usize,
    /// Total number of regions with at least one non-zero year (fitted).
    pub fitted: usize,
    /// Number of regions skipped because they had fewer than
    /// [`MIN_NON_ZERO_YEARS`] non-zero years (Issue #5 filter).
    pub skipped: usize,
}

/// The bound above which a CP return level is considered a **huge finite**
/// value. A meaningful 1-in-100 return level for these records is `O(1)`–`O(10)`.
pub const HUGE_CP_BOUND: f64 = 1e6;

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
/// A region with fewer than [`MIN_NON_ZERO_YEARS`] non-zero years is skipped
/// (Issue #5: the maintainer's real-data evidence is restricted to `>= 8`
/// non-zero years). A non-finite CP return level is counted as a non-finite and
/// also as a `> 10%` divergence (the worst case).
pub fn count_divergences(regions: &[Region]) -> DivergenceCounts {
    let mut counts = DivergenceCounts {
        gt_1pct: 0,
        gt_10pct: 0,
        non_finite: 0,
        degenerate: 0,
        huge_finite: 0,
        fitted: 0,
        skipped: 0,
    };

    for region in regions {
        let y = region.non_zero_years();
        if y.len() < MIN_NON_ZERO_YEARS {
            counts.skipped += 1;
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

        // Item B2: a degenerate fit is flagged and returns NaN CP output.
        if fit.degenerate_fit {
            counts.degenerate += 1;
        }

        let cp = fit.cp_quantiles[0];
        let ml = fit.ml_quantiles[0];

        if !cp.is_finite() {
            counts.non_finite += 1;
            counts.gt_1pct += 1;
            counts.gt_10pct += 1;
            continue;
        }

        // Item B: a huge finite CP return level is the degenerate-fit defect.
        if cp.abs() > HUGE_CP_BOUND {
            counts.huge_finite += 1;
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
    println!(
        "  fitted regions (n >= {MIN_NON_ZERO_YEARS} non-zero years): {}",
        counts.fitted
    );
    println!("  skipped (n < {MIN_NON_ZERO_YEARS}): {}", counts.skipped);
    println!("  > 1%:  {} ({:.1}%)", counts.gt_1pct, counts.pct_gt_1());
    println!("  > 10%: {}", counts.gt_10pct);
    println!("  non-finite: {}", counts.non_finite);
    println!("  degenerate (item B2 flag): {}", counts.degenerate);
    println!(
        "  huge finite (> {HUGE_CP_BOUND:e}): {}",
        counts.huge_finite
    );
    println!("  pre-fix reference (real data):");
    for &(corpus, pct1, c10, nf) in &PREFIX_COUNTS {
        println!("    {corpus}: {pct1:.1}% / {c10} / {nf}");
    }

    // Issue 2 re-check (item B1): the relative degenerate-fit guard rejects the
    // `scale -> 0` spurious maximum, so the multi-start finds a valid interior
    // candidate on **every** region and **no** fit is flagged degenerate. This
    // is the item-B guarantee and is asserted exactly (zero).
    assert_eq!(
        counts.degenerate, 0,
        "item B1: no region may be flagged degenerate (the multi-start must \
         find a valid interior candidate); got {} of {}",
        counts.degenerate, counts.fitted
    );

    // The remaining huge finite CP return levels are **not** the item-B
    // `scale -> 0` degeneracy. They are the DMGS expansion blowing up at a
    // large **positive** shape (all have shape >= 3, median ~7.5) on short
    // records — the GEV analogue of item C (the GPD near-−1 blow-up), which
    // item B does not address.
    //
    // Issue #5: the maintainer verified that these values are
    // **reference-matching** (the reference also reverts to the ML quantile
    // when `|xi| >= 1`, and a GEV with `xi > 1` has an extremely heavy tail, so
    // the ML 1-in-100 return level is huge for both implementations). The
    // count is therefore a **regression baseline**, not a defect count. It is
    // measured on the `n >= MIN_NON_ZERO_YEARS` regime (the records actually
    // fitted in practice) and pinned to the measured value so a regression that
    // reintroduces the item-B degeneracy (which would raise it) is caught. See
    // `tests/golden/TOLERANCE.md` §3.14b/§8.2 and the plan §3.2.
    const KNOWN_LARGE_POSITIVE_SHAPE_HUGE: usize = 24;
    assert_eq!(
        counts.huge_finite, KNOWN_LARGE_POSITIVE_SHAPE_HUGE,
        "the huge finite CP return levels must be exactly the pinned \
         reference-matching count (n >= {MIN_NON_ZERO_YEARS}); got {} of {}",
        counts.huge_finite, counts.fitted
    );

    // The non-finite CP return levels were the DMGS expansion being singular at
    // a near-−1 fitted shape (shape ~ -0.99, just above the `revert2ml`
    // threshold of 1) on very short records (n = 3-8) — item C's GEV analogue,
    // not item B.
    //
    // Issue 2 re-check (item A): the CP quantiles are no longer clamped to the
    // GEV support, so these near-−1 singularities surfaced as NaN rather than
    // being masked by the clamp. This matches the reference (which does not
    // clamp either).
    //
    // Issue #4: the GEV `revert2ml` guard now also fires for `shape <= -0.96`
    // (`GEV_NEAR_MINUS_ONE_DELTA`), so every near-−1 fit reverts to the (finite)
    // ML output and the non-finite count is **zero**. The count is asserted
    // exactly so a regression that removes the guard is caught.
    const KNOWN_NEAR_MINUS_ONE_NONFINITE: usize = 0;
    assert_eq!(
        counts.non_finite, KNOWN_NEAR_MINUS_ONE_NONFINITE,
        "the non-finite CP return levels must be zero (Issue #4: the near-−1 \
         GEV fits revert to the ML output); got {} of {}",
        counts.non_finite, counts.fitted
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

/// Historical record (Issue #5): the **unfiltered** surrogate (n >= 3, the
/// pre-Issue-#5 skip) produced 192 huge-finite regions. The filtered harness
/// (`n >= MIN_NON_ZERO_YEARS`) pins 24. This test reproduces the 192 figure so
/// the historical number stays reproducible and the before/after is auditable.
#[test]
fn hail_region_unfiltered_huge_finite_count_is_192() {
    let regions = synthetic_hail_regions(SEED, N_REGIONS);
    let mut huge_finite = 0usize;
    let mut fitted = 0usize;
    for region in &regions {
        let y = region.non_zero_years();
        if y.len() < 3 {
            continue;
        }
        fitted += 1;
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
        if cp.is_finite() && cp.abs() > HUGE_CP_BOUND {
            huge_finite += 1;
        }
    }
    println!("unfiltered (n >= 3): fitted = {fitted}, huge finite = {huge_finite}");
    assert_eq!(
        huge_finite, 192,
        "the unfiltered (n >= 3) huge-finite count must stay 192 (historical \
         Issue #5 record); got {huge_finite} of {fitted}"
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
