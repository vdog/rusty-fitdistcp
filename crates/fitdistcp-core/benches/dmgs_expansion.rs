//! Criterion benchmarks for the DMGS expansion (W16, design §6).
//!
//! §6 names the **per-grid-point DMGS expansion** as the other hot loop (besides
//! the derivative evaluations). This bench measures the two public entry points
//! that drive it:
//!
//! - [`DmgsIntegrator::cp_pdf`] — the CP predictive density over a prediction
//!   grid `x` (the reference `rh_pdf`),
//! - [`DmgsIntegrator::cp_quantiles`] — the CP predictive quantiles over a
//!   probability grid `p` (the reference `rh_quantiles`),
//! - [`DmgsIntegrator::cp_pdf_quantile_derived`] — the quantile-derived CP
//!   density used by the negative-shape GEV path (the reference `rh_flat_pdf`).
//!
//! The grid size is swept because the expansion is `O(p^4 · nalpha)` in the
//! `beps1` term (the `t, j, r, s` quadruple loop in
//! [`DmgsIntegrator::dmgs_expansion`]) and `O(p^2 · nalpha)` in the other three
//! terms, so the per-point cost grows with the parameter count `p` and the grid
//! length `nalpha`. We benchmark `norm` (p = 2), `gamma` (p = 2), and
//! `genextreme` (p = 3) to expose the `p^4` scaling.
//!
//! Run with: `cargo bench --bench dmgs_expansion`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use fitdistcp_core::dmgs::DmgsIntegrator;
use fitdistcp_core::{Gamma, GenExtreme, Normal};

/// Deterministic pseudo-random sample (LCG + Irwin–Hall), reproducible.
fn sample(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut u = 0.0;
        for _ in 0..12 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as f64;
            u += bits / (1u64 << 53) as f64;
        }
        out.push(u - 6.0);
    }
    out
}

fn positive_sample(n: usize, seed: u64) -> Vec<f64> {
    sample(n, seed)
        .into_iter()
        .map(|v| (v + 6.0).abs() + 0.1)
        .collect()
}

/// A probability grid in `(0, 1)` for the quantile benchmarks.
fn prob_grid(nalpha: usize) -> Vec<f64> {
    (0..nalpha)
        .map(|i| (i as f64 + 0.5) / nalpha as f64)
        .collect()
}

fn bench_cp_pdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("dmgs/cp_pdf");

    // norm: p = 2, analytic density/CDF derivative hooks.
    let norm = DmgsIntegrator::new(Normal);
    let y_norm = sample(200, 0x1111_2222);
    let params_norm = [0.4, 1.3];
    for &nalpha in &[10usize, 100, 1_000] {
        let x: Vec<f64> = (0..nalpha)
            .map(|i| -3.0 + 6.0 * i as f64 / nalpha as f64)
            .collect();
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(BenchmarkId::new("norm", nalpha), &x, |b, x| {
            b.iter(|| norm.cp_pdf(&params_norm, &y_norm, x))
        });
    }

    // gamma: p = 2, analytic density hooks, numerical CDF hooks.
    let gamma = DmgsIntegrator::new(Gamma);
    let y_gamma = positive_sample(200, 0x3333_4444);
    let params_gamma = [2.5, 1.2];
    for &nalpha in &[10usize, 100, 1_000] {
        let x: Vec<f64> = (0..nalpha)
            .map(|i| 0.05 + 5.0 * i as f64 / nalpha as f64)
            .collect();
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(BenchmarkId::new("gamma", nalpha), &x, |b, x| {
            b.iter(|| gamma.cp_pdf(&params_gamma, &y_gamma, x))
        });
    }

    // genextreme: p = 3, so the p^4 beps1 term is 81x the p=2 cost per point.
    let gev = DmgsIntegrator::new(GenExtreme);
    let y_gev = sample(200, 0x5555_6666);
    let params_gev = [0.0, 1.0, 0.1];
    for &nalpha in &[10usize, 100, 1_000] {
        let x: Vec<f64> = (0..nalpha)
            .map(|i| -3.0 + 6.0 * i as f64 / nalpha as f64)
            .collect();
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(BenchmarkId::new("genextreme", nalpha), &x, |b, x| {
            b.iter(|| gev.cp_pdf(&params_gev, &y_gev, x))
        });
    }

    group.finish();
}

fn bench_cp_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("dmgs/cp_quantiles");

    let norm = DmgsIntegrator::new(Normal);
    let y_norm = sample(200, 0x1111_2222);
    let params_norm = [0.4, 1.3];
    for &nalpha in &[10usize, 100, 1_000] {
        let p = prob_grid(nalpha);
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(BenchmarkId::new("norm", nalpha), &p, |b, p| {
            b.iter(|| norm.cp_quantiles(&params_norm, &y_norm, p))
        });
    }

    let gamma = DmgsIntegrator::new(Gamma);
    let y_gamma = positive_sample(200, 0x3333_4444);
    let params_gamma = [2.5, 1.2];
    for &nalpha in &[10usize, 100, 1_000] {
        let p = prob_grid(nalpha);
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(BenchmarkId::new("gamma", nalpha), &p, |b, p| {
            b.iter(|| gamma.cp_quantiles(&params_gamma, &y_gamma, p))
        });
    }

    let gev = DmgsIntegrator::new(GenExtreme);
    let y_gev = sample(200, 0x5555_6666);
    let params_gev = [0.0, 1.0, 0.1];
    for &nalpha in &[10usize, 100, 1_000] {
        let p = prob_grid(nalpha);
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(BenchmarkId::new("genextreme", nalpha), &p, |b, p| {
            b.iter(|| gev.cp_quantiles(&params_gev, &y_gev, p))
        });
    }

    group.finish();
}

/// The quantile-derived CP density (negative-shape GEV path). This calls
/// `cp_quantiles` twice (at `alpha ± dalpha`), so it is ~2x the quantile cost
/// plus the finite-difference bookkeeping.
fn bench_cp_pdf_quantile_derived(c: &mut Criterion) {
    let mut group = c.benchmark_group("dmgs/cp_pdf_quantile_derived");
    let gev = DmgsIntegrator::new(GenExtreme);
    let y_gev = sample(200, 0x5555_6666);
    let params_gev = [0.0, 1.0, -0.2];

    for &nalpha in &[10usize, 100, 1_000] {
        let alpha: Vec<f64> = prob_grid(nalpha);
        group.throughput(Throughput::Elements(nalpha as u64));
        group.bench_with_input(
            BenchmarkId::new("genextreme", nalpha),
            &alpha,
            |b, alpha| b.iter(|| gev.cp_pdf_quantile_derived(&params_gev, &y_gev, alpha, 0.01)),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cp_pdf,
    bench_cp_quantiles,
    bench_cp_pdf_quantile_derived
);
criterion_main!(benches);
