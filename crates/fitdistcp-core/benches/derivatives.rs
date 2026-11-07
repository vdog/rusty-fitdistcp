//! Criterion benchmarks for derivative evaluation (W16, design §6).
//!
//! §6 identifies the **derivative evaluations** as one of the two hot loops in
//! the DMGS path (the other being the per-grid-point expansion). This bench
//! measures the four derivative surfaces the expansion consumes, for a
//! representative analytic model (`norm`) and a representative DMGS model
//! (`gamma`):
//!
//! - `score` (first derivatives),
//! - `hessian` (second derivatives / observed information),
//! - `third_derivatives` (bias-correction tensor),
//! - `mixed_derivatives` (Haar-prior term).
//!
//! For `norm` all four are analytic; for `gamma` the score/Hessian/third are
//! analytic but the mixed derivatives fall back to finite differences. The
//! `AnalyticDerivatives` adapter is used to compare the analytic path against
//! the forced-numerical fallback, which quantifies the cost of the fallback and
//! tells W17 where an analytic override would pay off.
//!
//! Run with: `cargo bench --bench derivatives`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use fitdistcp_core::models::{AnalyticDerivatives, DerivativeMode, Derivatives};
use fitdistcp_core::{Gamma, Normal};

/// A deterministic pseudo-random sample (no external RNG dependency).
///
/// Uses a simple LCG so the benchmark is reproducible across runs and machines.
fn sample(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Numerical Recipes LCG constants.
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map the top 53 bits to [0, 1), then to a roughly standard-normal-ish
        // range via a sum of uniforms (Irwin–Hall, k = 12) minus 6.
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

/// Positive sample for the gamma model (shape/rate parameterization).
fn positive_sample(n: usize, seed: u64) -> Vec<f64> {
    sample(n, seed)
        .into_iter()
        .map(|v| (v + 6.0).abs() + 0.1)
        .collect()
}

fn bench_norm_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivatives/norm");
    let model = Normal;
    let params = [0.4, 1.3];

    for &n in &[100usize, 1_000, 10_000] {
        let y = sample(n, 0x1234_5678);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("score", n), &y, |b, y| {
            b.iter(|| model.score(&params, y))
        });
        group.bench_with_input(BenchmarkId::new("hessian", n), &y, |b, y| {
            b.iter(|| model.hessian(&params, y))
        });
        group.bench_with_input(BenchmarkId::new("third_derivatives", n), &y, |b, y| {
            b.iter(|| model.third_derivatives(&params, y))
        });
        group.bench_with_input(BenchmarkId::new("mixed_derivatives", n), &y, |b, y| {
            b.iter(|| model.mixed_derivatives(&params, y))
        });
    }
    group.finish();
}

fn bench_gamma_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivatives/gamma");
    let model = Gamma;
    let params = [2.5, 1.2];

    for &n in &[100usize, 1_000, 10_000] {
        let y = positive_sample(n, 0x9abc_def0);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("score", n), &y, |b, y| {
            b.iter(|| model.score(&params, y))
        });
        group.bench_with_input(BenchmarkId::new("hessian", n), &y, |b, y| {
            b.iter(|| model.hessian(&params, y))
        });
        group.bench_with_input(BenchmarkId::new("third_derivatives", n), &y, |b, y| {
            b.iter(|| model.third_derivatives(&params, y))
        });
        group.bench_with_input(BenchmarkId::new("mixed_derivatives", n), &y, |b, y| {
            b.iter(|| model.mixed_derivatives(&params, y))
        });
    }
    group.finish();
}

/// Analytic vs. forced-numerical comparison for the normal model.
///
/// The normal has analytic overrides for every derivative, so the numerical
/// path here is the *worst case* for the fallback: it shows the cost W17 would
/// pay if a model lacked an analytic override.
fn bench_analytic_vs_numerical(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivatives/norm_analytic_vs_numerical");
    let n = 1_000usize;
    let y = sample(n, 0xdead_beef);
    let params = [0.4, 1.3];

    let auto = AnalyticDerivatives::new(Normal, DerivativeMode::Auto);
    let num = AnalyticDerivatives::new(Normal, DerivativeMode::Numerical);

    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("score/analytic", |b| b.iter(|| auto.score(&params, &y)));
    group.bench_function("score/numerical", |b| b.iter(|| num.score(&params, &y)));
    group.bench_function("hessian/analytic", |b| b.iter(|| auto.hessian(&params, &y)));
    group.bench_function("hessian/numerical", |b| b.iter(|| num.hessian(&params, &y)));
    group.bench_function("third/analytic", |b| {
        b.iter(|| auto.third_derivatives(&params, &y))
    });
    group.bench_function("third/numerical", |b| {
        b.iter(|| num.third_derivatives(&params, &y))
    });
    group.bench_function("mixed/analytic", |b| {
        b.iter(|| auto.mixed_derivatives(&params, &y))
    });
    group.bench_function("mixed/numerical", |b| {
        b.iter(|| num.mixed_derivatives(&params, &y))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_norm_derivatives,
    bench_gamma_derivatives,
    bench_analytic_vs_numerical
);
criterion_main!(benches);
