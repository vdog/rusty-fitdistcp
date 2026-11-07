//! Criterion benchmarks for MLE fitting (W16, design §6).
//!
//! §6 asks for MLE fitting of representative models to be measured. The MLE
//! fit is the outer loop of the whole CP pipeline: it runs BFGS (with the
//! analytic score) plus two Newton–Raphson refinements, each of which evaluates
//! the score/Hessian repeatedly. This bench therefore measures the *end-to-end*
//! fit cost for `norm`, `genextreme`, and `gamma` across sample sizes, which is
//! the quantity W17's caching/parallelism work must move.
//!
//! It also benchmarks the full [`fitdistcp_core::predict`] entry point (MLE +
//! DMGS expansion) so the split between fit cost and expansion cost is visible.
//!
//! Run with: `cargo bench --bench mle_fitting`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use fitdistcp_core::mle::fit;
use fitdistcp_core::{predict, Gamma, GenExtreme, Normal, PredictionRequest};

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

fn bench_mle_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("mle/fit");

    for &n in &[50usize, 500, 5_000] {
        group.throughput(Throughput::Elements(n as u64));

        let y_norm = sample(n, 0xaaaa_0001);
        group.bench_with_input(BenchmarkId::new("norm", n), &y_norm, |b, y| {
            b.iter(|| fit(&Normal, y, &[0.0, 1.0]))
        });

        let y_gamma = positive_sample(n, 0xaaaa_0002);
        group.bench_with_input(BenchmarkId::new("gamma", n), &y_gamma, |b, y| {
            b.iter(|| fit(&Gamma, y, &[1.0, 1.0]))
        });

        let y_gev = sample(n, 0xaaaa_0003);
        group.bench_with_input(BenchmarkId::new("genextreme", n), &y_gev, |b, y| {
            b.iter(|| fit(&GenExtreme, y, &[0.0, 1.0, 0.1]))
        });
    }
    group.finish();
}

/// End-to-end `predict` (MLE + DMGS expansion) for the same models.
///
/// The difference between this and `mle/fit` is the DMGS expansion cost, which
/// is the other half of the §6 hot-loop story.
fn bench_predict_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("mle/predict_end_to_end");
    let n = 500usize;
    let x: Vec<f64> = (0..100).map(|i| -3.0 + 6.0 * i as f64 / 100.0).collect();
    let p: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();

    let y_norm = sample(n, 0xbbbb_0001);
    let req_norm = PredictionRequest {
        y: y_norm,
        x: x.clone(),
        p: p.clone(),
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("norm", |b| {
        b.iter(|| predict(Normal, &req_norm, &[0.0, 1.0]))
    });

    let y_gamma = positive_sample(n, 0xbbbb_0002);
    let req_gamma = PredictionRequest {
        y: y_gamma,
        x: x.clone(),
        p: p.clone(),
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    group.bench_function("gamma", |b| {
        b.iter(|| predict(Gamma, &req_gamma, &[1.0, 1.0]))
    });

    let y_gev = sample(n, 0xbbbb_0003);
    let req_gev = PredictionRequest {
        y: y_gev,
        x,
        p,
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    group.bench_function("genextreme", |b| {
        b.iter(|| predict(GenExtreme, &req_gev, &[0.0, 1.0, 0.1]))
    });

    group.finish();
}

criterion_group!(benches, bench_mle_fit, bench_predict_end_to_end);
criterion_main!(benches);
