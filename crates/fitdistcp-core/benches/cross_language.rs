//! Cross-language performance comparison — Rust Criterion bench (Group B).
//!
//! This target is registered with `harness = false` in
//! [`crates/fitdistcp-core/Cargo.toml`](../Cargo.toml) and is the Rust side of
//! the cross-language benchmark described in
//! [`PERF_COMPARISON.md`](PERF_COMPARISON.md). It consumes the shared inputs in
//! [`perf_inputs.json`](perf_inputs.json).
//!
//! The CSV-emitting timing binary lives at
//! `crates/fitdistcp-core/src/bin/perf_timing.rs` (auto-discovered by Cargo).
//!
//! # Workloads
//!
//! - pointwise `pdf`/`cdf`/`ppf`/`rvs` for `norm`/`genextreme`/`gamma` across
//!   n = 50/500/5000 on the 100-point grid, and
//! - end-to-end `predict` (MLE + DMGS) at n = 500.
//!
//! reltest is **not** covered here (it is a Monte Carlo workload that does not
//! fit Criterion's per-iteration model); it is emitted by the `perf_timing`
//! binary instead.
//!
//! # Core API entry points used
//!
//! | Operation | Entry point |
//! |---|---|
//! | `pdf` | `Normal::pdf` / `Gamma::pdf` / `GenExtreme::pdf` |
//! | `cdf` | `Normal::cdf` / `Gamma::cdf` / `GenExtreme::cdf` |
//! | `ppf` | `Normal::quantile` / `Gamma::quantile` / `GenExtreme::quantile` |
//! | `rvs` | inverse-CDF sampling via the model `quantile` (see [`rvs`]) |
//! | `predict` | [`fitdistcp_core::predict`] / [`fitdistcp_core::predict_gev`] |
//!
//! Run with: `cargo bench --bench cross_language`

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use fitdistcp_core::{predict, predict_gev, Gamma, GenExtreme, Normal, PredictionRequest};

// ---------------------------------------------------------------------------
// Shared fixture
// ---------------------------------------------------------------------------

/// The materialized inputs from `perf_inputs.json` (see PERF_COMPARISON.md §3.3).
///
/// Only the fields the Rust harness needs are deserialized; serde ignores the
/// rest (`meta`, `reltest`, ...).
#[derive(Debug, Clone, serde::Deserialize)]
struct Fixture {
    grid: Grid,
    samples: Samples,
    predict: PredictSamples,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Grid {
    x: Vec<f64>,
    p: Vec<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Samples {
    norm: PerN,
    gamma: PerN,
    genextreme: PerN,
}

/// The per-sample-size arrays. JSON object keys are strings, so the sizes are
/// named fields rather than a map.
#[derive(Debug, Clone, serde::Deserialize)]
struct PerN {
    #[serde(rename = "50")]
    n50: Vec<f64>,
    #[serde(rename = "500")]
    n500: Vec<f64>,
    #[serde(rename = "5000")]
    n5000: Vec<f64>,
}

impl PerN {
    fn get(&self, n: usize) -> &[f64] {
        match n {
            50 => &self.n50,
            500 => &self.n500,
            5_000 => &self.n5000,
            other => panic!("unsupported sample size {other}"),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PredictSamples {
    norm: PredictSample,
    gamma: PredictSample,
    genextreme: PredictSample,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PredictSample {
    #[allow(dead_code)]
    n: usize,
    y: Vec<f64>,
}

/// Resolve the fixture path relative to the crate root.
///
/// Cargo runs bench binaries with `CARGO_MANIFEST_DIR` set to the crate root, so
/// the fixture is always found regardless of the working directory.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("perf_inputs.json")
}

/// Load and parse the shared fixture.
fn load_fixture() -> Fixture {
    let path = fixture_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Model dispatch
// ---------------------------------------------------------------------------

/// The three benchmarked models, dispatched to their scalar core entry points.
///
/// The `pdf`/`cdf`/`ppf` methods call the model's inherent scalar methods
/// (`Normal::pdf`, `Gamma::quantile`, ...), which are the public core API.
#[derive(Debug, Clone, Copy)]
enum Model {
    Norm,
    GenExtreme,
    Gamma,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Model::Norm => "norm",
            Model::GenExtreme => "genextreme",
            Model::Gamma => "gamma",
        }
    }

    /// Initial parameter guess for the MLE fit (matches `mle_fitting.rs`).
    fn init(self) -> &'static [f64] {
        match self {
            Model::Norm => &[0.0, 1.0],
            Model::GenExtreme => &[0.0, 1.0, 0.1],
            Model::Gamma => &[1.0, 1.0],
        }
    }

    /// Density `f(x; params)` at a single point.
    fn pdf(self, x: f64, params: &[f64]) -> f64 {
        match self {
            Model::Norm => Normal.pdf(x, params[0], params[1]),
            Model::GenExtreme => GenExtreme.pdf(x, params[0], params[1], params[2]),
            Model::Gamma => Gamma.pdf(x, params[0], params[1]),
        }
    }

    /// CDF `F(x; params)` at a single point.
    fn cdf(self, x: f64, params: &[f64]) -> f64 {
        match self {
            Model::Norm => Normal.cdf(x, params[0], params[1]),
            Model::GenExtreme => GenExtreme.cdf(x, params[0], params[1], params[2]),
            Model::Gamma => Gamma.cdf(x, params[0], params[1]),
        }
    }

    /// Quantile `F^{-1}(p; params)`.
    fn quantile(self, p: f64, params: &[f64]) -> f64 {
        match self {
            Model::Norm => Normal.quantile(p, params[0], params[1]),
            Model::GenExtreme => GenExtreme.quantile(p, params[0], params[1], params[2]),
            Model::Gamma => Gamma.quantile(p, params[0], params[1]),
        }
    }
}

/// Fit the model once (MLE) **outside** the timed region and return the fitted
/// parameters, as PERF_COMPARISON.md §1.1 requires.
fn fit_params(model: Model, y: &[f64]) -> Vec<f64> {
    fitdistcp_core::mle::fit(&model_struct(model), y, model.init()).ml_params
}

/// The zero-sized model struct for `model` (used by the generic `mle::fit`).
fn model_struct(model: Model) -> ModelStruct {
    match model {
        Model::Norm => ModelStruct::Norm(Normal),
        Model::GenExtreme => ModelStruct::GenExtreme(GenExtreme),
        Model::Gamma => ModelStruct::Gamma(Gamma),
    }
}

/// A concrete model value so `mle::fit` can be called generically.
#[derive(Debug, Clone, Copy)]
enum ModelStruct {
    Norm(Normal),
    GenExtreme(GenExtreme),
    Gamma(Gamma),
}

impl fitdistcp_core::models::Derivatives for ModelStruct {
    fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
        match self {
            ModelStruct::Norm(m) => m.loglik(params, y),
            ModelStruct::GenExtreme(m) => m.loglik(params, y),
            ModelStruct::Gamma(m) => m.loglik(params, y),
        }
    }
    fn n_params(&self) -> usize {
        match self {
            ModelStruct::Norm(m) => m.n_params(),
            ModelStruct::GenExtreme(m) => m.n_params(),
            ModelStruct::Gamma(m) => m.n_params(),
        }
    }
    fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        match self {
            ModelStruct::Norm(m) => m.score(params, y),
            ModelStruct::GenExtreme(m) => m.score(params, y),
            ModelStruct::Gamma(m) => m.score(params, y),
        }
    }
    fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        match self {
            ModelStruct::Norm(m) => m.hessian(params, y),
            ModelStruct::GenExtreme(m) => m.hessian(params, y),
            ModelStruct::Gamma(m) => m.hessian(params, y),
        }
    }
    fn third_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        match self {
            ModelStruct::Norm(m) => m.third_derivatives(params, y),
            ModelStruct::GenExtreme(m) => m.third_derivatives(params, y),
            ModelStruct::Gamma(m) => m.third_derivatives(params, y),
        }
    }
    fn mixed_derivatives(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        match self {
            ModelStruct::Norm(m) => m.mixed_derivatives(params, y),
            ModelStruct::GenExtreme(m) => m.mixed_derivatives(params, y),
            ModelStruct::Gamma(m) => m.mixed_derivatives(params, y),
        }
    }
}

// ---------------------------------------------------------------------------
// rvs: inverse-CDF sampling
// ---------------------------------------------------------------------------

/// Deterministic uniform stream (the documented LCG from PERF_COMPARISON.md §3.1).
///
/// The core exposes **no** `rvs`/`cp_rvs` entry point, so `rvs` is implemented
/// as inverse-CDF sampling: draw `u ~ U(0,1)` from the documented LCG and map it
/// through the model quantile `F^{-1}(u; params)`. This is the same sampling
/// primitive the `reltest` test uses (`DmgsModel::quantile`). The uniform stream
/// is pre-generated **outside** the timed region, so the timed loop only performs
/// the quantile evaluations.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next uniform in `(0, 1)` via the documented 64-bit LCG.
    fn next_u01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) as f64;
        // Map to (0, 1) exclusive so the quantile is finite at both ends.
        (bits + 0.5) / (1u64 << 53) as f64
    }
}

/// Pre-generate the uniform stream used by the `rvs` workload.
fn rvs_uniforms(count: usize, seed: u64) -> Vec<f64> {
    let mut lcg = Lcg::new(seed);
    (0..count).map(|_| lcg.next_u01()).collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Pointwise `pdf`/`cdf`/`ppf`/`rvs` across models and sample sizes.
///
/// For each `(model, operation, n)` the model is fitted once outside the timed
/// region and the operation is timed over the 100-point grid.
fn bench_pointwise(c: &mut Criterion) {
    let fx = load_fixture();
    let sizes = [50usize, 500, 5_000];
    let models = [Model::Norm, Model::GenExtreme, Model::Gamma];

    // --- pdf / cdf over grid.x ---
    for op in ["pdf", "cdf"] {
        let mut group = c.benchmark_group(format!("cross_language/{op}"));
        for model in models {
            for &n in &sizes {
                let y = samples_for(&fx, model, n);
                let params = fit_params(model, y);
                let x = fx.grid.x.clone();
                group.throughput(Throughput::Elements(x.len() as u64));
                group.bench_with_input(
                    BenchmarkId::new(model.name(), n),
                    &(params, x),
                    |b, (params, x)| {
                        b.iter(|| {
                            let mut acc = 0.0;
                            for &xi in x.iter() {
                                acc += match op {
                                    "pdf" => model.pdf(xi, params),
                                    _ => model.cdf(xi, params),
                                };
                            }
                            acc
                        })
                    },
                );
            }
        }
        group.finish();
    }

    // --- ppf over grid.p ---
    {
        let mut group = c.benchmark_group("cross_language/ppf");
        for model in models {
            for &n in &sizes {
                let y = samples_for(&fx, model, n);
                let params = fit_params(model, y);
                let p = fx.grid.p.clone();
                group.throughput(Throughput::Elements(p.len() as u64));
                group.bench_with_input(
                    BenchmarkId::new(model.name(), n),
                    &(params, p),
                    |b, (params, p)| {
                        b.iter(|| {
                            let mut acc = 0.0;
                            for &pi in p.iter() {
                                acc += model.quantile(pi, params);
                            }
                            acc
                        })
                    },
                );
            }
        }
        group.finish();
    }

    // --- rvs: inverse-CDF sampling over a pre-generated uniform stream ---
    {
        let mut group = c.benchmark_group("cross_language/rvs");
        for model in models {
            for &n in &sizes {
                let y = samples_for(&fx, model, n);
                let params = fit_params(model, y);
                // One uniform per grid point; pre-generated outside the timed loop.
                let u = rvs_uniforms(fx.grid.x.len(), 0x5EED_1234_5678_9ABC);
                group.throughput(Throughput::Elements(u.len() as u64));
                group.bench_with_input(
                    BenchmarkId::new(model.name(), n),
                    &(params, u),
                    |b, (params, u)| {
                        b.iter(|| {
                            let mut acc = 0.0;
                            for &ui in u.iter() {
                                acc += model.quantile(ui, params);
                            }
                            acc
                        })
                    },
                );
            }
        }
        group.finish();
    }
}

/// End-to-end `predict` (MLE + DMGS) at n = 500, one workload per model.
fn bench_predict_end_to_end(c: &mut Criterion) {
    let fx = load_fixture();
    let mut group = c.benchmark_group("cross_language/predict");
    let n = 500usize;
    group.throughput(Throughput::Elements(n as u64));

    // norm
    let req_norm = PredictionRequest {
        y: fx.predict.norm.y.clone(),
        x: fx.grid.x.clone(),
        p: fx.grid.p.clone(),
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    group.bench_function("norm", |b| {
        b.iter(|| predict(Normal, &req_norm, &[0.0, 1.0]))
    });

    // gamma
    let req_gamma = PredictionRequest {
        y: fx.predict.gamma.y.clone(),
        x: fx.grid.x.clone(),
        p: fx.grid.p.clone(),
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    group.bench_function("gamma", |b| {
        b.iter(|| predict(Gamma, &req_gamma, &[1.0, 1.0]))
    });

    // genextreme (GEV-specific CP path)
    let req_gev = PredictionRequest {
        y: fx.predict.genextreme.y.clone(),
        x: fx.grid.x.clone(),
        p: fx.grid.p.clone(),
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    group.bench_function("genextreme", |b| {
        b.iter(|| predict_gev(GenExtreme, &req_gev, &[0.0, 1.0, 0.1], 0.01))
    });

    group.finish();
}

/// The sample vector for `(model, n)` from the fixture.
fn samples_for(fx: &Fixture, model: Model, n: usize) -> &[f64] {
    match model {
        Model::Norm => fx.samples.norm.get(n),
        Model::GenExtreme => fx.samples.genextreme.get(n),
        Model::Gamma => fx.samples.gamma.get(n),
    }
}

criterion_group!(benches, bench_pointwise, bench_predict_end_to_end);
criterion_main!(benches);
