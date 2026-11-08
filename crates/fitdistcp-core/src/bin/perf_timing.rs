//! Cross-language performance comparison — Rust CSV timing binary (Group B).
//!
//! This binary runs each canonical workload from
//! [`benches/PERF_COMPARISON.md`](../../benches/PERF_COMPARISON.md) in a
//! warmup + repeats loop timed with [`std::time::Instant`], and writes CSV rows
//! with the schema:
//!
//! ```text
//! implementation,model,operation,n,median_ns,min_ns,max_ns,repeats
//! ```
//!
//! with `implementation = rust`. It is the Rust side of the cross-implementation
//! comparison; the Criterion bench (`benches/cross_language.rs`) remains the
//! Rust-internal regression baseline.
//!
//! # Workloads
//!
//! - pointwise `pdf`/`cdf`/`ppf`/`rvs` for `norm`/`genextreme`/`gamma` across
//!   n = 50/500/5000 on the 100-point grid,
//! - end-to-end `predict` (MLE + DMGS) at n = 500, and
//! - `reltest` (reliability simulation) for each model.
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
//! | `reltest` | composed from `predict`/`predict_gev` + `DmgsModel` (see [`reltest`]) |
//!
//! # reltest
//!
//! The core exposes **no** dedicated public `reltest` entry point — the only
//! reltest code in the crate is the integration test
//! `crates/fitdistcp-core/tests/reltest.rs`. Rather than invent an API, this
//! binary reproduces the documented reltest *workload* (PERF_COMPARISON.md §1.3)
//! from public entry points: it simulates a training sample from a known model
//! via inverse-CDF sampling, fits it with `predict`/`predict_gev`, and scores the
//! MLE and CP predictive quantiles against the true distribution. The
//! `ntrials`/`nx`/`nrepeats`/`seed`/`params` come from the fixture's `reltest`
//! block.
//!
//! # Usage
//!
//! ```text
//! perf_timing [--out <path>] [--warmup <n>] [--repeats <n>] [--inputs <path>]
//! ```
//!
//! `--out` defaults to stdout; `--warmup` defaults to 3; `--repeats` defaults to
//! 20; `--inputs` defaults to `benches/perf_inputs.json` relative to the crate.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use fitdistcp_core::models::Derivatives;
use fitdistcp_core::{predict, predict_gev, Gamma, GenExtreme, Normal, PredictionRequest};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// The materialized inputs from `perf_inputs.json` (PERF_COMPARISON.md §3.3).
#[derive(Debug, Clone, serde::Deserialize)]
struct Fixture {
    grid: Grid,
    samples: Samples,
    predict: PredictSamples,
    reltest: Reltest,
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

#[derive(Debug, Clone, serde::Deserialize)]
struct Reltest {
    ntrials: usize,
    nx: usize,
    nrepeats: usize,
    seed: u64,
    params: ReltestParams,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ReltestParams {
    norm: Vec<f64>,
    genextreme: Vec<f64>,
    gamma: Vec<f64>,
}

/// Resolve the default fixture path relative to the crate root.
fn default_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("perf_inputs.json")
}

fn load_fixture(path: &PathBuf) -> Fixture {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Model dispatch
// ---------------------------------------------------------------------------

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

    fn init(self) -> &'static [f64] {
        match self {
            Model::Norm => &[0.0, 1.0],
            Model::GenExtreme => &[0.0, 1.0, 0.1],
            Model::Gamma => &[1.0, 1.0],
        }
    }

    fn pdf(self, x: f64, params: &[f64]) -> f64 {
        match self {
            Model::Norm => Normal.pdf(x, params[0], params[1]),
            Model::GenExtreme => GenExtreme.pdf(x, params[0], params[1], params[2]),
            Model::Gamma => Gamma.pdf(x, params[0], params[1]),
        }
    }

    fn cdf(self, x: f64, params: &[f64]) -> f64 {
        match self {
            Model::Norm => Normal.cdf(x, params[0], params[1]),
            Model::GenExtreme => GenExtreme.cdf(x, params[0], params[1], params[2]),
            Model::Gamma => Gamma.cdf(x, params[0], params[1]),
        }
    }

    fn quantile(self, p: f64, params: &[f64]) -> f64 {
        match self {
            Model::Norm => Normal.quantile(p, params[0], params[1]),
            Model::GenExtreme => GenExtreme.quantile(p, params[0], params[1], params[2]),
            Model::Gamma => Gamma.quantile(p, params[0], params[1]),
        }
    }
}

/// A concrete model value so `mle::fit` can be called generically.
#[derive(Debug, Clone, Copy)]
enum ModelStruct {
    Norm(Normal),
    GenExtreme(GenExtreme),
    Gamma(Gamma),
}

fn model_struct(model: Model) -> ModelStruct {
    match model {
        Model::Norm => ModelStruct::Norm(Normal),
        Model::GenExtreme => ModelStruct::GenExtreme(GenExtreme),
        Model::Gamma => ModelStruct::Gamma(Gamma),
    }
}

impl Derivatives for ModelStruct {
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

/// Fit the model once (MLE) outside the timed region.
fn fit_params(model: Model, y: &[f64]) -> Vec<f64> {
    fitdistcp_core::mle::fit(&model_struct(model), y, model.init()).ml_params
}

// ---------------------------------------------------------------------------
// rvs: inverse-CDF sampling
// ---------------------------------------------------------------------------

/// Deterministic uniform stream (the documented LCG from PERF_COMPARISON.md §3.1).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u01(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) as f64;
        (bits + 0.5) / (1u64 << 53) as f64
    }
}

fn rvs_uniforms(count: usize, seed: u64) -> Vec<f64> {
    let mut lcg = Lcg::new(seed);
    (0..count).map(|_| lcg.next_u01()).collect()
}

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

/// One CSV row.
struct Row {
    model: &'static str,
    operation: &'static str,
    n: usize,
    median_ns: u128,
    min_ns: u128,
    max_ns: u128,
    repeats: usize,
}

/// Run `work` `warmup` times (discarded) then `repeats` times, timing each
/// repeat with [`Instant`], and return the median/min/max nanoseconds.
fn time_workload<F: FnMut()>(warmup: usize, repeats: usize, mut work: F) -> (u128, u128, u128) {
    for _ in 0..warmup {
        work();
    }
    let mut samples: Vec<u128> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        work();
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let min = *samples.first().unwrap();
    let max = *samples.last().unwrap();
    (median, min, max)
}

// ---------------------------------------------------------------------------
// reltest workload
// ---------------------------------------------------------------------------

/// Nominal exceedance probabilities `alpha` at which reliability is measured.
///
/// Matches the Rust reltest test's coarse tail-covering grid (the reference uses
/// `seq(0.005, 0.995, 0.005)`; this keeps the workload tractable while still
/// dominated by the tails).
const RELTEST_ALPHAS: [f64; 9] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

/// Whether a fitted parameter vector is well-behaved enough to evaluate the
/// predictive quantiles without hitting a degenerate model.
///
/// The reference `reltest2` resamples until the fit is well-behaved; the Rust
/// reltest test does the same (`if !fit.is_converged() { continue; }` plus a
/// shape guard). We reproduce that rejection loop here: a trial whose fit does
/// not converge, or whose scale/shape is non-positive, is skipped. This is
/// necessary because `predict` evaluates the model quantile unconditionally, and
/// a degenerate fit (e.g. a non-positive gamma shape) would otherwise panic in
/// the incomplete-gamma routine.
fn fit_is_usable(model: Model, params: &[f64]) -> bool {
    if params.iter().any(|v| !v.is_finite()) {
        return false;
    }
    match model {
        // norm: [mu, sigma], sigma > 0.
        Model::Norm => params[1] > 0.0,
        // genextreme: [loc, scale, shape], scale > 0.
        Model::GenExtreme => params[1] > 0.0,
        // gamma: [shape, rate], both > 0.
        Model::Gamma => params[0] > 0.0 && params[1] > 0.0,
    }
}

/// Run the reltest reliability simulation for one model.
///
/// This reproduces the documented reltest workload (PERF_COMPARISON.md §1.3)
/// from public entry points: simulate a training sample of size `nx` from the
/// known model via inverse-CDF sampling, fit it with `predict`/`predict_gev`,
/// and score the MLE and CP predictive quantiles against the true distribution.
/// `nrepeats` is the reference's per-trial repeat count (the committed config
/// uses 1). Degenerate fits are skipped (the reference's rejection loop). The
/// return value is a checksum of the aggregate exceedance errors so the compiler
/// cannot elide the work.
fn reltest(
    model: Model,
    params: &[f64],
    ntrials: usize,
    nx: usize,
    nrepeats: usize,
    seed: u64,
) -> f64 {
    let mut rng = Lcg::new(seed);
    let p: Vec<f64> = RELTEST_ALPHAS.iter().map(|&a| 1.0 - a).collect();
    let mut checksum = 0.0;

    for _ in 0..ntrials {
        for _ in 0..nrepeats {
            // Simulate a training sample from the true model via inverse-CDF.
            let y: Vec<f64> = (0..nx)
                .map(|_| model.quantile(rng.next_u01(), params))
                .collect();

            // Fit first and reject degenerate fits before calling `predict`,
            // which evaluates the model quantile unconditionally.
            let mle = fitdistcp_core::mle::fit(&model_struct(model), &y, params);
            if !mle.convergence.is_converged() || !fit_is_usable(model, &mle.ml_params) {
                continue;
            }

            let req = PredictionRequest {
                y: y.clone(),
                x: vec![],
                p: p.clone(),
                t: None,
                t0: None,
                n0: None,
                means: None,
            };

            // Fit MLE + CP predictive quantiles. The GEV uses the GEV-specific CP
            // path; the others use the generic `predict`.
            let fit = match model {
                Model::GenExtreme => predict_gev(GenExtreme, &req, params, 0.01),
                Model::Norm => predict(Normal, &req, params),
                Model::Gamma => predict(Gamma, &req, params),
            };

            // Score the MLE and CP quantiles against the true distribution by
            // mapping each predicted quantile back through the true CDF to an
            // exceedance probability `ep = 1 - F_true(pred)` (the reference
            // `reltest2_makeep`).
            for (k, &alpha) in RELTEST_ALPHAS.iter().enumerate() {
                let true_cdf = |x: f64| model.cdf(x, params);
                let ep_mle = (1.0 - true_cdf(fit.ml_quantiles[k])).clamp(0.0, 1.0);
                let ep_cp = (1.0 - true_cdf(fit.cp_quantiles[k])).clamp(0.0, 1.0);
                checksum += (ep_mle - alpha) * (ep_mle - alpha);
                checksum += (ep_cp - alpha) * (ep_cp - alpha);
            }
        }
    }
    checksum
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

struct Args {
    out: Option<PathBuf>,
    warmup: usize,
    repeats: usize,
    inputs: PathBuf,
}

fn parse_args() -> Args {
    let mut out = None;
    let mut warmup = 3usize;
    let mut repeats = 20usize;
    let mut inputs = default_fixture_path();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().expect("--out requires a path argument"),
                ));
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .expect("--warmup requires a number")
                    .parse()
                    .expect("--warmup must be an integer");
            }
            "--repeats" => {
                repeats = args
                    .next()
                    .expect("--repeats requires a number")
                    .parse()
                    .expect("--repeats must be an integer");
            }
            "--inputs" => {
                inputs = PathBuf::from(args.next().expect("--inputs requires a path argument"));
            }
            other => panic!("unknown argument: {other}"),
        }
    }
    Args {
        out,
        warmup,
        repeats,
        inputs,
    }
}

fn main() {
    let args = parse_args();
    let fx = load_fixture(&args.inputs);
    let sizes = [50usize, 500, 5_000];
    let models = [Model::Norm, Model::GenExtreme, Model::Gamma];

    let mut rows: Vec<Row> = Vec::new();

    // --- pointwise pdf / cdf / ppf / rvs ---
    for model in models {
        // Domain restriction (PERF_COMPARISON.md §2 / §5): the shared grid.x
        // spans [-3, 3], but the gamma distribution has support x > 0. The
        // reference R and Python gamma routines reject non-positive x, so to
        // keep the three implementations comparable we evaluate the gamma
        // pdf/cdf/rvs on the strictly-positive subset of the grid (x > 0).
        // norm/genextreme use the full grid. The (model, operation, n) keys are
        // unchanged.
        let grid_x_model: Vec<f64> = if matches!(model, Model::Gamma) {
            fx.grid.x.iter().copied().filter(|&v| v > 0.0).collect()
        } else {
            fx.grid.x.clone()
        };
        for &n in &sizes {
            let y = samples_for(&fx, model, n);
            let params = fit_params(model, y);

            // pdf over grid.x (gamma: x > 0 subset)
            {
                let x = grid_x_model.clone();
                let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
                    let mut acc = 0.0;
                    for &xi in x.iter() {
                        acc += model.pdf(xi, &params);
                    }
                    std::hint::black_box(acc);
                });
                rows.push(Row {
                    model: model.name(),
                    operation: "pdf",
                    n,
                    median_ns,
                    min_ns,
                    max_ns,
                    repeats: args.repeats,
                });
            }

            // cdf over grid.x (gamma: x > 0 subset)
            {
                let x = grid_x_model.clone();
                let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
                    let mut acc = 0.0;
                    for &xi in x.iter() {
                        acc += model.cdf(xi, &params);
                    }
                    std::hint::black_box(acc);
                });
                rows.push(Row {
                    model: model.name(),
                    operation: "cdf",
                    n,
                    median_ns,
                    min_ns,
                    max_ns,
                    repeats: args.repeats,
                });
            }

            // ppf over grid.p
            {
                let p = fx.grid.p.clone();
                let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
                    let mut acc = 0.0;
                    for &pi in p.iter() {
                        acc += model.quantile(pi, &params);
                    }
                    std::hint::black_box(acc);
                });
                rows.push(Row {
                    model: model.name(),
                    operation: "ppf",
                    n,
                    median_ns,
                    min_ns,
                    max_ns,
                    repeats: args.repeats,
                });
            }

            // rvs: inverse-CDF sampling over a pre-generated uniform stream
            // (gamma: one deviate per strictly-positive grid point)
            {
                let u = rvs_uniforms(grid_x_model.len(), 0x5EED_1234_5678_9ABC);
                let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
                    let mut acc = 0.0;
                    for &ui in u.iter() {
                        acc += model.quantile(ui, &params);
                    }
                    std::hint::black_box(acc);
                });
                rows.push(Row {
                    model: model.name(),
                    operation: "rvs",
                    n,
                    median_ns,
                    min_ns,
                    max_ns,
                    repeats: args.repeats,
                });
            }
        }
    }

    // --- end-to-end predict at n = 500 ---
    {
        let n = 500usize;
        let x = fx.grid.x.clone();
        let p = fx.grid.p.clone();

        let req_norm = PredictionRequest {
            y: fx.predict.norm.y.clone(),
            x: x.clone(),
            p: p.clone(),
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
            std::hint::black_box(predict(Normal, &req_norm, &[0.0, 1.0]));
        });
        rows.push(Row {
            model: "norm",
            operation: "predict",
            n,
            median_ns,
            min_ns,
            max_ns,
            repeats: args.repeats,
        });

        let req_gamma = PredictionRequest {
            y: fx.predict.gamma.y.clone(),
            x: x.clone(),
            p: p.clone(),
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
            std::hint::black_box(predict(Gamma, &req_gamma, &[1.0, 1.0]));
        });
        rows.push(Row {
            model: "gamma",
            operation: "predict",
            n,
            median_ns,
            min_ns,
            max_ns,
            repeats: args.repeats,
        });

        let req_gev = PredictionRequest {
            y: fx.predict.genextreme.y.clone(),
            x,
            p,
            t: None,
            t0: None,
            n0: None,
            means: None,
        };
        let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
            std::hint::black_box(predict_gev(GenExtreme, &req_gev, &[0.0, 1.0, 0.1], 0.01));
        });
        rows.push(Row {
            model: "genextreme",
            operation: "predict",
            n,
            median_ns,
            min_ns,
            max_ns,
            repeats: args.repeats,
        });
    }

    // --- reltest (Monte Carlo reliability simulation) ---
    {
        let rt = &fx.reltest;
        let cases: [(Model, &[f64]); 3] = [
            (Model::Norm, &rt.params.norm),
            (Model::GenExtreme, &rt.params.genextreme),
            (Model::Gamma, &rt.params.gamma),
        ];
        for (model, params) in cases {
            let (median_ns, min_ns, max_ns) = time_workload(args.warmup, args.repeats, || {
                std::hint::black_box(reltest(
                    model,
                    params,
                    rt.ntrials,
                    rt.nx,
                    rt.nrepeats,
                    rt.seed,
                ));
            });
            rows.push(Row {
                model: model.name(),
                operation: "reltest",
                n: rt.nx,
                median_ns,
                min_ns,
                max_ns,
                repeats: args.repeats,
            });
        }
    }

    // --- emit CSV ---
    let mut out: Box<dyn Write> = match &args.out {
        Some(path) => Box::new(
            std::fs::File::create(path)
                .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display())),
        ),
        None => Box::new(std::io::stdout()),
    };

    writeln!(
        out,
        "implementation,model,operation,n,median_ns,min_ns,max_ns,repeats"
    )
    .expect("write header");
    for row in &rows {
        writeln!(
            out,
            "rust,{},{},{},{},{},{},{}",
            row.model, row.operation, row.n, row.median_ns, row.min_ns, row.max_ns, row.repeats
        )
        .expect("write row");
    }
    out.flush().expect("flush output");
}

/// The sample vector for `(model, n)` from the fixture.
fn samples_for(fx: &Fixture, model: Model, n: usize) -> &[f64] {
    match model {
        Model::Norm => fx.samples.norm.get(n),
        Model::GenExtreme => fx.samples.genextreme.get(n),
        Model::Gamma => fx.samples.gamma.get(n),
    }
}
