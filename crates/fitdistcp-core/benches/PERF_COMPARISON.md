# Cross-Language Performance Comparison — Canonical Workload Spec

This document is the **single source of truth** for the cross-language
execution-time benchmark described in
[`plans/cross-language-benchmark-plan.md`](../../../plans/cross-language-benchmark-plan.md).
All three harnesses — the Rust port in this workspace, the reference R package
(`stephenjewson/fitdistcp` v0.2.3, vendored at `tools/fitdistcp-ref/`), and the
Python module (`lynnejewson/fitdistcp`, cloned to `tools/fitdistcp-py-ref/`) —
MUST consume the exact inputs defined here so the numbers are comparable.

Nothing in this document changes numerics. It only fixes the *workload*.

---

## 1. Workload matrix

| Dimension | Values |
|---|---|
| Models | `norm`, `genextreme` (GEV), `gamma` |
| Operations | `pdf`, `cdf`, `ppf`, `rvs` |
| Sample sizes (n) | 50, 500, 5000 |
| Prediction grid | 100 points (`x` for pdf/cdf, `p` for ppf) |
| End-to-end | full `predict` (MLE + DMGS) at n = 500 |
| reltest | reliability simulation, matched `ntrials`/`nx`/`params` |
| Seeds | fixed; identical embedded data vectors for all three implementations |

The full cross product is `3 models × 4 operations × 3 sample sizes = 36`
pointwise workloads, plus 3 end-to-end `predict` workloads (one per model at
n = 500) and 3 reltest workloads (one per model).

### 1.1 Pointwise operations

For each `(model, operation, n)` the harness:

1. reads the embedded sample `y` of length `n` from the shared fixture
   ([`perf_inputs.json`](perf_inputs.json)),
2. fits the model once (MLE) **outside** the timed region,
3. times the operation over the 100-point prediction grid.

| Operation | Rust port | Reference R | Python module |
|---|---|---|---|
| `pdf` | `predict(...).cp_pdf` / `cp_pdf` | `dnorm_cp` / `dgev_cp` / `dgamma_cp` | `fitdistcp.<model>.pdf(x, y=...)['cp_pdf']` |
| `cdf` | `predict(...).cp_cdf` / `cp_cdf` | `pnorm_cp` / `pgev_cp` / `pgamma_cp` | `fitdistcp.<model>.cdf(x, y=...)['cp_cdf']` |
| `ppf` | `predict(...).cp_quantiles` / `cp_quantiles` | `qnorm_cp` / `qgev_cp` / `qgamma_cp` | `fitdistcp.<model>.ppf(x, p=...)['cp_quantiles']` |
| `rvs` | `predict(...).cp_rvs` / `cp_rvs` | `rnorm_cp` / `rgev_cp` / `rgamma_cp` | `fitdistcp.<model>.rvs(n, x)['cp_deviates']` |

### 1.2 End-to-end `predict` (MLE + DMGS)

One workload per model at **n = 500**, timing the *entire* pipeline (MLE fit +
DMGS expansion) on the same 100-point grid. This is the quantity that exposes
the algorithmic difference between the reference `optim` (Nelder–Mead) and the
Rust BFGS + Newton path.

| Model | Rust port | Reference R | Python module |
|---|---|---|---|
| `norm` | `predict(Normal, req, init)` | `qnorm_cp` (full pipeline) | `fitdistcp.norm.ppf(...)` |
| `genextreme` | `predict(GenExtreme, req, init)` | `qgev_cp` (full pipeline) | `fitdistcp.genextreme.ppf(...)` |
| `gamma` | `predict(Gamma, req, init)` | `qgamma_cp` (full pipeline) | `fitdistcp.gamma.ppf(...)` |

### 1.3 reltest (reliability simulation)

A Monte Carlo workload: repeatedly simulate a training sample from a *known*
model, fit it, form MLE and CP predictive quantiles, and score them against the
true distribution. Because it is Monte Carlo, the committed run uses a **small,
documented** configuration so it finishes in CI:

| Parameter | Value | Rationale |
|---|---|---|
| `ntrials` | **1000** | reference default; enough for a stable aggregate |
| `nx` | **20** | reference default; small training sample per trial |
| `nrepeats` | 1 | one repeat per trial (reference default) |
| `params` | per-model, see §1.3.1 | matched across implementations |
| seed | `0x5EED_1234_5678_9ABC` | fixed; matches the Rust `reltest` test seed |

> **Note.** The reference `reltest`/`reltest2` signature is
> `reltest(model, ntrials, nrepeats, nx, params, ...)`. The committed choice is
> `ntrials = 1000`, `nx = 20`, `nrepeats = 1`. The Rust `reltest` test uses a
> smaller `N_TRIALS = 400` for CI speed; the benchmark uses the reference
> default `ntrials = 1000` so the three implementations are matched.

#### 1.3.1 reltest parameters (matched across implementations)

| Model | `params` (Rust order) | Reference R order | Notes |
|---|---|---|---|
| `norm` | `[mu, sigma]` = `[0.0, 1.0]` | `[mean, sd]` | identical |
| `genextreme` | `[loc, scale, shape]` = `[0.0, 1.0, 0.1]` | `[loc, scale, shape]` | identical |
| `gamma` | `[shape, rate]` = `[2.0, 1.0]` | `[shape, scale]` = `[2.0, 1.0]` | **rate vs scale** — see §5 |

---

## 2. Prediction grid

The 100-point grid is defined identically in all three harnesses and is stored
in the shared fixture ([`perf_inputs.json`](perf_inputs.json)) as `grid.x` and
`grid.p`:

- **`x`** (for `pdf`/`cdf`): 100 equally spaced points on `[-3.0, 3.0]`,
  `x_i = -3.0 + 6.0 * i / 100` for `i = 0..99`.
- **`p`** (for `ppf`): 100 midpoints of equal-probability bins,
  `p_i = (i + 0.5) / 100` for `i = 0..99`.

These match the grid used by the existing
[`mle_fitting.rs`](mle_fitting.rs) end-to-end bench, so the Rust Criterion
baseline and the cross-language CSV are directly comparable.

---

## 3. Embedded data vectors (fixed seeds)

All three harnesses MUST consume the **same** data vectors. They are stored as
literal arrays in the shared fixture
[`crates/fitdistcp-core/benches/perf_inputs.json`](perf_inputs.json) so that R,
Python, and Rust read byte-identical inputs.

### 3.1 Generation procedure (documented LCG)

The vectors are generated by a 64-bit linear congruential generator (LCG) with
the constants used by the existing Rust benches:

```
state_0 = seed
state_{k+1} = state_k * 6364136223846793005 + 1442695040888963407   (mod 2^64)
u_k = (state_k >> 11) / 2^53
```

- **`norm` / `genextreme` samples** use the Irwin–Hall transform: sum 12
  successive `u_k` and subtract 6, giving an approximately standard-normal
  deviate `z`. The stored vector is `z` (mean ≈ 0, sd ≈ 1).
- **`gamma` samples** use `positive_sample`: `abs(z + 6.0) + 0.1`, giving a
  strictly positive vector suitable for a `(shape, rate)` fit.

### 3.2 Seeds per `(model, n)`

| Model | n | Seed (hex) | Transform |
|---|---|---|---|
| `norm` | 50 | `0xaaaa0001` | Irwin–Hall |
| `norm` | 500 | `0xaaaa0001` | Irwin–Hall |
| `norm` | 5000 | `0xaaaa0001` | Irwin–Hall |
| `gamma` | 50 | `0xaaaa0002` | positive |
| `gamma` | 500 | `0xaaaa0002` | positive |
| `gamma` | 5000 | `0xaaaa0002` | positive |
| `genextreme` | 50 | `0xaaaa0003` | Irwin–Hall |
| `genextreme` | 500 | `0xaaaa0003` | Irwin–Hall |
| `genextreme` | 5000 | `0xaaaa0003` | Irwin–Hall |

The end-to-end `predict` workloads at n = 500 use the `0xbbbb0001` /
`0xbbbb0002` / `0xbbbb0003` seeds (norm / gamma / genextreme respectively),
matching [`mle_fitting.rs`](mle_fitting.rs).

> **Fixture format.** `perf_inputs.json` stores the *materialized* arrays (not
> the seeds) so that R and Python — which do not share the Rust LCG — read the
> exact same numbers. The seeds above are recorded for provenance and for
> regenerating the fixture. The fixture is the contract; the seeds are the
> recipe.

### 3.3 Fixture schema

```jsonc
{
  "meta": {
    "lcg": { "multiplier": 6364136223846793005, "increment": 1442695040888963407, "shift": 11 },
    "generated_by": "crates/fitdistcp-core/benches/perf_inputs.json",
    "note": "Materialized arrays; seeds recorded for provenance only."
  },
  "grid": { "x": [ /* 100 f64 */ ], "p": [ /* 100 f64 */ ] },
  "samples": {
    "norm":       { "50": [ /* ... */ ], "500": [ /* ... */ ], "5000": [ /* ... */ ] },
    "gamma":      { "50": [ /* ... */ ], "500": [ /* ... */ ], "5000": [ /* ... */ ] },
    "genextreme": { "50": [ /* ... */ ], "500": [ /* ... */ ], "5000": [ /* ... */ ] }
  },
  "predict": {
    "norm":       { "n": 500, "y": [ /* ... */ ] },
    "gamma":      { "n": 500, "y": [ /* ... */ ] },
    "genextreme": { "n": 500, "y": [ /* ... */ ] }
  },
  "reltest": {
    "ntrials": 1000, "nx": 20, "nrepeats": 1, "seed": 0x5EED123456789ABC,
    "params": {
      "norm":       [0.0, 1.0],
      "genextreme": [0.0, 1.0, 0.1],
      "gamma":      [2.0, 1.0]
    }
  }
}
```

---

## 4. API mapping across implementations

| Operation | Rust port | Reference R | Python module |
|---|---|---|---|
| pdf | `predict(...)` / `cp_pdf` | `dnorm_cp` / `dgev_cp` / `dgamma_cp` | `fitdistcp.<model>.pdf(x, y=...)['cp_pdf']` |
| cdf | `predict(...)` / `cp_cdf` | `pnorm_cp` / `pgev_cp` / `pgamma_cp` | `fitdistcp.<model>.cdf(x, y=...)['cp_cdf']` |
| ppf | `predict(...)` / `cp_quantiles` | `qnorm_cp` / `qgev_cp` / `qgamma_cp` | `fitdistcp.<model>.ppf(x, p=...)['cp_quantiles']` |
| rvs | `predict(...)` / `cp_rvs` | `rnorm_cp` / `rgev_cp` / `rgamma_cp` | `fitdistcp.<model>.rvs(n, x)['cp_deviates']` |
| end-to-end | `predict(model, req, init)` | `q*_cp` (full pipeline) | `fitdistcp.<model>.ppf(...)` |
| reltest | `fitdistcp_core` reltest port | `reltest` / `reltest2` | `fitdistcp.reltest.reltest_<model>` |

Confirmed reference entry points:

- R: `reltest(model, ntrials, nrepeats, nx, params, ...)` and
  `reltest2(model, ntrials, nrepeats, nx, params, ...)` in
  `tools/fitdistcp-ref/R/002a_reltest.R` and `003a_reltest2.R`.
- Python: `fitdistcp.reltest.reltest_gumbel(file, ntrials=200, nx=50)` per the
  reference docs.

---

## 5. Fairness caveats

These are recorded in the results header so the numbers are not misread:

1. **Optimizer algorithm differs.** The reference R package uses `optim`
   (Nelder–Mead, `reltol = 1e-8`); the Rust port uses BFGS with an analytic
   score plus Newton–Raphson refinements. This is an **algorithmic** difference,
   not merely a language one, and dominates the end-to-end `predict` numbers.
2. **Interpreted vs compiled.** The Python module is pure Python (interpreted);
   the Rust port is compiled. Pointwise operations therefore show a large
   constant-factor gap that is a language/runtime effect, not an algorithmic one.
3. **Monte Carlo reltest.** reltest is a stochastic workload. The three
   implementations are matched on `ntrials`/`nx`/`params` and use a fixed seed
   where the implementation allows seeding; residual differences are sampling
   noise, not performance.
4. **Parameterization.** The reference gamma uses `(shape, scale)`; the Rust
   port uses `(shape, rate)`. The harness converts so the *same distribution* is
   fit (`scale = 1 / rate`). The reltest `params` in §1.3.1 are given in each
   implementation's native order.
5. **Criterion vs wall-clock.** Criterion numbers and the CSV `Instant` numbers
   may differ slightly. The CSV is the cross-implementation comparison source;
   Criterion remains the Rust-internal regression baseline.
6. **Gamma domain restriction.** The shared prediction grid `grid.x` spans
   `[-3, 3]`, but the gamma distribution has support `x > 0`. The reference R
   gamma routines hard-reject non-positive `x` (`stopifnot(!x < 0)`), and the
   reference Python gamma derivative code raises at `x = 0` (`log(0) = -inf`).
   To keep the three implementations comparable, the gamma `pdf`/`cdf`/`rvs`
   workloads are evaluated on the **strictly-positive subset** of the grid
   (`x > 0`, 49 of the 100 points) in all three harnesses. The `ppf` workload
   (which uses `p ∈ (0, 1)`) and the end-to-end `predict` workload (which uses
   `p`) are unaffected. The `(model, operation, n)` keys are identical across
   implementations.
7. **R timer resolution.** R's `proc.time()[["elapsed"]]` has ~1 ms resolution,
   so the R harness times a calibrated *batch* of calls per sample and divides
   by the batch size (see `time_workload` in `tools/bench/bench_ref_r.R`). This
   makes sub-millisecond workloads (e.g. the norm pointwise operations)
   measurable; the reported per-call nanoseconds are quantized to roughly
   `1 ms / batch_size`.

---

## 6. Methodology

- **Warmup**: discard the first N iterations (default **3**).
- **Repeats**: run each workload R times (default **20**); report **median** +
  min/max.
- **Timing source**:
  - Rust: Criterion (statistical) + `std::time::Instant` loop for the CSV.
  - Reference R: `system.time()` / `proc.time()` deltas.
  - Python module: `time.perf_counter_ns`.
- **Isolation**: inputs pre-generated outside the timed region; each workload
  timed in its own loop.
- **Determinism**: fixed seeds; identical embedded data vectors (§3) so all
  three implementations fit the same data.
- **Environment capture**: host, container image tag, Rust/R/Python versions,
  CPU count, and reference git commit SHAs in the results header.

---

## 7. Output CSV schema

Every harness emits rows with exactly this header:

```
implementation,model,operation,n,median_ns,min_ns,max_ns,repeats
```

- `implementation` ∈ {`rust`, `ref_r`, `py_module`}.
- `model` ∈ {`norm`, `genextreme`, `gamma`}.
- `operation` ∈ {`pdf`, `cdf`, `ppf`, `rvs`, `predict`, `reltest`}.
- `n` is the sample size (for `predict`/`reltest`, the configured `n`/`nx`).
- `median_ns`, `min_ns`, `max_ns` are integer nanoseconds.
- `repeats` is the number of timed iterations (after warmup).

Example:

```
implementation,model,operation,n,median_ns,min_ns,max_ns,repeats
rust,norm,pdf,500,123456,120000,130000,20
ref_r,norm,pdf,500,234567,230000,240000,20
py_module,norm,pdf,500,345678,340000,350000,20
```

The aggregator (`tools/bench/aggregate_perf.py`) pivots this into one row per
`(model, operation, n)` with columns `Rust`, `Reference R`, `Python module`,
and `R/Rust`, `Py/Rust` ratios.
