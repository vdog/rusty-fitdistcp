# fitdistcp — Rust Port

A Rust reimplementation of the R/Python package `fitdistcp` (calibrating prior
prediction), with thin R (extendr) and Python (PyO3) bindings over one shared,
binding-agnostic numerics core.

`fitdistcp` calibrates a prior prediction by correcting an MLE-based predictive
density with a DMGS asymptotic-expansion bias term, producing a **calibrated
prior (CP)** predictive density and quantiles. This port provides the same
`d/p/q/r` + `t***_cp` (R) and `pdf/cdf/ppf/rvs` (Python) surface, backed by a
single Rust core so both bindings return identical numerics.

See [`plans/fitdistcp-rust-port-design.md`](plans/fitdistcp-rust-port-design.md)
for the full design and phased plan.

## Status: Phase 1 complete

Phase 1 (the standard DMGS asymptotic-expansion integration path) is
**implemented, tested, benchmarked, and packaged**. All quality gates are green:

- **Core numerics** — special functions, MLE fitting, analytic-first
  derivatives, DMGS expansion, and 10 models are implemented in
  `fitdistcp-core`.
- **Bindings** — R (53 exported functions) and Python (submodule API, a drop-in
  replacement for the reference `fitdistcp` v0.0.4 Python package) are complete,
  with predictor handling for `norm_p1` / `genextreme_p1`.
- **Tests** — 144 core unit tests + 17 golden/integration tests + 141 R testthat
  assertions + 109 Python pytest tests (1 skipped), all passing.
- **Quality gates** — `cargo clippy --workspace --all-targets -- -D warnings`
  and `cargo fmt --check` are clean.
- **Packaging** — `R CMD build`/`INSTALL`/`check` succeeds (1 WARNING + 1 NOTE,
  both inherent to Rust-backed CRAN packages and documented in
  [`cran-comments.md`](crates/fitdistcp-r/cran-comments.md)); `maturin build`
  produces a working wheel.

**Phase 2 (posterior sampling) is deferred** — see
[Known differences / limitations](#known-differences--limitations).

## Repository layout

```
fitdistcp/
├── Cargo.toml                       # workspace manifest
├── rust-toolchain.toml              # pinned Rust 1.85.0
├── Dockerfile                       # Ubuntu + Rust + R + Python toolchains
├── docker-compose.yml               # dev wrapper + cache volumes
├── .devcontainer/                   # VS Code devcontainer
├── crates/
│   ├── fitdistcp-core/              # binding-agnostic numerics (no extendr/pyo3)
│   │   ├── src/
│   │   │   ├── special/             # ln_gamma, gamma, digamma, trigamma, tetragamma,
│   │   │   │                        #   ln_beta, erf, erfc, Student-t helpers
│   │   │   ├── models/              # 10 Phase-1 models + Derivatives trait + numerical fallback
│   │   │   ├── mle/                 # BFGS primary + Newton fallback, SEs, AIC, small-n flag
│   │   │   ├── dmgs/                # DMGS expansion + Haar-prior terms + bias correction
│   │   │   └── api.rs               # shared binding-agnostic structs + entry points
│   │   ├── benches/                 # Criterion benchmarks + baselines + HOT_LOOPS.md
│   │   └── tests/                   # golden/integration harnesses
│   ├── fitdistcp-r/                 # extendr binding (R package)
│   │   ├── R/extendr-wrappers.R     # auto-generated wrappers
│   │   ├── man/                     # 53 generated .Rd pages
│   │   ├── NAMESPACE, DESCRIPTION   # CRAN-ready
│   │   ├── configure, configure.win # vendored-crate build hooks
│   │   └── cran-comments.md
│   └── fitdistcp-py/                # PyO3 binding (maturin)
│       ├── src/lib.rs               # private extension module (fitdistcp._fitdistcp)
│       ├── python/fitdistcp/        # public submodule API (norm, genextreme, ...)
│       ├── tests/                   # pytest suite (drop-in compat, golden, examples)
│       └── pyproject.toml           # maturin + cibuildwheel config
├── tests/golden/                    # committed golden fixtures + tolerance policy
│   ├── mle_golden.json
│   ├── cp_golden.json
│   ├── gev_negshape_golden.json
│   ├── w11_cp_golden.json
│   ├── TOLERANCE.md
│   └── README.md
├── tools/                           # fixture generators + vendoring helpers
│   ├── generate_golden.R
│   ├── generate_cp_golden.R
│   ├── generate_gev_negshape_golden.R
│   ├── generate_w11_cp_golden.R
│   └── vendor-r-crate.sh
└── .github/workflows/               # ci.yml + release.yml
```

## Implemented API surface

### Models (10, Phase 1)

`norm`, `norm_p1`, `genextreme` (GEV), `genextreme_p1`, `genpareto` (GPD),
`gumbel`, `lnorm`, `weibull`, `gamma`, `expon`.

`norm_p1` and `genextreme_p1` are the predictor variants (handling `t`, `t0`,
`n0`).

### R binding (`crates/fitdistcp-r`) — 53 exported functions

For each of the 10 models, the `d/p/q/r` family plus the `t***_cp` calibrated
prior entry point:

- `dnorm_cp`, `pnorm_cp`, `qnorm_cp`, `rnorm_cp`, `tnorm_cp`
- `dnorm_p1_cp`, `pnorm_p1_cp`, `qnorm_p1_cp`, `rnorm_p1_cp`, `tnorm_p1_cp`
- `dgev_cp`, `pgev_cp`, `qgev_cp`, `rgev_cp`, `tgev_cp`
- `dgev_p1_cp`, `pgev_p1_cp`, `qgev_p1_cp`, `rgev_p1_cp`, `tgev_p1_cp`
- `dgpd_k1_cp`, `pgpd_k1_cp`, `qgpd_k1_cp`, `rgpd_k1_cp`, `tgpd_k1_cp`
- `dgumbel_cp`, `pgumbel_cp`, `qgumbel_cp`, `rgumbel_cp`, `tgumbel_cp`
- `dlnorm_cp`, `plnorm_cp`, `qlnorm_cp`, `rlnorm_cp`, `tlnorm_cp`
- `dweibull_cp`, `pweibull_cp`, `qweibull_cp`, `rweibull_cp`, `tweibull_cp`
- `dgamma_cp`, `pgamma_cp`, `qgamma_cp`, `rgamma_cp`, `tgamma_cp`
- `dexp_cp`, `pexp_cp`, `qexp_cp`, `rexp_cp`, `texp_cp`

Plus the shared helpers `core_version`, `add_vectors`, `sum_values`,
`fitdistcp_params`. The package uses the `rextendr` layout; `NAMESPACE` and 53
man pages are generated, and `DESCRIPTION` is finalized for CRAN.

### Python binding (`crates/fitdistcp-py`) — drop-in submodule API

The Python binding is a **drop-in replacement** for the reference `fitdistcp`
v0.0.4 Python package (vendored read-only at
[`tools/fitdistcp-py-ref/`](tools/fitdistcp-py-ref/fitdistcp/__init__.py:1)). The
compiled extension module is private (`fitdistcp._fitdistcp`); the public surface
is the **submodule API**, mirroring the reference:

```python
import fitdistcp

fitdistcp.norm.ppf([1.0, 2.0, 3.0], p=[0.1, 0.5, 0.9])
fitdistcp.genextreme.ppf(x, p=0.5)
fitdistcp.genpareto.pdf(x, y)
fitdistcp.reltest(x)
```

Each submodule (`norm`, `genextreme`, `genpareto`, `expon`, `gamma`, `gumbel`,
`lnorm`, `weibull`, `genextreme_p1`, `reltest`) exposes `ppf` / `pdf` / `cdf` /
`rvs` (and `reltest` where applicable) with the reference signatures, defaults,
and return-dict keys.

The previous **flat API** (`fitdistcp.ppf_norm`, `fitdistcp.pdf_gev`, ...) is
retained as **deprecated aliases**: accessing a flat name emits a
`DeprecationWarning` (module-level `__getattr__`) and forwards to the
corresponding submodule function.

Built with PyO3 + maturin; `cibuildwheel` configuration lives in
[`pyproject.toml`](crates/fitdistcp-py/pyproject.toml).

### Shared core API (`crates/fitdistcp-core/src/api.rs`)

Binding-agnostic structs and entry points used by both bindings:
`FitResult`, `MleFit`, `PredictionRequest`, `Convergence`, plus `predict`,
`predict_gev`, `predict_norm_p1`, and `predict_gev_p1`.

## Prerequisites

Only **Docker** (or Podman). No Rust, R, or Python installs are required on the
host — the container is the single source of truth for all toolchains.

The image is based on **Ubuntu** and pins **Rust 1.85.0** (via
`rust-toolchain.toml`), **R** (CRAN apt repo), and the system **Python** (with
PyO3 / numpy versions that support it), plus `maturin`, `pytest`, and
`cibuildwheel`.

## Quick start

```sh
# Build the image once.
docker compose build

# Interactive shell with the workspace bind-mounted.
docker compose run --rm dev bash
```

### Rust core: build, test, lint

```sh
docker compose run --rm dev cargo build --workspace
docker compose run --rm dev cargo test --workspace
docker compose run --rm dev cargo clippy --workspace --all-targets -- -D warnings
docker compose run --rm dev cargo fmt --check
```

### R binding: testthat

The R package's `Cargo.toml` is a member of the top-level Cargo workspace, so
`devtools::load_all()` is the supported path (rather than `R CMD INSTALL`):

```sh
docker compose run --rm dev bash -lc \
  "cd crates/fitdistcp-r && R -e \"devtools::load_all(quiet=TRUE); testthat::test_dir('tests/testthat')\""
```

### Python binding: maturin develop + pytest

```sh
docker compose run --rm dev bash -lc \
  "cd crates/fitdistcp-py && python3 -m venv .venv && . .venv/bin/activate \
   && pip install --quiet numpy pytest && maturin develop --quiet && pytest"
```

## Benchmarks

Criterion benchmarks live under
[`crates/fitdistcp-core/benches/`](crates/fitdistcp-core/benches/) —
`derivatives`, `dmgs_expansion`, and `mle_fitting` — with committed baselines
(`*_baseline.txt`) and a hand-off note in
[`HOT_LOOPS.md`](crates/fitdistcp-core/benches/HOT_LOOPS.md).

```sh
docker compose run --rm dev cargo bench -p fitdistcp-core --bench derivatives
docker compose run --rm dev cargo bench -p fitdistcp-core --bench dmgs_expansion
docker compose run --rm dev cargo bench -p fitdistcp-core --bench mle_fitting
```

Performance optimizations have been applied (up to **16×** on mixed
derivatives, **2.1×** on gamma/GEV quantiles) with no tolerance regression; the
before/after numbers are recorded in `HOT_LOOPS.md`.

A cross-language execution-time comparison of the Rust port vs the reference R
package vs the Python module is committed in
[`crates/fitdistcp-core/benches/perf_comparison_results.md`](crates/fitdistcp-core/benches/perf_comparison_results.md)
(with a machine-readable
[`perf_comparison_results.csv`](crates/fitdistcp-core/benches/perf_comparison_results.csv));
the workload and methodology are defined in
[`PERF_COMPARISON.md`](crates/fitdistcp-core/benches/PERF_COMPARISON.md).
Regenerate it with [`tools/run-perf-comparison.sh`](tools/run-perf-comparison.sh).

The Python bench harness [`tools/bench/bench_ref_py.py`](tools/bench/bench_ref_py.py:1)
supports `--impl {ref,rust}` (or the `BENCH_IMPL` environment variable): `ref`
benchmarks the vendored reference package (`implementation = "py_module"`) and
`rust` benchmarks the drop-in binding (`implementation = "rust_binding"`).

## Golden fixtures & tolerance policy

The committed fixtures in [`tests/golden/`](tests/golden/) are generated from
the reference **`fitdistcp` v0.2.3** R package and loaded by the test harnesses
from disk (no network or reference package needed at test time). The tolerance
policy is the single source of truth in
[`tests/golden/TOLERANCE.md`](tests/golden/TOLERANCE.md); see
[`tests/golden/README.md`](tests/golden/README.md) for the fixture inventory.

### Regenerating the fixtures

All generation runs inside the Docker dev container (no host R toolchain
required):

```sh
# 1. Build the dev image (once).
docker compose build

# 2. Clone the reference package at the pinned version (once).
docker compose run --rm dev bash -lc \
  'git clone --depth 1 --branch v0.2.3 \
     https://github.com/stephenjewson/fitdistcp.git tools/fitdistcp-ref'

# 3. Regenerate each fixture (order is independent).
docker compose run --rm dev Rscript tools/generate_golden.R
docker compose run --rm dev Rscript tools/generate_cp_golden.R
docker compose run --rm dev Rscript tools/generate_gev_negshape_golden.R
docker compose run --rm dev Rscript tools/generate_w11_cp_golden.R

# 4. Confirm the suite still passes at the documented tolerances.
docker compose run --rm dev cargo test --workspace
```

The generators `source()` the reference package's own routines and embed fixed
datasets, so the fixtures are reproducible byte-for-byte. Do **not** silently
loosen a tolerance; any intentional relaxation must be recorded in
`TOLERANCE.md` with its rationale.

## Continuous integration & packaging

CI lives in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and **reuses
the same Docker image as local development** — the runner only needs Docker, and
every job runs the exact commands verified locally via
`docker compose run --rm dev ...`. This keeps CI and local runs identical.

Jobs:

- **`build-image`** — builds the dev image from `Dockerfile` (Buildx + GHA layer
  cache) and uploads it as an artifact so the test jobs reuse it instead of
  rebuilding.
- **`rust`** (matrix: `pinned` = 1.85.0 from `rust-toolchain.toml`, `stable`) —
  `cargo build --workspace`, `cargo test --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`,
  and `cargo bench --workspace --no-run` (bench compile check).
- **`r`** — the testthat suite via the documented `devtools::load_all()` +
  `testthat::test_dir()` workflow.
- **`python`** — `maturin develop` + `pytest` in a venv.

Triggered on every `push` and `pull_request`.

[`.github/workflows/release.yml`](.github/workflows/release.yml) handles
packaging: the R source tarball, `cibuildwheel` wheels, and PyPI publishing via
Trusted Publishing.

### Packaging status

- **R:** `R CMD build` / `INSTALL` / `check` succeeds with **1 WARNING + 1
  NOTE**, both inherent to Rust-backed CRAN packages and documented in
  [`cran-comments.md`](crates/fitdistcp-r/cran-comments.md). Rust crates are
  vendored for standalone builds via
  [`tools/vendor-r-crate.sh`](tools/vendor-r-crate.sh), with `configure` /
  `configure.win` build hooks.
- **Python:** `maturin build` produces a working wheel.

## Known differences / limitations

These are deliberate, documented differences from the reference package, not
bugs:

- **Gamma DMGS is parameterization-dependent.** The reference uses
  `(shape, scale)`; the Rust port uses `(shape, rate)`. Results agree after the
  parameterization is accounted for.
- **GEV third derivatives remain numerical.** The reference's `gev_logfddd` is a
  ~180-line machine-generated expression, impractical to transcribe; the port
  uses the numerical fallback behind the `Derivatives` trait.
- **Small-n handling.** DMGS degrades below `n ≈ 10`; the port flags small-n
  results rather than silently degrading, and the golden tolerance is relaxed
  for `n < 10` (see `TOLERANCE.md`).

The Python binding is a **drop-in replacement** for the reference `fitdistcp`
v0.0.4 Python package (submodule API), with the flat API retained as deprecated
aliases.

### Phase 2 (deferred)

The following are **not** implemented in Phase 1 and are explicitly deferred:

- `t***_cp` posterior sampling (Northrop `rust` interop) / real `ru`.
- Two-predictor GEV (`genextreme_p12`).

## Roadmap

- **Phase 0** — Docker dev environment + scaffolding ✅
- **Phase 1a** — core numerics foundation (special functions, MLE, derivatives) ✅
- **Phase 1b** — DMGS expansion ✅
- **Phase 1c** — DMGS models + edge cases ✅
- **Phase 1d** — bindings (R `d/p/q/r` + `t***_cp`, Python `pdf/cdf/ppf/rvs`) ✅
- **Phase 1e** — performance (benchmarks, hot-loop optimizations) ✅
- **Phase 1f** — validation, golden fixtures, CI, packaging ✅
- **Phase 2** — posterior sampling / real `ru`, two-predictor GEV (deferred)
