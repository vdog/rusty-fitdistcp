# fitdistcp — Rust Port

A Rust reimplementation of the R/Python package `fitdistcp` (calibrating prior
prediction), with thin R (extendr) and Python (PyO3) bindings over one shared,
binding-agnostic numerics core.

See [`plans/fitdistcp-rust-port-design.md`](plans/fitdistcp-rust-port-design.md)
for the full design and phased plan.

## Status: Phase 0 — Docker dev environment + scaffolding

Phase 0 delivers the reproducible dev environment and the Cargo workspace
skeleton. No numerics yet — the goal is to prove that **extendr and PyO3 can
both build against one shared core crate** and return identical results, all
inside Docker.

## Layout

```
fitdistcp/
├── Cargo.toml                  # workspace manifest
├── rust-toolchain.toml         # pinned Rust 1.85.0
├── Dockerfile                  # Ubuntu 26.04 + Rust + R + Python toolchains
├── docker-compose.yml          # dev wrapper + cache volumes
├── .devcontainer/              # VS Code devcontainer
├── crates/
│   ├── fitdistcp-core/         # binding-agnostic numerics (no extendr/pyo3)
│   ├── fitdistcp-r/            # extendr binding (R package skeleton)
│   └── fitdistcp-py/           # PyO3 binding (maturin)
├── tests/golden/               # (Phase 1f) R + Python golden fixtures
└── benches/                    # (Phase 1e) criterion benchmarks
```

## Prerequisites

Only **Docker** (or Podman). No Rust, R, or Python installs are required on the
host — the container is the single source of truth for all toolchains.

The image is based on **Ubuntu 26.04 LTS** and pins **Rust 1.85.0** (via
`rust-toolchain.toml`), **R 4.6.x** (CRAN apt repo), and the system **Python
3.14** (with PyO3 0.30 / numpy 0.30, which support it).

## Quick start

```sh
# Build the image once.
docker compose build

# Interactive shell with the workspace bind-mounted.
docker compose run --rm dev bash

# Rust core: build, test, lint.
docker compose run --rm dev cargo build
docker compose run --rm dev cargo test
docker compose run --rm dev cargo clippy --all-targets -- -D warnings

# R binding (Phase 0 smoke test).
docker compose run --rm dev bash -lc "cd crates/fitdistcp-r && R -e \"devtools::load_all(quiet=TRUE); print(core_version()); print(add_vectors(c(1,2,3), c(4,5,6))); print(sum_values(c(1,2,3,4)))\""

# Python binding (Phase 0 smoke test).
docker compose run --rm dev bash -lc "cd crates/fitdistcp-py && python3 -m venv .venv && . .venv/bin/activate && pip install --quiet numpy && maturin develop --quiet && python -c 'import fitdistcp, numpy as np; print(fitdistcp.core_version()); print(fitdistcp.add_vectors(np.array([1.,2.,3.]), np.array([4.,5.,6.]))); print(fitdistcp.sum_values(np.array([1.,2.,3.,4.])))'"
```

## Continuous integration (W20)

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
  `testthat::test_dir()` workflow. (`devtools::install()` / `R CMD INSTALL` fail
  here because the R package's `Cargo.toml` is a member of the top-level Cargo
  workspace; `load_all` is the supported path.)
- **`python`** — `maturin develop` + `pytest` in a venv.

Triggered on every `push` and `pull_request`. Multi-platform R checks
(Linux/macOS/Windows) and `cibuildwheel` wheels are **W21 (packaging)** and are
intentionally out of scope for W20.

The workflow YAML is validated with `actionlint` (and `pyyaml`); the underlying
commands are verified locally inside the dev container.

## Phase 0 milestone

Container builds; both bindings call a trivial core function
(`add_vectors` / `sum_values`) and return identical results, all inside Docker.

## Roadmap

- **Phase 1a** — core numerics foundation (special functions, MLE, derivatives)
- **Phase 1b** — DMGS expansion
- **Phase 1c** — DMGS models + edge cases
- **Phase 1d** — bindings (R `d/p/q/r` + `t***_cp`, Python `pdf/cdf/ppf/rvs`)
- **Phase 1e** — performance (Rayon, SIMD, caching)
- **Phase 1f** — validation & CI
- **Phase 2** — posterior sampling (deferred)
