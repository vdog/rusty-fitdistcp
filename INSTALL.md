# Installing `rusty-fitdistcp` (Rust port)

This guide covers installing the Rust-backed `fitdistcp` bindings as **drop-in
replacements** for the reference R and Python packages.

There are two supported install paths for each language:

- **End-user install** — install a prebuilt artifact from a
  [GitHub release](https://github.com/stephenjewson/fitdistcp/releases). No
  repository checkout required.
- **Dev-container install** — build and load the bindings from source inside the
  project's Docker dev container. This is the supported path for development and
  for running the test suites.

> **Distribution model.** Artifacts are published to **GitHub releases only** —
> there is **no PyPI and no CRAN** publishing step. See
> [`.github/workflows/release.yml`](.github/workflows/release.yml:1).

## Which install path should I use?

| I want to… | Language | Path | Artifact |
|---|---|---|---|
| Use the package in my own project | Python | [End-user (Python)](#python-end-user-install) | `*.whl` wheel or `*.tar.gz` sdist |
| Use the package in my own project | R | [End-user (R)](#r-end-user-install) | `rusty.fitdistcp_*.tar.gz` source tarball |
| Develop / run the test suite | Python | [Dev container (Python)](#python-dev-container-install) | `maturin develop` |
| Develop / run the test suite | R | [Dev container (R)](#r-dev-container-install) | `devtools::load_all()` |

## Prerequisites

### End-user install

- **Python:** CPython **>= 3.9** and `pip`. Prebuilt wheels are published for
  **3.9–3.14** on manylinux, macOS (Intel + Apple Silicon), and Windows; on
  other versions install from the sdist, which requires a Rust toolchain +
  `maturin`. `numpy>=1.22` is the only runtime dependency. On Linux, the
  3.9–3.13 wheels target glibc 2.17 (`manylinux2014`) and the 3.14 wheels
  target glibc 2.28 (`manylinux_2_28`).
- **R:** **R >= 4.0** and a **Rust toolchain** (`cargo` + `rustc >= 1.85`). The R
  package compiles a Rust staticlib at install time, so a Rust toolchain must be
  on `PATH`. The Rust crates are vendored into the tarball, so **no network
  access is required** for a release tarball.

### Dev-container install

- **Docker** (or Podman). No Rust, R, or Python installs are required on the
  host — the container is the single source of truth for all toolchains. See the
  [root README](README.md:150) for the full environment description.

## Python: end-user install

Download the wheel (or sdist) for your platform from the
[GitHub releases page](https://github.com/stephenjewson/fitdistcp/releases), then
install it into your environment:

```sh
# From a downloaded wheel (recommended — no Rust toolchain needed).
pip install ./rusty_fitdistcp-0.0.4-cp312-cp312-manylinux_2_17_x86_64.whl

# Or from the source distribution (requires a Rust toolchain + maturin).
pip install ./rusty_fitdistcp-0.0.4.tar.gz
```

The package installs as `rusty_fitdistcp` and is a **drop-in replacement** for the
reference `fitdistcp` v0.0.4 Python package: the submodule API, function
signatures/defaults, and return-dict keys match the reference.

### Verify the drop-in install

```python
import rusty_fitdistcp

# Submodule API (reference-compatible).
rusty_fitdistcp.norm.ppf([1.0, 2.0, 3.0], p=[0.1, 0.5, 0.9])
rusty_fitdistcp.genextreme.ppf([1.0, 2.0, 3.0], p=0.5)
rusty_fitdistcp.genpareto.pdf([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
rusty_fitdistcp.reltest([1.0, 2.0, 3.0])
```

The legacy **flat** API (`rusty_fitdistcp.ppf_norm`, `rusty_fitdistcp.pdf_gev`, ...) is
retained as deprecated aliases and emits a `DeprecationWarning`; new code should
use the submodule API. See
[`crates/fitdistcp-py/README.md`](crates/fitdistcp-py/README.md:1) for the full
surface.

## Python: dev-container install

Build and load the extension from source inside the dev container:

```sh
# Build the image once.
docker compose build

# Create a venv, install test deps, build the extension, and run the suite.
docker compose run --rm dev bash -lc \
  "cd crates/fitdistcp-py && python3 -m venv .venv && . .venv/bin/activate \
   && pip install --quiet numpy pytest && maturin develop --quiet && pytest"
```

`maturin develop` compiles the PyO3 extension (`rusty_fitdistcp._fitdistcp`) and
installs the thin Python submodules from
[`python/rusty_fitdistcp/`](crates/fitdistcp-py/python/rusty_fitdistcp/__init__.py:1) into the
active virtualenv. After it completes, `import rusty_fitdistcp` works in that venv.

To build a distributable wheel locally:

```sh
docker compose run --rm dev bash -lc \
  "cd crates/fitdistcp-py && maturin build --out dist"
```

## R: end-user install

Download the source tarball (`rusty.fitdistcp_*.tar.gz`) from the
[GitHub releases page](https://github.com/stephenjewson/fitdistcp/releases), then
install it:

```sh
# Requires R >= 4.0 and a Rust toolchain (cargo + rustc >= 1.85) on PATH.
R CMD INSTALL rusty.fitdistcp_0.1.0.tar.gz
```

Or from within R:

```r
install.packages("rusty.fitdistcp_0.1.0.tar.gz", repos = NULL, type = "source")
```

The tarball ships the Rust crates **vendored** (via
[`tools/vendor-r-crate.sh`](tools/vendor-r-crate.sh:1)), so `R CMD INSTALL`
builds offline. The package's [`configure`](crates/fitdistcp-r/configure:1)
script prepares the Rust build and falls back to `cargo vendor` (which needs
network access) only if the vendor tree is missing.

The package is a **drop-in replacement** for the reference R `fitdistcp` API: it
provides the `d/p/q/r` family plus the `t***_cp` calibrated-prior entry point for
all ten Phase-1 models (53 exported functions).

### Verify the drop-in install

```r
library(rusty.fitdistcp)

# d/p/q/r family plus the calibrated-prior entry point.
dnorm_cp(0, x = c(-1, 0, 1))
pnorm_cp(0, x = c(-1, 0, 1))
qnorm_cp(c(0.1, 0.5, 0.9), x = c(-1, 0, 1))
rnorm_cp(3, x = c(-1, 0, 1))
tnorm_cp(x = c(-1, 0, 1))

# Shared helpers.
core_version()
```

See [`crates/fitdistcp-r/NAMESPACE`](crates/fitdistcp-r/NAMESPACE:1) for the full
exported surface.

## R: dev-container install

The R package's `Cargo.toml` is a member of the top-level Cargo workspace, so
`devtools::load_all()` is the supported development path (rather than
`R CMD INSTALL`):

```sh
# Build the image once.
docker compose build

# Load the package and run the testthat suite.
docker compose run --rm dev bash -lc \
  "cd crates/fitdistcp-r && R -e \"devtools::load_all(quiet=TRUE); testthat::test_dir('tests/testthat')\""
```

To build the CRAN-style source tarball locally (vendoring the Rust crates first):

```sh
docker compose run --rm dev bash -lc \
  "bash tools/vendor-r-crate.sh && cd crates/fitdistcp-r && R CMD build ."
```

## Troubleshooting

### R install fails: `cargo` / `rustc` not found

`R CMD INSTALL` compiles a Rust staticlib, so a Rust toolchain (`cargo` +
`rustc >= 1.85`) must be on `PATH`. Install Rust via
[rustup](https://rustup.rs/) or use the dev container, which ships the pinned
toolchain.

### R install fails: vendored crates missing / network error

Release tarballs ship the vendored crate tree, so no network is needed. If you
are installing from a **source checkout** rather than a release tarball, the
[`configure`](crates/fitdistcp-r/configure:1) script runs `cargo vendor`, which
requires network access. Run
[`tools/vendor-r-crate.sh`](tools/vendor-r-crate.sh:1) first (inside the dev
container) to populate the vendor tree.

### Python: no wheel for my platform / Python version

Prebuilt wheels are published for CPython **3.9–3.14** on manylinux, macOS
(Intel + Apple Silicon), and Windows. The package itself supports CPython
**>= 3.9** (`requires-python = ">=3.9"`), so newer versions work too — they
just have no prebuilt wheel yet. If no wheel matches your platform or Python
version, install from the **sdist** (`pip install ./rusty_fitdistcp-*.tar.gz`), which
requires a Rust toolchain and `maturin`.

### Python: `DeprecationWarning` on a flat name

Accessing a legacy flat name (e.g. `rusty_fitdistcp.ppf_norm`) emits a
`DeprecationWarning` and forwards to the corresponding submodule function. Switch
to the submodule API (`rusty_fitdistcp.norm.ppf(...)`).

### Python test suite: one skipped test

The full pytest suite is **109 passed, 1 skipped**. The skip is the gamma
`reltest` case — a pre-existing core robustness issue in
`crates/fitdistcp-core`'s `special::inc_gamma` when the gamma MLE optimizer
wanders to a non-positive shape. It is documented and is not a binding
regression.

### `ru` / posterior sampling is a placeholder

The `ru` argument is accepted but returns placeholder strings (e.g.
`"ru not selected"`) and is disabled by default. Real ratio-of-uniforms posterior
sampling is deferred to Phase 2. See the
[root README](README.md:321) for the full list of deferred items.
