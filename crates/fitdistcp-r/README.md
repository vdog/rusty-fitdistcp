# rusty.fitdistcp (R bindings)

extendr bindings over `fitdistcp-core`. The R package is a **drop-in
replacement** for the reference R `fitdistcp` API: it provides the `d/p/q/r`
family plus the `t***_cp` calibrated-prior entry point for all ten Phase-1
models (53 exported functions), returning the reference field names
(`ml_pdf`, `cp_pdf`, `ml_quantiles`, `cp_quantiles`, `ml_params`, AIC/MAIC,
standard errors, convergence, and the small-n flag).

The numerics live in the shared Rust core (`fitdistcp-core`); the R package
compiles that core into a staticlib via extendr.

## Models

`norm`, `norm_p1`, `genextreme` (GEV), `genextreme_p1`, `genpareto` (GPD),
`gumbel`, `lnorm`, `weibull`, `gamma`, `expon`.

For each model the package exports the `d/p/q/r` family plus the `t***_cp`
calibrated-prior entry point, e.g. `dnorm_cp`, `pnorm_cp`, `qnorm_cp`,
`rnorm_cp`, `tnorm_cp`. See [`NAMESPACE`](NAMESPACE:1) for the full exported
surface.

## Behaviour notes (Issue 2 re-check)

- **`_p1` / `_p12` models require exactly one of `t0` / `n0`.** `qgev_p1_cp`,
  `qnorm_p1_cp`, `qgev_p12_cp` (and the other `_p1`/`_p12` entry points) raise
  an R error when neither is supplied (`"Either t0 or n0 must be specified"`) or
  when both are (`"Only one of t0 or n0 must be specified"`), matching the
  reference `maket0`. Earlier versions silently defaulted `t0 = 0`.
- **`degenerate_fit`** is a new logical field in the `q/d/p/r_cp` return lists.
  It is `TRUE` when a GEV fit is degenerate (a spurious `scale -> 0` maximum on
  a very short record); in that case `cp_quantiles` / `cp_pdf` are `NaN` rather
  than huge finite numbers. It is `FALSE` for all other models/paths.
- **`cp_monotone_clamped`** is `TRUE` when the Rust-only monotonicity guard
  raised a CP quantile to keep the curve non-decreasing (see
  [`tests/golden/TOLERANCE.md`](../../tests/golden/TOLERANCE.md) §7).
- **GEV CP quantiles are not clamped** to the ML upper endpoint (the reference
  does not clamp either), and **GPD fits with shape `<= -0.98` revert to the ML
  curve**. See the top-level [`README.md`](../../README.md) "Known differences"
  section.

## Installation

See [`INSTALL.md`](../../INSTALL.md:1) for the full guide (including the Python
binding). The R-specific paths are below.

### End-user install (from a GitHub release)

Download the source tarball (`rusty.fitdistcp_*.tar.gz`) from the
[GitHub releases page](https://github.com/stephenjewson/fitdistcp/releases), then:

```sh
# Requires R >= 4.0 and a Rust toolchain (cargo + rustc >= 1.85) on PATH.
R CMD INSTALL rusty.fitdistcp_0.1.0.tar.gz
```

Or from within R:

```r
install.packages("rusty.fitdistcp_0.1.0.tar.gz", repos = NULL, type = "source")
```

The tarball ships the Rust crates **vendored** (via
[`tools/vendor-r-crate.sh`](../../tools/vendor-r-crate.sh:1)), so `R CMD INSTALL`
builds offline. The package's [`configure`](configure:1) script prepares the Rust
build and falls back to `cargo vendor` (which needs network access) only if the
vendor tree is missing. Artifacts are distributed via **GitHub releases only** —
there is no CRAN publishing step.

Verify the drop-in install:

```r
library(rusty.fitdistcp)
qnorm_cp(c(0.1, 0.5, 0.9), x = c(-1, 0, 1))
```

### Dev-container install (from source)

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

## Packaging status

`R CMD build` / `INSTALL` / `check` succeeds with **1 WARNING + 1 NOTE**, both
inherent to Rust-backed R packages. Rust crates are vendored for standalone
builds via [`tools/vendor-r-crate.sh`](../../tools/vendor-r-crate.sh:1), with
[`configure`](configure:1) / [`configure.win`](configure.win:1) build hooks.
