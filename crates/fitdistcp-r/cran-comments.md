# CRAN submission comments — fitdistcp 0.1.0

## Rust build / CRAN policy (design §11)

This package links a Rust static library built with Cargo (via `extendr`). To
satisfy CRAN's requirement that the package build in isolation from a source
tarball, the Rust sources are **vendored** into the package:

- `src/rust/` is a standalone Cargo workspace (it declares its own
  `[workspace]` table and does not inherit from any parent workspace).
- `src/rust/fitdistcp-core/` is a vendored copy of the numerics core crate.
- `src/rust/vendor/` contains all third-party crates, and
  `src/rust/.cargo/config.toml` redirects `crates.io` to that directory, so the
  build needs **no network access**.
- `configure` / `configure.win` verify the vendored tree is present (and
  regenerate it with `cargo vendor` only if a source checkout is missing it).

`SystemRequirements: Cargo (Rust's package manager), rustc (>= 1.85)`.

## Test environments

- Local: Ubuntu 26.04 (aarch64), R 4.6.2, rustc 1.85.0 (Docker dev image).

## R CMD check results

See the W21 verification notes in the repository (`plans/` and the CI workflow)
for the exact `R CMD build` / `R CMD INSTALL` / `R CMD check` output produced in
the Docker dev environment.

## Downstream dependencies

None (new package).
