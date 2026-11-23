#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Issue 2 re-check (item B) golden-fixture generator: degenerate GEV fits on
# ~10-point records.
#
# Runs the REFERENCE fitdistcp R package's own GEV MLE machinery
# (`gev_setics` + `gev_loglik` + `optim`) on fixed ~10-point records that make
# the Rust multi-start optimizer reach a spurious `scale -> 0` maximum, and
# emits the reference `ml_params`, `loglik`, `revert2ml`, `ml_max`, and the
# reference CP quantiles as JSON into tests/golden/gev_degenerate_golden.json.
#
# # What the reference actually does (important)
#
# The reference `qgev_cp` does NOT reject a degenerate fit: its
# `gev_checkmle(ml_params, minxi, maxxi)` call is **commented out** (110a_gev.R
# line 76), and `revert2ml = abs(v3hat) >= 1` only reverts the CP output to the
# MLE output. On these degenerate records the reference `optim` itself lands in
# the `scale -> 0` region (e.g. `tight_cluster` -> scale ~4e-4, shape ~7.8), so
# the reference returns **huge finite** CP quantiles (up to ~2e11). That is the
# defect the report describes ("huge finite numbers instead of NaN").
#
# The Rust multi-start used to find the same spurious point and return the same
# huge values. Item B1 tightens the degenerate-fit guard (a relative scale floor
# plus a runaway-location bound) so the Rust multi-start **rejects** the
# degenerate candidate and returns a valid interior fit (or, if every candidate
# is degenerate, NaN CP quantiles with `degenerate_fit = true`, item B2).
#
# This fixture therefore records the reference's degenerate `ml_params` and its
# huge `cp_quantiles` for documentation, plus a `reference_degenerate` flag
# (computed from the same relative scale floor the Rust guard uses) so the
# harness can assert that the Rust fit is NOT degenerate on these records and
# does NOT reproduce the huge values.
#
# The fixtures come from the reference code, not a re-implementation:
#   * `gev_setics` (110b_gev_libs.R) — the initial conditions,
#   * `gev_loglik` (110b_gev_libs.R) — the MLE objective,
#   * `gev_ldda` / `gev_lddda` (110c_gev_derivs.R) — the derivatives,
#   * `gev_mu1fa` / `gev_mu2fa` (110c_gev_derivs.R) — the CDF derivs,
#   * `dmgs` (000_evaluate_dmgs_equation3p3.R) — the expansion.
#
# Reproducibility:
#   * The datasets are FIXED LITERALS (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gev_degenerate_golden.R
# ---------------------------------------------------------------------------

local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}
suppressPackageStartupMessages(library(extraDistr))

ref_dir <- "tools/fitdistcp-ref/R"
if (!dir.exists(ref_dir)) {
  stop("Reference package not found at ", ref_dir, ". Clone it first.")
}

# Source the DMGS core + the GEV libs/derivs we need.
lib_files <- c(
  "000_evaluate_dmgs_equation3p3.R", # dmgs / bayesian_dq_4terms_v1
  "005_utils.R",                     # fixgevrange, movexiawayfromzero, make_se, make_maic
  "110b_gev_libs.R",                 # gev_loglik, gev_setics, dgevsub
  "110c_gev_derivs.R"                # gev_ldda, gev_lddda, gev_mu1fa, gev_mu2fa
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

# --- Fixed ~10-point degenerate records ------------------------------------
# These are the canonical `scale -> 0` degenerate records: a tight cluster with
# one outlier, a near-constant record, and a constant record. The Rust
# multi-start used to reach the spurious `scale -> 0` maximum on these; the
# reference `optim` returns a finite interior MLE. Fixed literals.
y_tight_cluster <- c(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001)
y_near_constant <- c(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0001)
y_constant <- rep(1.0, 12)

# A legitimate ~10-point record (the reference interior MLE) used as a control:
# the guard must NOT reject it.
y_legit <- c(1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38)

# Prediction grid and probability levels (fixed). p = 0.99 is the 1-in-100
# return level, the point that blew up to 1.2e8 / 2.0e9.
x_grid <- c(1.0, 1.3, 1.6, 1.9, 2.2)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")
json_bool <- function(b) if (isTRUE(b)) "true" else "false"

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)

# The Rust degenerate-fit guard thresholds (item B1), mirrored here so the
# fixture can record whether the reference fit is degenerate by the same rule.
# See crates/fitdistcp-core/src/models/mod.rs (GEV_SCALE_FLOOR_REL,
# GEV_LOCATION_RUNAWAY_REL).
GEV_SCALE_FLOOR_REL <- 1e-3
GEV_LOCATION_RUNAWAY_REL <- 1e3

# The data spread max(sd(y), |mean(y)|, 1) used by the Rust guard.
gev_data_spread <- function(y) {
  sd <- if (length(y) > 1) stats::sd(y) else 0
  max(sd, abs(mean(y)), 1)
}

# Whether a GEV (loc, scale) fit is degenerate by the Rust guard's rule.
gev_is_degenerate <- function(loc, scale, y) {
  spread <- gev_data_spread(y)
  !(is.finite(scale) && scale > GEV_SCALE_FLOOR_REL * spread &&
      is.finite(loc) && abs(loc) <= GEV_LOCATION_RUNAWAY_REL * spread)
}

# --- Reference qgev_cp core (quantiles), transcribed from 110a_gev.R --------
# We reproduce the exact reference computation for the CP quantiles, so the
# fixture is the reference's own output. The reference does NOT clamp the CP
# quantiles (Issue 2 re-check, item A), so `cp_quantiles` is the raw DMGS value.
gev_cp_fixture <- function(x, x_grid, p_grid) {
  nx <- length(x)
  alpha <- 1 - p_grid

  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1hat <- opt1$par[1]; v2hat <- opt1$par[2]; v3hat <- opt1$par[3]
  ml_params <- c(v1hat, v2hat, v3hat)
  loglik <- gev_loglik(ml_params, x)

  revert2ml <- abs(v3hat) >= 1
  ml_max <- if (v3hat < 0) v1hat - v2hat / v3hat else Inf

  ml_quantiles <- qgev((1 - alpha), mu = v1hat, sigma = v2hat, xi = v3hat)
  ml_pdf <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)

  if (revert2ml) {
    return(list(
      params = ml_params, loglik = loglik, revert2ml = revert2ml,
      ml_max = ml_max, ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
      cp_quantiles = ml_quantiles
    ))
  }

  # ldd / lddd
  ldd <- gev_ldda(x, v1hat, v2hat, v3hat); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1hat, v2hat, v3hat)
  lambdad_rh_flat <- c(0, -1 / v2hat, 0)

  # CP quantiles (rh_flat), raw DMGS — the reference does NOT clamp these.
  mu1 <- gev_mu1fa(alpha, v1hat, v2hat, v3hat)
  mu2 <- gev_mu2fa(alpha, v1hat, v2hat, v3hat)
  fhat <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)
  dq <- dmgs(lddi, lddd, mu1, lambdad_rh_flat, mu2, dim = 3)
  cp_quantiles <- ml_quantiles + dq / (nx * fhat)

  list(
    params = ml_params, loglik = loglik, revert2ml = revert2ml,
    ml_max = ml_max, ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
    cp_quantiles = cp_quantiles
  )
}

# Wrap gev_cp_fixture to also record whether the reference fit is degenerate.
gev_degenerate_fixture <- function(x, x_grid, p_grid) {
  m <- gev_cp_fixture(x, x_grid, p_grid)
  m$reference_degenerate <- gev_is_degenerate(m$params[1], m$params[2], x)
  m
}

models <- list()
models$gev_degenerate_tight_cluster <- gev_degenerate_fixture(y_tight_cluster, x_grid, p_grid)
models$gev_degenerate_near_constant <- gev_degenerate_fixture(y_near_constant, x_grid, p_grid)
models$gev_degenerate_constant <- gev_degenerate_fixture(y_constant, x_grid, p_grid)
models$gev_degenerate_legit <- gev_degenerate_fixture(y_legit, x_grid, p_grid)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gev_degenerate_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gev_degenerate_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-6, "atol": 1e-10},')
lines <- c(lines, paste0('  "x_grid": ', json_vec(x_grid), ","))
lines <- c(lines, paste0('  "p_grid": ', json_vec(p_grid), ","))
lines <- c(lines, '  "models": {')

model_names <- names(models)
for (mi in seq_along(model_names)) {
  nm <- model_names[[mi]]
  m <- models[[nm]]
  lines <- c(lines, paste0("    ", json_str(nm), ": {"))
  lines <- c(lines, paste0("      \"params\": ", json_vec(m$params), ","))
  lines <- c(lines, paste0("      \"loglik\": ", json_num(m$loglik), ","))
  lines <- c(lines, paste0("      \"revert2ml\": ", json_bool(m$revert2ml), ","))
  lines <- c(lines, paste0("      \"reference_degenerate\": ", json_bool(m$reference_degenerate), ","))
  lines <- c(lines, paste0("      \"ml_max\": ", json_num(m$ml_max), ","))
  lines <- c(lines, paste0("      \"ml_pdf\": ", json_vec(m$ml_pdf), ","))
  lines <- c(lines, paste0("      \"ml_quantiles\": ", json_vec(m$ml_quantiles), ","))
  lines <- c(lines, paste0("      \"cp_quantiles\": ", json_vec(m$cp_quantiles)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_tight_cluster": ', json_vec(y_tight_cluster), ","))
lines <- c(lines, paste0('    "y_near_constant": ', json_vec(y_near_constant), ","))
lines <- c(lines, paste0('    "y_constant": ', json_vec(y_constant), ","))
lines <- c(lines, paste0('    "y_legit": ', json_vec(y_legit)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
for (nm in model_names) {
  cat(nm, "params =", paste(models[[nm]]$params, collapse = ", "),
      "loglik =", models[[nm]]$loglik,
      "revert2ml =", models[[nm]]$revert2ml,
      "reference_degenerate =", models[[nm]]$reference_degenerate, "\n")
  cat("  cp_quantiles =", paste(models[[nm]]$cp_quantiles, collapse = ", "), "\n")
}
