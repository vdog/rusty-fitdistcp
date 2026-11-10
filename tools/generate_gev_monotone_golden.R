#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phase B (Issue 4) golden-fixture generator: the exact GEV monotonicity repro.
#
# Runs the REFERENCE fitdistcp R package's own GEV MLE + CP machinery
# (`gev_setics` + `gev_loglik` + `gev_ldda`/`gev_lddda` + `gev_mu1fa`/`gev_mu2fa`
# + `dmgs` + `fixgevrange`) on the literal repro dataset from the Phase-1
# numerical-issues report, and emits the reference `ml_params`, `ml_max`,
# `revert2ml`, and `cp_quantiles` as JSON into
# tests/golden/gev_monotone_golden.json.
#
# The defect (Issue 4): at small n the DMGS CP quantiles can be non-monotone in
# `p` (a decreasing tail) and the median can be negative. The reference does
# NOT sort the quantiles (its `optim` avoids the degenerate fit); the Rust core
# adds a monotonicity guard (Phase A) that restores the non-decreasing
# invariant and records whether it fired via `cp_monotone_clamped`. This fixture
# pins the reference result AND the Rust monotone-guard expectation so the
# deliberate Rust-only divergence is regression-tested.
#
# The fixtures come from the reference code, not a re-implementation:
#   * `gev_setics` (110b_gev_libs.R) — the initial conditions,
#   * `gev_loglik` (110b_gev_libs.R) — the MLE objective,
#   * `gev_ldda` / `gev_lddda` (110c_gev_derivs.R) — the derivatives,
#   * `gev_mu1fa` / `gev_mu2fa` (110c_gev_derivs.R) — the CDF derivs,
#   * `dmgs` (000_evaluate_dmgs_equation3p3.R) — the expansion,
#   * `fixgevrange` (005_utils.R) — the support clamp.
#
# Reproducibility:
#   * The dataset is a FIXED LITERAL (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gev_monotone_golden.R
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

# --- The exact repro dataset and probability levels (report literals) -------
repro_x <- c(
  1.426, 1.123, 1.447, 1.51, 0.814, 2.773, 1.907, 2.208, 0.904, 2.343,
  1.144, 0.69
)
repro_p <- c(0.5, 0.8, 0.9, 0.96, 0.98, 0.99)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")
json_bool <- function(b) if (isTRUE(b)) "true" else "false"

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)

# --- Reference GEV MLE + CP quantiles on the repro --------------------------
# Mirrors the reference `qgev_cp` computation (see
# tools/generate_short_record_golden.R) so the fixture is the reference's own
# output. We record the reference `ml_params`, `ml_max`, `revert2ml`, and the
# `fixgevrange`-clamped CP quantiles.
gev_monotone_fixture <- function(x, p_grid) {
  nx <- length(x)
  alpha <- 1 - p_grid

  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1hat <- opt1$par[1]; v2hat <- opt1$par[2]; v3hat <- opt1$par[3]
  ml_params <- c(v1hat, v2hat, v3hat)
  loglik <- opt1$val

  revert2ml <- abs(v3hat) >= 1
  ml_max <- if (v3hat < 0) v1hat - v2hat / v3hat else Inf

  ml_quantiles <- qgev((1 - alpha), mu = v1hat, sigma = v2hat, xi = v3hat)
  ml_pdf <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)

  if (revert2ml) {
    return(list(
      params = ml_params, loglik = loglik, revert2ml = revert2ml,
      ml_max = ml_max, ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
      cp_quantiles_raw = ml_quantiles, cp_quantiles = ml_quantiles
    ))
  }

  ldd <- gev_ldda(x, v1hat, v2hat, v3hat); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1hat, v2hat, v3hat)
  lambdad_rh_flat <- c(0, -1 / v2hat, 0)

  mu1 <- gev_mu1fa(alpha, v1hat, v2hat, v3hat)
  mu2 <- gev_mu2fa(alpha, v1hat, v2hat, v3hat)
  fhat <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)
  dq <- dmgs(lddi, lddd, mu1, lambdad_rh_flat, mu2, dim = 3)
  cp_quantiles_raw <- ml_quantiles + dq / (nx * fhat)
  cp_quantiles <- fixgevrange(cp_quantiles_raw, v1hat, v2hat, v3hat)

  list(
    params = ml_params, loglik = loglik, revert2ml = revert2ml,
    ml_max = ml_max, ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
    cp_quantiles_raw = cp_quantiles_raw, cp_quantiles = cp_quantiles
  )
}

m <- gev_monotone_fixture(repro_x, repro_p)

# The Rust monotone-guard expectation: the returned CP quantiles must be
# non-decreasing in `p` and the median (p = 0.5) non-negative. The reference
# does not sort, so the raw reference curve may be non-monotone; the Rust guard
# restores the invariant (a deliberate Rust-only divergence).
ref_monotone <- all(diff(m$cp_quantiles) >= 0)
rust_median_nonneg <- m$cp_quantiles[1] >= 0

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gev_monotone_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gev_monotone_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-6, "atol": 1e-10},')
lines <- c(lines, paste0('  "p_grid": ', json_vec(repro_p), ","))
lines <- c(lines, '  "models": {')
lines <- c(lines, '    "repro_gev": {')
lines <- c(lines, paste0('      "params": ', json_vec(m$params), ","))
lines <- c(lines, paste0('      "loglik": ', json_num(m$loglik), ","))
lines <- c(lines, paste0('      "revert2ml": ', json_bool(m$revert2ml), ","))
lines <- c(lines, paste0('      "ml_max": ', json_num(m$ml_max), ","))
lines <- c(lines, paste0('      "ml_pdf": ', json_vec(m$ml_pdf), ","))
lines <- c(lines, paste0('      "ml_quantiles": ', json_vec(m$ml_quantiles), ","))
lines <- c(lines, paste0('      "cp_quantiles_raw": ', json_vec(m$cp_quantiles_raw), ","))
lines <- c(lines, paste0('      "cp_quantiles": ', json_vec(m$cp_quantiles)))
lines <- c(lines, "    }")
lines <- c(lines, "  },")
lines <- c(lines, '  "rust_expectation": {')
lines <- c(lines, paste0('    "cp_quantiles_non_decreasing": ', json_bool(TRUE), ","))
lines <- c(lines, paste0('    "median_non_negative": ', json_bool(TRUE), ","))
lines <- c(lines, paste0('    "reference_cp_quantiles_monotone": ', json_bool(ref_monotone), ","))
lines <- c(lines, paste0('    "reference_median_non_negative": ', json_bool(rust_median_nonneg)))
lines <- c(lines, "  },")
lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "repro_x": ', json_vec(repro_x)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("params:", paste(m$params, collapse = ", "), "\n")
cat("revert2ml:", m$revert2ml, " ml_max:", m$ml_max, "\n")
cat("cp_quantiles:", paste(m$cp_quantiles, collapse = ", "), "\n")
cat("reference cp monotone:", ref_monotone,
    " reference median non-negative:", rust_median_nonneg, "\n")
