#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phase 3 (Issue 3) golden-fixture generator: robust ML optimiser on short
# records (the hail-region data shape).
#
# Runs the REFERENCE fitdistcp R package's own GEV MLE machinery
# (`gev_setics` + `gev_loglik`) on several FIXED short (n = 10-15) datasets that
# mimic the hail-region data shape (right-skewed, bounded-above maxima), and
# emits the reference `ml_params`, `loglik`, and `cp_quantiles` as JSON into
# tests/golden/short_record_golden.json.
#
# The defect (Issue 3): the hand-rolled BFGS lacks the reference's robust
# multi-start / init strategy, so on short records it stops at a worse fit
# (return levels change in 19-39% of hail regions; NaN in up to 121 regions).
# The Rust `fit_with_config` now seeds from `setics`-style heuristics in
# addition to the caller's `init`, adds a Nelder-Mead polish, and keeps the best
# log-likelihood. This fixture pins the reference result so the Rust fit's
# log-likelihood can be checked against it (and no region returns NaN).
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
#   * The datasets are FIXED LITERALS (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_short_record_golden.R
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

# --- Fixed short-record (hail-region shape) datasets ------------------------
# Each dataset is a short (n = 10-15) right-skewed, bounded-above sample that
# mimics a hail-region record (annual maximum hail size / loss). The short
# record is the regime where the hand-rolled BFGS stalled (Issue 3). The
# literals are fixed so the fixture is reproducible.
#
# `hail_region_a` (n = 12): a mild negative-shape record.
hail_region_a <- c(
  1.12, 1.48, 1.21, 1.77, 1.05, 1.63, 1.34, 1.19, 1.88, 1.27, 1.55, 1.41
)
# `hail_region_b` (n = 10): a stronger negative-shape record (bounded above).
hail_region_b <- c(
  1.31, 1.62, 1.44, 1.90, 1.22, 1.71, 1.53, 1.38, 1.84, 1.47
)
# `hail_region_c` (n = 15): a longer short record with a heavier upper tail.
hail_region_c <- c(
  1.08, 1.55, 1.29, 1.83, 1.14, 1.68, 1.42, 1.24, 1.96, 1.36,
  1.51, 1.19, 1.74, 1.33, 1.60
)

# Prediction grid and probability levels (fixed). p = 0.99 is the 1-in-100
# return level, the quantity the defect changed.
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

# --- Reference GEV MLE + CP quantiles on a short record ---------------------
# Reproduces the reference `qgev_cp` computation so the fixture is the
# reference's own output. We record the reference `ml_params`, the maximized
# `loglik` (the Phase 3 target), and the `fixgevrange`-clamped CP quantiles.
gev_short_record_fixture <- function(x, x_grid, p_grid) {
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

models <- list()
models$hail_region_a <- gev_short_record_fixture(hail_region_a, x_grid, p_grid)
models$hail_region_b <- gev_short_record_fixture(hail_region_b, x_grid, p_grid)
models$hail_region_c <- gev_short_record_fixture(hail_region_c, x_grid, p_grid)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "short_record_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_short_record_golden.R",')
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
  lines <- c(lines, paste0("      \"ml_max\": ", json_num(m$ml_max), ","))
  lines <- c(lines, paste0("      \"ml_pdf\": ", json_vec(m$ml_pdf), ","))
  lines <- c(lines, paste0("      \"ml_quantiles\": ", json_vec(m$ml_quantiles), ","))
  lines <- c(lines, paste0("      \"cp_quantiles_raw\": ", json_vec(m$cp_quantiles_raw), ","))
  lines <- c(lines, paste0("      \"cp_quantiles\": ", json_vec(m$cp_quantiles)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "hail_region_a": ', json_vec(hail_region_a), ","))
lines <- c(lines, paste0('    "hail_region_b": ', json_vec(hail_region_b), ","))
lines <- c(lines, paste0('    "hail_region_c": ', json_vec(hail_region_c)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
for (nm in model_names) {
  cat(nm, "n =", length(get(nm)), "shape =", models[[nm]]$params[3],
      "loglik =", models[[nm]]$loglik,
      "revert2ml =", models[[nm]]$revert2ml, "\n")
  cat("  cp_quantiles =", paste(models[[nm]]$cp_quantiles, collapse = ", "), "\n")
}
