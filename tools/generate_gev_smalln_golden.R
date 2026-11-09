#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phase 4 (Issue 4) golden-fixture generator: GEV CP support clamping at small n.
#
# Runs the REFERENCE fitdistcp R package's own GEV CP machinery
# (`gev_setics` + `gev_loglik` + `gev_ldda`/`gev_lddda` + `gev_mu1fa`/`gev_mu2fa`
# + `dmgs`) on a fixed small-n (n = 20) dataset and emits the reference
# `ml_params`, `ml_max`, `revert2ml`, and `cp_quantiles` (including the 1-in-100
# level, p = 0.99) as JSON into tests/golden/gev_smalln_golden.json.
#
# The defect (Issue 4): at small n the DMGS expansion can push a CP quantile
# outside the GEV support (e.g. a negative 1-in-100 return level for a
# positive-support fit). The reference `fixgevrange` (005_utils.R) clamps a
# value to the (nudged) support bound. The Rust `predict_gev` now applies the
# same clamp and falls back to the MLE quantile when a CP quantile still leaves
# the support. This fixture pins the reference result so the Rust clamp can be
# checked against it.
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
#   Rscript tools/generate_gev_smalln_golden.R
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

# --- Fixed small-n (n = 20) dataset ----------------------------------------
# A right-skewed, bounded-above sample (Weibull-like maxima) gives a negative
# GEV shape with a finite upper support bound. n = 20 is the small-n regime
# where the DMGS CP quantile can leave the support. These literals are fixed so
# the fixture is reproducible.
y_smalln <- c(
  1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38,
  1.09, 1.72, 1.28, 1.51, 1.83, 1.16, 1.61, 1.35, 1.99, 1.47
)

# A second small-n (n = 20) dataset whose raw DMGS CP quantile at p = 0.99
# EXCEEDS the finite upper support bound `ml_max`, so the reference
# `fixgevrange` clamp is genuinely exercised (raw != clamped). This is the
# defect case (Issue 4): a CP quantile outside the support. Fixed literal.
y_smalln_clamp <- c(
  1.27, 1.37, 1.57, 1.91, 1.20, 1.90, 1.94, 1.66, 1.63, 1.06,
  1.21, 1.18, 1.69, 1.38, 1.77, 1.50, 1.72, 1.99, 1.38, 1.78
)

# Prediction grid and probability levels (fixed). p = 0.99 is the 1-in-100
# return level, the point that can leave the support at small n.
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

# --- Reference qgev_cp core (quantiles), transcribed from 110a_gev.R --------
# We reproduce the exact reference computation for the CP quantiles, so the
# fixture is the reference's own output. We additionally record the reference
# `fixgevrange`-clamped CP quantiles (the Phase 4 target) alongside the raw
# DMGS CP quantiles.
gev_cp_fixture <- function(x, x_grid, p_grid) {
  nx <- length(x)
  alpha <- 1 - p_grid

  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1hat <- opt1$par[1]; v2hat <- opt1$par[2]; v3hat <- opt1$par[3]
  ml_params <- c(v1hat, v2hat, v3hat)

  revert2ml <- abs(v3hat) >= 1
  ml_max <- if (v3hat < 0) v1hat - v2hat / v3hat else Inf

  ml_quantiles <- qgev((1 - alpha), mu = v1hat, sigma = v2hat, xi = v3hat)
  ml_pdf <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)

  if (revert2ml) {
    return(list(
      params = ml_params, revert2ml = revert2ml, ml_max = ml_max,
      ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
      cp_quantiles_raw = ml_quantiles, cp_quantiles = ml_quantiles
    ))
  }

  # ldd / lddd
  ldd <- gev_ldda(x, v1hat, v2hat, v3hat); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1hat, v2hat, v3hat)
  lambdad_rh_flat <- c(0, -1 / v2hat, 0)

  # CP quantiles (rh_flat), raw DMGS
  mu1 <- gev_mu1fa(alpha, v1hat, v2hat, v3hat)
  mu2 <- gev_mu2fa(alpha, v1hat, v2hat, v3hat)
  fhat <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)
  dq <- dmgs(lddi, lddd, mu1, lambdad_rh_flat, mu2, dim = 3)
  cp_quantiles_raw <- ml_quantiles + dq / (nx * fhat)

  # Phase 4 target: the reference `fixgevrange` clamp applied to the CP
  # quantiles (the Rust `predict_gev` now does this).
  cp_quantiles <- fixgevrange(cp_quantiles_raw, v1hat, v2hat, v3hat)

  list(
    params = ml_params, revert2ml = revert2ml, ml_max = ml_max,
    ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
    cp_quantiles_raw = cp_quantiles_raw, cp_quantiles = cp_quantiles
  )
}

models <- list()
models$gev_smalln <- gev_cp_fixture(y_smalln, x_grid, p_grid)
models$gev_smalln_clamp <- gev_cp_fixture(y_smalln_clamp, x_grid, p_grid)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gev_smalln_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gev_smalln_golden.R",')
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
lines <- c(lines, paste0('    "y_smalln": ', json_vec(y_smalln), ","))
lines <- c(lines, paste0('    "y_smalln_clamp": ', json_vec(y_smalln_clamp)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
for (nm in model_names) {
  cat(nm, "shape =", models[[nm]]$params[3],
      "revert2ml =", models[[nm]]$revert2ml,
      "ml_max =", models[[nm]]$ml_max, "\n")
  cat("  cp_quantiles_raw =", paste(models[[nm]]$cp_quantiles_raw, collapse = ", "), "\n")
  cat("  cp_quantiles     =", paste(models[[nm]]$cp_quantiles, collapse = ", "), "\n")
}
