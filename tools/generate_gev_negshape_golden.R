#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# W10 golden-fixture generator for the fitdistcp Rust port.
#
# Negative-shape GEV quantile-derived CP density + small-n handling.
#
# Runs the REFERENCE fitdistcp R package's own `qgev_cp` (with pdf=TRUE) on
# fixed, seeded datasets whose MLE shape is NEGATIVE, and emits golden fixtures
# (CP pdf via the quantile-derived path, CP quantiles, ML pdf, ML quantiles,
# params, ml_max, revert2ml) as JSON into tests/golden/gev_negshape_golden.json.
#
# The fixtures come from the reference code, not a re-implementation:
#   * `qgev_cp` (110a_gev.R) — the negative-shape handling, `revert2ml`,
#     `ml_max`, and the quantile-derived `rh_flat_pdf = -(alphap-alpham)/(qp-qm)`,
#   * `gev_ldda` / `gev_lddda` / `gev_mu1fa` / `gev_mu2fa` (110c_gev_derivs.R),
#   * `dmgs` (000_evaluate_dmgs_equation3p3.R),
#   * `movexiawayfromzero` (005_utils.R).
#
# Reproducibility:
#   * The datasets are FIXED LITERALS, so the fixtures are byte-for-byte
#     reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gev_negshape_golden.R
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
  "005_utils.R",                     # deriv_copyldd, deriv_copylddd, make_se, make_maic,
                                     # movexiawayfromzero, fixgevrange
  "110b_gev_libs.R",                 # gev_loglik, gev_setics, dgevsub
  "110c_gev_derivs.R"                # gev_ldda, gev_lddda, gev_mu1fa, gev_mu2fa, gev_f1fa, gev_f2fa
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

# --- Fixed datasets with NEGATIVE MLE shape --------------------------------
# A right-skewed, bounded-above sample (Weibull-like maxima) gives a negative
# GEV shape. These literals are fixed so the fixtures are reproducible.
y_negshape <- c(
  1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38,
  1.09, 1.72, 1.28, 1.51, 1.83, 1.16, 1.61, 1.35, 1.99, 1.47,
  1.05, 1.78, 1.24, 1.58, 1.90, 1.19, 1.65, 1.41, 1.86, 1.33
)
# A small-n sample (n = 8) with negative shape, to exercise the small-n flag.
y_negshape_small <- c(1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21)

# Prediction grid and probability levels (fixed).
x_grid <- c(1.0, 1.3, 1.6, 1.9, 2.2)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")
json_bool <- function(b) if (isTRUE(b)) "true" else "false"

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)

# --- Reference qgev_cp core (pdf path), transcribed from 110a_gev.R ---------
# We reproduce the exact reference computation for the CP pdf (quantile-derived)
# and CP quantiles, so the fixture is the reference's own output.
gev_cp_fixture <- function(x, x_grid, p_grid, fdalpha = 0.01) {
  nx <- length(x)
  alpha <- 1 - p_grid
  nalpha <- length(alpha)

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
      cp_quantiles = ml_quantiles, cp_pdf = ml_pdf
    ))
  }

  # ldd / lddd
  ldd <- gev_ldda(x, v1hat, v2hat, v3hat); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1hat, v2hat, v3hat)
  lambdad_rh_flat <- c(0, -1 / v2hat, 0)

  # CP quantiles (rh_flat)
  mu1 <- gev_mu1fa(alpha, v1hat, v2hat, v3hat)
  mu2 <- gev_mu2fa(alpha, v1hat, v2hat, v3hat)
  fhat <- dgev(ml_quantiles, mu = v1hat, sigma = v2hat, xi = v3hat)
  dq <- dmgs(lddi, lddd, mu1, lambdad_rh_flat, mu2, dim = 3)
  cp_quantiles <- ml_quantiles + dq / (nx * fhat)

  # CP pdf via the quantile-derived path (reference qgev_cp with pdf=TRUE)
  dalpha <- pmin(fdalpha * alpha, fdalpha * (1 - alpha))
  alpham <- alpha - dalpha
  alphap <- alpha + dalpha
  ml_quantilesm <- qgev((1 - alpham), mu = v1hat, sigma = v2hat, xi = v3hat)
  ml_quantilesp <- qgev((1 - alphap), mu = v1hat, sigma = v2hat, xi = v3hat)
  fhatm <- dgev(ml_quantilesm, mu = v1hat, sigma = v2hat, xi = v3hat)
  fhatp <- dgev(ml_quantilesp, mu = v1hat, sigma = v2hat, xi = v3hat)
  mu1m <- gev_mu1fa(alpham, v1hat, v2hat, v3hat)
  mu1p <- gev_mu1fa(alphap, v1hat, v2hat, v3hat)
  mu2m <- gev_mu2fa(alpham, v1hat, v2hat, v3hat)
  mu2p <- gev_mu2fa(alphap, v1hat, v2hat, v3hat)
  dqm <- dmgs(lddi, lddd, mu1m, lambdad_rh_flat, mu2m, dim = 3)
  dqp <- dmgs(lddi, lddd, mu1p, lambdad_rh_flat, mu2p, dim = 3)
  quantilesm <- ml_quantilesm + dqm / (nx * fhatm)
  quantilesp <- ml_quantilesp + dqp / (nx * fhatp)
  cp_pdf <- -(alphap - alpham) / (quantilesp - quantilesm)

  list(
    params = ml_params, revert2ml = revert2ml, ml_max = ml_max,
    ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
    cp_quantiles = cp_quantiles, cp_pdf = cp_pdf
  )
}

models <- list()
models$gev_negshape <- gev_cp_fixture(y_negshape, x_grid, p_grid)
models$gev_negshape_small <- gev_cp_fixture(y_negshape_small, x_grid, p_grid)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gev_negshape_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gev_negshape_golden.R",')
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
  lines <- c(lines, paste0("      \"cp_pdf\": ", json_vec(m$cp_pdf), ","))
  lines <- c(lines, paste0("      \"ml_quantiles\": ", json_vec(m$ml_quantiles), ","))
  lines <- c(lines, paste0("      \"cp_quantiles\": ", json_vec(m$cp_quantiles)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_negshape": ', json_vec(y_negshape), ","))
lines <- c(lines, paste0('    "y_negshape_small": ', json_vec(y_negshape_small)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
for (nm in model_names) {
  cat(nm, "shape =", models[[nm]]$params[3],
      "revert2ml =", models[[nm]]$revert2ml, "\n")
}
