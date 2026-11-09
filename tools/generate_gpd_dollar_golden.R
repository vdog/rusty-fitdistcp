#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phase 2 (Issue 2) golden-fixture generator: GPD on raw dollar-scale data.
#
# Runs the REFERENCE fitdistcp R package's own GPD CP machinery
# (`gpd_k1_setics` + `gpd_k1_loglik` + `gpd_k1_ldda`/`gpd_k1_lddda` +
# `gpd_k1_mu1fa`/`gpd_k1_mu2fa` + `dmgs`) on a fixed dollar-scale dataset
# (values ~10^3-10^5) and emits the reference `ml_params` and `cp_quantiles`
# as JSON into tests/golden/gpd_dollar_golden.json.
#
# The defect (Issue 2): the binding hard-coded `init = [1.0, 0.0]`, which is
# many orders of magnitude below the scale of raw dollar data, so the optimizer
# started in a degenerate region and silently returned NaN. The reference seeds
# the scale from `gpd_k1_setics` = `[sd(x), 0]`. This fixture pins the correct
# reference result so the Rust `predict_gpd` (with the `setics`-derived init)
# can be checked against it.
#
# The fixtures come from the reference code, not a re-implementation:
#   * `gpd_k1_setics` (120b_gpd_k1_libs.R) — ics = [sd(x), 0] when ics = [0, 0],
#   * `gpd_k1_loglik` (120b_gpd_k1_libs.R) — the MLE objective,
#   * `gpd_k1_ldda` / `gpd_k1_lddda` (120c_gpd_k1_derivs.R) — the derivatives,
#   * `gpd_k1_mu1fa` / `gpd_k1_mu2fa` (120c_gpd_k1_derivs.R) — the CDF derivs,
#   * `dmgs` (000_evaluate_dmgs_equation3p3.R) — the expansion.
#
# Reproducibility:
#   * The dataset is a FIXED LITERAL (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gpd_dollar_golden.R
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
  "005_utils.R",                     # make_se, make_maic
  "120b_gpd_k1_libs.R",              # gpd_k1_loglik, gpd_k1_setics
  "120c_gpd_k1_derivs.R"             # gpd_k1_ldda, gpd_k1_lddda, gpd_k1_mu1fa, gpd_k1_mu2fa
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9)
alpha <- 1 - p_grid
fdalpha <- 0.01

# --- Fixed dollar-scale dataset (deterministic literal) ---------------------
# Raw dollar-scale exceedances (~10^3-10^5), the shape that triggered the NaN.
y_dollar <- c(
  1200, 45000, 8300, 99000, 21000, 6700, 33000, 15000, 58000, 9200,
  27000, 4100, 76000, 18000, 5200, 39000, 11000, 64000, 2400, 47000
)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")

# --- Reference GPD CP on the dollar-scale data ------------------------------
{
  x <- y_dollar
  nx <- length(x); kloc <- 0
  ics <- gpd_k1_setics(x, c(0, 0))
  opt1 <- optim(ics, gpd_k1_loglik, x = x, kloc = kloc, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]
  ldd <- gpd_k1_ldda(x, v1, v2, kloc); lddi <- solve(ldd)
  lddd <- gpd_k1_lddda(x, v1, v2, kloc)
  lambdad <- c(-1 / v1, 0)
  ml_q <- extraDistr::qgpd(1 - alpha, mu = kloc, sigma = v1, xi = v2)
  fhat <- extraDistr::dgpd(ml_q, mu = kloc, sigma = v1, xi = v2)
  mu1 <- gpd_k1_mu1fa(alpha, v1, v2, kloc); mu2 <- gpd_k1_mu2fa(alpha, v1, v2, kloc)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 2)
  cp_q <- ml_q + dq / (nx * fhat)
  dalpha <- pmin(fdalpha * alpha, fdalpha * (1 - alpha))
  alpham <- alpha - dalpha; alphap <- alpha + dalpha
  ml_qm <- extraDistr::qgpd(1 - alpham, mu = kloc, sigma = v1, xi = v2)
  ml_qp <- extraDistr::qgpd(1 - alphap, mu = kloc, sigma = v1, xi = v2)
  fhatm <- extraDistr::dgpd(ml_qm, mu = kloc, sigma = v1, xi = v2)
  fhatp <- extraDistr::dgpd(ml_qp, mu = kloc, sigma = v1, xi = v2)
  mu1m <- gpd_k1_mu1fa(alpham, v1, v2, kloc); mu1p <- gpd_k1_mu1fa(alphap, v1, v2, kloc)
  mu2m <- gpd_k1_mu2fa(alpham, v1, v2, kloc); mu2p <- gpd_k1_mu2fa(alphap, v1, v2, kloc)
  dqm <- dmgs(lddi, lddd, mu1m, lambdad, mu2m, dim = 2)
  dqp <- dmgs(lddi, lddd, mu1p, lambdad, mu2p, dim = 2)
  qm <- ml_qm + dqm / (nx * fhatm); qp <- ml_qp + dqp / (nx * fhatp)
  cp_pdf <- -(alphap - alpham) / (qp - qm)
  setics <- gpd_k1_setics(x, c(0, 0))
  gpd_dollar <- list(
    params = c(v1, v2), setics = setics, ml_pdf = fhat, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = paste("qgpd_k1_cp on dollar-scale data: setics = [sd(x), 0];",
                 "DMGS CP quantiles + quantile-derived CP density (fdalpha=0.01)")
  )
}

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gpd_dollar_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gpd_dollar_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-6, "atol": 1e-10},')
lines <- c(lines, paste0('  "p_grid": ', json_vec(p_grid), ","))
lines <- c(lines, '  "models": {')
lines <- c(lines, '    "genpareto_dollar": {')
lines <- c(lines, paste0('      "params": ', json_vec(gpd_dollar$params), ","))
lines <- c(lines, paste0('      "setics": ', json_vec(gpd_dollar$setics), ","))
lines <- c(lines, paste0('      "ml_pdf": ', json_vec(gpd_dollar$ml_pdf), ","))
lines <- c(lines, paste0('      "cp_pdf": ', json_vec(gpd_dollar$cp_pdf), ","))
lines <- c(lines, paste0('      "ml_quantiles": ', json_vec(gpd_dollar$ml_quantiles), ","))
lines <- c(lines, paste0('      "cp_quantiles": ', json_vec(gpd_dollar$cp_quantiles), ","))
lines <- c(lines, paste0('      "note": ', json_str(gpd_dollar$note)))
lines <- c(lines, "    }")
lines <- c(lines, "  },")
lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_dollar": ', json_vec(y_dollar)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("setics:", paste(gpd_dollar$setics, collapse = ", "), "\n")
cat("params:", paste(gpd_dollar$params, collapse = ", "), "\n")
