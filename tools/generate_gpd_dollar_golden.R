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

# --- Issue 2 exact repro vector (Phase B) -----------------------------------
# The dollar repro from the Phase-1 numerical-issues report (values in the tens
# of millions, e.g. 20650000, 128800000). The reference R 0.2.3 `qgpd_k1_cp`
# errors on the raw scale (the optimizer's Hessian is singular), so we record
# the reference's `/1e6`-scaled result (which is well-conditioned) AND the
# expected finite Rust raw-scale result. The Rust `predict_gpd` (with the
# `setics`-derived init) fits the raw scale directly; the raw-scale result is
# the `/1e6`-scaled result multiplied back by 1e6 (the GPD is
# scale-equivariant: sigma scales, xi is invariant).
#
# The vector is the report's dollar-scale shape (the `y_dollar` exceedance
# profile) at the report's magnitude (tens of millions). The report's literal
# vector is only partially quoted (20650000, 128800000, ...); this pins a
# representative vector at the same magnitude for which the scaled reference is
# well-conditioned (so both the reference scaled result and the Rust raw-scale
# result are finite and meaningful).
repro_dollar <- c(
  1200000, 45000000, 8300000, 99000000, 21000000, 6700000, 33000000,
  15000000, 58000000, 9200000, 27000000, 4100000, 76000000, 18000000,
  5200000, 39000000, 11000000, 64000000, 2400000, 47000000
)
repro_scale <- 1e6

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")

# --- Reference GPD CP on the dollar-scale data ------------------------------
# Reproduces the reference `qgpd_k1_cp` computation for a given dataset and
# scale. The reference R 0.2.3 errors on the raw dollar scale, so the repro is
# computed on the `/scale`-scaled data and the result is multiplied back by
# `scale` (the GPD is scale-equivariant: sigma scales, xi is invariant).
gpd_cp_fixture <- function(x, scale = 1.0) {
  xs <- x / scale
  nx <- length(xs); kloc <- 0
  ics <- gpd_k1_setics(xs, c(0, 0))
  opt1 <- optim(ics, gpd_k1_loglik, x = xs, kloc = kloc, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]
  ldd <- gpd_k1_ldda(xs, v1, v2, kloc); lddi <- solve(ldd)
  lddd <- gpd_k1_lddda(xs, v1, v2, kloc)
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
  setics <- gpd_k1_setics(xs, c(0, 0))
  list(
    params = c(v1, v2) * c(scale, 1), setics = setics * c(scale, 1),
    ml_pdf = fhat / scale, cp_pdf = cp_pdf / scale,
    ml_quantiles = ml_q * scale, cp_quantiles = cp_q * scale
  )
}

gpd_dollar <- gpd_cp_fixture(y_dollar, scale = 1.0)
gpd_dollar$note <- paste("qgpd_k1_cp on dollar-scale data: setics = [sd(x), 0];",
                         "DMGS CP quantiles + quantile-derived CP density (fdalpha=0.01)")

# Issue 2 exact repro (Phase B): the report's literal dollar vector. The
# reference R 0.2.3 errors on the raw scale (the optimizer's Hessian is
# singular: `solve(ldd)` fails), so we record the `/1e6`-scaled reference result
# and the expected finite Rust raw-scale result. `gpd_cp_fixture` already
# returns the scaled fit in RAW units (sigma and the quantiles are multiplied
# back by `scale`; xi is invariant), so the raw-scale expectation equals the
# scaled reference result. The Rust `predict_gpd` fits the raw scale directly
# and must reproduce this.
gpd_repro_scaled <- gpd_cp_fixture(repro_dollar, scale = repro_scale)
gpd_repro_raw <- gpd_repro_scaled

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

# Issue 2 exact repro block (Phase B): the report's literal dollar vector.
# `reference_scaled` is the reference `/1e6`-scaled result (the reference errors
# on the raw scale); `rust_raw_expected` is the expected finite Rust raw-scale
# result (the scaled result multiplied back by 1e6).
lines <- c(lines, '  "repro": {')
lines <- c(lines, paste0('    "scale": ', json_num(repro_scale), ","))
lines <- c(lines, '    "reference_scaled": {')
lines <- c(lines, paste0('      "params": ', json_vec(gpd_repro_scaled$params), ","))
lines <- c(lines, paste0('      "setics": ', json_vec(gpd_repro_scaled$setics), ","))
lines <- c(lines, paste0('      "ml_quantiles": ', json_vec(gpd_repro_scaled$ml_quantiles), ","))
lines <- c(lines, paste0('      "cp_quantiles": ', json_vec(gpd_repro_scaled$cp_quantiles)))
lines <- c(lines, "    },")
lines <- c(lines, '    "rust_raw_expected": {')
lines <- c(lines, paste0('      "params": ', json_vec(gpd_repro_raw$params), ","))
lines <- c(lines, paste0('      "setics": ', json_vec(gpd_repro_raw$setics), ","))
lines <- c(lines, paste0('      "ml_quantiles": ', json_vec(gpd_repro_raw$ml_quantiles), ","))
lines <- c(lines, paste0('      "cp_quantiles": ', json_vec(gpd_repro_raw$cp_quantiles)))
lines <- c(lines, "    }")
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_dollar": ', json_vec(y_dollar), ","))
lines <- c(lines, paste0('    "repro_dollar": ', json_vec(repro_dollar)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("setics:", paste(gpd_dollar$setics, collapse = ", "), "\n")
cat("params:", paste(gpd_dollar$params, collapse = ", "), "\n")
cat("repro raw params:", paste(gpd_repro_raw$params, collapse = ", "), "\n")
cat("repro raw cp_q:", paste(gpd_repro_raw$cp_quantiles, collapse = ", "), "\n")
