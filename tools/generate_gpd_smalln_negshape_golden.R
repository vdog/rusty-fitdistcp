#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Issue 2 re-check (item C1) golden-fixture generator: GPD small-n (n = 20)
# with the ML shape near -1.
#
# Runs the REFERENCE fitdistcp R package's own GPD CP machinery
# (`gpd_k1_setics` + `gpd_k1_loglik` + `gpd_k1_ldda`/`gpd_k1_lddda` +
# `gpd_k1_mu1fa`/`gpd_k1_mu2fa` + `dmgs`) on FIXED n = 20 datasets whose
# reference MLE shape is just below -1 (so the reference `revert2ml` fires and
# the CP output equals the MLE output), and emits the reference `ml_params`,
# `ml_quantiles`, and `cp_quantiles` as JSON into
# tests/golden/gpd_smalln_negshape_golden.json.
#
# The defect (item C): the Rust multi-start lands at a shape just ABOVE -1
# (e.g. -0.987), which the reference's `revert2ml = (abs(v2hat) >= 1) ||
# (v2hat > 30)` does NOT catch. At that shape the DMGS `1/fhat` term blows up,
# the CP quantiles leave the support `y < -scale/shape`, and the monotone guard
# flattens the curve far above the data (the reported n = 20 case: 19.8 vs the
# reference 2.06). The fix widens the revert to `shape <= -0.98` (see
# `GPD_NEAR_MINUS_ONE_DELTA` in crates/fitdistcp-core/src/models/mod.rs).
#
# The fixtures come from the reference code, not a re-implementation:
#   * `gpd_k1_setics` (120b_gpd_k1_libs.R) — ics = [sd(x), 0] when ics = [0, 0],
#   * `gpd_k1_loglik` (120b_gpd_k1_libs.R) — the MLE objective,
#   * `gpd_k1_ldda` / `gpd_k1_lddda` (120c_gpd_k1_derivs.R) — the derivatives,
#   * `gpd_k1_mu1fa` / `gpd_k1_mu2fa` (120c_gpd_k1_derivs.R) — the CDF derivs,
#   * `dmgs` (000_evaluate_dmgs_equation3p3.R) — the expansion.
#
# Reproducibility:
#   * The datasets are FIXED LITERALS (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gpd_smalln_negshape_golden.R
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
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
alpha <- 1 - p_grid
fdalpha <- 0.01

# --- Fixed n = 20 datasets (deterministic literals) -------------------------
# Both are bounded-looking records (max ~1) whose reference MLE shape is just
# below -1, so the reference `revert2ml` fires and the CP output equals the MLE
# output. The Rust multi-start lands just ABOVE -1 on both (shape ~ -0.987),
# which is the item-C blow-up regime.
#
# `y_negshape_big`: a concave record with a heavy low tail; the pre-fix Rust CP
# curve was flat at ~3.92 (vs data max 1.0) — the clearest blow-up.
y_negshape_big <- c(
  0.002, 0.3580957422515204, 0.45586561262981606, 0.5250692866983445,
  0.5804792399520996, 0.627469709675076, 0.668683518185681,
  0.7056409316494344, 0.739306862964518, 0.7703368437858562,
  0.7991990655326616, 0.8262410133590513, 0.8517285631621542,
  0.8758702495258178, 0.8988330272506132, 0.9207528972477227,
  0.9417422977868534, 0.9618953844791897, 0.9812918896082028, 1.0
)
# `y_negshape_uniform`: a near-uniform record; the pre-fix Rust CP curve was
# flat at ~1.45 (vs data max 1.0).
y_negshape_uniform <- c(
  0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
  0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.97, 1.00
)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")

# --- Reference GPD CP on a near--1 n = 20 record ----------------------------
# Reproduces the reference `qgpd_k1_cp` computation, including its
# `revert2ml = (abs(v2hat) >= 1) || (v2hat > 30)` guard.
gpd_cp_fixture <- function(x) {
  nx <- length(x); kloc <- 0
  ics <- gpd_k1_setics(x, c(0, 0))
  opt1 <- optim(ics, gpd_k1_loglik, x = x, kloc = kloc, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]
  revert <- (abs(v2) >= 1) || (v2 > 30)
  ml_q <- extraDistr::qgpd(1 - alpha, mu = kloc, sigma = v1, xi = v2)
  fhat <- extraDistr::dgpd(ml_q, mu = kloc, sigma = v1, xi = v2)
  if (revert) {
    cp_q <- ml_q
  } else {
    ldd <- gpd_k1_ldda(x, v1, v2, kloc); lddi <- solve(ldd)
    lddd <- gpd_k1_lddda(x, v1, v2, kloc)
    lambdad <- c(-1 / v1, 0)
    mu1 <- gpd_k1_mu1fa(alpha, v1, v2, kloc); mu2 <- gpd_k1_mu2fa(alpha, v1, v2, kloc)
    dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 2)
    cp_q <- ml_q + dq / (nx * fhat)
  }
  list(
    params = c(v1, v2), loglik = opt1$val, revert = revert,
    setics = ics, ml_quantiles = ml_q, cp_quantiles = cp_q
  )
}

big <- gpd_cp_fixture(y_negshape_big)
uni <- gpd_cp_fixture(y_negshape_uniform)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gpd_smalln_negshape_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gpd_smalln_negshape_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-6, "atol": 1e-10},')
lines <- c(lines, paste0('  "p_grid": ', json_vec(p_grid), ","))
lines <- c(lines, '  "models": {')
lines <- c(lines, '    "genpareto_negshape_big": {')
lines <- c(lines, paste0('      "params": ', json_vec(big$params), ","))
lines <- c(lines, paste0('      "loglik": ', json_num(big$loglik), ","))
lines <- c(lines, paste0('      "revert2ml": ', tolower(as.character(big$revert)), ","))
lines <- c(lines, paste0('      "setics": ', json_vec(big$setics), ","))
lines <- c(lines, paste0('      "ml_quantiles": ', json_vec(big$ml_quantiles), ","))
lines <- c(lines, paste0('      "cp_quantiles": ', json_vec(big$cp_quantiles)))
lines <- c(lines, "    },")
lines <- c(lines, '    "genpareto_negshape_uniform": {')
lines <- c(lines, paste0('      "params": ', json_vec(uni$params), ","))
lines <- c(lines, paste0('      "loglik": ', json_num(uni$loglik), ","))
lines <- c(lines, paste0('      "revert2ml": ', tolower(as.character(uni$revert)), ","))
lines <- c(lines, paste0('      "setics": ', json_vec(uni$setics), ","))
lines <- c(lines, paste0('      "ml_quantiles": ', json_vec(uni$ml_quantiles), ","))
lines <- c(lines, paste0('      "cp_quantiles": ', json_vec(uni$cp_quantiles)))
lines <- c(lines, "    }")
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_negshape_big": ', json_vec(y_negshape_big), ","))
lines <- c(lines, paste0('    "y_negshape_uniform": ', json_vec(y_negshape_uniform)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("big:     params =", paste(big$params, collapse = ", "),
    " revert =", big$revert, " cp99 =", big$cp_quantiles[length(big$cp_quantiles)], "\n")
cat("uniform: params =", paste(uni$params, collapse = ", "),
    " revert =", uni$revert, " cp99 =", uni$cp_quantiles[length(uni$cp_quantiles)], "\n")
