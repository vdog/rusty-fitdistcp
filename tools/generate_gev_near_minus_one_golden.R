#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Issue #4 golden-fixture generator: GEV short records (n = 5-8) whose Rust ML
# shape lands in the near--1 DMGS blow-up band.
#
# Runs the REFERENCE fitdistcp R package's own GEV CP machinery
# (`gev_setics` + `gev_loglik` + `gev_ldda`/`gev_lddda` + `gev_mu1fa`/`gev_mu2fa`
# + `dmgs`) on FIXED short records taken from the hail-region surrogate, and
# emits the reference `ml_params`, `revert2ml`, `ml_quantiles`, and
# `cp_quantiles` as JSON into tests/golden/gev_near_minus_one_golden.json.
#
# # The defect (Issue #4)
#
# The Rust multi-start lands at a shape just ABOVE -1 (e.g. -0.9669, -0.9767),
# which the reference's `revert2ml = abs(v3hat) >= 1` does NOT catch. At that
# shape the DMGS `1/fhat` term blows up, the CP quantiles leave the support
# `loc - scale/shape`, and the curve goes NaN. The reference `optim` never lands
# in this band (on these records it lands <= -1 or well above -0.90), so the
# reference reverts (or fits a benign shape) and returns a finite curve.
#
# The fix widens the GEV revert to `shape <= -0.96` (see
# `GEV_NEAR_MINUS_ONE_DELTA` in crates/fitdistcp-core/src/models/mod.rs), so the
# Rust CP output equals the (finite) ML output on these records.
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
#   Rscript tools/generate_gev_near_minus_one_golden.R
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

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
alpha <- 1 - p_grid

# --- Fixed short records (deterministic literals) ---------------------------
# These are the two hail-surrogate regions (824 and 1647) whose Rust ML shape
# lands in the near--1 band ABOVE -0.98 (so the GPD delta alone does not catch
# them). The Rust multi-start lands at shape ~ -0.9669 / -0.9767; the reference
# `optim` lands <= -1 (reverts) or well above -0.90.
#
# `y_near_minus_one_n8`: n = 8, Rust shape ~ -0.9669 (region 824).
y_near_minus_one_n8 <- c(
  0.7125294739446348, 0.4138741774114036, 0.980450294747705,
  1.1798904438994047, 1.6961481716254514, 0.97524963229813,
  1.6701931735981974, 1.7779094800244961
)
# `y_near_minus_one_n5`: n = 5, Rust shape ~ -0.9767 (region 1647).
y_near_minus_one_n5 <- c(
  1.0507941194109296, 0.9338334081992506, 0.06364190335675814,
  0.582510573474041, 0.3909819303691967
)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")
json_bool <- function(b) if (isTRUE(b)) "true" else "false"

# --- Reference qgev_cp core (quantiles), transcribed from 110a_gev.R --------
# We reproduce the exact reference computation for the CP quantiles, so the
# fixture is the reference's own output. The reference does NOT clamp the CP
# quantiles (Issue 2 re-check, item A), so `cp_quantiles` is the raw DMGS value
# (or the ML value when `revert2ml` fires).
gev_cp_fixture <- function(x, p_grid) {
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

  ldd <- gev_ldda(x, v1hat, v2hat, v3hat); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1hat, v2hat, v3hat)
  lambdad_rh_flat <- c(0, -1 / v2hat, 0)

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

models <- list()
models$gev_near_minus_one_n8 <- gev_cp_fixture(y_near_minus_one_n8, p_grid)
models$gev_near_minus_one_n5 <- gev_cp_fixture(y_near_minus_one_n5, p_grid)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gev_near_minus_one_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gev_near_minus_one_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-6, "atol": 1e-10},')
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
  lines <- c(lines, paste0("      \"ml_quantiles\": ", json_vec(m$ml_quantiles), ","))
  lines <- c(lines, paste0("      \"cp_quantiles\": ", json_vec(m$cp_quantiles)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_near_minus_one_n8": ', json_vec(y_near_minus_one_n8), ","))
lines <- c(lines, paste0('    "y_near_minus_one_n5": ', json_vec(y_near_minus_one_n5)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
for (nm in model_names) {
  cat(nm, "params =", paste(models[[nm]]$params, collapse = ", "),
      "revert2ml =", models[[nm]]$revert2ml,
      "cp99 =", models[[nm]]$cp_quantiles[length(models[[nm]]$cp_quantiles)], "\n")
}
