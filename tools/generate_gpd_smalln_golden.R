#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phase D golden-fixture generator: GPD small-n (n = 10) ML divergence.
#
# Runs the REFERENCE fitdistcp R package's own GPD MLE machinery
# (`gpd_k1_setics` + `gpd_k1_loglik`) on a FIXED small (n = 10) dataset and
# emits the reference `ml_params` and `loglik` as JSON into
# tests/golden/gpd_smalln_golden.json.
#
# The defect (Phase 3/5, Issue 3): the hand-rolled BFGS lacked the reference's
# robust multi-start / init strategy, so on short records (n = 10) it stopped at
# a worse fit — the report measured a log-likelihood divergence of up to 365% at
# n = 10. The Rust `fit_with_config` now seeds from `setics`-style heuristics in
# addition to the caller's `init`, adds a Nelder-Mead polish, and keeps the best
# log-likelihood. This fixture pins the reference result so the Rust fit's
# log-likelihood can be checked against it (within 1e-8).
#
# The fixtures come from the reference code, not a re-implementation:
#   * `gpd_k1_setics` (120b_gpd_k1_libs.R) — ics = [sd(x), 0] when ics = [0, 0],
#   * `gpd_k1_loglik` (120b_gpd_k1_libs.R) — the MLE objective.
#
# Reproducibility:
#   * The dataset is a FIXED LITERAL (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gpd_smalln_golden.R
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
  "120b_gpd_k1_libs.R" # gpd_k1_loglik, gpd_k1_setics
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)

# --- Fixed small-n (n = 10) GPD dataset (deterministic literal) -------------
# A short record of positive threshold exceedances (the regime where the
# hand-rolled BFGS stalled: the report measured up to 365% log-likelihood
# divergence at n = 10). The literal is fixed so the fixture is reproducible.
#
# The dataset is chosen so the reference `optim` from the `setics` start
# `[sd(x), 0]` converges to the **global** optimum (shape ~ -0.62, inside the
# GPD support `xi > -1`), verified against a multi-start search in R. This is
# the regime where the reference and the Rust multi-start agree, so the
# reference log-likelihood is a meaningful cross-check target. (Datasets with a
# heavier tail drive the reference `optim` to the `xi -> -1` boundary, where the
# reference's own `gpd_k1_checkmle` would reject the fit; those are not a
# meaningful cross-check target.)
y_smalln <- c(
  0.31, 0.79, 1.16, 2.45, 1.07, 3.62, 0.14, 1.94, 2.62, 0.11
)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")

# --- Reference GPD MLE on the small-n record --------------------------------
# Reproduces the reference `gpd_k1_setics` + `gpd_k1_loglik` MLE so the fixture
# is the reference's own output. We record the reference `ml_params` and the
# maximized `loglik` (the Phase D target).
gpd_smalln_fixture <- function(x, kloc = 0) {
  ics <- gpd_k1_setics(x, c(0, 0))
  opt1 <- optim(ics, gpd_k1_loglik, x = x, kloc = kloc, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]
  list(
    params = c(v1, v2), loglik = opt1$val, setics = ics,
    note = paste("gpd_k1_setics + gpd_k1_loglik MLE on a fixed n = 10 record;",
                 "params [scale, shape]")
  )
}

gpd_smalln <- gpd_smalln_fixture(y_smalln)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gpd_smalln_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gpd_smalln_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-8, "atol": 1e-10},')
lines <- c(lines, '  "models": {')
lines <- c(lines, '    "genpareto_smalln": {')
lines <- c(lines, paste0('      "params": ', json_vec(gpd_smalln$params), ","))
lines <- c(lines, paste0('      "loglik": ', json_num(gpd_smalln$loglik), ","))
lines <- c(lines, paste0('      "setics": ', json_vec(gpd_smalln$setics), ","))
lines <- c(lines, paste0('      "note": ', json_str(gpd_smalln$note)))
lines <- c(lines, "    }")
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_smalln": ', json_vec(y_smalln)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("n =", length(y_smalln), "\n")
cat("setics:", paste(gpd_smalln$setics, collapse = ", "), "\n")
cat("params:", paste(gpd_smalln$params, collapse = ", "), "\n")
cat("loglik:", gpd_smalln$loglik, "\n")
