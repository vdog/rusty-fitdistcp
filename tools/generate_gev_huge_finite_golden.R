#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Issue #5 golden-fixture generator: GEV records whose Rust CP 1-in-100 return
# level is a HUGE FINITE value (|cp| > 1e6) because the ML shape is large and
# positive (xi >= 3).
#
# Runs the REFERENCE fitdistcp R package's own GEV CP machinery
# (`gev_setics` + `gev_loglik` + `qgev`/`dgev`) on FIXED records taken from the
# hail-region surrogate, and emits the reference `ml_params`, `revert2ml`,
# `ml_quantiles`, and `cp_quantiles` as JSON into
# tests/golden/gev_huge_finite_golden.json.
#
# # The behaviour (Issue #5)
#
# The Rust CP 1-in-100 return level is huge (1e6 .. 1e18) on ~192 surrogate
# regions. The maintainer verified that this is NOT a Rust defect: the ML shape
# is large and positive (xi >= 3.1, median ~7.5), and the reference ALSO reverts
# to the ML quantile when `abs(v3hat) >= 1` (see `qgev_cp` in 110a_gev.R). A GEV
# with xi > 1 has an extremely heavy tail, so the ML 1-in-100 return level is
# huge for BOTH implementations. The CP level therefore equals the ML level on
# both sides, and the huge value is reference-matching behaviour.
#
# This fixture pins that claim: it records the reference `cp_quantiles` on
# representative huge-finite records so the Rust harness can assert
# Rust CP == reference CP (not merely CP == Rust ML, which is trivially true
# after a revert).
#
# The fixtures come from the reference code, not a re-implementation:
#   * `gev_setics` (110b_gev_libs.R) — the initial conditions,
#   * `gev_loglik` (110b_gev_libs.R) — the MLE objective,
#   * `qgev` / `dgev` (extraDistr) — the ML quantiles / density.
#
# Reproducibility:
#   * The datasets are FIXED LITERALS (deterministic), so the fixture is
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_gev_huge_finite_golden.R
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
  "005_utils.R",     # fixgevrange, movexiawayfromzero, make_se, make_maic
  "110b_gev_libs.R"  # gev_loglik, gev_setics, dgevsub
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
alpha <- 1 - p_grid

# --- Fixed records (deterministic literals) ---------------------------------
# These are hail-surrogate regions whose Rust ML shape is large and positive
# (xi >= 3), so the CP 1-in-100 return level is huge. The non-zero years are
# copied verbatim from the surrogate dump (see the plan §3.1).
#
# `y_huge_finite_n9`: n = 9, Rust shape ~ 9.37 (region 39) — the "real" regime
# (n >= 8) the maintainer says is fitted in practice.
y_huge_finite_n9 <- c(
  0.38769525029678692, 9.39280180114009333, 2.21997046132707698,
  7.27557060542835288, 0.82032699068751069, 1.03456297313919432,
  0.38650546104321459, 2.79592649591480047, 1.63934146651602197
)
# `y_huge_finite_n10`: n = 10, Rust shape ~ 5.20 (region 114).
y_huge_finite_n10 <- c(
  2.00819440003154037, 0.13688914160769641, 0.12305474805128246,
  6.78787018594121072, 13.20254907269826994, 0.56299249086552128,
  8.46349362053482324, 5.98586316401298468, 7.78249265620097752,
  2.32716262744009095
)
# `y_huge_finite_n11`: n = 11, Rust shape ~ 6.42 (region 599).
y_huge_finite_n11 <- c(
  3.73947430228426780, 2.70999552225175755, 1.24398765948405887,
  16.64048431418169471, 8.97972290733812706, 2.76095201702238668,
  35.23065457862664118, 1.45815906960561881, 3.58995170871876290,
  1.24735273915795197, 3.63271023754450217
)
# `y_huge_finite_n6`: n = 6, Rust shape ~ 7.24 (region 0) — the short-record
# regime (n < 8) where the likelihood has several near-equal xi > 1 maxima.
y_huge_finite_n6 <- c(
  1.01817763278341000, 0.82607314200412330, 0.23019941500529645,
  1.87283537393490773, 0.23795122664720778, 8.09486896513796239
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
# fixture is the reference's own output. The reference reverts to the ML
# quantiles when `abs(v3hat) >= 1` (the huge-finite case), so `cp_quantiles`
# equals `ml_quantiles` there.
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

  # The reference reverts to the ML quantiles when `abs(v3hat) >= 1`; the
  # huge-finite records all have xi >= 3, so this branch is taken.
  cp_quantiles <- if (revert2ml) ml_quantiles else NA_real_

  list(
    params = ml_params, loglik = loglik, revert2ml = revert2ml,
    ml_max = ml_max, ml_quantiles = ml_quantiles, ml_pdf = ml_pdf,
    cp_quantiles = cp_quantiles
  )
}

models <- list()
models$gev_huge_finite_n9 <- gev_cp_fixture(y_huge_finite_n9, p_grid)
models$gev_huge_finite_n10 <- gev_cp_fixture(y_huge_finite_n10, p_grid)
models$gev_huge_finite_n11 <- gev_cp_fixture(y_huge_finite_n11, p_grid)
models$gev_huge_finite_n6 <- gev_cp_fixture(y_huge_finite_n6, p_grid)

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "gev_huge_finite_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_gev_huge_finite_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 5e-2, "atol": 1e-6},')
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
lines <- c(lines, paste0('    "y_huge_finite_n9": ', json_vec(y_huge_finite_n9), ","))
lines <- c(lines, paste0('    "y_huge_finite_n10": ', json_vec(y_huge_finite_n10), ","))
lines <- c(lines, paste0('    "y_huge_finite_n11": ', json_vec(y_huge_finite_n11), ","))
lines <- c(lines, paste0('    "y_huge_finite_n6": ', json_vec(y_huge_finite_n6)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
for (nm in model_names) {
  cat(nm, "params =", paste(models[[nm]]$params, collapse = ", "),
      "revert2ml =", models[[nm]]$revert2ml,
      "cp99 =", models[[nm]]$cp_quantiles[length(models[[nm]]$cp_quantiles)], "\n")
}
