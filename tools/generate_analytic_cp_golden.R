#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Phase 1 (Issue 1) golden-fixture generator: exact analytic CP quantiles for
# norm / lnorm / expon at n = 10, 20, 30.
#
# Runs the REFERENCE fitdistcp R package's own analytic CP routines
# (`qnorm_cp`, `qlnorm_cp`, `qexp_cp`) on fixed, seeded datasets and emits the
# CP quantiles as JSON into tests/golden/analytic_cp_golden.json.
#
# The fixtures come from the reference code, not a re-implementation:
#   * `qnorm_cp`  (030a_norm.R)  — Student-t with df = n-1, sg = v2hat*sqrt(n/(n-1))*sqrt((n+1)/n)
#   * `qlnorm_cp` (035a_lnorm.R) — exp(mu + qt * s1 * sqrt(1+1/n))
#   * `qexp_cp`   (010a_exp.R)   — sumx * (alpha^(-1/n) - 1)
#
# Reproducibility:
#   * The datasets are FIXED LITERALS (deterministic), so the fixtures are
#     byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_analytic_cp_golden.R
# ---------------------------------------------------------------------------

local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

ref_dir <- "tools/fitdistcp-ref/R"
if (!dir.exists(ref_dir)) {
  stop("Reference package not found at ", ref_dir, ". Clone it first.")
}

# Source the reference analytic CP routines and their dependencies.
lib_files <- c(
  "005_utils.R",        # make_maic, make_se
  "030b_norm_libs.R",   # norm_ml_params, norm_waic, norm_logscores
  "030a_norm.R",        # qnorm_cp
  "035b_lnorm_libs.R",  # lnorm_waic, lnorm_logscores
  "035a_lnorm.R",       # qlnorm_cp
  "010b_exp_libs.R",    # exp_waic, exp_logscores
  "010a_exp.R"          # qexp_cp
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

# --- Fixed datasets (deterministic literals) --------------------------------
# norm: general-scale values; lnorm/expon: positive values.
y_norm <- c(
  9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41,
  10.93, 8.21, 12.05, 10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.90,
  7.63, 10.81, 9.29, 12.44, 8.98, 10.17, 11.53, 9.60, 10.72, 8.40
)
y_positive <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)

# Probability levels (fixed).
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")

# --- Compute the analytic CP quantiles for each model at each n -------------
# The reference `q*_cp` returns a list with `cp_quantiles` (the analytic CP).
compute <- function(fn, x) {
  r <- fn(x, p = p_grid)
  list(
    params = as.numeric(r$ml_params),
    ml_quantiles = as.numeric(r$ml_quantiles),
    cp_quantiles = as.numeric(r$cp_quantiles)
  )
}

ns <- c(10L, 20L, 30L)
models <- list()
for (n in ns) {
  key <- paste0("n", n)
  models[[key]] <- list(
    norm = compute(qnorm_cp, y_norm[seq_len(n)]),
    lnorm = compute(qlnorm_cp, y_positive[seq_len(n)]),
    expon = compute(qexp_cp, y_positive[seq_len(n)])
  )
}

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "analytic_cp_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_analytic_cp_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-8, "atol": 1e-10},')
lines <- c(lines, paste0('  "p_grid": ', json_vec(p_grid), ","))
lines <- c(lines, '  "models": {')

model_names <- names(models)
for (mi in seq_along(model_names)) {
  nm <- model_names[[mi]]
  m <- models[[nm]]
  lines <- c(lines, paste0("    ", json_str(nm), ": {"))
  sub_names <- names(m)
  for (si in seq_along(sub_names)) {
    sn <- sub_names[[si]]
    s <- m[[sn]]
    lines <- c(lines, paste0("      ", json_str(sn), ": {"))
    lines <- c(lines, paste0("        \"params\": ", json_vec(s$params), ","))
    lines <- c(lines, paste0("        \"ml_quantiles\": ", json_vec(s$ml_quantiles), ","))
    lines <- c(lines, paste0("        \"cp_quantiles\": ", json_vec(s$cp_quantiles)))
    lines <- c(lines, paste0("      }", if (si < length(sub_names)) "," else ""))
  }
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_norm": ', json_vec(y_norm), ","))
lines <- c(lines, paste0('    "y_positive": ', json_vec(y_positive)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
