#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# W8 golden-fixture generator for the fitdistcp Rust port (DMGS / CP path).
#
# Runs the REFERENCE fitdistcp R package's own DMGS machinery on fixed, seeded
# datasets and emits golden fixtures (CP pdf, CP quantiles, ML pdf, ML
# quantiles, params) as JSON into tests/golden/cp_golden.json.
#
# The fixtures come from the reference code, not a re-implementation:
#   * the DMGS expansion `dmgs` / `bayesian_dq_4terms_v1` from
#     000_evaluate_dmgs_equation3p3.R,
#   * the per-model `*_ldda` / `*_lddda` / `*_mu1fa` / `*_mu2fa` / `*_f1fa` /
#     `*_f2fa` derivative helpers from the `*_derivs.R` files,
#   * the analytic CP routines (`dnormsub`, `dexpsub`, `dlnormsub`) for the
#     analytic models.
#
# Reproducibility:
#   * The datasets are FIXED LITERALS (identical to tools/generate_golden.R),
#     so the fixtures are byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_cp_golden.R
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

# Source the DMGS core + the per-model libs/derivs we need.
lib_files <- c(
  "000_evaluate_dmgs_equation3p3.R", # dmgs / bayesian_dq_4terms_v1
  "005_utils.R",                     # deriv_copyldd, deriv_copylddd, make_se, make_maic
  "030b_norm_libs.R",                # norm_ml_params, dnormsub
  "030c_norm_derivs.R",              # norm_ldda, norm_lddda, norm_mu1fa, norm_mu2fa, norm_f1fa, norm_f2fa
  "010b_exp_libs.R",                 # dexpsub
  "010c_exp_derivs.R",               # exp_ldda, exp_lddda, exp_mu1fa, exp_mu2fa, exp_f1fa, exp_f2fa
  "035b_lnorm_libs.R",               # dlnormsub
  "035c_lnorm_derivs.R",             # lnorm_ldda, lnorm_lddda, lnorm_mu1fa, lnorm_mu2fa, lnorm_f1fa, lnorm_f2fa
  "050b_gumbel_libs.R",              # gumbel_loglik, dgumbelsub
  "050c_gumbel_derivs.R",            # gumbel_ldda, gumbel_lddda, gumbel_mu1fa, gumbel_mu2fa, gumbel_f1fa, gumbel_f2fa
  "052b_weibull_libs.R",             # weibull_loglik
  "052c_weibull_derivs.R",           # weibull_ldda, weibull_lddda, weibull_mu1fa, weibull_mu2fa, weibull_f1fa, weibull_f2fa
  "100b_gamma_libs.R",               # gamma_loglik
  "100c_gamma_derivs.R"              # gamma_ldda, gamma_lddda, gamma_mu1fa, gamma_mu2fa, gamma_f1fa, gamma_f2fa
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

# --- Fixed datasets (identical to tools/generate_golden.R) ------------------
y_general <- c(
  9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41,
  10.93, 8.21, 12.05, 10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.90,
  7.63, 10.81, 9.29, 12.44, 8.98, 10.17, 11.53, 9.60, 10.72, 8.40
)
y_positive <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)
# Phase D (Issue 5): the gamma ~90% outlier case. A fixed positive dataset with
# a heavy right tail (a few large values) that drives the gamma MLE to a small
# shape / large scale, the regime where the pre-Phase-5 (shape, rate) DMGS
# expansion diverged from the reference by ~90% (the report's headline gamma
# outlier). The literal is fixed so the fixture is reproducible.
y_gamma_outlier <- c(
  0.05, 0.08, 0.11, 0.14, 0.19, 0.23, 0.31, 0.42, 0.58, 0.77,
  1.05, 1.44, 2.10, 3.35, 5.90, 11.20, 24.50, 61.30, 180.40, 640.70
)

# Prediction grid and probability levels (fixed).
x_grid <- c(8.0, 9.0, 10.0, 11.0, 12.0)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9)

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)

models <- list()

# --- norm (analytic CP + DMGS CP) -------------------------------------------
{
  x <- y_general
  nx <- length(x)
  ml <- norm_ml_params(x)
  v1 <- ml[1]; v2 <- ml[2]
  dd <- dnormsub(x = x, y = x_grid)
  ml_pdf <- dd$ml_pdf
  cp_pdf <- dd$rh_pdf
  # The reference uses alpha = 1 - p and evaluates the quantile at (1 - alpha)
  # = p, so the quantiles are at probability p_grid.
  ml_q <- qnorm(p_grid, mean = v1, sd = v2)
  # analytic rhp quantiles (reference qnorm_cp)
  sgu <- v2 * sqrt(nx / (nx - 1))
  sg <- sgu * sqrt((nx + 1) / nx)
  cp_q <- v1 + qt(p_grid, df = nx - 1) * sg
  # DMGS-based CP (same expansion the Rust port uses), for an apples-to-apples
  # comparison at the §7.2 tolerance.
  ldd <- norm_ldda(x, v1, v2); lddi <- solve(ldd)
  lddd <- norm_lddda(x, v1, v2)
  lambdad_rhp <- c(0, -1 / v2)
  f1 <- norm_f1fa(x_grid, v1, v2); f2 <- norm_f2fa(x_grid, v1, v2)
  dmgs_pdf <- pmax(ml_pdf + dmgs(lddi, lddd, f1, lambdad_rhp, f2, dim = 2) / nx, 0)
  alpha <- 1 - p_grid
  mu1 <- norm_mu1fa(alpha, v1, v2); mu2 <- norm_mu2fa(alpha, v1, v2)
  fhat <- dnorm(ml_q, mean = v1, sd = v2)
  dmgs_q <- ml_q + dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim = 2) / (nx * fhat)
  models$norm <- list(
    params = as.numeric(ml), ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    dmgs_pdf = dmgs_pdf, dmgs_quantiles = dmgs_q,
    note = "analytic CP (dnormsub / qnorm_cp) + DMGS CP (norm_* + dmgs)"
  )
}

# --- expon (analytic CP) ----------------------------------------------------
{
  x <- y_positive
  nx <- length(x); sx <- sum(x)
  rate <- nx / sx
  dd <- dexpsub(x = x, y = x_grid)
  ml_pdf <- dd$ml_pdf
  cp_pdf <- dd$rh_pdf
  ml_q <- qexp(p_grid, rate = rate)
  # reference qexp_cp: rh_quantiles = sumx*((alpha^(-1/nx))-1) with alpha = 1-p.
  alpha <- 1 - p_grid
  cp_q <- sx * ((alpha^(-1 / nx)) - 1)
  # DMGS-based CP (same expansion the Rust port uses).
  ldd <- exp_ldda(x, rate); lddi <- solve(ldd)
  lddd <- exp_lddda(x, rate)
  lambdad_rhp <- -1 / rate
  f1 <- exp_f1fa(x_grid, rate); f2 <- exp_f2fa(x_grid, rate)
  dmgs_pdf <- pmax(ml_pdf + dmgs(lddi, lddd, f1, lambdad_rhp, f2, dim = 1) / nx, 0)
  # The reference has no exp_mu1fa/exp_mu2fa; the CDF derivatives are
  # exp_p1fa/exp_p2fa and mu1 = -p1, mu2 = -p2 (matching the *_mu1fa convention).
  mu1 <- -exp_p1fa(ml_q, rate)
  mu2 <- -exp_p2fa(ml_q, rate)
  fhat <- dexp(ml_q, rate = rate)
  dmgs_q <- ml_q + dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim = 1) / (nx * fhat)
  models$expon <- list(
    params = rate, ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    dmgs_pdf = dmgs_pdf, dmgs_quantiles = dmgs_q,
    note = "analytic CP (dexpsub / qexp_cp) + DMGS CP (exp_* + dmgs)"
  )
}

# --- lnorm (analytic CP) ----------------------------------------------------
{
  x <- y_positive
  nx <- length(x)
  y <- log(x)
  ml <- norm_ml_params(y)
  v1 <- ml[1]; v2 <- ml[2]
  dd <- dlnormsub(x = x, y = x_grid)
  ml_pdf <- dd$ml_pdf
  cp_pdf <- dd$rh_pdf
  ml_q <- qlnorm(p_grid, meanlog = v1, sdlog = v2)
  mu <- mean(y); s1 <- sqrt(var(y))
  cp_q <- exp(mu + qt(p_grid, df = nx - 1) * s1 * sqrt(1 + 1 / nx))
  # DMGS-based CP (same expansion the Rust port uses).
  ldd <- lnorm_ldda(x, v1, v2); lddi <- solve(ldd)
  lddd <- lnorm_lddda(x, v1, v2)
  lambdad_rhp <- c(0, -1 / v2)
  f1 <- lnorm_f1fa(x_grid, v1, v2); f2 <- lnorm_f2fa(x_grid, v1, v2)
  dmgs_pdf <- pmax(ml_pdf + dmgs(lddi, lddd, f1, lambdad_rhp, f2, dim = 2) / nx, 0)
  alpha <- 1 - p_grid
  mu1 <- lnorm_mu1fa(alpha, v1, v2); mu2 <- lnorm_mu2fa(alpha, v1, v2)
  fhat <- dlnorm(ml_q, meanlog = v1, sdlog = v2)
  dmgs_q <- ml_q + dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim = 2) / (nx * fhat)
  models$lnorm <- list(
    params = as.numeric(ml), ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    dmgs_pdf = dmgs_pdf, dmgs_quantiles = dmgs_q,
    note = "analytic CP (dlnormsub / qlnorm_cp) + DMGS CP (lnorm_* + dmgs)"
  )
}

# --- gumbel (DMGS) ----------------------------------------------------------
{
  x <- y_general
  nx <- length(x)
  opt <- optim(c(mean(x), sd(x)), gumbel_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]
  ldd <- gumbel_ldda(x, v1, v2); lddi <- solve(ldd)
  lddd <- gumbel_lddda(x, v1, v2)
  lambdad_rhp <- c(0, -1 / v2)
  # CP pdf: rh_pdf = ml_pdf + dmgs(lddi,lddd,f1,lambdad,f2)/nx
  f1 <- gumbel_f1fa(x_grid, v1, v2)
  f2 <- gumbel_f2fa(x_grid, v1, v2)
  ml_pdf <- extraDistr::dgumbel(x_grid, mu = v1, sigma = v2)
  df1 <- dmgs(lddi, lddd, f1, lambdad_rhp, f2, dim = 2)
  cp_pdf <- pmax(ml_pdf + df1 / nx, 0)
  # CP quantiles: rh_q = ml_q + dmgs(lddi,lddd,mu1,lambdad,mu2)/(nx*fhat).
  # The reference uses alpha = 1 - p and evaluates at (1 - alpha) = p.
  alpha <- 1 - p_grid
  ml_q <- qgumbel(1 - alpha, mu = v1, sigma = v2)
  mu1 <- gumbel_mu1fa(alpha, v1, v2)
  mu2 <- gumbel_mu2fa(alpha, v1, v2)
  fhat <- extraDistr::dgumbel(ml_q, mu = v1, sigma = v2)
  dq <- dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim = 2)
  cp_q <- ml_q + dq / (nx * fhat)
  models$gumbel <- list(
    params = c(v1, v2), ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = "DMGS (gumbel_ldda/lddda/mu1fa/mu2fa/f1fa/f2fa + dmgs)"
  )
}

# --- weibull (DMGS) ---------------------------------------------------------
{
  x <- y_positive
  nx <- length(x)
  opt <- optim(c(1, 1), weibull_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]  # [shape, scale]
  ldd <- weibull_ldda(x, v1, v2); lddi <- solve(ldd)
  lddd <- weibull_lddda(x, v1, v2)
  lambdad_rhp <- c(0, -1 / v2)
  f1 <- weibull_f1fa(x_grid, v1, v2)
  f2 <- weibull_f2fa(x_grid, v1, v2)
  ml_pdf <- dweibull(x_grid, shape = v1, scale = v2)
  df1 <- dmgs(lddi, lddd, f1, lambdad_rhp, f2, dim = 2)
  cp_pdf <- pmax(ml_pdf + df1 / nx, 0)
  alpha <- 1 - p_grid
  ml_q <- qweibull(1 - alpha, shape = v1, scale = v2)
  mu1 <- weibull_mu1fa(alpha, v1, v2)
  mu2 <- weibull_mu2fa(alpha, v1, v2)
  fhat <- dweibull(ml_q, shape = v1, scale = v2)
  dq <- dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim = 2)
  cp_q <- ml_q + dq / (nx * fhat)
  models$weibull <- list(
    params = c(v1, v2), ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = "DMGS (weibull_* + dmgs); params [shape, scale]"
  )
}

# --- gamma (DMGS) -----------------------------------------------------------
# The reference uses R's dgamma scale parameterization and a NON-homogeneous
# ad-hoc "type 1" prior lambdad_cp = c(-1/shape, -1/scale). The Rust model uses
# (shape, rate). We emit the reference (shape, scale) params and the CP pdf
# computed in the reference parameterization; the Rust harness converts.
#
# NOTE: the reference has no analytic gamma_mu1fa/gamma_mu2fa (it uses the
# numerical gamma_mu1f/gamma_mu2f defined in 100a_gamma.R), so we emit the CP
# *pdf* only for the gamma and leave the CP quantiles to the Rust-side
# finite-difference path (validated separately).
{
  x <- y_positive
  nx <- length(x)
  opt <- optim(c(1, 1), gamma_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]  # [shape, SCALE]
  ldd <- gamma_ldda(x, v1, v2); lddi <- solve(ldd)
  lddd <- gamma_lddda(x, v1, v2)
  lambdad_cp <- c(-1 / v1, -1 / v2)  # reference "type 1" prior
  f1 <- gamma_f1fa(x_grid, v1, v2)
  f2 <- gamma_f2fa(x_grid, v1, v2)
  ml_pdf <- dgamma(x_grid, shape = v1, scale = v2)
  df1 <- dmgs(lddi, lddd, f1, lambdad_cp, f2, dim = 2)
  cp_pdf <- pmax(ml_pdf + df1 / nx, 0)
  ml_q <- qgamma(p_grid, shape = v1, scale = v2)
  models$gamma <- list(
    params = c(v1, v2), ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = rep(NA_real_, length(p_grid)),
    note = paste("DMGS (gamma_* + dmgs); reference params [shape, scale];",
                 "type-1 prior c(-1/shape,-1/scale); CP quantiles not emitted",
                 "(reference uses numerical gamma_mu1f/mu2f)")
  )
}

# --- gamma_outlier (DMGS; Phase D, Issue 5) ---------------------------------
# The ~90% outlier case: a heavy-tailed positive dataset that drives the gamma
# MLE to a small shape / large scale. This is the regime where the pre-Phase-5
# (shape, rate) DMGS expansion diverged from the reference by ~90%. The fixture
# pins the reference CP pdf (computed in the reference (shape, scale)
# parameterization) so the Rust `GammaScale` residual can be asserted below
# `GAMMA_RTOL`. As with `gamma`, the reference emits no CP quantiles.
{
  x <- y_gamma_outlier
  nx <- length(x)
  opt <- optim(c(1, 1), gamma_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]  # [shape, SCALE]
  ldd <- gamma_ldda(x, v1, v2); lddi <- solve(ldd)
  lddd <- gamma_lddda(x, v1, v2)
  lambdad_cp <- c(-1 / v1, -1 / v2)  # reference "type 1" prior
  f1 <- gamma_f1fa(x_grid, v1, v2)
  f2 <- gamma_f2fa(x_grid, v1, v2)
  ml_pdf <- dgamma(x_grid, shape = v1, scale = v2)
  df1 <- dmgs(lddi, lddd, f1, lambdad_cp, f2, dim = 2)
  cp_pdf <- pmax(ml_pdf + df1 / nx, 0)
  ml_q <- qgamma(p_grid, shape = v1, scale = v2)
  models$gamma_outlier <- list(
    params = c(v1, v2), ml_pdf = ml_pdf, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = rep(NA_real_, length(p_grid)),
    note = paste("Phase D gamma ~90% outlier case; DMGS (gamma_* + dmgs);",
                 "reference params [shape, scale]; type-1 prior",
                 "c(-1/shape,-1/scale); CP quantiles not emitted")
  )
}

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "cp_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_cp_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-8, "atol": 1e-10},')
lines <- c(lines, paste0('  "x_grid": ', json_vec(x_grid), ","))
lines <- c(lines, paste0('  "p_grid": ', json_vec(p_grid), ","))
lines <- c(lines, '  "models": {')

model_names <- names(models)
for (mi in seq_along(model_names)) {
  nm <- model_names[[mi]]
  m <- models[[nm]]
  lines <- c(lines, paste0("    ", json_str(nm), ": {"))
  lines <- c(lines, paste0("      \"params\": ", json_vec(m$params), ","))
  lines <- c(lines, paste0("      \"ml_pdf\": ", json_vec(m$ml_pdf), ","))
  lines <- c(lines, paste0("      \"cp_pdf\": ", json_vec(m$cp_pdf), ","))
  lines <- c(lines, paste0("      \"ml_quantiles\": ", json_vec(m$ml_quantiles), ","))
  lines <- c(lines, paste0("      \"cp_quantiles\": ", json_vec(m$cp_quantiles), ","))
  if (!is.null(m$dmgs_pdf)) {
    lines <- c(lines, paste0("      \"dmgs_pdf\": ", json_vec(m$dmgs_pdf), ","))
    lines <- c(lines, paste0("      \"dmgs_quantiles\": ", json_vec(m$dmgs_quantiles), ","))
  }
  lines <- c(lines, paste0("      \"note\": ", json_str(m$note)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_general": ', json_vec(y_general), ","))
lines <- c(lines, paste0('    "y_positive": ', json_vec(y_positive), ","))
lines <- c(lines, paste0('    "y_gamma_outlier": ', json_vec(y_gamma_outlier)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
