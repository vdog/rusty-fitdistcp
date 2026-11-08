#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# W11 golden-fixture generator for the fitdistcp Rust port.
#
# Extends the CP golden coverage to the four Phase-1 models not covered by
# cp_golden.rs: genextreme (GEV), genextreme_p1, genpareto (GPD), norm_p1.
#
# Runs the REFERENCE fitdistcp R package's own CP machinery on fixed datasets
# and emits golden fixtures (CP pdf, CP quantiles, ML pdf, ML quantiles, params)
# as JSON into tests/golden/w11_cp_golden.json.
#
# The fixtures come from the reference code, not a re-implementation:
#   * `qgev_cp` (110a_gev.R) — GEV CP quantiles + quantile-derived CP density,
#   * `qgpd_k1_cp` (120a_gpd_k1.R) — GPD CP quantiles + quantile-derived density,
#   * `qgev_p1_cp` (150a_gev_p1.R) — GEV-with-predictor CP quantiles,
#   * `qnorm_p1_cp` (060a_norm_p1.R) — normal-with-predictor analytic CP,
#   * the per-model `*_ldda` / `*_lddda` / `*_mu1fa` / `*_mu2fa` derivative
#     helpers from the `*_derivs.R` files,
#   * `dmgs` (000_evaluate_dmgs_equation3p3.R).
#
# # Conventions (documented, so the Rust harness can match exactly)
#
# * **GEV / GPD**: the reference predicts at the MLE quantile; the Rust DMGS
#   path is identical. The CP density is the reference's quantile-derived
#   `rh_flat_pdf = -(Δα)/(Δq)` with `fdalpha = 0.01`.
# * **norm_p1 / gev_p1**: the Rust `_p1` models evaluate the DMGS expansion at
#   `PREDICTION_INDEX = 0`, i.e. at the centered predictor `(t[0]-t0)/n0`. We
#   therefore emit the reference CP at the prediction point `t0 = t[0]` (in the
#   mean-centered predictor space) so the two coincide. The Rust harness builds
#   the model with `t0 = 0, n0 = 1` and passes the mean-centered `t`.
# * **norm_p1 MLE convention**: the reference `norm_p1_mlparams` uses the
#   UNBIASED residual SD (`sd(lm$residuals)`, n-1 denominator), whereas the Rust
#   `fit` converges to the true MLE (biased, n denominator). We emit BOTH the
#   reference params and the reference DMGS CP evaluated at the *Rust* params,
#   so the harness can isolate the DMGS expansion from the MLE convention.
#
# Reproducibility:
#   * The datasets are FIXED LITERALS (the reference example data), so the
#     fixtures are byte-for-byte reproducible.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_w11_cp_golden.R
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
  "005_utils.R",                     # deriv_copyldd, deriv_copylddd, make_se, make_maic,
                                     # movexiawayfromzero, maket0, maketresid0, makemuhat0
  "110b_gev_libs.R",                 # gev_loglik, gev_setics
  "110c_gev_derivs.R",               # gev_ldda, gev_lddda, gev_mu1fa, gev_mu2fa
  "120b_gpd_k1_libs.R",              # gpd_k1_loglik, gpd_k1_setics
  "120c_gpd_k1_derivs.R",            # gpd_k1_ldda, gpd_k1_lddda, gpd_k1_mu1fa, gpd_k1_mu2fa
  "060b_norm_p1_libs.R",             # norm_p1_mlparams, qnorm_p1_formula
  "060c_norm_p1_derivs.R",           # norm_p1_ldda, norm_p1_lddda, norm_p1_mu1fa, norm_p1_mu2fa
  "150b_gev_p1_libs.R",              # gev_p1_loglik, gev_p1_setics, qgev_p1, dgev_p1
  "153c_gev_p1a_derivs.R"            # gev_p1a_ldda, gev_p1a_lddda, gev_p1a_mu1fa, gev_p1a_mu2fa
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

# --- JSON helpers -----------------------------------------------------------
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")
json_bool <- function(b) if (isTRUE(b)) "true" else "false"

models <- list()

# --- genextreme (GEV, negative shape, reference example data) ---------------
{
  load("tools/fitdistcp-ref/data/d110gev_example_data_v1.rda")
  x <- d110gev_example_data_v1
  nx <- length(x)
  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]; v3 <- opt1$par[3]
  ldd <- gev_ldda(x, v1, v2, v3); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1, v2, v3)
  lambdad <- c(0, -1 / v2, 0)
  ml_q <- qgev(1 - alpha, mu = v1, sigma = v2, xi = v3)
  fhat <- dgev(ml_q, mu = v1, sigma = v2, xi = v3)
  mu1 <- gev_mu1fa(alpha, v1, v2, v3); mu2 <- gev_mu2fa(alpha, v1, v2, v3)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 3)
  cp_q <- ml_q + dq / (nx * fhat)
  dalpha <- pmin(fdalpha * alpha, fdalpha * (1 - alpha))
  alpham <- alpha - dalpha; alphap <- alpha + dalpha
  ml_qm <- qgev(1 - alpham, mu = v1, sigma = v2, xi = v3)
  ml_qp <- qgev(1 - alphap, mu = v1, sigma = v2, xi = v3)
  fhatm <- dgev(ml_qm, mu = v1, sigma = v2, xi = v3)
  fhatp <- dgev(ml_qp, mu = v1, sigma = v2, xi = v3)
  mu1m <- gev_mu1fa(alpham, v1, v2, v3); mu1p <- gev_mu1fa(alphap, v1, v2, v3)
  mu2m <- gev_mu2fa(alpham, v1, v2, v3); mu2p <- gev_mu2fa(alphap, v1, v2, v3)
  dqm <- dmgs(lddi, lddd, mu1m, lambdad, mu2m, dim = 3)
  dqp <- dmgs(lddi, lddd, mu1p, lambdad, mu2p, dim = 3)
  qm <- ml_qm + dqm / (nx * fhatm); qp <- ml_qp + dqp / (nx * fhatp)
  cp_pdf <- -(alphap - alpham) / (qp - qm)
  models$genextreme <- list(
    params = c(v1, v2, v3), ml_pdf = fhat, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = "qgev_cp: DMGS CP quantiles + quantile-derived CP density (fdalpha=0.01)"
  )
}

# --- genextreme (GEV, positive shape, synthetic maxima) ---------------------
{
  x <- c(
    0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
    0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
    0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
  )
  gev_pos_data <- x
  nx <- length(x)
  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]; v3 <- opt1$par[3]
  ldd <- gev_ldda(x, v1, v2, v3); lddi <- solve(ldd)
  lddd <- gev_lddda(x, v1, v2, v3)
  lambdad <- c(0, -1 / v2, 0)
  ml_q <- qgev(1 - alpha, mu = v1, sigma = v2, xi = v3)
  fhat <- dgev(ml_q, mu = v1, sigma = v2, xi = v3)
  mu1 <- gev_mu1fa(alpha, v1, v2, v3); mu2 <- gev_mu2fa(alpha, v1, v2, v3)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 3)
  cp_q <- ml_q + dq / (nx * fhat)
  dalpha <- pmin(fdalpha * alpha, fdalpha * (1 - alpha))
  alpham <- alpha - dalpha; alphap <- alpha + dalpha
  ml_qm <- qgev(1 - alpham, mu = v1, sigma = v2, xi = v3)
  ml_qp <- qgev(1 - alphap, mu = v1, sigma = v2, xi = v3)
  fhatm <- dgev(ml_qm, mu = v1, sigma = v2, xi = v3)
  fhatp <- dgev(ml_qp, mu = v1, sigma = v2, xi = v3)
  mu1m <- gev_mu1fa(alpham, v1, v2, v3); mu1p <- gev_mu1fa(alphap, v1, v2, v3)
  mu2m <- gev_mu2fa(alpham, v1, v2, v3); mu2p <- gev_mu2fa(alphap, v1, v2, v3)
  dqm <- dmgs(lddi, lddd, mu1m, lambdad, mu2m, dim = 3)
  dqp <- dmgs(lddi, lddd, mu1p, lambdad, mu2p, dim = 3)
  qm <- ml_qm + dqm / (nx * fhatm); qp <- ml_qp + dqp / (nx * fhatp)
  cp_pdf <- -(alphap - alpham) / (qp - qm)
  models$genextreme_pos <- list(
    params = c(v1, v2, v3), ml_pdf = fhat, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = "positive-shape GEV; qgev_cp DMGS CP + quantile-derived density"
  )
}

# --- genpareto (GPD) --------------------------------------------------------
{
  load("tools/fitdistcp-ref/data/d120gpd_k1_example_data_v1.rda")
  x <- d120gpd_k1_example_data_v1
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
  models$genpareto <- list(
    params = c(v1, v2), ml_pdf = fhat, cp_pdf = cp_pdf,
    ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = "qgpd_k1_cp: DMGS CP quantiles + quantile-derived CP density (kloc=0)"
  )
}

# --- norm_p1 (analytic CP + DMGS CP) ----------------------------------------
{
  load("tools/fitdistcp-ref/data/d060norm_p1_example_data_v1_x.rda")
  load("tools/fitdistcp-ref/data/d060norm_p1_example_data_v1_t.rda")
  x <- d060norm_p1_example_data_v1_x
  t <- d060norm_p1_example_data_v1_t
  nx <- length(x)
  meant <- mean(t); t <- t - meant
  norm_p1_t_centered <- t
  # Predict at the first centered predictor so the Rust PREDICTION_INDEX = 0
  # (centered(0) = t[0] with t0 = 0, n0 = 1) coincides with t0.
  t0 <- t[1]
  tresid <- t - mean(t)
  tresid0 <- t0 - mean(t)
  ml <- norm_p1_mlparams(x, t)
  v1 <- ml[1]; v2 <- ml[2]; v3 <- ml[3]
  muhat0 <- v1 + v2 * t0
  ml_q <- qnorm(1 - alpha, mean = muhat0, sd = v3)
  # Analytic CP (what qnorm_p1_cp actually returns).
  cp_q_analytic <- qnorm_p1_formula(alpha, tresid, tresid0, nx, muhat0, v3)
  # DMGS-based CP (same expansion the Rust port uses).
  ldd <- norm_p1_ldda(x, t, v1, v2, v3); lddi <- solve(ldd)
  lddd <- norm_p1_lddda(x, t, v1, v2, v3)
  lambdad <- c(0, 0, -1 / v3)
  fhat <- dnorm(ml_q, mean = muhat0, sd = v3)
  mu1 <- norm_p1_mu1fa(alpha, t0, v1, v2, v3)
  mu2 <- norm_p1_mu2fa(alpha, t0, v1, v2, v3)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 3)
  cp_q_dmgs <- ml_q + dq / (nx * fhat)
  # Reference DMGS CP evaluated at the RUST params (true MLE, biased sigma),
  # so the harness can isolate the DMGS expansion from the MLE convention.
  v1r <- 0.6886651644002889
  v2r <- 1.1061138551923522
  v3r <- 0.9304085868285676
  muhat0r <- v1r + v2r * t0
  ml_qr <- qnorm(1 - alpha, mean = muhat0r, sd = v3r)
  lddr <- norm_p1_ldda(x, t, v1r, v2r, v3r); lddir <- solve(lddr)
  ldddr <- norm_p1_lddda(x, t, v1r, v2r, v3r)
  lambdadr <- c(0, 0, -1 / v3r)
  fhatr <- dnorm(ml_qr, mean = muhat0r, sd = v3r)
  mu1r <- norm_p1_mu1fa(alpha, t0, v1r, v2r, v3r)
  mu2r <- norm_p1_mu2fa(alpha, t0, v1r, v2r, v3r)
  dqr <- dmgs(lddir, ldddr, mu1r, lambdadr, mu2r, dim = 3)
  cp_q_dmgs_rustparams <- ml_qr + dqr / (nx * fhatr)
  models$norm_p1 <- list(
    params = c(v1, v2, v3), ml_quantiles = ml_q,
    cp_quantiles_analytic = cp_q_analytic,
    cp_quantiles_dmgs = cp_q_dmgs,
    cp_quantiles_dmgs_rustparams = cp_q_dmgs_rustparams,
    rust_params = c(v1r, v2r, v3r),
    note = paste("qnorm_p1_cp analytic CP (qnorm_p1_formula) + DMGS CP;",
                 "reference sigma is the UNBIASED sd (n-1), Rust uses the true MLE")
  )
}

# --- gev_p1 (DMGS) ----------------------------------------------------------
{
  load("tools/fitdistcp-ref/data/d150gev_p1_example_data_v1_x.rda")
  load("tools/fitdistcp-ref/data/d150gev_p1_example_data_v1_t.rda")
  x <- d150gev_p1_example_data_v1_x
  t <- d150gev_p1_example_data_v1_t
  nx <- length(x)
  meant <- mean(t); t <- t - meant
  gev_p1_t_centered <- t
  # Predict at the first centered predictor so the Rust PREDICTION_INDEX = 0
  # (centered(0) = t[0] with t0 = 0, n0 = 1) coincides with t0.
  t0 <- t[1]
  ics <- gev_p1_setics(x, t, c(0, 0, 0, 0))
  opt1 <- optim(ics, gev_p1_loglik, x = x, t = t, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]; v3 <- opt1$par[3]; v4 <- opt1$par[4]
  ldd <- gev_p1a_ldda(x, t, v1, v2, v3, v4); lddi <- solve(ldd)
  lddd <- gev_p1a_lddda(x, t, v1, v2, v3, v4)
  lambdad <- c(0, 0, -1 / v3, 0)
  ml_q <- qgev_p1(1 - alpha, t0, ymn = v1, slope = v2, sigma = v3, xi = v4)
  fhat <- dgev_p1(ml_q, t0, ymn = v1, slope = v2, sigma = v3, xi = v4)
  mu1 <- gev_p1a_mu1fa(alpha, t0, v1, v2, v3, v4)
  mu2 <- gev_p1a_mu2fa(alpha, t0, v1, v2, v3, v4)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 4)
  cp_q <- ml_q + dq / (nx * fhat)
  models$genextreme_p1 <- list(
    params = c(v1, v2, v3, v4), ml_quantiles = ml_q, cp_quantiles = cp_q,
    note = paste("qgev_p1_cp: DMGS CP quantiles; prediction at t0 = t[0]",
                 "(centered predictor space)")
  )
}

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "w11_cp_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_w11_cp_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-8, "atol": 1e-10},')
lines <- c(lines, paste0('  "p_grid": ', json_vec(p_grid), ","))
lines <- c(lines, '  "models": {')

model_names <- names(models)
for (mi in seq_along(model_names)) {
  nm <- model_names[[mi]]
  m <- models[[nm]]
  lines <- c(lines, paste0("    ", json_str(nm), ": {"))
  lines <- c(lines, paste0("      \"params\": ", json_vec(m$params), ","))
  if (!is.null(m$ml_pdf)) {
    lines <- c(lines, paste0("      \"ml_pdf\": ", json_vec(m$ml_pdf), ","))
  }
  if (!is.null(m$cp_pdf)) {
    lines <- c(lines, paste0("      \"cp_pdf\": ", json_vec(m$cp_pdf), ","))
  }
  lines <- c(lines, paste0("      \"ml_quantiles\": ", json_vec(m$ml_quantiles), ","))
  if (!is.null(m$cp_quantiles)) {
    lines <- c(lines, paste0("      \"cp_quantiles\": ", json_vec(m$cp_quantiles), ","))
  }
  if (!is.null(m$cp_quantiles_analytic)) {
    lines <- c(lines, paste0("      \"cp_quantiles_analytic\": ",
                             json_vec(m$cp_quantiles_analytic), ","))
  }
  if (!is.null(m$cp_quantiles_dmgs)) {
    lines <- c(lines, paste0("      \"cp_quantiles_dmgs\": ",
                             json_vec(m$cp_quantiles_dmgs), ","))
  }
  if (!is.null(m$cp_quantiles_dmgs_rustparams)) {
    lines <- c(lines, paste0("      \"cp_quantiles_dmgs_rustparams\": ",
                             json_vec(m$cp_quantiles_dmgs_rustparams), ","))
  }
  if (!is.null(m$rust_params)) {
    lines <- c(lines, paste0("      \"rust_params\": ", json_vec(m$rust_params), ","))
  }
  lines <- c(lines, paste0("      \"note\": ", json_str(m$note)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

# Datasets (fixed literals, so the harness can rebuild the models).
lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "gev_neg": ', json_vec(d110gev_example_data_v1), ","))
lines <- c(lines, paste0('    "gev_pos": ', json_vec(gev_pos_data), ","))
lines <- c(lines, paste0('    "gpd": ', json_vec(d120gpd_k1_example_data_v1), ","))
lines <- c(lines, paste0('    "norm_p1_x": ', json_vec(d060norm_p1_example_data_v1_x), ","))
# Emit the MEAN-CENTERED predictor so the Rust model (t0 = 0, n0 = 1) matches
# the reference, which centers t by mean(t) internally.
lines <- c(lines, paste0('    "norm_p1_t": ', json_vec(norm_p1_t_centered), ","))
lines <- c(lines, paste0('    "gev_p1_x": ', json_vec(d150gev_p1_example_data_v1_x), ","))
lines <- c(lines, paste0('    "gev_p1_t": ', json_vec(gev_p1_t_centered)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
