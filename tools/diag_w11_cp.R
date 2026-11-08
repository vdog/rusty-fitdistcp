#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# W11 diagnostic: reference CP pdf/quantiles for the four models not yet
# covered by cp_golden.rs (genextreme, genextreme_p1, genpareto, norm_p1).
#
# Runs the REFERENCE fitdistcp R package's own CP routines on fixed datasets
# and prints the reference CP pdf / CP quantiles so we can compare against the
# Rust DMGS path and choose tolerances.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/diag_w11_cp.R
# ---------------------------------------------------------------------------

local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}
suppressPackageStartupMessages(library(extraDistr))

ref_dir <- "tools/fitdistcp-ref/R"
lib_files <- c(
  "000_evaluate_dmgs_equation3p3.R",
  "005_utils.R",
  "110b_gev_libs.R",
  "110c_gev_derivs.R",
  "120b_gpd_k1_libs.R",
  "120c_gpd_k1_derivs.R",
  "060b_norm_p1_libs.R",
  "060c_norm_p1_derivs.R",
  "150b_gev_p1_libs.R",
  "153c_gev_p1a_derivs.R"
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

# --- GEV (negative shape, reference example data) ---------------------------
{
  load("tools/fitdistcp-ref/data/d110gev_example_data_v1.rda")
  x <- d110gev_example_data_v1
  nx <- length(x)
  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]; v3 <- opt1$par[3]
  cat("GEV params:", v1, v2, v3, " shape sign:", sign(v3), "\n")
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
  cat("GEV cp_q:", cp_q, "\n")
  cat("GEV cp_pdf:", cp_pdf, "\n")
}

# --- GEV (positive shape, synthetic maxima) ---------------------------------
{
  # A heavy-tailed sample gives a positive GEV shape.
  x <- c(
    0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
    0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
    0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
  )
  nx <- length(x)
  ics <- gev_setics(x, c(0, 0, 0))
  opt1 <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]; v3 <- opt1$par[3]
  cat("GEVpos params:", v1, v2, v3, " shape sign:", sign(v3), "\n")
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
  cat("GEVpos cp_q:", cp_q, "\n")
  cat("GEVpos cp_pdf:", cp_pdf, "\n")
}

# --- GPD --------------------------------------------------------------------
{
  load("tools/fitdistcp-ref/data/d120gpd_k1_example_data_v1.rda")
  x <- d120gpd_k1_example_data_v1
  nx <- length(x); kloc <- 0
  ics <- gpd_k1_setics(x, c(0, 0))
  opt1 <- optim(ics, gpd_k1_loglik, x = x, kloc = kloc, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]
  cat("GPD params:", v1, v2, " shape sign:", sign(v2), "\n")
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
  cat("GPD cp_q:", cp_q, "\n")
  cat("GPD cp_pdf:", cp_pdf, "\n")
}

# --- norm_p1 (analytic CP + DMGS CP) ----------------------------------------
{
  load("tools/fitdistcp-ref/data/d060norm_p1_example_data_v1_x.rda")
  load("tools/fitdistcp-ref/data/d060norm_p1_example_data_v1_t.rda")
  x <- d060norm_p1_example_data_v1_x
  t <- d060norm_p1_example_data_v1_t
  nx <- length(x)
  meant <- mean(t); t <- t - meant
  # Predict at the FIRST centered predictor value so the Rust model's
  # PREDICTION_INDEX = 0 coincides with t0.
  t0 <- t[1]
  tresid <- t - mean(t)
  tresid0 <- t0 - mean(t)
  ml <- norm_p1_mlparams(x, t)
  v1 <- ml[1]; v2 <- ml[2]; v3 <- ml[3]
  muhat0 <- v1 + v2 * t0
  cat("norm_p1 params:", v1, v2, v3, "\n")
  ml_q <- qnorm(1 - alpha, mean = muhat0, sd = v3)
  # Analytic CP (reference qnorm_p1_cp uses this, NOT DMGS).
  cp_q_analytic <- qnorm_p1_formula(alpha, tresid, tresid0, nx, muhat0, v3)
  # DMGS-based CP (same expansion the Rust port uses), for apples-to-apples.
  ldd <- norm_p1_ldda(x, t, v1, v2, v3); lddi <- solve(ldd)
  lddd <- norm_p1_lddda(x, t, v1, v2, v3)
  lambdad <- c(0, 0, -1 / v3)
  fhat <- dnorm(ml_q, mean = muhat0, sd = v3)
  mu1 <- norm_p1_mu1fa(alpha, t0, v1, v2, v3)
  mu2 <- norm_p1_mu2fa(alpha, t0, v1, v2, v3)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 3)
  cp_q_dmgs <- ml_q + dq / (nx * fhat)
  cat("norm_p1 ml_q:", ml_q, "\n")
  cat("norm_p1 cp_q_analytic:", cp_q_analytic, "\n")
  cat("norm_p1 cp_q_dmgs:", cp_q_dmgs, "\n")
  cat("norm_p1 x:", x, "\n")
  cat("norm_p1 t:", t, "\n")

  # --- Apples-to-apples: reference DMGS with the RUST params ----------------
  # Rust params (beta0 at t0=t[1], biased sigma) and prediction at t0 = t[1].
  v1r <- 0.15773051390769322
  v2r <- 1.1061138551923082
  v3r <- 0.9304085868348183
  # The Rust model (t0=0, n0=1) predicts at centered(0) = t[1]. The reference
  # predicts at mu = v1r + v2r*t0r, so t0r = t[1] gives the same point.
  t0r <- t[1]
  muhat0r <- v1r + v2r * t0r
  ml_qr <- qnorm(1 - alpha, mean = muhat0r, sd = v3r)
  lddr <- norm_p1_ldda(x, t, v1r, v2r, v3r); lddir <- solve(lddr)
  ldddr <- norm_p1_lddda(x, t, v1r, v2r, v3r)
  lambdadr <- c(0, 0, -1 / v3r)
  fhatr <- dnorm(ml_qr, mean = muhat0r, sd = v3r)
  mu1r <- norm_p1_mu1fa(alpha, t0r, v1r, v2r, v3r)
  mu2r <- norm_p1_mu2fa(alpha, t0r, v1r, v2r, v3r)
  dqr <- dmgs(lddir, ldddr, mu1r, lambdadr, mu2r, dim = 3)
  cp_q_dmgs_rustparams <- ml_qr + dqr / (nx * fhatr)
  cat("norm_p1 cp_q_dmgs_rustparams:", cp_q_dmgs_rustparams, "\n")
}

# --- gev_p1 (DMGS) ----------------------------------------------------------
{
  load("tools/fitdistcp-ref/data/d150gev_p1_example_data_v1_x.rda")
  load("tools/fitdistcp-ref/data/d150gev_p1_example_data_v1_t.rda")
  x <- d150gev_p1_example_data_v1_x
  t <- d150gev_p1_example_data_v1_t
  nx <- length(x)
  meant <- mean(t); t <- t - meant
  # Predict at the FIRST centered predictor value so that the Rust model's
  # PREDICTION_INDEX = 0 (centered(0) = (t[0]-t0)/n0) coincides with t0.
  t0 <- t[1]
  ics <- gev_p1_setics(x, t, c(0, 0, 0, 0))
  opt1 <- optim(ics, gev_p1_loglik, x = x, t = t, control = OPT_CTRL)
  v1 <- opt1$par[1]; v2 <- opt1$par[2]; v3 <- opt1$par[3]; v4 <- opt1$par[4]
  cat("gev_p1 params:", v1, v2, v3, v4, " shape sign:", sign(v4), "\n")
  ldd <- gev_p1a_ldda(x, t, v1, v2, v3, v4); lddi <- solve(ldd)
  lddd <- gev_p1a_lddda(x, t, v1, v2, v3, v4)
  lambdad <- c(0, 0, -1 / v3, 0)
  ml_q <- qgev_p1(1 - alpha, t0, ymn = v1, slope = v2, sigma = v3, xi = v4)
  fhat <- dgev_p1(ml_q, t0, ymn = v1, slope = v2, sigma = v3, xi = v4)
  mu1 <- gev_p1a_mu1fa(alpha, t0, v1, v2, v3, v4)
  mu2 <- gev_p1a_mu2fa(alpha, t0, v1, v2, v3, v4)
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 4)
  cp_q <- ml_q + dq / (nx * fhat)
  cat("gev_p1 ml_q:", ml_q, "\n")
  cat("gev_p1 cp_q:", cp_q, "\n")
  cat("gev_p1 x:", x, "\n")
  cat("gev_p1 t:", t, "\n")
}
