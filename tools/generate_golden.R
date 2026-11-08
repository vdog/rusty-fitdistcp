#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# W7 golden-fixture generator for the fitdistcp Rust port.
#
# Runs the REFERENCE fitdistcp R package's own MLE routines on fixed, seeded
# datasets (one per model) and emits golden fixtures (params, SEs, AIC, loglik)
# as JSON into tests/golden/mle_golden.json.
#
# Reproducibility:
#   * The datasets are FIXED LITERALS embedded below (no RNG dependence), so the
#     fixtures are byte-for-byte reproducible on any machine.
#   * The reference MLE routines are sourced directly from the cloned package
#     (tools/fitdistcp-ref/R) so the fixtures come from the reference code, not
#     from a re-implementation.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/generate_golden.R
#
# Reference package: https://github.com/stephenjewson/fitdistcp
#   clone: git clone --depth 1 https://github.com/stephenjewson/fitdistcp.git \
#              tools/fitdistcp-ref
# ---------------------------------------------------------------------------

# The dev container is ephemeral (docker compose run --rm), so R packages
# installed into the image's site-library do not persist. We install extraDistr
# into a workspace-local library (tools/Rlib, gitignored) that survives across
# container runs, and put it first on the search path.
local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

suppressPackageStartupMessages(library(extraDistr))

# --- Locate the cloned reference package -----------------------------------
ref_dir <- "tools/fitdistcp-ref/R"
if (!dir.exists(ref_dir)) {
  stop("Reference package not found at ", ref_dir,
       ". Clone it first (see header).")
}

# Source only the library files that define the MLE log-likelihoods and the
# closed-form MLE helpers. Sourcing the whole R/ directory would pull in the
# DMGS/rust/plotting code and its heavy dependencies (mev, rust, fExtremes,
# gnorm, fdrtool, pracma, actuar), which are not needed for MLE fixtures.
lib_files <- c(
  "005_utils.R",          # make_maic, make_se
  "030b_norm_libs.R",     # norm_ml_params
  "050b_gumbel_libs.R",   # gumbel_loglik
  "052b_weibull_libs.R",  # weibull_loglik
  "100b_gamma_libs.R",    # gamma_loglik
  "110b_gev_libs.R",      # gev_loglik, gev_setics
  "120b_gpd_k1_libs.R",   # gpd_k1_loglik, gpd_k1_setics
  "060b_norm_p1_libs.R"   # norm_p1_mlparams, norm_p1_loglik
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference lib file: ", path)
  source(path)
}

# --- Fixed, documented datasets --------------------------------------------
# n = 30 for every model. Values are fixed literals so the fixtures do not
# depend on R's RNG. Positive-support models use strictly positive data.
n <- 30

# General-purpose data (normal / GEV / Gumbel): roughly N(10, 2).
y_general <- c(
  9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41,
  10.93, 8.21, 12.05, 10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.90,
  7.63, 10.81, 9.29, 12.44, 8.98, 10.17, 11.53, 9.60, 10.72, 8.40
)

# Positive data (Weibull / Gamma / Log-normal / Exponential / GPD): roughly
# log-normal-ish, strictly positive.
y_positive <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)

# Predictor for norm_p1: fixed, roughly N(0, 1).
t_p1 <- c(
  -1.21, 0.44, -0.87, 1.33, -0.15, 0.92, -1.55, 0.28, 1.71, -0.63,
   0.05, -1.02, 1.18, -0.39, 0.77, -1.44, 0.61, -0.08, 1.49, -0.91,
   0.22, -1.28, 0.98, -0.52, 1.05, -0.71, 0.36, -1.11, 0.83, -0.24
)

# --- JSON helpers -----------------------------------------------------------
# Minimal JSON writer (no jsonlite dependency) so the script runs with only
# extraDistr installed.
json_num <- function(x) {
  if (!is.finite(x)) return("null")
  formatC(x, digits = 17, format = "g")
}
json_vec <- function(v) paste0("[", paste(vapply(v, json_num, ""), collapse = ", "), "]")
json_str <- function(s) paste0("\"", s, "\"")
json_str_vec <- function(v) paste0("[", paste(vapply(v, json_str, ""), collapse = ", "), "]")

# --- Per-model MLE extraction ----------------------------------------------
# Each entry returns list(params=..., se=..., aic=..., loglik=..., n=...,
#                          param_names=..., note=...).
# SEs follow the reference convention: sqrt(-(ldd/nx)^{-1}_{ii}) where ldd is
# the second derivative of the NORMALIZED log-likelihood (see make_se). This is
# algebraically identical to sqrt((H^{-1})_{ii}) with H the full observed
# information, which is the Rust convention.

models <- list()

# Tight optimizer control for the reference MLEs.
#
# The reference package calls `optim(...)` with DEFAULT control, i.e. Nelder-
# Mead with reltol = 1e-8. On the flat likelihood surfaces of the DMGS models
# (Weibull/Gamma/Gumbel/GEV/GPD) that stops well short of the true MLE: the
# default gamma fit is shape = 2.008119, whereas a tightly-converged fit is
# shape = 2.008469 (verified with reltol = 1e-14). The Rust BFGS/Newton
# optimizer converges to the true MLE, so comparing against the under-converged
# default would produce a spurious "discrepancy".
#
# We therefore generate the golden values with a TIGHTLY-CONVERGED reference
# optimizer so the fixtures represent the true MLE. This is a deliberate,
# documented choice: the fixtures are the reference *model* evaluated at its
# true optimum, not the reference *optimizer's* default stopping point.
OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)

# --- norm -------------------------------------------------------------------
{
  x <- y_general
  ml <- norm_ml_params(x)                 # [mu, sigma] with sigma = sqrt((n-1)/n) sd
  ll <- sum(dnorm(x, mean = ml[1], sd = ml[2], log = TRUE))
  # Observed information (analytic): diag(n/sigma^2, 2n/sigma^2).
  info <- diag(c(length(x) / ml[2]^2, 2 * length(x) / ml[2]^2))
  se <- sqrt(diag(solve(info)))
  models$norm <- list(
    params = as.numeric(ml), se = as.numeric(se),
    aic = 2 * 2 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("mu", "sigma"),
    note = "closed-form MLE; sigma uses n denominator (true MLE)"
  )
}

# --- expon ------------------------------------------------------------------
{
  x <- y_positive
  rate <- length(x) / sum(x)
  ll <- sum(dexp(x, rate = rate, log = TRUE))
  info <- matrix(length(x) / rate^2, 1, 1)
  se <- sqrt(diag(solve(info)))
  models$expon <- list(
    params = rate, se = as.numeric(se),
    aic = 2 * 1 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("rate"),
    note = "closed-form MLE rate = n/sum(x)"
  )
}

# --- lnorm ------------------------------------------------------------------
{
  x <- y_positive
  ml <- norm_ml_params(log(x))            # [meanlog, sdlog] on log scale
  ll <- sum(dlnorm(x, meanlog = ml[1], sdlog = ml[2], log = TRUE))
  info <- diag(c(length(x) / ml[2]^2, 2 * length(x) / ml[2]^2))
  se <- sqrt(diag(solve(info)))
  models$lnorm <- list(
    params = as.numeric(ml), se = as.numeric(se),
    aic = 2 * 2 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("meanlog", "sdlog"),
    note = "closed-form MLE on log scale (norm_ml_params of log(x))"
  )
}

# --- weibull ----------------------------------------------------------------
{
  x <- y_positive
  opt <- optim(c(1, 1), weibull_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]     # [shape, scale]
  ll <- opt$val
  # Observed information from the reference's analytic ldd (normalized), then
  # SE via make_se. weibull_ldda is defined in 052c_weibull_derivs.R.
  derivs_file <- file.path(ref_dir, "052c_weibull_derivs.R")
  if (file.exists(derivs_file)) source(derivs_file)
  ldd <- weibull_ldda(x, v1, v2)
  lddi <- solve(ldd)
  se <- make_se(length(x), lddi)
  models$weibull <- list(
    params = c(v1, v2), se = as.numeric(se),
    aic = 2 * 2 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("shape", "scale"),
    note = "optim(c(1,1), weibull_loglik, fnscale=-1); SE from analytic ldd"
  )
}

# --- gamma ------------------------------------------------------------------
{
  x <- y_positive
  opt <- optim(c(1, 1), gamma_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]     # [shape, SCALE] (R dgamma scale=)
  ll <- opt$val
  derivs_file <- file.path(ref_dir, "100c_gamma_derivs.R")
  if (file.exists(derivs_file)) source(derivs_file)
  ldd <- gamma_ldda(x, v1, v2)
  lddi <- solve(ldd)
  se <- make_se(length(x), lddi)
  models$gamma <- list(
    params = c(v1, v2), se = as.numeric(se),
    aic = 2 * 2 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("shape", "scale"),
    note = paste("optim(c(1,1), gamma_loglik, fnscale=-1); reference uses R",
                 "dgamma scale parameterization (scale = 1/rate). Rust uses",
                 "rate; convert rate = 1/scale before comparing.")
  )
}

# --- gumbel -----------------------------------------------------------------
{
  x <- y_general
  v1start <- mean(x); v2start <- sd(x)
  opt <- optim(c(v1start, v2start), gumbel_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]     # [loc, scale]
  ll <- opt$val
  derivs_file <- file.path(ref_dir, "050c_gumbel_derivs.R")
  if (file.exists(derivs_file)) source(derivs_file)
  ldd <- gumbel_ldda(x, v1, v2)
  lddi <- solve(ldd)
  se <- make_se(length(x), lddi)
  models$gumbel <- list(
    params = c(v1, v2), se = as.numeric(se),
    aic = 2 * 2 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("location", "scale"),
    note = "optim(c(mean,sd), gumbel_loglik, fnscale=-1); SE from analytic ldd"
  )
}

# --- genextreme (GEV) -------------------------------------------------------
{
  x <- y_general
  ics <- gev_setics(x, c(0, 0, 0))
  opt <- optim(ics, gev_loglik, x = x, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]; v3 <- opt$par[3]  # [mu, sigma, xi]
  ll <- opt$val
  derivs_file <- file.path(ref_dir, "110c_gev_derivs.R")
  if (file.exists(derivs_file)) source(derivs_file)
  ldd <- gev_ldda(x, v1, v2, v3)
  lddi <- solve(ldd)
  se <- make_se(length(x), lddi)
  models$genextreme <- list(
    params = c(v1, v2, v3), se = as.numeric(se),
    aic = 2 * 3 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("location", "scale", "shape"),
    note = "optim(gev_setics, gev_loglik, fnscale=-1); SE from analytic ldd"
  )
}

# --- genpareto (GPD) --------------------------------------------------------
{
  x <- y_positive
  kloc <- 0
  ics <- gpd_k1_setics(x, c(0, 0))
  opt <- optim(ics, gpd_k1_loglik, x = x, kloc = kloc, control = OPT_CTRL)
  v1 <- opt$par[1]; v2 <- opt$par[2]     # [sigma, xi]
  ll <- opt$val
  derivs_file <- file.path(ref_dir, "120c_gpd_k1_derivs.R")
  if (file.exists(derivs_file)) source(derivs_file)
  ldd <- gpd_k1_ldda(x, v1, v2, kloc)
  lddi <- solve(ldd)
  se <- make_se(length(x), lddi)
  models$genpareto <- list(
    params = c(v1, v2), se = as.numeric(se),
    aic = 2 * 2 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("scale", "shape"),
    note = "optim(gpd_k1_setics, gpd_k1_loglik, fnscale=-1); threshold kloc=0"
  )
}

# --- norm_p1 ----------------------------------------------------------------
{
  x <- y_general
  t <- t_p1
  ml <- norm_p1_mlparams(x, t)           # [intercept, slope, sd(residuals)]
  # The reference's own ml_value uses the UNBIASED sigma (ml[3]); the true MLE
  # log-likelihood uses the MLE sigma sqrt((n-1)/n)*sd(resid). We store the
  # TRUE MLE loglik/AIC so the fixture is self-consistent with the MLE params
  # (the reference's ml_value is documented as a quirk in the note).
  sigma_mle <- ml[3] * sqrt((length(x) - 1) / length(x))
  ll <- norm_p1_loglik(c(ml[1], ml[2], sigma_mle), x, t)
  # Observed information exactly as the reference `qnorm_p1_cp` builds it:
  #   expinfmat = diag(nx/sigma^2, sum(tresid^2)/sigma^2, 2*nx/sigma^2)
  # with tresid = t - mean(t) (the CENTERED predictor) and sigma = the
  # UNBIASED sd (ml[3]). The reference reports the UNcentered intercept in
  # ml_params but the CENTERED information here, so its SEs are for the
  # centered parameterization. We reproduce that convention verbatim.
  nx <- length(x)
  tresid <- t - mean(t)
  s <- ml[3]
  info <- diag(c(nx / s^2, sum(tresid * tresid) / s^2, 2 * nx / s^2))
  se <- sqrt(diag(solve(info)))
  models$norm_p1 <- list(
    params = as.numeric(ml), se = as.numeric(se),
    aic = 2 * 3 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("beta0", "beta1", "sigma"),
    note = paste("reference norm_p1_mlparams uses lm(x~t): [intercept, slope,",
                 "sd(residuals)]. Two reference quirks: (1) ml_params[1] is the",
                 "UNcentered intercept, but the SEs use the CENTERED predictor",
                 "tresid = t - mean(t) (orthogonal design); (2) sigma is the",
                 "UNBIASED sd (n-1 denominator), not the true MLE",
                 "sqrt((n-1)/n)*sd. The harness runs Rust with t0=mean(t) and",
                 "converts both quirks before comparing.")
  )
}

# --- genextreme_p1 ----------------------------------------------------------
# The reference `qgev_p1_cp` centers the predictor by its mean before fitting
# (`t = t - mean(t)`), so the model is mu_i = beta0 + beta1 * (t_i - mean(t)).
# We reproduce that here using the reference's own dgev density
# (extraDistr::dgev), matching the reference parameterization exactly (W14).
{
  x <- y_general
  t <- t_p1 - mean(t_p1)                  # reference centering
  gev_p1_loglik <- function(vv, x, t) {
    mu <- vv[1] + vv[2] * t
    sum(extraDistr::dgev(x, mu = mu, sigma = max(vv[3], .Machine$double.eps),
                         xi = vv[4], log = TRUE))
  }
  start <- c(mean(x), 0, sd(x), 0)
  opt <- optim(start, gev_p1_loglik, x = x, t = t, control = OPT_CTRL)
  v <- opt$par                            # [beta0, beta1, scale, shape]
  ll <- opt$val
  # Numerical observed information (central differences) on the full loglik.
  num_hess <- function(f, p, h = 1e-4) {
    k <- length(p); H <- matrix(0, k, k)
    for (i in 1:k) for (j in 1:k) {
      pp <- p; pp[i] <- pp[i] + h; pp[j] <- pp[j] + h; fpp <- f(pp)
      pm <- p; pm[i] <- pm[i] + h; pm[j] <- pm[j] - h; fpm <- f(pm)
      mp <- p; mp[i] <- mp[i] - h; mp[j] <- mp[j] + h; fmp <- f(mp)
      mm <- p; mm[i] <- mm[i] - h; mm[j] <- mm[j] - h; fmm <- f(mm)
      H[i, j] <- (fpp - fpm - fmp + fmm) / (4 * h * h)
    }
    H
  }
  H <- num_hess(function(p) gev_p1_loglik(p, x, t), v)
  se <- sqrt(diag(solve(-H)))
  models$genextreme_p1 <- list(
    params = as.numeric(v), se = as.numeric(se),
    aic = 2 * 4 - 2 * ll, loglik = ll, n = length(x),
    param_names = c("beta0", "beta1", "scale", "shape"),
    note = paste("optim over [beta0, beta1, scale, shape] with mu_i = beta0 +",
                 "beta1*(t_i - mean(t)) using extraDistr::dgev (reference",
                 "density). Reference centering (W14). SE from numerical",
                 "observed information.")
  )
}

# --- Emit JSON --------------------------------------------------------------
out_dir <- "tests/golden"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(out_dir, "mle_golden.json")

lines <- c("{")
lines <- c(lines, '  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",')
lines <- c(lines, '  "generator": "tools/generate_golden.R",')
lines <- c(lines, '  "tolerance": {"rtol": 1e-8, "atol": 1e-10},')
lines <- c(lines, '  "models": {')

model_names <- names(models)
for (mi in seq_along(model_names)) {
  nm <- model_names[[mi]]
  m <- models[[nm]]
  lines <- c(lines, paste0("    ", json_str(nm), ": {"))
  lines <- c(lines, paste0("      \"param_names\": ",
                           json_str_vec(m$param_names), ","))
  lines <- c(lines, paste0("      \"params\": ", json_vec(m$params), ","))
  lines <- c(lines, paste0("      \"se\": ", json_vec(m$se), ","))
  lines <- c(lines, paste0("      \"aic\": ", json_num(m$aic), ","))
  lines <- c(lines, paste0("      \"loglik\": ", json_num(m$loglik), ","))
  lines <- c(lines, paste0("      \"n\": ", m$n, ","))
  lines <- c(lines, paste0("      \"note\": ", json_str(m$note)))
  lines <- c(lines, paste0("    }", if (mi < length(model_names)) "," else ""))
}
lines <- c(lines, "  },")

# Also emit the fixed datasets so the Rust harness uses identical inputs.
lines <- c(lines, '  "datasets": {')
lines <- c(lines, paste0('    "y_general": ', json_vec(y_general), ","))
lines <- c(lines, paste0('    "y_positive": ', json_vec(y_positive), ","))
lines <- c(lines, paste0('    "t_p1": ', json_vec(t_p1)))
lines <- c(lines, "  }")
lines <- c(lines, "}")

writeLines(lines, out_path)
cat("Wrote", out_path, "\n")
cat("Models:", paste(model_names, collapse = ", "), "\n")
