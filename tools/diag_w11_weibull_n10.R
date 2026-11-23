#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# D1 diagnostic: Weibull CP at n = 10 — is the ~3% residual an algorithmic
# difference (the prior slope) or inherent O(1/n) truncation?
#
# The reference `qweibull_cp` uses `lambdad_rhp = c(-1/v1hat, -1/v2hat)`
# (shape component -1/shape), whereas the Rust `Weibull` DmgsModel uses the
# default right-Haar slope `c(0, -1/scale)`. This script computes the reference
# DMGS CP quantiles with BOTH priors at the same MLE, so the difference isolates
# the prior term from the O(1/n) truncation.
#
# Usage (inside the Docker dev container, from the workspace root):
#   Rscript tools/diag_w11_weibull_n10.R
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
  "052b_weibull_libs.R",
  "052c_weibull_derivs.R"
)
for (f in lib_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) stop("Missing reference file: ", path)
  source(path)
}

OPT_CTRL <- list(fnscale = -1, reltol = 1e-14, maxit = 10000)
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9)
alpha <- 1 - p_grid

# Fixed n = 10 positive dataset (a subset of the y_positive literal used by
# tools/generate_cp_golden.R, so it is reproducible).
x <- c(0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55)
nx <- length(x)

opt <- optim(c(1, 1), weibull_loglik, x = x, control = OPT_CTRL)
v1 <- opt$par[1]; v2 <- opt$par[2]  # [shape, scale]
cat(sprintf("n = %d  shape = %.17g  scale = %.17g\n", nx, v1, v2))

ldd <- weibull_ldda(x, v1, v2); lddi <- solve(ldd)
lddd <- weibull_lddda(x, v1, v2)
ml_q <- qweibull(1 - alpha, shape = v1, scale = v2)
fhat <- dweibull(ml_q, shape = v1, scale = v2)
mu1 <- weibull_mu1fa(alpha, v1, v2)
mu2 <- weibull_mu2fa(alpha, v1, v2)

cp_q <- function(lambdad) {
  dq <- dmgs(lddi, lddd, mu1, lambdad, mu2, dim = 2)
  ml_q + dq / (nx * fhat)
}

# Reference prior (what qweibull_cp actually uses).
lambdad_ref <- c(-1 / v1, -1 / v2)
# Rust prior (the DmgsModel default right-Haar slope).
lambdad_rust <- c(0, -1 / v2)

q_ref <- cp_q(lambdad_ref)
q_rust <- cp_q(lambdad_rust)

cat("\nml_q:            ", sprintf("%.17g ", ml_q), "\n")
cat("cp_q (ref prior):", sprintf("%.17g ", q_ref), "\n")
cat("cp_q (rust pri): ", sprintf("%.17g ", q_rust), "\n")
cat("rel diff (rust vs ref prior):",
    sprintf("%.3e ", abs(q_rust - q_ref) / abs(q_ref)), "\n")

# Also report the O(1/n) truncation scale: the DMGS correction relative to the
# ML quantile (the correction is O(1/n), so this shrinks ~1/n).
cat("\nDMGS correction / ml_q (ref prior):",
    sprintf("%.3e ", abs(q_ref - ml_q) / abs(ml_q)), "\n")

# And the same at n = 30 (the full y_positive literal) to show the 1/n scaling.
x30 <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)
n30 <- length(x30)
opt30 <- optim(c(1, 1), weibull_loglik, x = x30, control = OPT_CTRL)
w1 <- opt30$par[1]; w2 <- opt30$par[2]
ldd30 <- weibull_ldda(x30, w1, w2); lddi30 <- solve(ldd30)
lddd30 <- weibull_lddda(x30, w1, w2)
ml_q30 <- qweibull(1 - alpha, shape = w1, scale = w2)
fhat30 <- dweibull(ml_q30, shape = w1, scale = w2)
mu1_30 <- weibull_mu1fa(alpha, w1, w2)
mu2_30 <- weibull_mu2fa(alpha, w1, w2)
dq30_ref <- dmgs(lddi30, lddd30, mu1_30, c(-1 / w1, -1 / w2), mu2_30, dim = 2)
dq30_rust <- dmgs(lddi30, lddd30, mu1_30, c(0, -1 / w2), mu2_30, dim = 2)
q30_ref <- ml_q30 + dq30_ref / (n30 * fhat30)
q30_rust <- ml_q30 + dq30_rust / (n30 * fhat30)
cat(sprintf("\nn = %d  shape = %.17g  scale = %.17g\n", n30, w1, w2))
cat("rel diff (rust vs ref prior), n=30:",
    sprintf("%.3e ", abs(q30_rust - q30_ref) / abs(q30_ref)), "\n")
cat("DMGS correction / ml_q (ref prior), n=30:",
    sprintf("%.3e ", abs(q30_ref - ml_q30) / abs(ml_q30)), "\n")
