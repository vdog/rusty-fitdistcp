#!/usr/bin/env Rscript
# Diagnostic: GPD SE discrepancy — analytic vs numerical Hessian at the SAME params.
local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) .libPaths(c(normalizePath(local_lib), .libPaths()))
suppressPackageStartupMessages(library(extraDistr))
ref_dir <- "tools/fitdistcp-ref/R"
source(file.path(ref_dir, "005_utils.R"))
source(file.path(ref_dir, "120b_gpd_k1_libs.R"))
source(file.path(ref_dir, "120c_gpd_k1_derivs.R"))

y_positive <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)
x <- y_positive
kloc <- 0
nx <- length(x)

# Rust params (scale, shape).
rust_scale <- 2.6754610952051294
rust_shape <- -0.5786995537956616

# Reference analytic ldd at the Rust params.
ldd <- gpd_k1_ldda(x, rust_scale, rust_shape, kloc)
lddi <- solve(ldd)
se_analytic <- make_se(nx, lddi)
cat("ref analytic ldd at Rust params:\n"); print(ldd)
cat("ref analytic SE at Rust params:", se_analytic, "\n")

# Numerical Hessian of the FULL loglik at the Rust params (central diff),
# matching the Rust numerical fallback convention.
full_loglik <- function(p) gpd_k1_loglik(p, x, kloc)
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
H <- num_hess(full_loglik, c(rust_scale, rust_shape))
cat("numerical full-loglik Hessian at Rust params (h=1e-4):\n"); print(H)
se_num <- sqrt(diag(solve(-H)))
cat("numerical SE at Rust params (h=1e-4):", se_num, "\n")

# Now use the EXACT Rust finite-difference steps:
#   h_i = rel_step_2 * max(1, |theta_i|) + abs_step
# with rel_step_2 = 1.2207031e-4, abs_step = 1e-6.
rel_step_2 <- 1.2207031e-4
abs_step <- 1e-6
h_scale <- rel_step_2 * max(1, abs(rust_scale)) + abs_step
h_shape <- rel_step_2 * max(1, abs(rust_shape)) + abs_step
cat("Rust steps: h_scale =", h_scale, " h_shape =", h_shape, "\n")
num_hess_steps <- function(f, p, h) {
  k <- length(p); H <- matrix(0, k, k)
  for (i in 1:k) for (j in 1:k) {
    pp <- p; pp[i] <- pp[i] + h[i]; pp[j] <- pp[j] + h[j]; fpp <- f(pp)
    pm <- p; pm[i] <- pm[i] + h[i]; pm[j] <- pm[j] - h[j]; fpm <- f(pm)
    mp <- p; mp[i] <- mp[i] - h[i]; mp[j] <- mp[j] + h[j]; fmp <- f(mp)
    mm <- p; mm[i] <- mm[i] - h[i]; mm[j] <- mm[j] - h[j]; fmm <- f(mm)
    H[i, j] <- (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
  }
  H
}
H2 <- num_hess_steps(full_loglik, c(rust_scale, rust_shape), c(h_scale, h_shape))
cat("numerical Hessian with Rust steps:\n"); print(H2)
se_num2 <- sqrt(diag(solve(-H2)))
cat("numerical SE with Rust steps:", se_num2, "\n")

# Rust reported SE.
cat("Rust reported SE: 0.6753563327451457 0.20382374496972858\n")
