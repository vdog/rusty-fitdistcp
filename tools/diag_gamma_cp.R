#!/usr/bin/env Rscript
# Diagnostic: is the gamma CP-pdf difference due to (a) the prior slope or
# (b) the parameterization (shape,scale) vs (shape,rate)?
local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) .libPaths(c(normalizePath(local_lib), .libPaths()))
suppressPackageStartupMessages(library(extraDistr))
ref_dir <- "tools/fitdistcp-ref/R"
for (f in c("000_evaluate_dmgs_equation3p3.R", "005_utils.R",
            "100b_gamma_libs.R", "100c_gamma_derivs.R")) {
  source(file.path(ref_dir, f))
}
y_positive <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)
x <- y_positive
nx <- length(x)
x_grid <- c(8.0, 9.0, 10.0, 11.0, 12.0)
opt <- optim(c(1, 1), gamma_loglik, x = x,
             control = list(fnscale = -1, reltol = 1e-14, maxit = 10000))
v1 <- opt$par[1]; v2 <- opt$par[2]  # [shape, SCALE]
cat("shape =", v1, " scale =", v2, " rate =", 1/v2, "\n")

# Reference DMGS in (shape, scale) with type-1 prior.
ldd <- gamma_ldda(x, v1, v2); lddi <- solve(ldd)
lddd <- gamma_lddda(x, v1, v2)
f1 <- gamma_f1fa(x_grid, v1, v2); f2 <- gamma_f2fa(x_grid, v1, v2)
ml_pdf <- dgamma(x_grid, shape = v1, scale = v2)
for (prior in list(c(-1/v1, -1/v2), c(0, -1/v2))) {
  df1 <- dmgs(lddi, lddd, f1, prior, f2, dim = 2)
  cp_pdf <- pmax(ml_pdf + df1 / nx, 0)
  cat("prior =", prior, " -> cp_pdf =", cp_pdf, "\n")
}

# Now compute the SAME expansion in the (shape, rate) parameterization by
# transforming the derivatives via the chain rule, to test whether the
# difference is purely parameterization.
# rate r = 1/scale. d/dr = -scale^2 d/dscale = -(1/r^2) d/dscale.
# We rebuild ldd, lddd, f1, f2, prior in (shape, rate) numerically.
r <- 1 / v2
# log-density in (shape, rate): lgamma(shape) + shape*log(rate) + (shape-1)*log(y) - rate*y
logf_rate <- function(shape, rate, y) {
  sum(dgamma(y, shape = shape, rate = rate, log = TRUE))
}
# numerical derivatives of the normalized log-likelihood in (shape, rate)
num_ldd <- function(p, h = 1e-5) {
  f <- function(q) logf_rate(q[1], q[2], x) / nx
  H <- matrix(0, 2, 2)
  for (i in 1:2) for (j in 1:2) {
    pp <- p; pp[i] <- pp[i] + h; pp[j] <- pp[j] + h; fpp <- f(pp)
    pm <- p; pm[i] <- pm[i] + h; pm[j] <- pm[j] - h; fpm <- f(pm)
    mp <- p; mp[i] <- mp[i] - h; mp[j] <- mp[j] + h; fmp <- f(mp)
    mm <- p; mm[i] <- mm[i] - h; mm[j] <- mm[j] - h; fmm <- f(mm)
    H[i, j] <- (fpp - fpm - fmp + fmm) / (4 * h * h)
  }
  H
}
ldd_r <- num_ldd(c(v1, r)); lddi_r <- solve(ldd_r)
cat("ldd (shape,scale):\n"); print(ldd)
cat("ldd (shape,rate):\n"); print(ldd_r)
