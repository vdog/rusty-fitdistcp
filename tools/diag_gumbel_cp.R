#!/usr/bin/env Rscript
# Diagnostic: print the reference gumbel DMGS internals for the CP quantiles.
local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) .libPaths(c(normalizePath(local_lib), .libPaths()))
suppressPackageStartupMessages(library(extraDistr))
ref_dir <- "tools/fitdistcp-ref/R"
for (f in c("000_evaluate_dmgs_equation3p3.R", "005_utils.R",
            "050b_gumbel_libs.R", "050c_gumbel_derivs.R")) {
  source(file.path(ref_dir, f))
}
y_general <- c(
  9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41,
  10.93, 8.21, 12.05, 10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.90,
  7.63, 10.81, 9.29, 12.44, 8.98, 10.17, 11.53, 9.60, 10.72, 8.40
)
x <- y_general
nx <- length(x)
opt <- optim(c(mean(x), sd(x)), gumbel_loglik, x = x,
             control = list(fnscale = -1, reltol = 1e-14, maxit = 10000))
v1 <- opt$par[1]; v2 <- opt$par[2]
cat("v1 =", v1, " v2 =", v2, "\n")
ldd <- gumbel_ldda(x, v1, v2); lddi <- solve(ldd)
lddd <- gumbel_lddda(x, v1, v2)
cat("ldd:\n"); print(ldd)
cat("lddi:\n"); print(lddi)
cat("lddd dim:", dim(lddd), "\n")
p_grid <- c(0.1, 0.25, 0.5, 0.75, 0.9)
alpha <- 1 - p_grid
ml_q <- qgumbel(1 - alpha, mu = v1, sigma = v2)
cat("ml_q:", ml_q, "\n")
mu1 <- gumbel_mu1fa(alpha, v1, v2)
mu2 <- gumbel_mu2fa(alpha, v1, v2)
cat("mu1 dim:", dim(mu1), "\n"); print(mu1)
cat("mu2 dim:", dim(mu2), "\n"); print(mu2)
lambdad_rhp <- c(0, -1 / v2)
dq <- dmgs(lddi, lddd, mu1, lambdad_rhp, mu2, dim = 2)
cat("dq:", dq, "\n")
fhat <- extraDistr::dgumbel(ml_q, mu = v1, sigma = v2)
cat("fhat:", fhat, "\n")
cp_q <- ml_q + dq / (nx * fhat)
cat("cp_q:", cp_q, "\n")
