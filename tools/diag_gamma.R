#!/usr/bin/env Rscript
# Diagnostic: is the gamma discrepancy a reference-optimizer convergence issue?
local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) .libPaths(c(normalizePath(local_lib), .libPaths()))
suppressPackageStartupMessages(library(extraDistr))
ref_dir <- "tools/fitdistcp-ref/R"
source(file.path(ref_dir, "100b_gamma_libs.R"))

y_positive <- c(
  0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
  0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
  0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93
)
x <- y_positive

# Reference default optim (Nelder-Mead).
opt_def <- optim(c(1, 1), gamma_loglik, x = x, control = list(fnscale = -1))
cat("default optim par:", opt_def$par, " val:", opt_def$val, "\n")

# Tighter optim.
opt_tight <- optim(c(1, 1), gamma_loglik, x = x,
                   control = list(fnscale = -1, reltol = 1e-14, maxit = 10000))
cat("tight   optim par:", opt_tight$par, " val:", opt_tight$val, "\n")

# Rust result (shape, rate) -> convert rate to scale = 1/rate.
rust_shape <- 2.00846887587350542
rust_rate  <- NA  # fill from Rust output below
# Evaluate reference loglik at the Rust params (converted to scale).
# We'll pass scale = 1/rate; get rate from the Rust test output.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) rust_rate <- as.numeric(args[1])
if (!is.na(rust_rate)) {
  rust_scale <- 1 / rust_rate
  cat("Rust params (shape, rate):", rust_shape, rust_rate, "\n")
  cat("Rust params as (shape, scale):", rust_shape, rust_scale, "\n")
  cat("ref loglik at Rust params:", gamma_loglik(c(rust_shape, rust_scale), x), "\n")
}
cat("ref loglik at default params:", gamma_loglik(opt_def$par, x), "\n")
cat("ref loglik at tight params:", gamma_loglik(opt_tight$par, x), "\n")
