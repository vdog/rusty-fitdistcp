# Test harness for the fitdistcp R package (W12).
#
# Runs the testthat suite under `R CMD check` / `devtools::test()`.
library(testthat)
library(fitdistcp)

test_check("fitdistcp")
