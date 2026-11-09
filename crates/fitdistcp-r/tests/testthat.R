# Test harness for the rusty.fitdistcp R package (W12).
#
# Runs the testthat suite under `R CMD check` / `devtools::test()`.
library(testthat)
library(rusty.fitdistcp)

test_check("rusty.fitdistcp")
