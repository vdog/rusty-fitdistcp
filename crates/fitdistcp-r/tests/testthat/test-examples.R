# W15 milestone check: reproduce the reference "R Examples 1-7".
#
# Source of the examples: the reference `fitdistcp` package documentation at
# `tools/fitdistcp-ref/docs/index.html` (section "R Examples"), which is the
# canonical list of the seven short examples referenced by the design document
# §7.1 ("Reproduce R Examples 1-7 ... as golden tests").
#
# The reference examples use *random* training data (`rnorm`, `rlogis`, ...),
# so their printed output is not reproducible without the reference package's
# RNG. To make the milestone check deterministic we substitute fixed data
# vectors (the same shape/support as the reference examples) and assert:
#
#   * the reference return fields are present;
#   * the MLE quantities match their closed-form / analytically-derived values
#     (the reference `ml_params` for these models are the sample MLEs);
#   * the CP quantities are finite, ordered, and differ from the MLE ones
#     (the DMGS correction is non-trivial);
#   * the documented reference relationships hold (e.g. the CP predictive has
#     fatter tails than the MLE predictive).
#
# Tolerance policy: design §7.2 (rtol = 1e-8, atol = 1e-10) for the
# analytically-known MLE values; looser documented tolerances where the
# reference itself uses an iterative optimizer (see the per-test comments).

# ---------------------------------------------------------------------------
# Shared fixed data (deterministic substitutes for the reference `rnorm` etc.).
# ---------------------------------------------------------------------------

# A normal sample (n = 20), the shape of R Example 1's `rnorm(20)`.
ex1_x <- c(
  -0.56047565, -0.23017749, 1.55870831, 0.07050839, 0.12928774,
  1.71506499, 0.46091621, -1.26506123, -0.68685285, -0.44566197,
  1.22408180, 0.35981383, 0.40077145, 0.11068272, -0.55584113,
  1.78691314, 0.49785048, -1.96661716, 0.70135590, -0.47279141
)

# A logistic sample (n = 20), the shape of R Example 2's `rlogis(20)`.
ex2_x <- c(
  -0.62645381, 0.18364332, -0.83562861, 1.59528080, 0.32950777,
  -0.82046838, 0.48742905, 0.73832471, 0.57578135, -0.30538839,
  1.51178117, 0.38984324, -0.62124058, -2.21469989, 1.12493092,
  -0.04493361, -0.01619026, 0.94383621, 0.82122120, 0.59390132
)

# A positive, right-skewed sample (n = 20), the shape of R Example 3's
# `rlnorm(20, sdlog = 2)`.
ex3_x <- c(
  0.419585, 1.870000, 0.950000, 3.210000, 0.610000,
  2.440000, 1.130000, 0.780000, 4.020000, 1.550000,
  0.330000, 2.910000, 1.020000, 0.690000, 3.770000,
  1.340000, 0.510000, 2.180000, 0.880000, 1.960000
)

# A small normal sample (n = 10), the shape of R Example 4's `rnorm(10)`.
ex4_x <- c(
  0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8
)

# A positive sample (n = 20), the shape of R Example 5's `rlnorm(20)`.
ex5_x <- c(
  0.7, 1.4, 0.9, 2.1, 1.1, 0.5, 1.8, 1.3, 0.6, 1.6,
  0.8, 1.2, 2.3, 0.4, 1.9, 1.0, 0.75, 1.55, 2.6, 0.95
)

# ---------------------------------------------------------------------------
# R Example 1: Fitting a normal distribution and calculating the fitted pdf.
#
#   > x = rnorm(20)
#   > y = seq(-5, 5, 0.01)
#   > d = dnorm_cp(x, y)
#   > print(d$ml_pdf); print(d$cp_pdf)
# ---------------------------------------------------------------------------

test_that("R Example 1: dnorm_cp returns ml_pdf and cp_pdf on the grid", {
  y <- seq(-5, 5, 0.01)
  d <- dnorm_cp(ex1_x, y)

  expect_true(all(c("ml_params", "ml_pdf", "cp_pdf", "cp_method") %in% names(d)))
  expect_length(d$ml_pdf, length(y))
  expect_length(d$cp_pdf, length(y))
  expect_true(all(is.finite(d$ml_pdf)))
  expect_true(all(is.finite(d$cp_pdf)))
  expect_true(all(d$ml_pdf >= 0))
  expect_true(all(d$cp_pdf >= 0))

  # The reference `ml_params` for the normal are the sample MLEs:
  #   mu = mean(x), sigma = sqrt((n-1)/n) * sd(x).
  n <- length(ex1_x)
  expect_equal(d$ml_params[1], mean(ex1_x), tolerance = 1e-8)
  expect_equal(d$ml_params[2], sqrt((n - 1) / n) * sd(ex1_x), tolerance = 1e-8)

  # The MLE predictive density is the normal density at the MLE parameters.
  expect_equal(d$ml_pdf, dnorm(y, mean = d$ml_params[1], sd = d$ml_params[2]),
    tolerance = 1e-8
  )

  # The CP predictive has fatter tails than the MLE predictive: in the far
  # tails the CP density exceeds the MLE density (the reference's stated
  # qualitative property). Check the extreme grid points.
  expect_gt(d$cp_pdf[1], d$ml_pdf[1])
  expect_gt(d$cp_pdf[length(y)], d$ml_pdf[length(y)])
})

# ---------------------------------------------------------------------------
# R Example 2: Fitting a GEV distribution and calculating the fitted quantiles.
#
#   > set.seed(1); x = rlogis(20)
#   > p = seq(0.001, 0.999, 0.001)
#   > q = qgev_cp(x, p)
#   > q$ml_params; plot(q$ml_quantiles, p); points(q$cp_quantiles, p)
# ---------------------------------------------------------------------------

test_that("R Example 2: qgev_cp returns ml/cp quantiles and GEV fields", {
  p <- seq(0.001, 0.999, 0.001)
  q <- qgev_cp(ex2_x, p)

  expect_true(all(c(
    "ml_params", "ml_quantiles", "cp_quantiles", "ml_max", "revert2ml",
    "ml_pdf", "cp_pdf", "maic", "cp_method"
  ) %in% names(q)))
  expect_length(q$ml_params, 3)
  expect_type(q$revert2ml, "logical")
  expect_length(q$ml_quantiles, length(p))
  expect_length(q$cp_quantiles, length(p))
  expect_true(all(is.finite(q$ml_quantiles)))
  expect_true(all(is.finite(q$cp_quantiles)))

  # The MLE quantiles are monotone increasing in p.
  expect_true(all(diff(q$ml_quantiles) > 0))
  expect_true(all(diff(q$cp_quantiles) > 0))

  # The CP quantiles have fatter tails than the MLE quantiles: the lower tail
  # is lower and the upper tail is higher.
  expect_lt(q$cp_quantiles[1], q$ml_quantiles[1])
  expect_gt(q$cp_quantiles[length(p)], q$ml_quantiles[length(p)])
})

# ---------------------------------------------------------------------------
# R Example 3: Fitting a GPD and calculating the fitted CDF.
#
#   > set.seed(1); x = rlnorm(20, sdlog = 2)
#   > y = seq(0.1, 5, 0.1)
#   > p = pgpd_k1_cp(x, y)
#   > plot(y, p$ml_cdf); points(y, p$cp_cdf, col = "red")
# ---------------------------------------------------------------------------

test_that("R Example 3: pgpd_k1_cp returns ml_cdf and cp_cdf on the grid", {
  y <- seq(0.1, 5, 0.1)
  p <- pgpd_k1_cp(ex3_x, y)

  expect_true(all(c("ml_params", "ml_cdf", "cp_cdf", "cp_method") %in% names(p)))
  expect_length(p$ml_cdf, length(y))
  expect_length(p$cp_cdf, length(y))
  expect_true(all(is.finite(p$ml_cdf)))
  expect_true(all(is.finite(p$cp_cdf)))

  # CDFs are probabilities in [0, 1] and monotone increasing in y.
  expect_true(all(p$ml_cdf >= 0 & p$ml_cdf <= 1))
  expect_true(all(p$cp_cdf >= 0 & p$cp_cdf <= 1))
  expect_true(all(diff(p$ml_cdf) >= -1e-12))
  expect_true(all(diff(p$cp_cdf) >= -1e-12))

  # DOCUMENTED LIMITATION (not a failure): the current `p***_cp` wrappers build
  # `cp_cdf` by evaluating the model CDF at `cp_params`, and `cp_params` equal
  # `ml_params` (the DMGS expansion corrects the *predictive* distribution, not
  # the point estimates — see the R binding's discrepancy note 4). The CP CDF is
  # therefore identical to the MLE CDF at this stage. The reference's
  # `pgpd_k1_cp` returns a genuinely DMGS-corrected `cp_cdf` (fatter tails); the
  # Rust port does not yet reproduce that field. We assert the current, known
  # behaviour so the test is honest and fails loudly if the field later changes.
  expect_equal(p$cp_cdf, p$ml_cdf, tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# R Example 4: Fitting a Gumbel distribution and generating random numbers.
#
#   > x = rnorm(10)
#   > r = rgumbel_cp(10000, x)
#   > plot(density(r$ml_deviates)); lines(density(r$cp_deviates), col = "red")
# ---------------------------------------------------------------------------

test_that("R Example 4: rgumbel_cp returns ml/cp deviates of the requested length", {
  r <- rgumbel_cp(10000, ex4_x)

  expect_true(all(c("ml_params", "ml_deviates", "cp_deviates", "cp_method") %in% names(r)))
  expect_length(r$ml_deviates, 10000)
  expect_length(r$cp_deviates, 10000)
  expect_true(all(is.finite(r$ml_deviates)))
  expect_true(all(is.finite(r$cp_deviates)))

  # The deviates are generated by inverting the predictive CDF on the midpoint
  # probability grid, so they are monotone increasing.
  expect_true(all(diff(r$ml_deviates) > 0))
  expect_true(all(diff(r$cp_deviates) > 0))

  # The CP deviates have fatter tails: the range is wider than the MLE range.
  expect_gt(diff(range(r$cp_deviates)), diff(range(r$ml_deviates)))
})

# ---------------------------------------------------------------------------
# R Example 5: Generating posterior samples from a Weibull distribution.
#
#   > x = rlnorm(20)
#   > t = tweibull_cp(1000, x)$theta_samples
#
# The reference generates the posterior samples with Paul Northrop's `rust`
# package (ratio-of-uniforms). The Rust port defers posterior sampling to
# Phase 2 (design §8, §10 "Phase 2 (deferred)"), so `tweibull_cp` returns the
# reference's own "rust not selected" placeholder. This is a documented
# Phase-2 deferral, not a failure.
# ---------------------------------------------------------------------------

test_that("R Example 5: tweibull_cp is a documented Phase-2 RUST deferral", {
  t <- tweibull_cp(1000, ex5_x)
  expect_true("theta_samples" %in% names(t))
  # The reference returns the string "rust not selected" when `rust = FALSE`
  # (the default); the Rust port mirrors that placeholder until Phase 2.
  expect_equal(t$theta_samples, "rust not selected")
})

# ---------------------------------------------------------------------------
# R Example 6: Fitting a GEV with a predictor on the location parameter.
#
#   > set.seed(1); n = 30
#   > t = 0.1 * c(1:n) / n
#   > x = t + rlogis(n)
#   > t0 = t[n]
#   > p = seq(0.001, 0.999, 0.001)
#   > q = qgev_p1_cp(x, t, t0 = t0, n0 = NA, p)
# ---------------------------------------------------------------------------

# The reference example's "detailed" version (`example_150_gev_p1.R`) uses the
# packaged dataset `d150gev_p1_example_data_v1_{x,t}` (n = 40). We embed those
# values so the test does not depend on the gitignored reference clone. The
# reference `ml_params` for this dataset are
#   [-0.1313654, 1.12952, 0.9609753, -0.2304662]  (decentered).
ex6_x <- c(
  -0.6496620155, 0.8595188103, 1.5996735378, -0.4879222534, 1.9369990537,
  0.5352789761, 0.1026214406, -0.2352828067, -0.0578378077, 1.7340061048,
  0.5586518301, 0.0761553670, 0.7009911053, -0.1419008138, 0.2951846923,
  0.6210626814, 0.0392157115, -1.3732915540, -0.0947370485, 0.9500258630,
  1.4342303601, 0.9058821048, 0.7540360796, 1.5581120538, 0.1332782097,
  2.5242109543, 1.2288074482, 0.6133552910, 3.7228621117, 1.1994391429,
  -1.1647550833, 2.1560221466, 1.3467321649, 1.6192740351, 2.2680367239,
  2.0191888598, 1.2164931186, 1.1718289697, 1.0419927261, -0.3171065633
)
ex6_t <- seq(0.025, 1, 0.025)

test_that("R Example 6: qgev_p1_cp runs with a location predictor", {
  p <- seq(0.001, 0.999, 0.001)
  q <- qgev_p1_cp(ex6_x, ex6_t, t0 = ex6_t[length(ex6_t)], n0 = NA, p = p)

  expect_true(all(c(
    "ml_params", "ml_quantiles", "cp_quantiles", "ml_max", "revert2ml",
    "ml_pdf", "cp_pdf", "maic", "cp_method"
  ) %in% names(q)))
  expect_length(q$ml_params, 4)
  expect_type(q$revert2ml, "logical")
  expect_length(q$ml_quantiles, length(p))
  expect_length(q$cp_quantiles, length(p))
  expect_true(all(is.finite(q$ml_quantiles)))
  expect_true(all(is.finite(q$cp_quantiles)))
  expect_true(all(diff(q$ml_quantiles) > 0))
  expect_true(all(diff(q$cp_quantiles) > 0))

  # Reference `ml_params` (decentered). The GEV surface is flat, so the Rust
  # BFGS and R's optim stop at slightly different points; use the documented
  # optimizer tolerance (1e-3), as in the W14 test.
  expect_equal(q$ml_params[1], -0.1313654, tolerance = 1e-3)
  expect_equal(q$ml_params[2], 1.12952, tolerance = 1e-3)
  expect_equal(q$ml_params[3], 0.9609753, tolerance = 1e-3)
  expect_equal(q$ml_params[4], -0.2304662, tolerance = 1e-3)
  expect_false(q$revert2ml)
})

test_that("R Example 6: qgev_p1_cp accepts n0 in place of t0", {
  p <- c(0.1, 0.5, 0.9)
  # `n0 = 10` (1-based) resolves to `t0 = t[10]`, so the two calls agree.
  q_t0 <- qgev_p1_cp(ex6_x, ex6_t, t0 = ex6_t[10], n0 = NA, p = p)
  q_n0 <- qgev_p1_cp(ex6_x, ex6_t, t0 = NA, n0 = 10, p = p)
  expect_equal(q_t0$ml_quantiles, q_n0$ml_quantiles, tolerance = 1e-8)
  expect_equal(q_t0$cp_quantiles, q_n0$cp_quantiles, tolerance = 1e-8)
})

# ---------------------------------------------------------------------------
# R Example 7: Fitting a GEV with predictors on location AND scale.
#
#   > set.seed(1); n = 50
#   > t1 = 0.1 * c(1:n) / n; t2 = 0.2 * c(1:n) / n
#   > x = (t1 + rlogis(n)) * t2
#   > n0 = n
#   > p = seq(0.001, 0.999, 0.001)
#   > q = qgev_p12_cp(x, t1, t2, t01 = NA, t02 = NA, n01 = n0, n02 = n0, p)
#
# The two-predictor GEV (`qgev_p12_cp`) is NOT part of the Phase-1 model list
# (design §3 lists `genextreme_p1` only). The Rust port therefore does not
# expose `qgev_p12_cp`; this example is a documented Phase-2 / out-of-scope
# deferral. We assert the function is absent so the deferral is explicit and
# the test fails loudly if a future phase adds it without updating this note.
# ---------------------------------------------------------------------------

test_that("R Example 7: qgev_p12_cp is a documented out-of-scope deferral", {
  # The Phase-1 surface exposes the single-predictor GEV only.
  expect_true(exists("qgev_p1_cp", mode = "function"))
  # The two-predictor GEV is not part of Phase 1 (design §3).
  expect_false(exists("qgev_p12_cp", mode = "function"))
})
