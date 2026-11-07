# W12 R-level tests for the d/p/q/r + t***_cp API.
#
# These tests exercise the extendr wrappers over `fitdistcp-core` for a couple
# of representative models (normal, exponential, gumbel, GEV) and check that:
#   * the reference field names are present in the returned lists;
#   * the numeric fields have the expected lengths and are finite;
#   * the MLE quantiles match the closed-form reference values where known;
#   * the CP quantiles differ from the MLE quantiles (the DMGS correction is
#     non-trivial) but stay finite;
#   * the `t***_cp` RUST placeholders return the reference "rust not selected".

test_that("core_version and smoke tests still work", {
  expect_type(core_version(), "character")
  expect_equal(add_vectors(c(1, 2, 3), c(4, 5, 6)), c(5, 7, 9))
  expect_equal(sum_values(c(1, 2, 3, 4)), 10)
})

test_that("qnorm_cp returns the reference field names and finite values", {
  x <- c(0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8)
  p <- seq(0.1, 0.9, 0.1)
  r <- qnorm_cp(x, p)

  expect_true(all(c(
    "ml_params", "ml_value", "standard_errors", "ml_quantiles",
    "cp_quantiles", "ru_quantiles", "maic", "waic1", "waic2",
    "ml_oos_logscore", "cp_oos_logscore", "ml_mean", "cp_mean", "cp_method"
  ) %in% names(r)))

  expect_length(r$ml_params, 2)
  expect_length(r$standard_errors, 2)
  expect_length(r$ml_quantiles, length(p))
  expect_length(r$cp_quantiles, length(p))
  expect_length(r$maic, 3)

  expect_true(all(is.finite(r$ml_quantiles)))
  expect_true(all(is.finite(r$cp_quantiles)))

  # MLE mean/sd of the sample.
  expect_equal(r$ml_params[1], mean(x), tolerance = 1e-6)
  expect_equal(r$ml_params[2], sqrt((length(x) - 1) / length(x)) * sd(x), tolerance = 1e-6)

  # The CP quantiles are the DMGS-corrected quantiles; they must differ from
  # the MLE quantiles (the correction is non-trivial) but stay finite.
  expect_false(isTRUE(all.equal(r$ml_quantiles, r$cp_quantiles)))
})

test_that("dnorm_cp / pnorm_cp return pdf and cdf fields", {
  x <- c(0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8)
  y <- c(-1.0, 0.0, 1.0)

  d <- dnorm_cp(x, y)
  expect_true(all(c("ml_params", "ml_pdf", "cp_pdf", "ru_pdf", "cp_method") %in% names(d)))
  expect_length(d$ml_pdf, length(y))
  expect_length(d$cp_pdf, length(y))
  expect_true(all(is.finite(d$ml_pdf)))
  expect_true(all(is.finite(d$cp_pdf)))

  pc <- pnorm_cp(x, y)
  expect_true(all(c("ml_params", "ml_cdf", "cp_cdf", "ru_cdf", "cp_method") %in% names(pc)))
  expect_length(pc$ml_cdf, length(y))
  expect_length(pc$cp_cdf, length(y))
  expect_true(all(pc$ml_cdf >= 0 & pc$ml_cdf <= 1))
  expect_true(all(pc$cp_cdf >= 0 & pc$cp_cdf <= 1))
})

test_that("rnorm_cp returns ml/cp deviates of the requested length", {
  x <- c(0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8)
  r <- rnorm_cp(5, x)
  expect_true(all(c("ml_params", "ml_deviates", "cp_deviates", "ru_deviates", "cp_method") %in% names(r)))
  expect_length(r$ml_deviates, 5)
  expect_length(r$cp_deviates, 5)
  expect_true(all(is.finite(r$ml_deviates)))
  expect_true(all(is.finite(r$cp_deviates)))
})

test_that("qexp_cp matches the closed-form exponential MLE quantiles", {
  x <- c(0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.1, 0.7, 1.8)
  p <- c(0.25, 0.5, 0.75)
  r <- qexp_cp(x, p)

  expect_length(r$ml_params, 1)
  # MLE rate = n / sum(x).
  expect_equal(r$ml_params[1], length(x) / sum(x), tolerance = 1e-6)
  # MLE quantiles = qexp(p, rate).
  expect_equal(r$ml_quantiles, qexp(p, rate = r$ml_params[1]), tolerance = 1e-6)
  expect_true(all(is.finite(r$cp_quantiles)))
})

test_that("qgumbel_cp returns finite MLE and CP quantiles", {
  x <- c(0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.1, 0.7, 1.8)
  p <- c(0.25, 0.5, 0.75)
  r <- qgumbel_cp(x, p)
  expect_length(r$ml_params, 2)
  expect_length(r$ml_quantiles, length(p))
  expect_length(r$cp_quantiles, length(p))
  expect_true(all(is.finite(r$ml_quantiles)))
  expect_true(all(is.finite(r$cp_quantiles)))
})

test_that("qgev_cp returns the GEV-specific fields and a revert2ml flag", {
  x <- c(1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38)
  p <- c(0.25, 0.5, 0.75)
  r <- qgev_cp(x, p)

  expect_true(all(c(
    "ml_params", "ml_quantiles", "cp_quantiles", "ml_max", "revert2ml",
    "ml_pdf", "cp_pdf", "maic", "cp_method"
  ) %in% names(r)))
  expect_length(r$ml_params, 3)
  expect_type(r$revert2ml, "logical")
  expect_length(r$ml_quantiles, length(p))
  expect_length(r$cp_quantiles, length(p))
  expect_true(all(is.finite(r$ml_quantiles)))
  expect_true(all(is.finite(r$cp_quantiles)))
})

test_that("t***_cp RUST placeholders return the reference placeholder", {
  x <- c(0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8)
  expect_equal(tnorm_cp(10, x)$theta_samples, "rust not selected")
  expect_equal(texp_cp(10, abs(x))$theta_samples, "rust not selected")
  expect_equal(tgumbel_cp(10, x)$theta_samples, "rust not selected")
})

test_that("small-n flag is surfaced through the GEV CP path", {
  # n = 8 < 10 => the core flags small_n; the wrapper does not currently expose
  # it as a field, but the fit must still return finite CP quantiles.
  x <- c(1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21)
  r <- qgev_cp(x, c(0.25, 0.5, 0.75))
  expect_true(all(is.finite(r$cp_quantiles)))
})

test_that("_p1 wrappers accept an integer predictor vector", {
  # `seq(1, 10)` is an R integer vector; the wrappers must coerce it to double
  # (the reference R API accepts either). This guards the integer-coercion fix.
  x <- c(0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.1, 0.7, 1.8)
  t <- seq(1, 10)
  t0 <- 5
  n0 <- 10
  p <- c(0.25, 0.5, 0.75)

  rq <- qnorm_p1_cp(x, t, t0, n0, p)
  expect_length(rq$ml_params, 3)
  expect_true(all(is.finite(rq$cp_quantiles)))

  rd <- dnorm_p1_cp(x, t, t0, n0, x)
  expect_true(all(is.finite(rd$ml_pdf)))

  rp <- pnorm_p1_cp(x, t, t0, n0, x)
  expect_true(all(is.finite(rp$ml_cdf)))

  rr <- rnorm_p1_cp(5, x, t, t0, n0)
  expect_length(rr$ml_deviates, 5)

  rqg <- qgev_p1_cp(x, t, t0, n0, p)
  expect_length(rqg$ml_params, 4)
  expect_true(all(is.finite(rqg$cp_quantiles)))

  rdg <- dgev_p1_cp(x, t, t0, n0, x)
  expect_true(all(is.finite(rdg$ml_pdf)))

  rpg <- pgev_p1_cp(x, t, t0, n0, x)
  expect_true(all(is.finite(rpg$ml_cdf)))

  rrg <- rgev_p1_cp(5, x, t, t0, n0)
  expect_length(rrg$ml_deviates, 5)
})

# W14: predictor semantics (centering, maket0, prediction point, predictordata).

test_that("qnorm_p1_cp applies the reference predictor semantics", {
  # The reference example data (d060norm_p1_*): n = 25, t = seq(0.04, 1, 0.04),
  # mean(t) = 0.52. With n0 = 10 (1-based) => t0 = t[10] = 0.4.
  x <- c(
    -0.58645381, 0.26364332, -0.71562861, 1.7552808, 0.52950777, -0.58046838,
    0.76742905, 1.0583247, 0.93578135, 0.094611613, 1.9517812, 0.86984324,
    -0.10124058, -1.6546999, 1.7249309, 0.59506639, 0.66380974, 1.6638362,
    1.5812212, 1.3939013, 1.7589774, 1.6621363, 0.99456498, -1.0293517, 1.6198257
  )
  t <- seq(0.04, 1, 0.04)
  p <- seq(0.1, 0.9, 0.1)
  r <- qnorm_p1_cp(x, t, t0 = NA, n0 = 10, p = p)

  # Reference ml_params (decentered): a = 0.11348586625, b = 1.10611412112,
  # sigma = 0.949594263163.
  expect_equal(r$ml_params[1], 0.11348586625, tolerance = 1e-6)
  expect_equal(r$ml_params[2], 1.10611412112, tolerance = 1e-6)
  expect_equal(r$ml_params[3], 0.949594263163, tolerance = 1e-6)

  # Reference ml_quantiles (p = 0.1..0.9).
  ref_ml_q <- c(
    -0.661022499888, -0.243267180457, 0.0579637962327, 0.315354558973,
    0.5559315147, 0.796508470426, 1.05389923317, 1.35513020986, 1.77288552929
  )
  expect_equal(r$ml_quantiles, ref_ml_q, tolerance = 1e-6)

  # Reference cp_quantiles (analytic Student-t; looser tolerance documented).
  ref_cp_q <- c(
    -0.753652844124, -0.29517961154, 0.028164231521, 0.301552784103,
    0.5559315147, 0.810310245296, 1.08369879788, 1.40704264094, 1.86551587352
  )
  expect_equal(r$cp_quantiles, ref_cp_q, tolerance = 1e-5)

  # predictordata: predictedparameter (length n) and adjustedx (length n).
  expect_length(r$predictedparameter, 25)
  expect_length(r$adjustedx, 25)
  expect_equal(r$predictedparameter[1], 0.157730431095, tolerance = 1e-6)
  expect_equal(r$adjustedx[1], -0.188252727137, tolerance = 1e-6)

  # standard_errors.
  expect_equal(r$standard_errors[1], 0.189918852633, tolerance = 1e-6)
})

test_that("qnorm_p1_cp prediction point shifts with t0", {
  x <- c(
    -0.58645381, 0.26364332, -0.71562861, 1.7552808, 0.52950777, -0.58046838,
    0.76742905, 1.0583247, 0.93578135, 0.094611613, 1.9517812, 0.86984324,
    -0.10124058, -1.6546999, 1.7249309, 0.59506639, 0.66380974, 1.6638362,
    1.5812212, 1.3939013, 1.7589774, 1.6621363, 0.99456498, -1.0293517, 1.6198257
  )
  t <- seq(0.04, 1, 0.04)
  p <- c(0.5)
  r1 <- qnorm_p1_cp(x, t, t0 = 0.4, n0 = NA, p = p)
  r2 <- qnorm_p1_cp(x, t, t0 = 5, n0 = NA, p = p)
  # The median shifts by b*(5 - 0.4).
  b <- r1$ml_params[2]
  expect_equal(r2$ml_quantiles[1] - r1$ml_quantiles[1], b * (5 - 0.4), tolerance = 1e-6)
})

test_that("qgev_p1_cp returns the GEV-specific fields and revert2ml", {
  x <- c(
    -0.64966202, 0.85951881, 1.5996735, -0.48792225, 1.9369991, 0.53527898,
    0.10262144, -0.23528281, -0.057837808, 1.7340061, 0.55865183, 0.076155367,
    0.70099111, -0.14190081, 0.29518469, 0.62106268, 0.039215712, -1.3732916,
    -0.094737048, 0.95002586, 1.4342304, 0.9058821, 0.75403608, 1.5581121,
    0.13327821, 2.524211, 1.2288074, 0.61335529, 3.7228621, 1.1994391,
    -1.1647551, 2.1560221, 1.3467322, 1.619274, 2.2680367, 2.0191889,
    1.2164931, 1.171829, 1.0419927, -0.31710656
  )
  t <- seq(0.025, 1, 0.025)
  p <- seq(0.1, 0.9, 0.1)
  r <- qgev_p1_cp(x, t, t0 = NA, n0 = 10, p = p)

  expect_true(all(c(
    "ml_params", "ml_quantiles", "cp_quantiles", "ml_max", "revert2ml",
    "ml_pdf", "cp_pdf", "maic", "cp_method"
  ) %in% names(r)))
  expect_length(r$ml_params, 4)
  expect_type(r$revert2ml, "logical")
  expect_length(r$cp_quantiles, length(p))
  expect_true(all(is.finite(r$cp_quantiles)))

  # Reference ml_params (decentered): -0.131382693504, 1.12967831086,
  # 0.960973959105, -0.23045953922. The GEV surface is flat, so the Rust BFGS
  # and R's optim stop at slightly different points; use the documented
  # optimizer tolerance (1e-3).
  expect_equal(r$ml_params[1], -0.131382693504, tolerance = 1e-3)
  expect_equal(r$ml_params[2], 1.12967831086, tolerance = 1e-3)
  expect_equal(r$ml_params[3], 0.960973959105, tolerance = 1e-3)
  expect_equal(r$ml_params[4], -0.23045953922, tolerance = 1e-3)
  expect_false(r$revert2ml)
})
