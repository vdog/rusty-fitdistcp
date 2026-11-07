"""Python-level tests for the `fitdistcp` PyO3 bindings (W13).

These tests exercise the §5.2 `pdf`/`cdf`/`ppf`/`rvs` API surface for the ten
Phase-1 models. They check that each function returns the expected fields and
that the values are finite / self-consistent for a couple of representative
models (normal, exponential, GEV, gamma).

Run inside the dev container after `maturin develop`:

    docker compose run --rm dev maturin develop
    docker compose run --rm dev pytest
"""

import numpy as np
import pytest

import fitdistcp


# ---------------------------------------------------------------------------
# Phase 0 smoke tests (kept intact).
# ---------------------------------------------------------------------------


def test_core_version():
    assert isinstance(fitdistcp.core_version(), str)
    assert fitdistcp.core_version() == "0.1.0"


def test_add_vectors():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    out = fitdistcp.add_vectors(a, b)
    np.testing.assert_allclose(out, [5.0, 7.0, 9.0])


def test_sum_values():
    assert fitdistcp.sum_values(np.array([1.0, 2.0, 3.0, 4.0])) == 10.0


# ---------------------------------------------------------------------------
# Shared fixtures / helpers.
# ---------------------------------------------------------------------------

# A well-behaved sample for the normal / exponential / gamma models.
Y = np.array([0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8])
# A positive sample for the positive-support models (gamma, weibull, lnorm, gpd).
Y_POS = np.array([0.7, 1.4, 0.9, 2.1, 1.1, 0.5, 1.8, 1.3, 0.6, 1.6])
# A sample for the GEV / gumbel models.
Y_GEV = np.array([1.02, 1.55, 1.31, 1.88, 1.12, 1.67, 1.44, 1.21, 1.95, 1.38])

X = np.array([-1.0, 0.0, 1.0])
# A strictly-positive prediction grid for the positive-support models
# (gamma, weibull, lnorm, gpd), whose density is undefined at x <= 0.
X_POS = np.array([0.5, 1.0, 1.5])
P = np.array([0.25, 0.5, 0.75])

# The common return fields every function must expose.
COMMON_FIELDS = [
    "ml_params",
    "cp_params",
    "ml_se",
    "cp_se",
    "ml_value",
    "aic",
    "maic",
    "maic_vector",
    "waic",
    "log_scores",
    "convergence",
    "small_n",
    "cp_method",
]


def _check_common(res):
    """Assert the common §5.2 fields are present and well-typed."""
    for field in COMMON_FIELDS:
        assert field in res, f"missing field: {field}"
    assert isinstance(res["ml_params"], np.ndarray)
    assert isinstance(res["cp_params"], np.ndarray)
    assert isinstance(res["ml_se"], np.ndarray)
    assert isinstance(res["cp_se"], np.ndarray)
    assert isinstance(res["ml_value"], float)
    assert isinstance(res["aic"], float)
    assert isinstance(res["maic"], float)
    assert isinstance(res["maic_vector"], list)
    assert len(res["maic_vector"]) == 3
    assert res["waic"] is None
    assert res["log_scores"] is None
    assert isinstance(res["convergence"], str)
    assert isinstance(res["small_n"], bool)
    assert isinstance(res["cp_method"], str)


# ---------------------------------------------------------------------------
# Normal (`norm`).
# ---------------------------------------------------------------------------


def test_norm_ppf_fields_and_values():
    res = fitdistcp.ppf_norm(X, P)
    _check_common(res)
    assert "ml_quantiles" in res and "cp_quantiles" in res
    assert res["ml_quantiles"].shape == (3,)
    assert res["cp_quantiles"].shape == (3,)
    assert res["ml_params"].shape == (2,)
    # CP parameters equal the MLE parameters (documented discrepancy 3).
    np.testing.assert_allclose(res["cp_params"], res["ml_params"])
    np.testing.assert_allclose(res["cp_se"], res["ml_se"])
    # Quantiles must be finite and ordered for increasing p.
    assert np.all(np.isfinite(res["ml_quantiles"]))
    assert np.all(np.isfinite(res["cp_quantiles"]))
    assert res["ml_quantiles"][0] < res["ml_quantiles"][1] < res["ml_quantiles"][2]
    assert res["cp_quantiles"][0] < res["cp_quantiles"][1] < res["cp_quantiles"][2]
    # MAIC = loglik - n_params, and aic = 2k - 2*loglik => maic = -aic/2.
    assert abs(res["maic"] - (-res["aic"] / 2.0)) < 1e-9


def test_norm_pdf_fields_and_values():
    res = fitdistcp.pdf_norm(X, Y)
    _check_common(res)
    assert "ml_pdf" in res and "cp_pdf" in res
    # The predictive density is evaluated at the prediction grid `y` (mirroring
    # the R `dnorm_cp(x, y)` semantics), so its length matches `y`, not `x`.
    assert res["ml_pdf"].shape == Y.shape
    assert res["cp_pdf"].shape == Y.shape
    assert np.all(res["ml_pdf"] > 0.0)
    assert np.all(res["cp_pdf"] > 0.0)
    assert np.all(np.isfinite(res["cp_pdf"]))


def test_norm_cdf_fields_and_values():
    res = fitdistcp.cdf_norm(X, Y)
    _check_common(res)
    assert "ml_cdf" in res and "cp_cdf" in res
    # The CDFs are evaluated at the observed data `y` (mirroring the R
    # `p***_cp` semantics), so their length matches `y`, not the grid `x`.
    assert res["ml_cdf"].shape == Y.shape
    assert res["cp_cdf"].shape == Y.shape
    # CDFs are probabilities in [0, 1].
    assert np.all((res["ml_cdf"] >= 0.0) & (res["ml_cdf"] <= 1.0))
    assert np.all((res["cp_cdf"] >= 0.0) & (res["cp_cdf"] <= 1.0))


def test_norm_rvs_fields_and_values():
    res = fitdistcp.rvs_norm(5, Y)
    _check_common(res)
    assert "ml_deviates" in res and "cp_deviates" in res
    assert res["ml_deviates"].shape == (5,)
    assert res["cp_deviates"].shape == (5,)
    assert np.all(np.isfinite(res["cp_deviates"]))


def test_norm_means_optional_arg_accepted():
    # `means` is accepted and passed through (not yet consumed by the core).
    res = fitdistcp.ppf_norm(X, P, means=np.array([0.0, 1.0]))
    _check_common(res)
    assert res["ml_quantiles"].shape == (3,)


# ---------------------------------------------------------------------------
# Exponential (`exp`).
# ---------------------------------------------------------------------------


def test_exp_ppf_fields_and_values():
    res = fitdistcp.ppf_exp(X, P)
    _check_common(res)
    assert res["ml_params"].shape == (1,)
    assert res["ml_quantiles"].shape == (3,)
    assert res["cp_quantiles"].shape == (3,)
    assert np.all(np.isfinite(res["cp_quantiles"]))
    assert res["ml_quantiles"][0] < res["ml_quantiles"][1] < res["ml_quantiles"][2]


def test_exp_pdf_cdf_rvs():
    # The exponential has positive support, so the training data must be
    # strictly positive (the MLE rate = n / sum(x) is degenerate otherwise).
    # Use `Y_POS` as the data and a positive prediction grid.
    grid = np.array([0.5, 1.0, 1.5, 2.0])
    pdf = fitdistcp.pdf_exp(Y_POS, grid)
    _check_common(pdf)
    # The density is evaluated at the grid `y` (length matches `y`).
    assert pdf["ml_pdf"].shape == grid.shape
    assert pdf["cp_pdf"].shape == grid.shape
    assert np.all(pdf["ml_pdf"] > 0.0)
    assert np.all(pdf["cp_pdf"] > 0.0)

    cdf = fitdistcp.cdf_exp(Y_POS, grid)
    _check_common(cdf)
    assert np.all((cdf["ml_cdf"] >= 0.0) & (cdf["ml_cdf"] <= 1.0))

    rvs = fitdistcp.rvs_exp(4, Y_POS)
    _check_common(rvs)
    assert rvs["cp_deviates"].shape == (4,)


# ---------------------------------------------------------------------------
# GEV (`gev`) — model-specific extras.
# ---------------------------------------------------------------------------


def test_gev_ppf_fields_and_extras():
    res = fitdistcp.ppf_gev(X, P)
    _check_common(res)
    assert res["ml_params"].shape == (3,)
    assert "ml_max" in res
    assert "revert2ml" in res
    assert isinstance(res["revert2ml"], bool)
    assert np.all(np.isfinite(res["cp_quantiles"]))


def test_gev_pdf_cdf_rvs():
    pdf = fitdistcp.pdf_gev(X, Y_GEV)
    _check_common(pdf)
    assert "revert2ml" in pdf
    # The density is evaluated at the grid `y` (length matches `y`).
    assert pdf["ml_pdf"].shape == Y_GEV.shape
    assert pdf["cp_pdf"].shape == Y_GEV.shape
    assert np.all(np.isfinite(pdf["cp_pdf"]))

    cdf = fitdistcp.cdf_gev(X, Y_GEV)
    _check_common(cdf)
    assert np.all((cdf["ml_cdf"] >= 0.0) & (cdf["ml_cdf"] <= 1.0))

    rvs = fitdistcp.rvs_gev(3, Y_GEV)
    _check_common(rvs)
    assert rvs["cp_deviates"].shape == (3,)


# ---------------------------------------------------------------------------
# Gamma (`gamma`).
# ---------------------------------------------------------------------------


def test_gamma_ppf_fields_and_values():
    res = fitdistcp.ppf_gamma(X, P)
    _check_common(res)
    assert res["ml_params"].shape == (2,)
    assert np.all(np.isfinite(res["cp_quantiles"]))
    assert res["ml_quantiles"][0] < res["ml_quantiles"][1] < res["ml_quantiles"][2]


def test_gamma_pdf_cdf_rvs():
    # Use a strictly-positive grid: the gamma density is undefined at x <= 0.
    pdf = fitdistcp.pdf_gamma(X_POS, Y_POS)
    _check_common(pdf)
    # The density is evaluated at the grid `y` (length matches `y`).
    assert pdf["ml_pdf"].shape == Y_POS.shape
    assert pdf["cp_pdf"].shape == Y_POS.shape
    assert np.all(pdf["ml_pdf"] > 0.0)
    # The CP density is a density: finite and non-negative (the DMGS expansion
    # can legitimately return 0 at an individual grid point).
    assert np.all(np.isfinite(pdf["cp_pdf"]))
    assert np.all(pdf["cp_pdf"] >= 0.0)

    cdf = fitdistcp.cdf_gamma(X_POS, Y_POS)
    _check_common(cdf)
    # CDFs are evaluated at `y` (length 10) and are finite probabilities.
    assert cdf["cp_cdf"].shape == Y_POS.shape
    assert np.all(np.isfinite(cdf["cp_cdf"]))
    assert np.all((cdf["cp_cdf"] >= 0.0) & (cdf["cp_cdf"] <= 1.0))

    rvs = fitdistcp.rvs_gamma(6, Y_POS)
    _check_common(rvs)
    assert rvs["cp_deviates"].shape == (6,)


# ---------------------------------------------------------------------------
# Predictor models (`norm_p1`, `gev_p1`) — W14 predictor semantics.
# ---------------------------------------------------------------------------

# The reference example data (d060norm_p1_*): n = 25, t = seq(0.04, 1, 0.04).
X_NORM_P1 = np.array([
    -0.58645381, 0.26364332, -0.71562861, 1.7552808, 0.52950777, -0.58046838,
    0.76742905, 1.0583247, 0.93578135, 0.094611613, 1.9517812, 0.86984324,
    -0.10124058, -1.6546999, 1.7249309, 0.59506639, 0.66380974, 1.6638362,
    1.5812212, 1.3939013, 1.7589774, 1.6621363, 0.99456498, -1.0293517, 1.6198257,
])
T_NORM_P1 = np.arange(0.04, 1.0 + 1e-9, 0.04)


def test_norm_p1_accepts_predictor_args():
    t = np.linspace(0.0, 1.0, Y.size)
    res = fitdistcp.ppf_norm_p1(X, P, t=t, t0=0.0, n0=1.0)
    assert res["ml_params"].shape == (3,)
    assert np.all(np.isfinite(res["cp_quantiles"]))


def test_norm_p1_predictor_semantics():
    # n0 = 10 (1-based) => t0 = t[10] = 0.4; mean(t) = 0.52.
    p9 = np.arange(0.1, 1.0, 0.1)
    res = fitdistcp.ppf_norm_p1(X_NORM_P1, p9, t=T_NORM_P1, n0=10.0)
    assert res["ml_params"].shape == (3,)
    # Reference ml_params (decentered).
    np.testing.assert_allclose(
        res["ml_params"], [0.11348586625, 1.10611412112, 0.949594263163], rtol=1e-6
    )
    # Reference ml_quantiles.
    ref_ml_q = np.array([
        -0.661022499888, -0.243267180457, 0.0579637962327, 0.315354558973,
        0.5559315147, 0.796508470426, 1.05389923317, 1.35513020986, 1.77288552929,
    ])
    np.testing.assert_allclose(res["ml_quantiles"], ref_ml_q, rtol=1e-6)
    # predictordata fields.
    assert res["predictedparameter"].shape == (25,)
    assert res["adjustedx"].shape == (25,)
    np.testing.assert_allclose(res["predictedparameter"][0], 0.157730431095, rtol=1e-6)
    np.testing.assert_allclose(res["adjustedx"][0], -0.188252727137, rtol=1e-6)


def test_gev_p1_accepts_predictor_args():
    t = np.linspace(0.0, 1.0, Y_GEV.size)
    res = fitdistcp.pdf_gev_p1(X, Y_GEV, t=t, t0=0.0, n0=1.0)
    assert res["ml_params"].shape == (4,)
    # `ml_pdf` is evaluated at the prediction grid `y` (length matches `y`).
    assert res["ml_pdf"].shape == Y_GEV.shape
    # DOCUMENTED LIMITATION: the reference `dgev_p1_cp` emits no `cp_pdf` (it
    # returns `cp_method = "no pdf/cdf"`). The Rust port emits a non-reference
    # `cp_pdf` derived from the CP quantiles at the `p` grid, so its length is
    # `len(p)` (here 1, the default `p = [0.5]`), not `len(y)`.
    assert res["cp_pdf"].shape == (1,)


def test_gev_p1_predictor_semantics():
    # The GEV `_p1` CP uses the DMGS CRHP/flat path with revert2ml.
    t = np.linspace(0.025, 1.0, Y_GEV.size)
    res = fitdistcp.ppf_gev_p1(X, P, t=t, n0=10.0)
    assert res["ml_params"].shape == (4,)
    assert "revert2ml" in res
    assert "ml_max" in res
    assert np.all(np.isfinite(res["cp_quantiles"]))


# ---------------------------------------------------------------------------
# Remaining models: smoke-check the full surface is registered and returns
# the expected fields (values are covered by the core golden tests).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix,y",
    [
        ("gpd_k1", Y_POS),
        ("gumbel", Y_GEV),
        ("lnorm", Y_POS),
        ("weibull", Y_POS),
    ],
)
def test_remaining_models_surface(suffix, y):
    ppf = getattr(fitdistcp, f"ppf_{suffix}")
    pdf = getattr(fitdistcp, f"pdf_{suffix}")
    cdf = getattr(fitdistcp, f"cdf_{suffix}")
    rvs = getattr(fitdistcp, f"rvs_{suffix}")

    # Use `y` as the training data (it is a valid sample for each model's
    # support) and as the prediction grid. `X` contains non-positive values and
    # is invalid data for the positive-support models (gpd_k1, lnorm, weibull).
    r_ppf = ppf(y, P)
    _check_common(r_ppf)
    assert r_ppf["cp_quantiles"].shape == (3,)

    r_pdf = pdf(y, y)
    _check_common(r_pdf)
    # The density is evaluated at the grid `y`, so its length matches `y`.
    assert r_pdf["cp_pdf"].shape == y.shape

    r_cdf = cdf(y, y)
    _check_common(r_cdf)
    # CDFs are evaluated at `y`, so their length matches `y`.
    assert r_cdf["cp_cdf"].shape == y.shape

    r_rvs = rvs(3, y)
    _check_common(r_rvs)
    assert r_rvs["cp_deviates"].shape == (3,)
