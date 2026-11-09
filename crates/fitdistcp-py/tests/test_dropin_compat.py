"""Phase C2 — drop-in compatibility test.

Verifies that the Rust-backed ``rusty_fitdistcp`` package is a drop-in replacement for
the reference ``fitdistcp`` v0.0.4 submodule API:

* ``import rusty_fitdistcp; rusty_fitdistcp.norm.ppf(x)`` works with the reference default
  ``p = np.arange(0.1, 1.0, 0.1)``.
* The reference return-dict keys are present for ``ppf``/``pdf``/``cdf``/``rvs``.
* ``maic`` is a length-3 array; ``standard_errors`` is object-dtype;
  ``ru_quantiles == "ru not selected"``.
* ``pdf``/``cdf`` ``y=None`` defaults to ``y=x``.
* ``rvs`` returns ``ml_deviates``/``cp_deviates``/``ru_deviates``.
* The flat API still works and emits ``DeprecationWarning``.
* ``rusty_fitdistcp.__all__`` includes ``genextreme_p12`` (Phase 6, Issue 6).

Run inside the dev container after ``maturin develop``::

    docker compose run --rm dev bash -lc "cd crates/fitdistcp-py && \
        . .venv/bin/activate && maturin develop && python -m pytest tests/ -q"
"""

import warnings

import numpy as np
import pytest

import rusty_fitdistcp


# A well-behaved sample for the normal model.
X = np.array([0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8])

# The reference `norm.ppf` return-dict keys (reference `norm.py`).
NORM_PPF_KEYS = (
    "ml_params",
    "ml_value",
    "standard_errors",
    "ml_quantiles",
    "cp_quantiles",
    "ru_quantiles",
    "maic",
    "waic1",
    "waic2",
    "ml_oos_logscore",
    "cp_oos_logscore",
    "ml_mean",
    "cp_mean",
    "cp_method",
    # norm-specific extras.
    "uv_params",
    "uv_quantiles",
)


# ---------------------------------------------------------------------------
# Submodule import + reference default `p`.
# ---------------------------------------------------------------------------


def test_import_fitdistcp_norm_and_default_p():
    """``import rusty_fitdistcp; rusty_fitdistcp.norm.ppf(x)`` uses the reference default p."""
    res = rusty_fitdistcp.norm.ppf(X)
    # The reference default is `p = np.arange(0.1, 1.0, 0.1)` (9 values).
    assert res["ml_quantiles"].shape == (9,)
    assert res["cp_quantiles"].shape == (9,)


def test_all_reference_submodules_importable():
    """Every reference submodule is importable as ``rusty_fitdistcp.<name>``."""
    for name in (
        "expon",
        "gamma",
        "genextreme",
        "genextreme_p1",
        "genextreme_p12",
        "genpareto",
        "gumbel",
        "lnorm",
        "norm",
        "weibull",
        "reltest",
    ):
        assert hasattr(rusty_fitdistcp, name), f"missing submodule: rusty_fitdistcp.{name}"


# ---------------------------------------------------------------------------
# Reference return-dict keys.
# ---------------------------------------------------------------------------


def test_norm_ppf_reference_keys():
    res = rusty_fitdistcp.norm.ppf(X)
    for k in NORM_PPF_KEYS:
        assert k in res, f"missing reference key: {k}"


def test_norm_pdf_cdf_rvs_reference_keys():
    pdf = rusty_fitdistcp.norm.pdf(X)
    for k in ("ml_params", "ml_pdf", "cp_pdf", "ru_pdf", "cp_method"):
        assert k in pdf, f"missing pdf key: {k}"

    cdf = rusty_fitdistcp.norm.cdf(X)
    for k in ("ml_params", "ml_cdf", "cp_cdf", "ru_cdf", "cp_method"):
        assert k in cdf, f"missing cdf key: {k}"

    rvs = rusty_fitdistcp.norm.rvs(5, X)
    for k in ("ml_params", "ml_deviates", "cp_deviates", "ru_deviates", "cp_method"):
        assert k in rvs, f"missing rvs key: {k}"


# ---------------------------------------------------------------------------
# Reference value shapes / dtypes / placeholders.
# ---------------------------------------------------------------------------


def test_maic_is_length_three_array():
    res = rusty_fitdistcp.norm.ppf(X)
    maic = np.asarray(res["maic"])
    assert maic.shape == (3,)
    # maic == [ml_value, -nparams, ml_value - nparams].
    ml_value = res["ml_value"]
    nparams = res["ml_params"].size
    np.testing.assert_allclose(maic[0], ml_value)
    np.testing.assert_allclose(maic[1], -nparams)
    np.testing.assert_allclose(maic[2], ml_value - nparams)


def test_standard_errors_is_object_dtype():
    res = rusty_fitdistcp.norm.ppf(X)
    assert res["standard_errors"].dtype == object


def test_ru_quantiles_placeholder():
    res = rusty_fitdistcp.norm.ppf(X)
    assert res["ru_quantiles"] == "ru not selected"


def test_norm_uv_placeholders():
    res = rusty_fitdistcp.norm.ppf(X)
    assert res["uv_params"] == "unbiasedv not selected"
    assert res["uv_quantiles"] == "unbiasedv not selected"


# ---------------------------------------------------------------------------
# `y=None` defaults to `y=x` for pdf/cdf.
# ---------------------------------------------------------------------------


def test_pdf_y_none_defaults_to_x():
    res_default = rusty_fitdistcp.norm.pdf(X)
    res_explicit = rusty_fitdistcp.norm.pdf(X, y=X)
    np.testing.assert_allclose(res_default["ml_pdf"], res_explicit["ml_pdf"])
    np.testing.assert_allclose(res_default["cp_pdf"], res_explicit["cp_pdf"])
    # The density is evaluated at `y` (length matches `x` when `y=None`).
    assert res_default["ml_pdf"].shape == X.shape


def test_cdf_y_none_defaults_to_x():
    res_default = rusty_fitdistcp.norm.cdf(X)
    res_explicit = rusty_fitdistcp.norm.cdf(X, y=X)
    np.testing.assert_allclose(res_default["ml_cdf"], res_explicit["ml_cdf"])
    np.testing.assert_allclose(res_default["cp_cdf"], res_explicit["cp_cdf"])
    assert res_default["ml_cdf"].shape == X.shape


# ---------------------------------------------------------------------------
# `rvs` returns the reference deviate keys.
# ---------------------------------------------------------------------------


def test_rvs_returns_deviate_keys():
    res = rusty_fitdistcp.norm.rvs(5, X)
    assert res["ml_deviates"].shape == (5,)
    assert res["cp_deviates"].shape == (5,)
    assert res["ru_deviates"] == "ru not selected"


# ---------------------------------------------------------------------------
# Flat API still works and emits DeprecationWarning.
# ---------------------------------------------------------------------------


def test_flat_api_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = rusty_fitdistcp.ppf_norm(X, np.array([0.5]))
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "flat API must emit a DeprecationWarning"
    # The flat API still returns a usable result.
    assert res["ml_quantiles"].shape == (1,)


def test_flat_api_matches_submodule_api():
    """The flat alias and the submodule API return the same numerics."""
    p = np.array([0.25, 0.5, 0.75])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        flat = rusty_fitdistcp.ppf_norm(X, p)
    sub = rusty_fitdistcp.norm.ppf(X, p)
    np.testing.assert_allclose(flat["ml_quantiles"], sub["ml_quantiles"])
    np.testing.assert_allclose(flat["cp_quantiles"], sub["cp_quantiles"])


def test_undeprecated_flat_names_do_not_warn():
    """The Phase 0 smoke-test names are not deprecated."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rusty_fitdistcp.core_version()
        rusty_fitdistcp.sum_values(np.array([1.0, 2.0, 3.0]))
    assert not any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "core_version/sum_values must not emit a DeprecationWarning"


# ---------------------------------------------------------------------------
# `__all__` includes `genextreme_p12` (Phase 6, Issue 6).
# ---------------------------------------------------------------------------


def test_all_includes_genextreme_p12():
    assert "genextreme_p12" in rusty_fitdistcp.__all__
    # The other reference submodules are present.
    for name in (
        "genextreme",
        "genextreme_p1",
        "genpareto",
        "expon",
        "gamma",
        "gumbel",
        "lnorm",
        "norm",
        "weibull",
        "reltest",
    ):
        assert name in rusty_fitdistcp.__all__


def test_genextreme_p12_importable():
    """Phase 6 (Issue 6): the two-predictor model is exposed as a submodule."""
    assert hasattr(rusty_fitdistcp, "genextreme_p12")
    assert hasattr(rusty_fitdistcp.genextreme_p12, "ppf")
    assert hasattr(rusty_fitdistcp.genextreme_p12, "pdf")
    assert hasattr(rusty_fitdistcp.genextreme_p12, "cdf")
    assert hasattr(rusty_fitdistcp.genextreme_p12, "rvs")


# ---------------------------------------------------------------------------
# Scalar arguments (regression: `TypeError: dimensionality mismatch`).
#
# The reference `fitdistcp` accepts scalar `p`/`y`/`x` and treats them as a
# single-element vector. The Rust extension's `PyReadonlyArray1` extractor
# requires a 1-D array, so a bare `np.asarray(0.5)` (0-D) was rejected with
#   TypeError: dimensionality mismatch: from=0, to=1
# The wrappers now normalise scalar-capable args with `np.atleast_1d`.
# ---------------------------------------------------------------------------


def test_scalar_p_accepted():
    """A scalar ``p`` is accepted and yields a length-1 quantile vector."""
    res = rusty_fitdistcp.genextreme.ppf([1.0, 2.0, 3.0], p=0.5)
    assert np.asarray(res["ml_quantiles"]).shape == (1,)
    assert np.asarray(res["cp_quantiles"]).shape == (1,)


def test_scalar_y_accepted_in_pdf_and_cdf():
    """A scalar ``y`` is accepted by ``pdf``/``cdf``."""
    pdf = rusty_fitdistcp.genextreme.pdf([1.0, 2.0, 3.0], y=2.0)
    cdf = rusty_fitdistcp.genextreme.cdf([1.0, 2.0, 3.0], y=2.0)
    assert np.asarray(pdf["ml_pdf"]).shape == (1,)
    assert np.asarray(cdf["ml_cdf"]).shape == (1,)


def test_scalar_x_accepted():
    """A scalar ``x`` is accepted and treated as a single-element sample."""
    res = rusty_fitdistcp.genextreme.ppf(2.0, p=0.5)
    assert np.asarray(res["ml_quantiles"]).shape == (1,)


@pytest.mark.parametrize(
    "model",
    ["norm", "gamma", "weibull", "lnorm", "expon", "gumbel", "genpareto"],
)
def test_scalar_p_accepted_all_models(model):
    """Every homogeneous model accepts a scalar ``p``."""
    mod = getattr(rusty_fitdistcp, model)
    res = mod.ppf([1.0, 2.0, 3.0], p=0.5)
    assert np.asarray(res["ml_quantiles"]).shape == (1,)


def test_scalar_p_accepted_genextreme_p1():
    """The predictor model accepts a scalar ``p``."""
    res = rusty_fitdistcp.genextreme_p1.ppf(
        [1.0, 2.0, 3.0], t=[0.0, 1.0, 2.0], p=0.5
    )
    assert np.asarray(res["ml_quantiles"]).shape == (1,)


def test_scalar_p_matches_single_element_array():
    """A scalar ``p`` gives the same result as a 1-element array ``p``."""
    x = [1.0, 2.0, 3.0]
    scalar = rusty_fitdistcp.genextreme.ppf(x, p=0.5)["ml_quantiles"]
    array = rusty_fitdistcp.genextreme.ppf(x, p=np.array([0.5]))["ml_quantiles"]
    np.testing.assert_allclose(scalar, array)
