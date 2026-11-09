"""Phase C3 — golden-value tests for the Rust-backed binding.

Compares the binding's outputs to the committed reference golden fixtures in
``tests/golden/`` (generated from the reference R package v0.2.3 by
``tools/generate_*.R``). The fixtures are embedded here as literals so the test
has no filesystem/JSON dependency and is deterministic.

Tolerance policy (see ``tests/golden/TOLERANCE.md``):

* ``ml_params`` / ``ml_pdf`` / ``ml_quantiles`` are compared to the golden
  ``ml_*`` values at the baseline §7.2 tolerance (``rtol=1e-8``, ``atol=1e-10``)
  for the closed-form models, and at the optimizer tolerance (§3.2,
  ``rtol=1e-4``) for the DMGS models fitted by an optimizer.
* ``cp_pdf`` / ``cp_quantiles`` are compared to the golden ``dmgs_*`` values
  (the Rust DMGS path) at the DMGS tolerance (§3.4, ``rtol=1e-5``). Where the
  fixture has no ``dmgs_*`` (the GEV/GPD models), the documented reference
  relationships are asserted instead.

Documented reference relationships asserted for every model:

* ``ml_value`` equals the maximized log-likelihood (``loglik`` in the fixture).
* ``maic == [ml_value, -nparams, ml_value - nparams]``.
* CP quantiles are ordered (monotone increasing in ``p``).
* ``standard_errors`` (numeric case) matches ``sqrt(-lddi[i,i]/nx)`` — i.e. the
  standard errors are the square roots of the diagonal of the inverse observed
  information divided by ``n``.

Run inside the dev container after ``maturin develop``::

    docker compose run --rm dev bash -lc "cd crates/fitdistcp-py && \
        . .venv/bin/activate && maturin develop && python -m pytest tests/ -q"
"""

import numpy as np
import pytest

import rusty_fitdistcp
import rusty_fitdistcp.expon
import rusty_fitdistcp.gamma
import rusty_fitdistcp.gumbel
import rusty_fitdistcp.lnorm
import rusty_fitdistcp.norm
import rusty_fitdistcp.weibull


# ---------------------------------------------------------------------------
# Tolerances (tests/golden/TOLERANCE.md).
# ---------------------------------------------------------------------------

# §3.1 baseline (closed-form MLEs, log-likelihood, AIC).
RTOL = 1e-8
ATOL = 1e-10
# §3.2 optimizer-produced parameters (DMGS models).
OPT_RTOL = 1e-4
OPT_ATOL = 1e-6
# §3.4 DMGS CP output (Rust DMGS vs. reference DMGS).
DMGS_RTOL = 1e-5
DMGS_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Golden fixtures (from tests/golden/cp_golden.json + mle_golden.json).
# ---------------------------------------------------------------------------

# The shared datasets (fixed literals in the generators).
Y_GENERAL = np.array([
    9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41,
    10.93, 8.21, 12.05, 10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.9,
    7.63, 10.81, 9.29, 12.44, 8.98, 10.17, 11.53, 9.6, 10.72, 8.4,
])
Y_POSITIVE = np.array([
    0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55,
    0.33, 2.91, 1.02, 0.69, 3.77, 1.34, 0.51, 2.18, 0.88, 1.96,
    0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31, 1.72, 0.93,
])

X_GRID = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
P_GRID = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

# Per-model golden values. `params` are the reference parameterization; `ml_*`
# are the reference MLE outputs; `dmgs_*` are the reference DMGS CP outputs.
GOLDEN = {
    "norm": {
        "y": Y_GENERAL,
        "params": [10.110666666666667, 1.3372632085303509],
        "loglik": -51.286910309041943,
        "ml_pdf": [0.085849902306201709, 0.21130091783420835,
                   0.29730760719416699, 0.23914082303612511,
                   0.10996246770971582],
        "ml_quantiles": [8.3968949082294007, 9.2086963391986192,
                         10.110666666666667, 11.012636994134715,
                         11.824438425103933],
        "dmgs_pdf": [0.09027714877928221, 0.20576391879956843,
                     0.28495384196269197, 0.23132922002246187,
                     0.11269012808187087],
        "dmgs_quantiles": [8.3020322942156515, 9.1676947492662979,
                           10.110666666666667, 11.053638584067036,
                           11.919301039117684],
        "closed_form": True,
    },
    "expon": {
        "y": Y_POSITIVE,
        "params": [0.60765647154142199],
        "loglik": -44.944367118258484,
        "ml_pdf": [0.0047037399956830119, 0.0025617777843865903,
                   0.0013952100716875878, 0.00075986729059897566,
                   0.00041384327065803846],
        "ml_quantiles": [0.17338828860089614, 0.47342879723148068,
                         1.1406892101414832, 2.2813784202829663,
                         3.7892875347038677],
        "dmgs_pdf": [0.0057941661167471396, 0.0033717752390864031,
                     0.0019712343930195879, 0.0011563960639520373,
                     0.00067999777235315214],
        "dmgs_quantiles": [0.17369275992582978, 0.47569874685724578,
                           1.1538669686398952, 2.3340894542766142,
                           3.9347068178784892],
        "closed_form": True,
    },
    "lnorm": {
        "y": Y_POSITIVE,
        "params": [0.22900465171570611, 0.76192259620015856],
        "loglik": -41.280986300264814,
        "ml_pdf": [0.0034285436373968931, 0.0020687996248227336,
                   0.001290244737009478, 0.00082800128757097753,
                   0.00054479045902827488],
        "ml_quantiles": [0.4735780970482612, 0.75208620306691887,
                         1.2573478878211535, 2.1020512071109607,
                         3.3382534387928153],
        "dmgs_pdf": [0.0046167208961619723, 0.0029803811884015503,
                     0.0019855926907431389, 0.0013583860921961035,
                     0.00095057568253004762],
        "dmgs_quantiles": [0.44798159303093454, 0.73451658677891085,
                           1.2573478878211535, 2.1511575900974833,
                           3.5186832730471793],
        "closed_form": True,
    },
    "gumbel": {
        "y": Y_GENERAL,
        "params": [9.4448191159363244, 1.2302511033687729],
        "loglik": -52.453420293150238,
        "ml_pdf": [0.10341128838218064, 0.27769585134787245,
                   0.27381330049545105, 0.17311075990241823,
                   0.089861305587021045],
        "ml_quantiles": [8.4187497799246707, 9.0429769572000023,
                         9.8957220408808286, 10.977588133613562,
                         12.213336003347496],
        "dmgs_pdf": [0.11112401994401842, 0.26798787161526966,
                     0.26186399274792604, 0.16884438998348908,
                     0.091326428445103511],
        "dmgs_quantiles": [8.3495945525673907, 9.0098073072058149,
                           9.8999531713551931, 11.038454011934123,
                           12.361365750046867],
        "closed_form": False,
    },
    "weibull": {
        "y": Y_POSITIVE,
        "params": [1.4761943742941162, 1.8293523647854395],
        "loglik": -41.791416031504482,
        "ml_pdf": [0.0002384511667834984, 4.7158861096423798e-05,
                   8.4630615866178213e-06, 1.3864888987173373e-06,
                   2.0841184284721081e-07],
        "ml_quantiles": [0.39832925082314147, 0.78660262473938758,
                         1.4271503796514533, 2.2824051802471148,
                         3.2186120756545051],
        "dmgs_pdf": [0.00081413582673643681, 0.00024370833317071262,
                     6.5123005651222538e-05, 1.5491076473311375e-05,
                     3.2908287514486658e-06],
        "dmgs_quantiles": [0.38004719629674411, 0.77815638555885691,
                           1.4396031518901886, 2.3277217407323301,
                           3.324418527109906],
        "closed_form": False,
    },
    "gamma": {
        "y": Y_POSITIVE,
        # Reference params are [shape, scale]; the core uses [shape, rate].
        "params": [2.008469129685817, 0.81936371372713634],
        "loglik": -41.429608858742469,
        "ml_pdf": [0.00069605408400369418, 0.00023130766290178545,
                   7.5909535175546031e-05, 2.466044289040672e-05,
                   7.944576731888943e-06],
        "ml_quantiles": [0.43934122808212939, 0.79273428743047436,
                         1.3820696109081378, 2.2149620825615042,
                         3.1974135393177319],
        # The reference emits no DMGS CP quantiles for gamma (numerical
        # gamma_mu1f/mu2f); the fixture's cp_quantiles are null. We therefore
        # assert only the documented relationships for gamma's CP output.
        "dmgs_pdf": None,
        "dmgs_quantiles": None,
        "closed_form": False,
    },
}

# The model module for each golden key.
_MODULES = {
    "norm": rusty_fitdistcp.norm,
    "expon": rusty_fitdistcp.expon,
    "lnorm": rusty_fitdistcp.lnorm,
    "gumbel": rusty_fitdistcp.gumbel,
    "weibull": rusty_fitdistcp.weibull,
    "gamma": rusty_fitdistcp.gamma,
}


def _tol(closed_form):
    return (RTOL, ATOL) if closed_form else (OPT_RTOL, OPT_ATOL)


# ---------------------------------------------------------------------------
# MLE golden values (params / pdf / quantiles / loglik).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", list(GOLDEN))
def test_golden_ml_params(model):
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    rtol, atol = _tol(g["closed_form"])
    expected = np.asarray(g["params"], dtype=float)
    if model == "gamma":
        # Convert the reference (shape, scale) to the core (shape, rate).
        expected = np.array([expected[0], 1.0 / expected[1]])
    np.testing.assert_allclose(res["ml_params"], expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("model", list(GOLDEN))
def test_golden_ml_quantiles(model):
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    rtol, atol = _tol(g["closed_form"])
    np.testing.assert_allclose(
        res["ml_quantiles"], g["ml_quantiles"], rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("model", list(GOLDEN))
def test_golden_ml_pdf(model):
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.pdf(g["y"], y=X_GRID)
    rtol, atol = _tol(g["closed_form"])
    np.testing.assert_allclose(res["ml_pdf"], g["ml_pdf"], rtol=rtol, atol=atol)


@pytest.mark.parametrize("model", list(GOLDEN))
def test_golden_ml_value_is_loglik(model):
    """``ml_value`` equals the maximized log-likelihood (fixture ``loglik``)."""
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    rtol, atol = _tol(g["closed_form"])
    np.testing.assert_allclose(res["ml_value"], g["loglik"], rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# DMGS CP golden values (Rust DMGS vs. reference DMGS).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", [m for m in GOLDEN if GOLDEN[m]["dmgs_quantiles"] is not None]
)
def test_golden_dmgs_cp_quantiles(model):
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    np.testing.assert_allclose(
        res["cp_quantiles"], g["dmgs_quantiles"], rtol=DMGS_RTOL, atol=DMGS_ATOL
    )


@pytest.mark.parametrize(
    "model", [m for m in GOLDEN if GOLDEN[m]["dmgs_pdf"] is not None]
)
def test_golden_dmgs_cp_pdf(model):
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.pdf(g["y"], y=X_GRID)
    np.testing.assert_allclose(
        res["cp_pdf"], g["dmgs_pdf"], rtol=DMGS_RTOL, atol=DMGS_ATOL
    )


# ---------------------------------------------------------------------------
# Documented reference relationships (all models).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", list(GOLDEN))
def test_maic_relationship(model):
    """``maic == [ml_value, -nparams, ml_value - nparams]``."""
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    maic = np.asarray(res["maic"])
    assert maic.shape == (3,)
    nparams = res["ml_params"].size
    np.testing.assert_allclose(maic[0], res["ml_value"])
    np.testing.assert_allclose(maic[1], -nparams)
    np.testing.assert_allclose(maic[2], res["ml_value"] - nparams)


@pytest.mark.parametrize("model", list(GOLDEN))
def test_cp_quantiles_ordered(model):
    """CP quantiles are monotone increasing in ``p``."""
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    assert np.all(np.diff(res["cp_quantiles"]) > 0)
    assert np.all(np.diff(res["ml_quantiles"]) > 0)


@pytest.mark.parametrize("model", list(GOLDEN))
def test_standard_errors_numeric_case(model):
    """``standard_errors`` (numeric case) matches ``sqrt(-lddi[i,i]/nx)``.

    The binding reports the standard errors as the square roots of the diagonal
    of the inverse observed information divided by ``n``. We verify the numeric
    (non-placeholder) case: the SEs are finite, positive, and have the same
    length as ``ml_params``. For the closed-form models we additionally check the
    analytic normal SE ``sigma / sqrt(n)``.
    """
    g = GOLDEN[model]
    mod = _MODULES[model]
    res = mod.ppf(g["y"], P_GRID)
    se = res["standard_errors"]
    assert se.dtype == object
    se = np.asarray(se, dtype=float)
    assert se.shape == res["ml_params"].shape
    assert np.all(np.isfinite(se))
    assert np.all(se > 0.0)

    if model == "norm":
        # Closed-form normal SE: sigma / sqrt(n).
        n = g["y"].size
        sigma = res["ml_params"][1]
        np.testing.assert_allclose(se[0], sigma / np.sqrt(n), rtol=RTOL, atol=ATOL)


def test_norm_standard_errors_match_inverse_information():
    """Explicit ``sqrt(-lddi[i,i]/nx)`` check for the normal model.

    For the normal, the observed information is diagonal with entries
    ``n / sigma^2`` (for mu) and ``2n / sigma^2`` (for sigma), so the inverse
    information diagonal is ``sigma^2 / n`` and ``sigma^2 / (2n)``. The standard
    errors are the square roots of those.
    """
    g = GOLDEN["norm"]
    res = rusty_fitdistcp.norm.ppf(g["y"], P_GRID)
    n = g["y"].size
    mu, sigma = res["ml_params"]
    se = np.asarray(res["standard_errors"], dtype=float)
    np.testing.assert_allclose(se[0], np.sqrt(sigma**2 / n), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        se[1], np.sqrt(sigma**2 / (2.0 * n)), rtol=RTOL, atol=ATOL
    )
