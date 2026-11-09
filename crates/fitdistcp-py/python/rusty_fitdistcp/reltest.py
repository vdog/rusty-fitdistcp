"""Reliability test for the calibrating-prior predictions.

Drop-in replacement for the reference ``fitdistcp.reltest`` submodule. The
reference exposes a per-model ``reltest`` (``reltest_dep.py``); this module
provides a single ``reltest`` entry point that delegates to the core reliability
test and returns ``{'actual_p_ml', 'actual_p_cp', 'p'}`` as Python lists.
"""

import numpy as np

from . import _fitdistcp

# The models the core reliability test covers (reference `reltest_dep.py`).
_MODELS = (
    "expon",
    "genextreme",
    "genpareto",
    "gumbel",
    "norm",
    "gamma",
    "weibull",
    "lnorm",
)


def reltest(model="norm", ntrials=100, nx=30,
            p=0.0001 * np.asarray(range(1, 10000)),
            true_params=None, init=None):
    """Run the reliability test for ``model``.

    Parameters
    ----------
    model : str, default "norm"
        One of ``expon, genextreme, genpareto, gumbel, norm, gamma, weibull,
        lnorm``.
    ntrials : int, default 100
        Number of Monte Carlo trials to average over.
    nx : int, default 30
        Number of samples per trial.
    p : array_like, default 0.0001*arange(1, 10000)
        Nominal probabilities at which to evaluate the quantiles.
    true_params : array_like, optional
        True model parameters (defaults to the reference values).
    init : array_like, optional
        Initial guess for each MLE fit (defaults to the reference values).

    Returns
    -------
    dict
        ``{'actual_p_ml', 'actual_p_cp', 'p'}`` (Python lists).
    """
    if model not in _MODELS:
        raise ValueError(
            f"unknown reltest model: {model!r}; expected one of {_MODELS}"
        )
    p = np.asarray(p, dtype=float)
    true_params = (
        None if true_params is None else np.asarray(true_params, dtype=float)
    )
    init = None if init is None else np.asarray(init, dtype=float)
    return _fitdistcp.reltest_ref(
        model, int(ntrials), int(nx), p, true_params, init,
    )
