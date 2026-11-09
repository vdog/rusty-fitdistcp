"""Weibull distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.weibull`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=np.arange(0.1, 1.0, 0.1),
        means=False, waicscores=False, logscores=False,
        dmgs=True, ru=False, ru_nsamples=100000,
        debug=False):
    """Weibull CP predictive quantiles (reference ``weibull.ppf``)."""
    return _fitdistcp.ppf_weibull_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(p, dtype=float)),
        means=means, waicscores=waicscores, logscores=logscores,
        dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
    )


def rvs(n, x, ru=False, mlcp=True, debug=False):
    """Weibull CP random deviates (reference ``weibull.rvs``)."""
    return _fitdistcp.rvs_weibull_ref(
        int(n), np.atleast_1d(np.asarray(x, dtype=float)), ru=ru, mlcp=mlcp,
    )


def pdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Weibull CP predictive density (reference ``weibull.pdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    return _fitdistcp.pdf_weibull_ref(
        np.atleast_1d(np.asarray(x, dtype=float)), y, ru=ru,
        ru_nsamples=ru_nsamples,
    )


def cdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Weibull CP predictive CDF (reference ``weibull.cdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    return _fitdistcp.cdf_weibull_ref(
        np.atleast_1d(np.asarray(x, dtype=float)), y, ru=ru,
        ru_nsamples=ru_nsamples,
    )
