"""Exponential distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.expon`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=np.arange(0.1, 1.0, 0.1),
        means=False, waicscores=False, logscores=False,
        ru=False, ru_nsamples=100000):
    """Exponential CP predictive quantiles (reference ``expon.ppf``)."""
    return _fitdistcp.ppf_expon_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(p, dtype=float)),
        means=means, waicscores=waicscores, logscores=logscores,
        ru=ru, ru_nsamples=ru_nsamples,
    )


def rvs(n, x, ru=False, mlcp=True):
    """Exponential CP random deviates (reference ``expon.rvs``)."""
    return _fitdistcp.rvs_expon_ref(
        int(n), np.atleast_1d(np.asarray(x, dtype=float)), ru=ru, mlcp=mlcp,
    )


def pdf(x, y=None, ru=False, ru_nsamples=1000):
    """Exponential CP predictive density (reference ``expon.pdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    return _fitdistcp.pdf_expon_ref(
        np.atleast_1d(np.asarray(x, dtype=float)), y, ru=ru,
        ru_nsamples=ru_nsamples,
    )


def cdf(x, y=None, ru=False, ru_nsamples=1000):
    """Exponential CP predictive CDF (reference ``expon.cdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    return _fitdistcp.cdf_expon_ref(
        np.atleast_1d(np.asarray(x, dtype=float)), y, ru=ru,
        ru_nsamples=ru_nsamples,
    )
