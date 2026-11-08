"""Gamma distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.gamma`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=None, fd1=0.01, fd2=0.01, means=False, waicscores=False,
        logscores=False, dmgs=True, ru=False, ru_nsamples=100000,
        prior="type 1", debug=False, aderivs=True):
    """Gamma CP predictive quantiles (reference ``gamma.ppf``)."""
    if p is None:
        p = np.arange(0.1, 1.0, 0.1)
    return _fitdistcp.ppf_gamma_ref(
        np.asarray(x, dtype=float), np.asarray(p, dtype=float),
        fd1=fd1, fd2=fd2, means=means, waicscores=waicscores,
        logscores=logscores, dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
        prior=prior, aderivs=aderivs,
    )


def rvs(n, x, ru=False, mlcp=True, debug=False):
    """Gamma CP random deviates (reference ``gamma.rvs``)."""
    return _fitdistcp.rvs_gamma_ref(
        int(n), np.asarray(x, dtype=float), ru=ru, mlcp=mlcp,
    )


def pdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Gamma CP predictive density (reference ``gamma.pdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    return _fitdistcp.pdf_gamma_ref(
        np.asarray(x, dtype=float), y, ru=ru, ru_nsamples=ru_nsamples,
    )


def cdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Gamma CP predictive CDF (reference ``gamma.cdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    return _fitdistcp.cdf_gamma_ref(
        np.asarray(x, dtype=float), y, ru=ru, ru_nsamples=ru_nsamples,
    )
