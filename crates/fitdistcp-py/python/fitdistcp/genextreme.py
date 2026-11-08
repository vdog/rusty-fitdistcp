"""Generalized extreme value distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.genextreme`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=np.arange(0.1, 1.0, 0.1), ics=None, fdalpha=0.01,
        means=False, waicscores=False, extramodels=False, pdf=False,
        customprior=0.0, dmgs=True, ru=False, ru_nsamples=100000,
        debug=False):
    """GEV CP predictive quantiles (reference ``genextreme.ppf``)."""
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.ppf_genextreme_ref(
        np.asarray(x, dtype=float), np.asarray(p, dtype=float),
        ics=ics, fdalpha=fdalpha, means=means, waicscores=waicscores,
        extramodels=extramodels, pdf=pdf, customprior=customprior,
        dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
    )


def rvs(n, x, ics=None, extramodels=False, mlcp=True):
    """GEV CP random deviates (reference ``genextreme.rvs``)."""
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.rvs_genextreme_ref(
        int(n), np.asarray(x, dtype=float), ics=ics,
        extramodels=extramodels, mlcp=mlcp,
    )


def pdf(x, y=None, ics=None):
    """GEV CP predictive density (reference ``genextreme.pdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.pdf_genextreme_ref(
        np.asarray(x, dtype=float), y, ics=ics,
    )


def cdf(x, y=None, ics=None, extramodels=False, debug=False):
    """GEV CP predictive CDF (reference ``genextreme.cdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.cdf_genextreme_ref(
        np.asarray(x, dtype=float), y, ics=ics, extramodels=extramodels,
    )
