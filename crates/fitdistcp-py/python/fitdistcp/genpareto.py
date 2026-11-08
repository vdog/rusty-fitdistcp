"""Generalized Pareto distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.genpareto`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=None, kloc=0, ics=None, fdalpha=0.01, customprior=0,
        minxi=-1, maxxi=2.0, means=False, waicscores=False, extramodels=False,
        pdf=False, dmgs=True, ru=False, ru_nsamples=100000,
        debug=False):
    """GPD CP predictive quantiles (reference ``genpareto.ppf``)."""
    if p is None:
        p = np.arange(0.1, 1.0, 0.1)
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.ppf_genpareto_ref(
        np.asarray(x, dtype=float), np.asarray(p, dtype=float),
        kloc=float(kloc), ics=ics, fdalpha=fdalpha, customprior=customprior,
        minxi=float(minxi), maxxi=float(maxxi), means=means,
        waicscores=waicscores, extramodels=extramodels, pdf=pdf,
        dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
    )


def rvs(n, x, kloc=0, ics=None, minxi=-1, maxxi=2.0,
        extramodels=False, ru=False, mlcp=True, debug=False):
    """GPD CP random deviates (reference ``genpareto.rvs``)."""
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.rvs_genpareto_ref(
        int(n), np.asarray(x, dtype=float), kloc=float(kloc), ics=ics,
        minxi=float(minxi), maxxi=float(maxxi), extramodels=extramodels,
        ru=ru, mlcp=mlcp,
    )


def pdf(x, y=None, kloc=0, ics=None, minxi=-1, maxxi=2.0,
        ru=False, ru_nsamples=1000):
    """GPD CP predictive density (reference ``genpareto.pdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.pdf_genpareto_ref(
        np.asarray(x, dtype=float), y, kloc=float(kloc), ics=ics,
        minxi=float(minxi), maxxi=float(maxxi), ru=ru, ru_nsamples=ru_nsamples,
    )


def cdf(x, y=None, kloc=0, ics=None, minxi=-1, maxxi=2.0,
        ru=False, ru_nsamples=1000):
    """GPD CP predictive CDF (reference ``genpareto.cdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    ics = None if ics is None else np.asarray(ics, dtype=float)
    return _fitdistcp.cdf_genpareto_ref(
        np.asarray(x, dtype=float), y, kloc=float(kloc), ics=ics,
        minxi=float(minxi), maxxi=float(maxxi), ru=ru, ru_nsamples=ru_nsamples,
    )
