"""GEV-with-predictor predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.genextreme_p1`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, t, t0=None, n0=None, p=np.arange(0.1, 1.0, 0.1), ics=None,
        fdalpha=0.01, minxi=-1, maxxi=1, means=False, waicscores=False,
        extramodels=False, pdf=False, dmgs=True, ru=False, ru_nsamples=100000,
        predictordata=True, centering=True, debug=False):
    """GEV-with-predictor CP predictive quantiles (reference ``genextreme_p1.ppf``)."""
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.ppf_genextreme_p1_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t, dtype=float)),
        t0=t0, n0=n0, p=np.atleast_1d(np.asarray(p, dtype=float)), ics=ics,
        fdalpha=fdalpha,
        minxi=float(minxi), maxxi=float(maxxi), means=means,
        waicscores=waicscores, extramodels=extramodels, pdf=pdf, dmgs=dmgs,
        ru=ru, ru_nsamples=ru_nsamples, predictordata=predictordata,
        centering=centering,
    )


def rvs(n, x, t, t0=None, n0=None, ics=None, minxi=-1, maxxi=1,
        extramodels=False, ru=False, mlcp=True, debug=False):
    """GEV-with-predictor CP random deviates (reference ``genextreme_p1.rvs``)."""
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.rvs_genextreme_p1_ref(
        int(n), np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t, dtype=float)),
        t0=t0, n0=n0, ics=ics, minxi=float(minxi), maxxi=float(maxxi),
        extramodels=extramodels, ru=ru, mlcp=mlcp,
    )


def pdf(x, t, t0=None, n0=None, y=None, ics=None, fdalpha=0.01,
        minxi=-1, maxxi=1, extramodels=False, pdf=False, dmgs=True,
        ru=False, ru_nsamples=1000, predictordata=True, centering=True,
        debug=False):
    """GEV-with-predictor CP predictive density (reference ``genextreme_p1.pdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.pdf_genextreme_p1_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t, dtype=float)),
        t0=t0, n0=n0, y=y, ics=ics, fdalpha=fdalpha, minxi=float(minxi),
        maxxi=float(maxxi), extramodels=extramodels, pdf=pdf, dmgs=dmgs,
        ru=ru, ru_nsamples=ru_nsamples, predictordata=predictordata,
        centering=centering,
    )


def cdf(x, t, t0=None, n0=None, y=None, ics=None, fdalpha=0.01,
        minxi=-1, maxxi=1, extramodels=False, dmgs=True, ru=False,
        ru_nsamples=1000, predictordata=True, centering=True, debug=False):
    """GEV-with-predictor CP predictive CDF (reference ``genextreme_p1.cdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.cdf_genextreme_p1_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t, dtype=float)),
        t0=t0, n0=n0, y=y, ics=ics, fdalpha=fdalpha, minxi=float(minxi),
        maxxi=float(maxxi), extramodels=extramodels, dmgs=dmgs, ru=ru,
        ru_nsamples=ru_nsamples, predictordata=predictordata,
        centering=centering,
    )
