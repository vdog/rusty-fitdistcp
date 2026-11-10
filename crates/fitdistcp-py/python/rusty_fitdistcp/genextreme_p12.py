"""GEV-with-two-predictors predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.genextreme_p12`` submodule.

The model has ``params = [beta0, beta1, beta2, scale, shape]`` with location
``mu_i = beta0 + beta1*(t1_i - mean(t1))`` and scale
``sigma_i = exp(beta2 + scale*(t2_i - mean(t2)))``.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, t1, t2, t01=None, t02=None, n01=None, n02=None,
        p=np.arange(0.1, 1.0, 0.1), ics=None, fdalpha=0.01, minxi=-1, maxxi=1,
        means=False, waicscores=False, extramodels=False, pdf=False, dmgs=True,
        ru=False, ru_nsamples=100000, predictordata=True, centering=True,
        debug=False):
    """GEV-with-two-predictors CP predictive quantiles (reference ``genextreme_p12.ppf``)."""
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.ppf_genextreme_p12_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t1, dtype=float)),
        np.atleast_1d(np.asarray(t2, dtype=float)),
        t01=t01, t02=t02, n01=n01, n02=n02,
        p=np.atleast_1d(np.asarray(p, dtype=float)), ics=ics,
        fdalpha=fdalpha,
        minxi=float(minxi), maxxi=float(maxxi), means=means,
        waicscores=waicscores, extramodels=extramodels, pdf=pdf, dmgs=dmgs,
        ru=ru, ru_nsamples=ru_nsamples, predictordata=predictordata,
        centering=centering,
    )


def rvs(n, x, t1, t2, t01=None, t02=None, n01=None, n02=None, ics=None,
        minxi=-1, maxxi=1, extramodels=False, ru=False, mlcp=True, debug=False):
    """GEV-with-two-predictors CP random deviates (reference ``genextreme_p12.rvs``)."""
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.rvs_genextreme_p12_ref(
        int(n), np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t1, dtype=float)),
        np.atleast_1d(np.asarray(t2, dtype=float)),
        t01=t01, t02=t02, n01=n01, n02=n02, ics=ics,
        minxi=float(minxi), maxxi=float(maxxi),
        extramodels=extramodels, ru=ru, mlcp=mlcp,
    )


def pdf(x, t1, t2, t01=None, t02=None, n01=None, n02=None, y=None, ics=None,
        fdalpha=0.01, minxi=-1, maxxi=1, extramodels=False, pdf=False,
        dmgs=True, ru=False, ru_nsamples=1000, predictordata=True,
        centering=True, debug=False):
    """GEV-with-two-predictors CP predictive density (reference ``genextreme_p12.pdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.pdf_genextreme_p12_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t1, dtype=float)),
        np.atleast_1d(np.asarray(t2, dtype=float)),
        t01=t01, t02=t02, n01=n01, n02=n02, y=y, ics=ics, fdalpha=fdalpha,
        minxi=float(minxi), maxxi=float(maxxi), extramodels=extramodels,
        pdf=pdf, dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
        predictordata=predictordata, centering=centering,
    )


def cdf(x, t1, t2, t01=None, t02=None, n01=None, n02=None, y=None, ics=None,
        fdalpha=0.01, minxi=-1, maxxi=1, extramodels=False, dmgs=True, ru=False,
        ru_nsamples=1000, predictordata=True, centering=True, debug=False):
    """GEV-with-two-predictors CP predictive CDF (reference ``genextreme_p12.cdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    ics = None if ics is None else np.atleast_1d(np.asarray(ics, dtype=float))
    return _fitdistcp.cdf_genextreme_p12_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(t1, dtype=float)),
        np.atleast_1d(np.asarray(t2, dtype=float)),
        t01=t01, t02=t02, n01=n01, n02=n02, y=y, ics=ics, fdalpha=fdalpha,
        minxi=float(minxi), maxxi=float(maxxi), extramodels=extramodels,
        dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
        predictordata=predictordata, centering=centering,
    )
