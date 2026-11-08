"""Log-normal distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.lnorm`` submodule.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=np.arange(0.1, 1.0, 0.1),
        means=False, waicscores=False, logscores=False,
        dmgs=True, ru=False, ru_nsamples=100000,
        debug=False):
    """Log-normal CP predictive quantiles (reference ``lnorm.ppf``)."""
    return _fitdistcp.ppf_lnorm_ref(
        np.asarray(x, dtype=float), np.asarray(p, dtype=float),
        means=means, waicscores=waicscores, logscores=logscores,
        dmgs=dmgs, ru=ru, ru_nsamples=ru_nsamples,
    )


def rvs(n, x, ru=False, mlcp=True, debug=False):
    """Log-normal CP random deviates (reference ``lnorm.rvs``)."""
    return _fitdistcp.rvs_lnorm_ref(
        int(n), np.asarray(x, dtype=float), ru=ru, mlcp=mlcp,
    )


def pdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Log-normal CP predictive density (reference ``lnorm.pdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    return _fitdistcp.pdf_lnorm_ref(
        np.asarray(x, dtype=float), y, ru=ru, ru_nsamples=ru_nsamples,
    )


def cdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Log-normal CP predictive CDF (reference ``lnorm.cdf``)."""
    y = None if y is None else np.asarray(y, dtype=float)
    return _fitdistcp.cdf_lnorm_ref(
        np.asarray(x, dtype=float), y, ru=ru, ru_nsamples=ru_nsamples,
    )
