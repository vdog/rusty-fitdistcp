"""Normal distribution predictions based on a calibrating prior.

Drop-in replacement for the reference ``fitdistcp.norm`` submodule. Delegates to
the compiled extension ``rusty_fitdistcp._fitdistcp``.
"""

import numpy as np

from . import _fitdistcp


def ppf(x, p=np.arange(0.1, 1.0, 0.1),
        means=False, waicscores=False, logscores=False,
        ru=False, ru_nsamples=100000,
        unbiasedv=False):
    """Normal CP predictive quantiles (reference ``norm.ppf``)."""
    return _fitdistcp.ppf_norm_ref(
        np.atleast_1d(np.asarray(x, dtype=float)),
        np.atleast_1d(np.asarray(p, dtype=float)),
        means=means, waicscores=waicscores, logscores=logscores,
        ru=ru, ru_nsamples=ru_nsamples, unbiasedv=unbiasedv,
    )


def rvs(n, x, ru=False, mlcp=True, debug=False):
    """Normal CP random deviates (reference ``norm.rvs``)."""
    return _fitdistcp.rvs_norm_ref(
        int(n), np.atleast_1d(np.asarray(x, dtype=float)), ru=ru, mlcp=mlcp,
    )


def pdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Normal CP predictive density (reference ``norm.pdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    return _fitdistcp.pdf_norm_ref(
        np.atleast_1d(np.asarray(x, dtype=float)), y, ru=ru,
        ru_nsamples=ru_nsamples,
    )


def cdf(x, y=None, ru=False, ru_nsamples=1000, debug=False):
    """Normal CP predictive CDF (reference ``norm.cdf``)."""
    y = None if y is None else np.atleast_1d(np.asarray(y, dtype=float))
    return _fitdistcp.cdf_norm_ref(
        np.atleast_1d(np.asarray(x, dtype=float)), y, ru=ru,
        ru_nsamples=ru_nsamples,
    )
