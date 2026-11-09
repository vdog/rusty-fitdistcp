"""``rusty_fitdistcp`` — calibrating-prior prediction for distributions.

This is the Rust-backed drop-in replacement for the reference ``fitdistcp``
package (v0.0.4). The public API is **submodule-based**, mirroring the
reference:

    import rusty_fitdistcp
    rusty_fitdistcp.norm.ppf(x, p)

The numerics live in the compiled extension module ``rusty_fitdistcp._fitdistcp``;
the thin Python submodules in this package provide the reference signatures,
defaults, and return-dict keys.

The legacy **flat** API (``rusty_fitdistcp.ppf_norm``, ``rusty_fitdistcp.pdf_gev``, ...) is
kept as deprecated aliases for backwards compatibility (see B5).
"""

from . import _fitdistcp

# ---------------------------------------------------------------------------
# Reference submodules (B4). Imported eagerly so `rusty_fitdistcp.norm` etc. work
# after a bare `import rusty_fitdistcp`.
# ---------------------------------------------------------------------------
from . import (  # noqa: F401  (re-exported for `rusty_fitdistcp.<name>` access)
    expon,
    gamma,
    genextreme,
    genextreme_p1,
    genpareto,
    gumbel,
    lnorm,
    norm,
    reltest,
    weibull,
)

# The reference `__all__` includes `genextreme_p12`; we deliberately OMIT it
# (deferred — see the Phase B scope decisions).
__all__ = [
    "genextreme",
    "genextreme_p1",
    "genpareto",
    "expon",
    "gamma",
    "gumbel",
    "lnorm",
    "norm",
    "weibull",
    "reltest",
]

# ---------------------------------------------------------------------------
# B5: flat API as deprecated aliases.
#
# The flat functions (`ppf_norm`, `pdf_gev`, ...) are re-exported from the
# extension module. Accessing them emits a `DeprecationWarning` pointing at the
# reference submodule API. The names are resolved lazily via module
# `__getattr__` so the warning fires on use, not on import.
# ---------------------------------------------------------------------------

# The flat names the extension module exposes (kept in sync with `lib.rs`).
_FLAT_NAMES = frozenset(
    [
        "core_version",
        "add_vectors",
        "sum_values",
        # norm
        "ppf_norm",
        "pdf_norm",
        "cdf_norm",
        "rvs_norm",
        # norm_p1
        "ppf_norm_p1",
        "pdf_norm_p1",
        "cdf_norm_p1",
        "rvs_norm_p1",
        # gev
        "ppf_gev",
        "pdf_gev",
        "cdf_gev",
        "rvs_gev",
        # gev_p1
        "ppf_gev_p1",
        "pdf_gev_p1",
        "cdf_gev_p1",
        "rvs_gev_p1",
        # gpd_k1
        "ppf_gpd_k1",
        "pdf_gpd_k1",
        "cdf_gpd_k1",
        "rvs_gpd_k1",
        # gumbel
        "ppf_gumbel",
        "pdf_gumbel",
        "cdf_gumbel",
        "rvs_gumbel",
        # lnorm
        "ppf_lnorm",
        "pdf_lnorm",
        "cdf_lnorm",
        "rvs_lnorm",
        # weibull
        "ppf_weibull",
        "pdf_weibull",
        "cdf_weibull",
        "rvs_weibull",
        # gamma
        "ppf_gamma",
        "pdf_gamma",
        "cdf_gamma",
        "rvs_gamma",
        # exp
        "ppf_exp",
        "pdf_exp",
        "cdf_exp",
        "rvs_exp",
    ]
)

# The flat names that are NOT deprecated (Phase 0 smoke tests).
_FLAT_UNDEPRECATED = frozenset(["core_version", "add_vectors", "sum_values"])


def __getattr__(name):
    """Resolve flat API names lazily, emitting a deprecation warning (B5)."""
    if name in _FLAT_NAMES:
        import warnings

        if name not in _FLAT_UNDEPRECATED:
            warnings.warn(
                f"rusty_fitdistcp.{name} is deprecated; use the reference submodule "
                f"API (e.g. rusty_fitdistcp.norm.ppf) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return getattr(_fitdistcp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
