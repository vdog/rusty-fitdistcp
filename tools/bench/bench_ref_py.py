#!/usr/bin/env python3
"""Cross-language performance comparison — Python module harness (Group C).

Times a Python ``fitdistcp`` implementation on the canonical workload defined in
``crates/fitdistcp-core/benches/PERF_COMPARISON.md``, consuming the shared
inputs in ``crates/fitdistcp-core/benches/perf_inputs.json``.

Two implementations are supported (selected with ``--impl`` or the
``BENCH_IMPL`` environment variable):

* ``ref`` (default) — the vendored reference Python module
  ``lynnejewson/fitdistcp`` (cloned to ``tools/fitdistcp-py-ref/``). Emits
  ``implementation = "py_module"``.
* ``rust`` — the Rust-backed drop-in binding (``crates/fitdistcp-py``), which
  exposes the same reference submodule API (``rusty_fitdistcp.norm.ppf`` etc.).
  Emits ``implementation = "rust_binding"``.

Both modes emit CSV rows with the schema::

    implementation,model,operation,n,median_ns,min_ns,max_ns,repeats

Workload (PERF_COMPARISON.md §1):

* pointwise ``pdf``/``cdf``/``ppf``/``rvs`` for ``norm``/``genextreme``/``gamma``
  at n = 50/500/5000 over the 100-point grid (fit once outside the timed region),
* end-to-end ``predict`` (MLE + DMGS) at n = 500 (one per model),
* ``reltest`` (ntrials = 1000, nx = 20, nrepeats = 1) per model.

Python API mapping (PERF_COMPARISON.md §4)::

    pdf  -> rusty_fitdistcp.<model>.pdf(x, y=...)['cp_pdf']
    cdf  -> rusty_fitdistcp.<model>.cdf(x, y=...)['cp_cdf']
    ppf  -> rusty_fitdistcp.<model>.ppf(x, p=...)['cp_quantiles']
    rvs  -> rusty_fitdistcp.<model>.rvs(n, x)['cp_deviates']
    predict -> rusty_fitdistcp.<model>.ppf(...)   (full pipeline)
    reltest -> rusty_fitdistcp.reltest.reltest_<model>

The reference mode imports the vendored package's *flat* modules (``import
norm``); the Rust mode imports the drop-in submodule API (``import
rusty_fitdistcp.norm``). Both are mapped onto the same workload by the dispatch
helpers below.

Gamma parameterization (PERF_COMPARISON.md §5.4): the reference gamma uses
``(shape, scale)``; the Rust port uses ``(shape, rate)``. The fixture's reltest
``params`` are given in each implementation's native order, so the Python
harness passes them through unchanged. For the pointwise workloads the module
fits its own ``(shape, scale)`` MLE from the same data, so no conversion is
needed there either.

Usage::

    python3 tools/bench/bench_ref_py.py [--impl {ref,rust}] [--out <path>]
                                        [--warmup <n>] [--repeats <n>]
                                        [--inputs <path>]

    --impl     implementation to benchmark: 'ref' (vendored reference) or
               'rust' (Rust-backed drop-in binding). Default: 'ref' (or the
               BENCH_IMPL environment variable if set).
    --out      output CSV path (default: stdout)
    --warmup   discarded warmup iterations (default: 3)
    --repeats  timed iterations (default: 20)
    --inputs   fixture path
               (default: crates/fitdistcp-core/benches/perf_inputs.json)

Reference module: https://github.com/lynnejewson/fitdistcp
    clone: git clone --depth 1 https://github.com/lynnejewson/fitdistcp.git \
               tools/fitdistcp-py-ref
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Where the (gitignored) Python reference module is expected to live.
PY_REF_DIR = "tools/fitdistcp-py-ref"

#: The reference package's inner module directory. The reference modules use
#: *flat* imports (e.g. ``import norm_libs``), so this directory — not its
#: parent — must be on ``sys.path`` and the models imported as top-level
#: modules (``import norm``), not as ``fitdistcp.norm``.
PY_REF_PKG_DIR = os.path.join(PY_REF_DIR, "fitdistcp")

#: Workspace-local Python library (gitignored) holding the reference deps
#: (numpy/scipy/scikit-learn/matplotlib/rusampling). Mirrors the R ``tools/Rlib``
#: approach: the dev container is ephemeral, so deps installed into the image's
#: site-packages do not survive across ``docker compose run`` invocations.
PY_LIB = "tools/pylib"

#: Default fixture path, relative to the workspace root.
DEFAULT_INPUTS = "crates/fitdistcp-core/benches/perf_inputs.json"

#: The two supported implementations and their CSV ``implementation`` labels.
IMPL_REF = "ref"
IMPL_RUST = "rust"
IMPL_LABELS = {IMPL_REF: "py_module", IMPL_RUST: "rust_binding"}

#: The Rust-backed drop-in binding's importable package name.
RUST_PKG = "rusty_fitdistcp"

MODELS = ("norm", "genextreme", "gamma")
SIZES = (50, 500, 5000)
PREDICT_N = 500


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    """Print an actionable error and exit non-zero (never a raw traceback)."""
    sys.stderr.write("bench_ref_py.py: " + message + "\n")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Import the implementation's model modules
# ---------------------------------------------------------------------------


def import_ref_modules():
    """Import the vendored reference model modules (flat imports).

    The reference package's modules use *flat* imports (e.g. ``import
    norm_libs``), so the inner package directory
    ``tools/fitdistcp-py-ref/fitdistcp`` must be on ``sys.path`` and the models
    imported as top-level modules (``import norm``), not as ``fitdistcp.norm``.

    Returns a dict mapping the canonical model name to its module.
    """
    # Make the workspace-local dependency library importable (mirrors R's
    # tools/Rlib). This is where the reference deps are installed so they
    # survive across ephemeral `docker compose run` invocations.
    if os.path.isdir(PY_LIB):
        sys.path.insert(0, os.path.abspath(PY_LIB))

    if not os.path.isdir(PY_REF_PKG_DIR):
        fail(
            "the Python reference package was not found at '{}'.\n"
            "  Clone it first:\n"
            "    git clone --depth 1 "
            "https://github.com/lynnejewson/fitdistcp.git {}\n"
            "  (or pip-install it).".format(PY_REF_PKG_DIR, PY_REF_DIR)
        )

    sys.path.insert(0, os.path.abspath(PY_REF_PKG_DIR))
    modules = {}
    for model in MODELS:
        try:
            modules[model] = __import__(model)
        except ImportError as exc:
            fail(
                "found the cloned repo at '{}' but could not import the "
                "'{}' module: {}\n"
                "  Ensure its dependencies (numpy, scipy, scikit-learn, "
                "matplotlib, rusampling) are installed (Group E).".format(
                    PY_REF_PKG_DIR, model, exc
                )
            )
    return modules


def import_rust_modules():
    """Import the Rust-backed drop-in binding's submodules.

    The drop-in exposes the reference submodule API (``rusty_fitdistcp.norm``
    etc.), so we import the top-level ``rusty_fitdistcp`` package and return its
    submodules.
    The package must be installed (``maturin develop``) in the active
    environment.

    Returns a dict mapping the canonical model name to its submodule.
    """
    try:
        import rusty_fitdistcp  # noqa: F401
    except ImportError as exc:
        fail(
            "could not import the Rust-backed '{}' package: {}\n"
            "  Build/install it first (inside the dev container):\n"
            "    cd crates/fitdistcp-py && maturin develop".format(
                RUST_PKG, exc
            )
        )

    modules = {}
    for model in MODELS:
        try:
            modules[model] = __import__(
                "{}.{}".format(RUST_PKG, model), fromlist=[model]
            )
        except ImportError as exc:
            fail(
                "the Rust-backed '{}' package has no '{}' submodule: {}".format(
                    RUST_PKG, model, exc
                )
            )
    return modules


def import_fitdistcp(impl):
    """Import the model modules for ``impl`` (``ref`` or ``rust``).

    Returns ``(modules, implementation_label)`` where ``modules`` maps the
    canonical model name to its module and ``implementation_label`` is the CSV
    ``implementation`` value (``py_module`` for the reference, ``rust_binding``
    for the Rust-backed drop-in).
    """
    if impl == IMPL_REF:
        return import_ref_modules(), IMPL_LABELS[IMPL_REF]
    if impl == IMPL_RUST:
        return import_rust_modules(), IMPL_LABELS[IMPL_RUST]
    fail("unknown implementation: {!r} (expected 'ref' or 'rust')".format(impl))


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def load_fixture(path: str) -> dict:
    if not os.path.exists(path):
        fail(
            "fixture not found at '{}'.\n"
            "  Pass --inputs <path> or run from the workspace root.".format(path)
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def time_workload(
    warmup: int, repeats: int, work: Callable[[], object]
) -> Dict[str, int]:
    """Run ``work`` ``warmup`` times (discarded) then ``repeats`` times.

    Each repeat is timed with :func:`time.perf_counter_ns`; returns the
    median/min/max in integer nanoseconds.
    """
    for _ in range(warmup):
        work()
    samples: List[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        work()
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    return {
        "median_ns": int(statistics.median(samples)),
        "min_ns": int(samples[0]),
        "max_ns": int(samples[-1]),
    }


# ---------------------------------------------------------------------------
# Model dispatch
# ---------------------------------------------------------------------------


def model_ops(modules: Dict[str, object], model: str) -> Dict[str, Callable]:
    """Return the pointwise operation closures for ``model``.

    Each closure takes the training sample ``y`` and the grid, and returns the
    operation result. The module's ``pdf``/``cdf``/``ppf``/``rvs`` entry points
    fit the MLE internally on every call (the reference API does not expose a
    fit-then-evaluate split), so the timed region includes the per-call fit.
    This matches the reference R harness and is recorded as a fairness caveat
    in the spec.
    """
    mod = modules.get(model)
    if mod is None:
        fail(
            "the Python module has no '{}' module.\n"
            "  Check the cloned repo layout at '{}'.".format(model, PY_REF_PKG_DIR)
        )

    def pdf(y, x):
        return mod.pdf(x, y=y)["cp_pdf"]

    def cdf(y, x):
        return mod.cdf(x, y=y)["cp_cdf"]

    def ppf(y, p):
        return mod.ppf(y, p=p)["cp_quantiles"]

    def rvs(y, u):
        return mod.rvs(len(u), y)["cp_deviates"]

    return {"pdf": pdf, "cdf": cdf, "ppf": ppf, "rvs": rvs}


def predict_op(modules: Dict[str, object], model: str) -> Callable:
    """End-to-end ``predict`` (MLE + DMGS): the full ``ppf`` pipeline."""
    mod = modules.get(model)
    if mod is None:
        fail("the Python module has no '{}' module.".format(model))

    def predict(y, p):
        return mod.ppf(y, p=p)

    return predict


def reltest_op(modules: Dict[str, object], model: str, impl: str) -> Callable:
    """The reltest entry point for ``model`` under implementation ``impl``.

    * ``ref`` — the reference exposes reltest as a *method on each model module*
      (``norm.reltest``, ``gamma.reltest``, ``genextreme.reltest``). The
      per-model signatures differ in the parameter names (``loc``/``scale`` for
      norm, ``a``/``scale`` for gamma, ``xi``/``loc``/``scale`` for genextreme),
      so we return a closure that maps the fixture's native-order ``params``
      onto the right keyword arguments.
    * ``rust`` — the Rust-backed drop-in exposes a single
      ``rusty_fitdistcp.reltest.reltest(model, ntrials, nx, p, true_params, init)``
      entry point; we pass the fixture's native-order ``params`` as both
      ``true_params`` and ``init``.
    """
    # Matched probability grid. The reference Python reltest defaults to
    # p = 0.0001*range(1,10000) (9999 points), which makes each trial's ppf
    # evaluation ~50x more expensive than the reference R reltest default
    # (alpha = seq(0.005, 0.995, 0.005), i.e. 199 points). To keep the workload
    # tractable AND matched across implementations we pass the same 199-point
    # grid the R harness uses (p = 1 - alpha).
    import numpy as np

    p_grid = np.arange(0.005, 0.995, 0.005)

    if impl == IMPL_RUST:
        # The drop-in's reltest lives in the `rusty_fitdistcp.reltest` submodule.
        try:
            reltest_mod = __import__(
                "{}.reltest".format(RUST_PKG), fromlist=["reltest"]
            )
        except ImportError as exc:
            fail(
                "the Rust-backed '{}' package has no 'reltest' submodule: "
                "{}".format(RUST_PKG, exc)
            )
        fn = getattr(reltest_mod, "reltest", None)
        if fn is None:
            fail("the Rust-backed reltest submodule has no 'reltest' entry point.")

        def run_rust(params, ntrials, nx):
            return fn(
                model=model,
                ntrials=ntrials,
                nx=nx,
                p=p_grid,
                true_params=params,
                init=params,
            )

        return run_rust

    # --- reference mode ---
    mod = modules.get(model)
    if mod is None:
        fail("the Python module has no '{}' module.".format(model))
    fn = getattr(mod, "reltest", None)
    if fn is None:
        fail(
            "the Python module '{}' has no 'reltest' entry point.".format(model)
        )

    def run(params, ntrials, nx):
        # plot=False so the harness never opens a figure (headless container).
        if model == "norm":
            # params = [mean, sd]
            return fn(plot=False, ntrials=ntrials, nx=nx, p=p_grid,
                      loc=params[0], scale=params[1])
        if model == "gamma":
            # params = [shape, scale]
            return fn(plot=False, ntrials=ntrials, nx=nx, p=p_grid,
                      a=params[0], scale=params[1])
        if model == "genextreme":
            # params = [loc, scale, shape]
            return fn(plot=False, ntrials=ntrials, nx=nx, p=p_grid,
                      loc=params[0], scale=params[1], xi=params[2])
        fail("unknown model for reltest: {}".format(model))

    return run


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Python fitdistcp performance harness (Group C)."
    )
    parser.add_argument(
        "--impl",
        choices=(IMPL_REF, IMPL_RUST),
        default=os.environ.get("BENCH_IMPL", IMPL_REF),
        help=(
            "implementation to benchmark: 'ref' (vendored reference module) or "
            "'rust' (Rust-backed drop-in binding). Default: 'ref' (or the "
            "BENCH_IMPL environment variable if set)."
        ),
    )
    parser.add_argument("--out", default=None, help="output CSV path (default: stdout)")
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
    parser.add_argument("--repeats", type=int, default=20, help="timed iterations")
    parser.add_argument("--inputs", default=DEFAULT_INPUTS, help="fixture path")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.warmup < 0:
        fail("--warmup must be non-negative")
    if args.repeats < 1:
        fail("--repeats must be positive")

    # The reference gamma derivative code emits RuntimeWarnings (divide-by-zero
    # in log) at the grid boundary; they are benign for timing and would flood
    # stderr. Suppress them.
    import warnings

    warnings.filterwarnings("ignore")

    modules, impl_label = import_fitdistcp(args.impl)
    fx = load_fixture(args.inputs)

    grid_x = list(fx["grid"]["x"])
    grid_p = list(fx["grid"]["p"])
    if not grid_x or not grid_p:
        fail("fixture is missing grid.x / grid.p")

    rows: List[str] = []

    def add_row(model: str, operation: str, n: int, t: Dict[str, int]) -> None:
        rows.append(
            "{},{},{},{},{},{},{},{}".format(
                impl_label,
                model,
                operation,
                n,
                t["median_ns"],
                t["min_ns"],
                t["max_ns"],
                args.repeats,
            )
        )

    # --- pointwise pdf / cdf / ppf / rvs ---
    for model in MODELS:
        ops = model_ops(modules, model)
        # Domain restriction (PERF_COMPARISON.md §2 / §5): the shared grid.x
        # spans [-3, 3], but the gamma distribution has support x > 0. The
        # reference gamma routines require strictly positive x (the Python
        # gamma derivative code raises at x = 0 because log(0) = -inf). To keep
        # the three implementations comparable we evaluate the gamma
        # pdf/cdf/rvs on the strictly-positive subset of the grid (x > 0).
        # norm/genextreme use the full grid. The (model, operation, n) keys are
        # unchanged.
        grid_x_model = [v for v in grid_x if v > 0] if model == "gamma" else grid_x
        for n in SIZES:
            y = list(fx["samples"][model][str(n)])
            if len(y) != n:
                fail(
                    "fixture sample {}/{} has length {} (expected {})".format(
                        model, n, len(y), n
                    )
                )

            add_row(
                model,
                "pdf",
                n,
                time_workload(
                    args.warmup, args.repeats, lambda: ops["pdf"](y, grid_x_model)
                ),
            )
            add_row(
                model,
                "cdf",
                n,
                time_workload(
                    args.warmup, args.repeats, lambda: ops["cdf"](y, grid_x_model)
                ),
            )
            add_row(
                model,
                "ppf",
                n,
                time_workload(args.warmup, args.repeats, lambda: ops["ppf"](y, grid_p)),
            )
            add_row(
                model,
                "rvs",
                n,
                time_workload(
                    args.warmup, args.repeats, lambda: ops["rvs"](y, grid_x_model)
                ),
            )

    # --- end-to-end predict at n = 500 ---
    for model in MODELS:
        y = list(fx["predict"][model]["y"])
        if not y:
            fail("fixture is missing predict.{}.y".format(model))
        op = predict_op(modules, model)
        add_row(
            model,
            "predict",
            PREDICT_N,
            time_workload(args.warmup, args.repeats, lambda: op(y, grid_p)),
        )

    # --- reltest ---
    rt = fx["reltest"]
    rt_ntrials = int(rt["ntrials"])
    rt_nx = int(rt["nx"])
    for model in MODELS:
        params = list(rt["params"][model])
        run = reltest_op(modules, model, args.impl)

        def run_reltest(run=run, params=params):
            return run(params, rt_ntrials, rt_nx)

        # DOCUMENTED LIMITATION: the Rust-backed gamma reltest path reliably
        # panics inside the `special` crate's `inc_gamma` (a pre-existing core
        # robustness bug, out of scope for this harness). Skip that one row with
        # a stderr note rather than aborting the whole run. The reference mode is
        # unaffected (it uses the reference's own gamma reltest).
        try:
            timing = time_workload(args.warmup, args.repeats, run_reltest)
        except BaseException as exc:  # noqa: BLE001 (PanicException is not Exception)
            sys.stderr.write(
                "bench_ref_py.py: skipping reltest for {}/{} ({}: {})\n".format(
                    args.impl, model, type(exc).__name__, exc
                )
            )
            continue

        add_row(model, "reltest", rt_nx, timing)

    # --- emit CSV ---
    header = "implementation,model,operation,n,median_ns,min_ns,max_ns,repeats"
    lines = [header] + rows
    text = "\n".join(lines) + "\n"

    if args.out is None:
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
