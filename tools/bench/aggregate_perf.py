#!/usr/bin/env python3
"""Cross-language performance comparison — results aggregator (Group D).

Merges the CSV outputs of the four harnesses (``rust``, ``ref_r``,
``py_module``, ``rust_binding``) into a single Markdown table plus a combined
CSV, as described in ``crates/fitdistcp-core/benches/PERF_COMPARISON.md``.

The four implementations are:

* ``rust`` — the Rust port, calling the core directly.
* ``ref_r`` — the reference R package.
* ``py_module`` — the vendored reference Python package (pure Python).
* ``rust_binding`` — the Rust-backed Python drop-in binding (compiled Rust
  behind a Python boundary; the drop-in replacement for the reference Python
  package).

Input CSV schema (one file per implementation)::

    implementation,model,operation,n,median_ns,min_ns,max_ns,repeats

Outputs (paths fixed by the spec):

* ``crates/fitdistcp-core/benches/perf_comparison_results.csv``
* ``crates/fitdistcp-core/benches/perf_comparison_results.md`` — one row per
  ``(model, operation, n)`` with columns ``Rust``, ``Reference R``,
  ``Python module``, ``Rust binding``, and ``R/Rust``, ``Py/Rust``,
  ``RustBinding/Rust`` ratios (computed from the ``median_ns`` values).

The Markdown output is prefixed with an environment-capture header (host, image
tag, Rust/R/Python versions, CPU count, and the git SHAs of the workspace and the
two reference repos). Every field degrades gracefully to ``unavailable`` when it
cannot be determined.

Missing implementations are handled gracefully: if, say, the Python module was
not vendored, the table is still produced with the available columns and a note
explaining which implementation(s) were absent.

Usage::

    python3 tools/bench/aggregate_perf.py \\
        [--rust <path>] [--ref-r <path>] [--ref-py <path>] \\
        [--rust-binding <path>] \\
        [--out-md <path>] [--out-csv <path>] \\
        [--workspace <dir>] [--ref-r-repo <dir>] [--ref-py-repo <dir>]

Defaults point at ``tools/bench/out/`` for the four input CSVs and at the
committed ``crates/fitdistcp-core/benches/`` paths for the two outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default directory holding the four harness CSV outputs.
DEFAULT_OUT_DIR = "tools/bench/out"

#: Default input CSV paths (one per implementation).
DEFAULT_RUST_CSV = os.path.join(DEFAULT_OUT_DIR, "perf_rust.csv")
DEFAULT_REF_R_CSV = os.path.join(DEFAULT_OUT_DIR, "perf_ref_r.csv")
DEFAULT_REF_PY_CSV = os.path.join(DEFAULT_OUT_DIR, "perf_py.csv")
DEFAULT_RUST_BINDING_CSV = os.path.join(DEFAULT_OUT_DIR, "perf_rust_binding.csv")

#: Default output paths (fixed by PERF_COMPARISON.md §7).
DEFAULT_OUT_MD = "crates/fitdistcp-core/benches/perf_comparison_results.md"
DEFAULT_OUT_CSV = "crates/fitdistcp-core/benches/perf_comparison_results.csv"

#: Default reference repo locations (gitignored clones).
DEFAULT_WORKSPACE = "."
DEFAULT_REF_R_REPO = "tools/fitdistcp-ref"
DEFAULT_REF_PY_REPO = "tools/fitdistcp-py-ref"

#: The four implementations, in display order, keyed by their CSV
#: ``implementation`` value.
IMPLEMENTATIONS: Tuple[Tuple[str, str], ...] = (
    ("rust", "Rust"),
    ("ref_r", "Reference R"),
    ("py_module", "Python module"),
    ("rust_binding", "Rust binding"),
)

#: Canonical ordering of models and operations for a stable table.
MODEL_ORDER = ("norm", "genextreme", "gamma")
OPERATION_ORDER = ("pdf", "cdf", "ppf", "rvs", "predict", "reltest")

#: The exact CSV header every harness emits.
CSV_HEADER = [
    "implementation",
    "model",
    "operation",
    "n",
    "median_ns",
    "min_ns",
    "max_ns",
    "repeats",
]

UNAVAILABLE = "unavailable"


def warn(message: str) -> None:
    """Print a non-fatal warning to stderr."""
    sys.stderr.write("aggregate_perf.py: warning: " + message + "\n")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_csv(path: str) -> Dict[Tuple[str, str, str, int], int]:
    """Parse one harness CSV into a ``(model, operation, n) -> median_ns`` map.

    The ``implementation`` column is validated against the expected schema but
    is not part of the returned key (each file holds a single implementation).
    Returns an empty map (with a warning) if the file is missing or empty, so a
    partial run still produces a table.
    """
    if not os.path.exists(path):
        warn("input CSV not found: '{}' (treating as absent)".format(path))
        return {}

    rows: Dict[Tuple[str, str, str, int], int] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            warn("input CSV is empty: '{}'".format(path))
            return {}
        missing = [c for c in CSV_HEADER if c not in reader.fieldnames]
        if missing:
            warn(
                "input CSV '{}' is missing column(s) {}; expected schema {}".format(
                    path, ", ".join(missing), ",".join(CSV_HEADER)
                )
            )
            return {}
        for line_no, row in enumerate(reader, start=2):
            try:
                model = row["model"].strip()
                operation = row["operation"].strip()
                n = int(row["n"])
                median_ns = int(row["median_ns"])
            except (KeyError, ValueError, AttributeError) as exc:
                warn(
                    "skipping malformed row {} in '{}': {}".format(line_no, path, exc)
                )
                continue
            key = (model, operation, n)
            if key in rows:
                warn(
                    "duplicate row for {} in '{}'; keeping the last value".format(
                        key, path
                    )
                )
            rows[key] = median_ns
    return rows


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
    """Run ``cmd`` and return stripped stdout, or ``None`` on any failure."""
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text.splitlines()[0].strip() or None


def capture_host() -> str:
    """Best-effort host identifier (hostname, else platform node)."""
    host = _run(["hostname"])
    if host:
        return host
    node = platform.node()
    return node or UNAVAILABLE


def capture_image_tag() -> str:
    """Container image tag, from ``FITDISTCP_IMAGE_TAG`` or the compose image.

    The runner exports ``FITDISTCP_IMAGE_TAG``; when running outside the
    container we fall back to the ``image:`` line of ``docker-compose.yml``.
    """
    env_tag = os.environ.get("FITDISTCP_IMAGE_TAG")
    if env_tag:
        return env_tag
    compose = "docker-compose.yml"
    if os.path.exists(compose):
        try:
            with open(compose, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith("image:"):
                        return stripped.split(":", 1)[1].strip()
        except OSError:
            pass
    return UNAVAILABLE


def capture_rust_version() -> str:
    """Rust compiler version (``rustc --version``)."""
    return _first_line(_run(["rustc", "--version"])) or UNAVAILABLE


def capture_r_version() -> str:
    """R version (``R --version``)."""
    return _first_line(_run(["R", "--version"])) or UNAVAILABLE


def capture_python_version() -> str:
    """Python version (``python3 --version``)."""
    return _first_line(_run(["python3", "--version"])) or UNAVAILABLE


def capture_cpu_count() -> str:
    """Logical CPU count (``os.cpu_count``)."""
    count = os.cpu_count()
    return str(count) if count else UNAVAILABLE


def capture_git_sha(repo_dir: str) -> str:
    """Short git SHA of ``repo_dir``, or ``unavailable`` if not a git repo."""
    if not os.path.isdir(repo_dir):
        return UNAVAILABLE
    sha = _run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"])
    return sha or UNAVAILABLE


def capture_environment(
    workspace: str, ref_r_repo: str, ref_py_repo: str
) -> List[Tuple[str, str]]:
    """Capture the environment header as ordered ``(label, value)`` pairs."""
    return [
        ("Host", capture_host()),
        ("Image tag", capture_image_tag()),
        ("Rust", capture_rust_version()),
        ("R", capture_r_version()),
        ("Python", capture_python_version()),
        ("CPU count", capture_cpu_count()),
        ("Workspace git SHA", capture_git_sha(workspace)),
        ("Reference R git SHA", capture_git_sha(ref_r_repo)),
        ("Python module git SHA", capture_git_sha(ref_py_repo)),
    ]


# ---------------------------------------------------------------------------
# Pivot + formatting
# ---------------------------------------------------------------------------


def sort_key(key: Tuple[str, str, int]) -> Tuple[int, int, int]:
    """Stable ordering: model, then operation, then n."""
    model, operation, n = key
    model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    op_rank = (
        OPERATION_ORDER.index(operation)
        if operation in OPERATION_ORDER
        else len(OPERATION_ORDER)
    )
    return (model_rank, op_rank, n)


def format_ns(value: Optional[int]) -> str:
    """Format a nanosecond median for the Markdown table."""
    if value is None:
        return "—"
    return "{:,}".format(value)


def format_ratio(numerator: Optional[int], denominator: Optional[int]) -> str:
    """Format ``numerator / denominator`` as a 2-decimal ratio, or ``—``."""
    if numerator is None or denominator is None or denominator == 0:
        return "—"
    return "{:.2f}×".format(numerator / denominator)


def build_rows(
    rust: Dict[Tuple[str, str, str, int], int],
    ref_r: Dict[Tuple[str, str, str, int], int],
    ref_py: Dict[Tuple[str, str, str, int], int],
    rust_binding: Dict[Tuple[str, str, str, int], int],
) -> List[
    Tuple[
        Tuple[str, str, int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]
]:
    """Pivot the four maps into one row per ``(model, operation, n)``."""
    keys = set(rust) | set(ref_r) | set(ref_py) | set(rust_binding)
    rows = []
    for key in sorted(keys, key=sort_key):
        rows.append(
            (
                key,
                rust.get(key),
                ref_r.get(key),
                ref_py.get(key),
                rust_binding.get(key),
            )
        )
    return rows


def render_markdown(
    env: List[Tuple[str, str]],
    rows: List[
        Tuple[
            Tuple[str, str, int],
            Optional[int],
            Optional[int],
            Optional[int],
            Optional[int],
        ]
    ],
    present: Dict[str, bool],
) -> str:
    """Render the full Markdown document (header + table + notes)."""
    lines: List[str] = []
    lines.append("# Cross-Language Performance Comparison — Results")
    lines.append("")
    lines.append(
        "Generated by [`tools/bench/aggregate_perf.py`](../../../tools/bench/aggregate_perf.py) "
        "from the four harness CSVs. Workload and methodology are defined in "
        "[`PERF_COMPARISON.md`](PERF_COMPARISON.md)."
    )
    lines.append("")

    # --- environment header ---
    lines.append("## Environment")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for label, value in env:
        lines.append("| {} | {} |".format(label, value))
    lines.append("")

    # --- missing-implementation note ---
    absent = [display for impl, display in IMPLEMENTATIONS if not present.get(impl, False)]
    if absent:
        lines.append(
            "> **Note.** The following implementation(s) were not available and "
            "their columns are omitted/empty: {}.".format(", ".join(absent))
        )
        lines.append("")

    # --- results table ---
    lines.append("## Results")
    lines.append("")
    lines.append(
        "Medians in nanoseconds (lower is better). Ratios are "
        "`reference median / Rust median`; a value > 1 means the reference is "
        "slower than the Rust port."
    )
    lines.append("")
    lines.append(
        "| Model | Operation | n | Rust (ns) | Reference R (ns) | "
        "Python module (ns) | Rust binding (ns) | R/Rust | Py/Rust | "
        "RustBinding/Rust |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (model, operation, n), rust_ns, ref_ns, py_ns, rb_ns in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                model,
                operation,
                n,
                format_ns(rust_ns),
                format_ns(ref_ns),
                format_ns(py_ns),
                format_ns(rb_ns),
                format_ratio(ref_ns, rust_ns),
                format_ratio(py_ns, rust_ns),
                format_ratio(rb_ns, rust_ns),
            )
        )
    lines.append("")

    # --- fairness caveats (mirrors PERF_COMPARISON.md §5) ---
    lines.append("## Fairness caveats")
    lines.append("")
    lines.append(
        "1. **Optimizer algorithm differs.** The reference R package uses "
        "`optim` (Nelder–Mead); the Rust port uses BFGS + Newton. This is an "
        "algorithmic difference that dominates the end-to-end `predict` numbers."
    )
    lines.append(
        "2. **Interpreted vs compiled.** The Python module is pure Python; the "
        "Rust port is compiled. Pointwise operations show a large constant-factor "
        "gap that is a language/runtime effect."
    )
    lines.append(
        "3. **Monte Carlo reltest.** reltest is stochastic; the implementations "
        "are matched on `ntrials`/`nx`/`params` and use a fixed seed where "
        "possible. Residual differences are sampling noise."
    )
    lines.append(
        "4. **Parameterization.** The reference gamma uses `(shape, scale)`; the "
        "Rust port uses `(shape, rate)`. The harness converts so the same "
        "distribution is fit."
    )
    lines.append(
        "5. **Criterion vs wall-clock.** Criterion numbers and the CSV `Instant` "
        "numbers may differ slightly; the CSV is the cross-implementation source."
    )
    lines.append(
        "6. **Gamma domain restriction.** The shared grid.x spans [-3, 3], but "
        "the gamma distribution has support x > 0. The reference R and Python "
        "gamma routines reject non-positive x, so the gamma pdf/cdf/rvs "
        "workloads are evaluated on the strictly-positive subset of the grid "
        "(x > 0, 49 of 100 points) in all three harnesses. ppf and predict are "
        "unaffected; the (model, operation, n) keys are identical."
    )
    lines.append(
        "7. **R timer resolution.** R's proc.time() elapsed has ~1 ms "
        "resolution, so the R harness times a calibrated batch of calls per "
        "sample and divides by the batch size. Sub-millisecond workloads are "
        "therefore quantized to roughly 1 ms / batch_size."
    )
    lines.append(
        "8. **Rust binding boundary.** The Rust binding is compiled Rust behind "
        "a Python boundary (the drop-in replacement for the reference Python "
        "package). Its numbers should sit between the pure-Rust core (`rust`) "
        "and the pure-Python reference (`py_module`): the Python↔Rust call and "
        "marshalling overhead is the gap to `rust`, while the compiled core is "
        "the gap to `py_module`."
    )
    lines.append("")
    return "\n".join(lines)


def render_csv(
    rows: List[
        Tuple[
            Tuple[str, str, int],
            Optional[int],
            Optional[int],
            Optional[int],
            Optional[int],
        ]
    ],
) -> str:
    """Render the combined CSV (one row per ``(model, operation, n)``)."""
    out_lines = [
        "model,operation,n,rust_ns,ref_r_ns,py_module_ns,rust_binding_ns,"
        "r_over_rust,py_over_rust,rust_binding_over_rust"
    ]
    for (model, operation, n), rust_ns, ref_ns, py_ns, rb_ns in rows:
        r_ratio = (
            "{:.6f}".format(ref_ns / rust_ns)
            if ref_ns is not None and rust_ns not in (None, 0)
            else ""
        )
        py_ratio = (
            "{:.6f}".format(py_ns / rust_ns)
            if py_ns is not None and rust_ns not in (None, 0)
            else ""
        )
        rb_ratio = (
            "{:.6f}".format(rb_ns / rust_ns)
            if rb_ns is not None and rust_ns not in (None, 0)
            else ""
        )
        out_lines.append(
            "{},{},{},{},{},{},{},{},{},{}".format(
                model,
                operation,
                n,
                "" if rust_ns is None else rust_ns,
                "" if ref_ns is None else ref_ns,
                "" if py_ns is None else py_ns,
                "" if rb_ns is None else rb_ns,
                r_ratio,
                py_ratio,
                rb_ratio,
            )
        )
    return "\n".join(out_lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the four cross-language harness CSVs into a "
        "Markdown table + combined CSV (Group D)."
    )
    parser.add_argument("--rust", default=DEFAULT_RUST_CSV, help="Rust harness CSV")
    parser.add_argument("--ref-r", default=DEFAULT_REF_R_CSV, help="Reference R harness CSV")
    parser.add_argument("--ref-py", default=DEFAULT_REF_PY_CSV, help="Python module harness CSV")
    parser.add_argument(
        "--rust-binding",
        default=DEFAULT_RUST_BINDING_CSV,
        help="Rust-backed Python binding harness CSV",
    )
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD, help="output Markdown path")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="output CSV path")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="workspace git repo dir")
    parser.add_argument("--ref-r-repo", default=DEFAULT_REF_R_REPO, help="reference R repo dir")
    parser.add_argument("--ref-py-repo", default=DEFAULT_REF_PY_REPO, help="Python module repo dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    rust = parse_csv(args.rust)
    ref_r = parse_csv(args.ref_r)
    ref_py = parse_csv(args.ref_py)
    rust_binding = parse_csv(args.rust_binding)

    present = {
        "rust": bool(rust),
        "ref_r": bool(ref_r),
        "py_module": bool(ref_py),
        "rust_binding": bool(rust_binding),
    }
    if not any(present.values()):
        warn("no input CSVs contained any rows; writing an empty results table")

    rows = build_rows(rust, ref_r, ref_py, rust_binding)
    env = capture_environment(args.workspace, args.ref_r_repo, args.ref_py_repo)

    md = render_markdown(env, rows, present)
    csv_text = render_csv(rows)

    for path, text in ((args.out_md, md), (args.out_csv, csv_text)):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    sys.stderr.write(
        "aggregate_perf.py: wrote {} row(s) to '{}' and '{}'\n".format(
            len(rows), args.out_md, args.out_csv
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
