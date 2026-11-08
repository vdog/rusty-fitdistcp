#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tools/run-perf-comparison.sh
#
# One-command runner for the cross-language performance comparison (Group D).
#
# It builds the Rust port, installs both reference implementations (R deps via
# Posit Public Package Manager; Python deps via pip), runs all three harnesses
# writing CSVs to tools/bench/out/, then invokes the aggregator to produce the
# committed results table.
#
# The project's single source of truth for toolchains is the Docker dev
# container (see Dockerfile / docker-compose.yml). This script is therefore
# intended to be run INSIDE that container:
#
#     docker compose run --rm dev bash tools/run-perf-comparison.sh
#
# (Running it on the host works only if Rust, R, and Python are all installed
# with the same versions as the image; the container is strongly recommended.)
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   tools/run-perf-comparison.sh [--help]
#
# Environment flags (all optional):
#
#   SKIP_BUILD=1        skip `cargo build --release`
#   SKIP_R=1            skip the reference R stage (install + harness)
#   SKIP_PY=1           skip the Python module stage (clone + install + harness)
#   SKIP_AGGREGATE=1    skip the final aggregation step
#
#   WARMUP=<n>          discarded warmup iterations (default: 3)
#   REPEATS=<n>         timed iterations (default: 20)
#   INPUTS=<path>       fixture path
#                       (default: crates/fitdistcp-core/benches/perf_inputs.json)
#
#   PY_REF_REPO=<url>   Python reference clone URL
#                       (default: https://github.com/lynnejewson/fitdistcp.git)
#   PY_REF_REF=<ref>    Python reference git ref/branch/tag to check out
#                       (default: main)
#   PY_REF_DIR=<dir>    where to clone the Python reference
#                       (default: tools/fitdistcp-py-ref)
#
#   R_REF_DIR=<dir>     vendored reference R package dir
#                       (default: tools/fitdistcp-ref)
#   R_LIB=<dir>         workspace-local R library for extra packages
#                       (default: tools/Rlib)
#
#   OUT_DIR=<dir>       directory for the harness CSVs
#                       (default: tools/bench/out)
#
# Outputs (fixed by PERF_COMPARISON.md §7):
#   crates/fitdistcp-core/benches/perf_comparison_results.md
#   crates/fitdistcp-core/benches/perf_comparison_results.csv
#
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Resolve the workspace root (this script lives in tools/) ---------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

# --- Configuration (env-overridable) ----------------------------------------

SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_R="${SKIP_R:-0}"
SKIP_PY="${SKIP_PY:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

WARMUP="${WARMUP:-3}"
REPEATS="${REPEATS:-20}"
INPUTS="${INPUTS:-crates/fitdistcp-core/benches/perf_inputs.json}"

PY_REF_REPO="${PY_REF_REPO:-https://github.com/lynnejewson/fitdistcp.git}"
PY_REF_REF="${PY_REF_REF:-main}"
PY_REF_DIR="${PY_REF_DIR:-tools/fitdistcp-py-ref}"

R_REF_DIR="${R_REF_DIR:-tools/fitdistcp-ref}"
R_LIB="${R_LIB:-tools/Rlib}"

# Workspace-local Python library for the reference module's dependencies
# (mirrors R_LIB; gitignored, survives across ephemeral containers).
PY_LIB="${PY_LIB:-tools/pylib}"

OUT_DIR="${OUT_DIR:-tools/bench/out}"

RUST_CSV="${OUT_DIR}/perf_rust.csv"
REF_R_CSV="${OUT_DIR}/perf_ref_r.csv"
REF_PY_CSV="${OUT_DIR}/perf_py.csv"

# --- Helpers ----------------------------------------------------------------

log()  { printf '\n=== %s ===\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

usage() {
  sed -n '2,80p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

# --- Preflight --------------------------------------------------------------

log "Preflight"
info "workspace root : ${WORKSPACE_ROOT}"
info "output dir     : ${OUT_DIR}"
info "fixture        : ${INPUTS}"
info "warmup/repeats : ${WARMUP}/${REPEATS}"
info "skip flags     : build=${SKIP_BUILD} r=${SKIP_R} py=${SKIP_PY} aggregate=${SKIP_AGGREGATE}"

if [[ ! -f "${INPUTS}" ]]; then
  warn "fixture not found at '${INPUTS}'; the harnesses will fail."
fi

mkdir -p "${OUT_DIR}"

# Export the image tag so the aggregator can record it in the results header.
export FITDISTCP_IMAGE_TAG="${FITDISTCP_IMAGE_TAG:-fitdistcp-dev:latest}"

# --- Stage 1: build the Rust port -------------------------------------------

if [[ "${SKIP_BUILD}" == "1" ]]; then
  log "Stage 1/5: build Rust port — SKIPPED (SKIP_BUILD=1)"
else
  log "Stage 1/5: build Rust port (cargo build --release)"
  cargo build --release -p fitdistcp-core --bin perf_timing
fi

# --- Stage 2: run the Rust harness ------------------------------------------

log "Stage 2/5: run Rust harness -> ${RUST_CSV}"
if [[ "${SKIP_BUILD}" == "1" && ! -x "target/release/perf_timing" ]]; then
  warn "SKIP_BUILD=1 but target/release/perf_timing is missing; skipping Rust harness."
else
  cargo run --release -p fitdistcp-core --bin perf_timing -- \
    --out "${RUST_CSV}" --warmup "${WARMUP}" --repeats "${REPEATS}" --inputs "${INPUTS}"
fi

# --- Stage 3: reference R ---------------------------------------------------

if [[ "${SKIP_R}" == "1" ]]; then
  log "Stage 3/5: reference R — SKIPPED (SKIP_R=1)"
else
  log "Stage 3/5: reference R (install deps + harness)"
  if [[ ! -d "${R_REF_DIR}/R" ]]; then
    warn "reference R package not found at '${R_REF_DIR}/R'; skipping the R stage."
    warn "  Clone it with:"
    warn "    git clone --depth 1 https://github.com/stephenjewson/fitdistcp.git ${R_REF_DIR}"
  else
    mkdir -p "${R_LIB}"
    info "installing R deps (reference package Imports) via P3M into ${R_LIB}"
    # The reference package's DESCRIPTION Imports are:
    #   stats, mev, extraDistr, gnorm, fdrtool, pracma, rust, actuar, fExtremes
    # The harness sources the reference R files directly, so the packages whose
    # symbols those files use unqualified must be installed (extraDistr for the
    # GEV d/p/q/r routines, fExtremes for gevFit, actuar/gnorm/fdrtool for the
    # reltest model dispatch, rust for `ru`). jsonlite is used for fixture
    # parsing when available.
    Rscript -e '
      options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
              paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])));
      options(repos = c(P3M = "https://packagemanager.posit.co/cran/__linux__/resolute/latest"),
              Ncpus = max(1L, parallel::detectCores()));
      lib <- Sys.getenv("R_LIB", unset = "tools/Rlib");
      dir.create(lib, showWarnings = FALSE, recursive = TRUE);
      need <- c("extraDistr", "jsonlite", "mev", "gnorm", "fdrtool",
                "pracma", "rust", "actuar", "fExtremes");
      miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)];
      if (length(miss)) install.packages(miss, lib = lib);
    ' || warn "R dependency installation failed; the R harness may fail."
    info "running R harness -> ${REF_R_CSV}"
    R_LIB="${R_LIB}" Rscript tools/bench/bench_ref_r.R \
      --out "${REF_R_CSV}" --warmup "${WARMUP}" --repeats "${REPEATS}" --inputs "${INPUTS}" \
      || warn "R harness failed; continuing without reference R results."
  fi
fi

# --- Stage 4: Python module -------------------------------------------------

if [[ "${SKIP_PY}" == "1" ]]; then
  log "Stage 4/5: Python module — SKIPPED (SKIP_PY=1)"
else
  log "Stage 4/5: Python module (clone + install deps + harness)"

  # Clone the reference module if it is not already present. The clone is
  # best-effort: if it fails we warn and skip the Python stage rather than
  # aborting the whole run.
  py_available=1
  if [[ ! -d "${PY_REF_DIR}" ]]; then
    info "cloning ${PY_REF_REPO} (ref: ${PY_REF_REF}) -> ${PY_REF_DIR}"
    if git clone --depth 1 --branch "${PY_REF_REF}" "${PY_REF_REPO}" "${PY_REF_DIR}"; then
      info "clone succeeded"
    else
      warn "failed to clone the Python reference from '${PY_REF_REPO}' (ref '${PY_REF_REF}')."
      warn "  Set PY_REF_REPO / PY_REF_REF to override, or vendor it manually at '${PY_REF_DIR}'."
      warn "  Skipping the Python stage."
      py_available=0
    fi
  else
    info "Python reference already present at '${PY_REF_DIR}'"
  fi

  if [[ "${py_available}" == "1" ]]; then
    # Install the reference module's dependencies into a workspace-local
    # directory (tools/pylib, gitignored) so they survive across the ephemeral
    # `docker compose run` containers. Installing into the image's
    # site-packages would be lost when the container exits. This mirrors the R
    # tools/Rlib approach. The reference pyproject dependencies are:
    #   numpy, scipy, scikit-learn, matplotlib, rusampling
    mkdir -p "${PY_LIB}"
    info "installing Python deps (numpy, scipy, scikit-learn, matplotlib, rusampling) into ${PY_LIB}"
    python3 -m pip install --no-cache-dir --break-system-packages --target "${PY_LIB}" \
      numpy scipy scikit-learn matplotlib rusampling \
      || warn "pip install of Python deps failed; the Python harness may fail."
    info "running Python harness -> ${REF_PY_CSV}"
    PYTHONPATH="${WORKSPACE_ROOT}/${PY_LIB}" python3 tools/bench/bench_ref_py.py \
      --out "${REF_PY_CSV}" --warmup "${WARMUP}" --repeats "${REPEATS}" --inputs "${INPUTS}" \
      || warn "Python harness failed; continuing without Python module results."
  fi
fi

# --- Stage 5: aggregate -----------------------------------------------------

if [[ "${SKIP_AGGREGATE}" == "1" ]]; then
  log "Stage 5/5: aggregate — SKIPPED (SKIP_AGGREGATE=1)"
else
  log "Stage 5/5: aggregate results"
  python3 tools/bench/aggregate_perf.py \
    --rust "${RUST_CSV}" \
    --ref-r "${REF_R_CSV}" \
    --ref-py "${REF_PY_CSV}" \
    --workspace "${WORKSPACE_ROOT}" \
    --ref-r-repo "${R_REF_DIR}" \
    --ref-py-repo "${PY_REF_DIR}"
fi

log "Done"
info "CSVs   : ${RUST_CSV} ${REF_R_CSV} ${REF_PY_CSV}"
info "results: crates/fitdistcp-core/benches/perf_comparison_results.md"
info "results: crates/fitdistcp-core/benches/perf_comparison_results.csv"
