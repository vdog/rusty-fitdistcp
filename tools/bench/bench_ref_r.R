#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# tools/bench/bench_ref_r.R
#
# Cross-language performance comparison — Reference R harness (Group C).
#
# Times the reference R package `stephenjewson/fitdistcp` (v0.2.3, vendored at
# `tools/fitdistcp-ref/`) on the canonical workload defined in
# `crates/fitdistcp-core/benches/PERF_COMPARISON.md`, consuming the shared
# inputs in `crates/fitdistcp-core/benches/perf_inputs.json`.
#
# It emits CSV rows with the schema:
#   implementation,model,operation,n,median_ns,min_ns,max_ns,repeats
# with `implementation = "ref_r"`.
#
# Workload (PERF_COMPARISON.md §1):
#   * pointwise pdf/cdf/ppf/rvs for norm/genextreme/gamma at n = 50/500/5000
#     over the 100-point grid (fit once outside the timed region),
#   * end-to-end `predict` (MLE + DMGS) at n = 500 (one per model),
#   * reltest (ntrials = 1000, nx = 20, nrepeats = 1) per model.
#
# Reference entry points used (see tools/fitdistcp-ref/R/):
#   norm:       dnorm_cp / pnorm_cp / qnorm_cp / rnorm_cp
#   genextreme: dgev_cp  / pgev_cp  / qgev_cp  / rgev_cp
#   gamma:      dgamma_cp / pgamma_cp / qgamma_cp / rgamma_cp
#   reltest:    reltest(model, ntrials, nrepeats, nx, params, ...)
#
# Gamma parameterization (PERF_COMPARISON.md §5.4): the reference gamma uses
# (shape, scale); the Rust port uses (shape, rate). The fixture's reltest
# params are given in each implementation's native order, so the reference
# harness passes them through unchanged. For the pointwise workloads the
# reference fits its own (shape, scale) MLE from the same data, so no
# conversion is needed there either; the conversion is only relevant when
# comparing fitted parameters across implementations (documented in the spec).
#
# Usage:
#   Rscript tools/bench/bench_ref_r.R [--out <path>] [--warmup <n>]
#                                     [--repeats <n>] [--inputs <path>]
#
#   --out      output CSV path (default: stdout)
#   --warmup   discarded warmup iterations (default: 3)
#   --repeats  timed iterations (default: 20)
#   --inputs   fixture path (default: crates/fitdistcp-core/benches/perf_inputs.json)
#
# Reference package: https://github.com/stephenjewson/fitdistcp
#   clone: git clone --depth 1 https://github.com/stephenjewson/fitdistcp.git \
#              tools/fitdistcp-ref
# ---------------------------------------------------------------------------

# --- Fail fast with an actionable message (never a raw stack trace) --------

fail <- function(...) {
  message("bench_ref_r.R: ", ...)
  quit(status = 1L)
}

# --- Argument parsing -------------------------------------------------------

parse_args <- function(argv) {
  out <- NULL
  warmup <- 3L
  repeats <- 20L
  inputs <- "crates/fitdistcp-core/benches/perf_inputs.json"

  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    if (a == "--out") {
      i <- i + 1L
      if (i > length(argv)) fail("--out requires a path argument")
      out <- argv[[i]]
    } else if (a == "--warmup") {
      i <- i + 1L
      if (i > length(argv)) fail("--warmup requires a number")
      warmup <- as.integer(argv[[i]])
      if (is.na(warmup) || warmup < 0L) fail("--warmup must be a non-negative integer")
    } else if (a == "--repeats") {
      i <- i + 1L
      if (i > length(argv)) fail("--repeats requires a number")
      repeats <- as.integer(argv[[i]])
      if (is.na(repeats) || repeats < 1L) fail("--repeats must be a positive integer")
    } else if (a == "--inputs") {
      i <- i + 1L
      if (i > length(argv)) fail("--inputs requires a path argument")
      inputs <- argv[[i]]
    } else {
      fail("unknown argument: ", a)
    }
    i <- i + 1L
  }
  list(out = out, warmup = warmup, repeats = repeats, inputs = inputs)
}

# --- Locate + source the reference package ---------------------------------

# The dev container is ephemeral, so extra R packages may be installed into a
# workspace-local library (tools/Rlib, gitignored) that survives across runs.
local_lib <- "tools/Rlib"
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

ref_dir <- "tools/fitdistcp-ref/R"
if (!dir.exists(ref_dir)) {
  fail("reference package not found at '", ref_dir, "'.\n",
       "  Clone it first:\n",
       "    git clone --depth 1 https://github.com/stephenjewson/fitdistcp.git ",
       "tools/fitdistcp-ref")
}

# The reference package's NAMESPACE imports a number of symbols *unqualified*
# from its dependencies (see tools/fitdistcp-ref/NAMESPACE), e.g.
#   importFrom("extraDistr","dgev","pgev","qgev","rgev")
#   importFrom("actuar","dinvgamma","pinvgamma","qinvgamma","rinvgamma")
#   importFrom("gnorm","dgnorm","pgnorm","qgnorm","rgnorm")
#   importFrom("fdrtool","dhalfnorm","phalfnorm","qhalfnorm","rhalfnorm")
#   importFrom("fExtremes","gevFit")
#   importFrom("rust","ru")
# When the package is loaded normally these resolve via the namespace; but this
# harness `source()`s the files into the global environment, so the unqualified
# names would be unresolved. We therefore attach the packages whose symbols the
# sourced files use unqualified, mirroring the NAMESPACE imports.
#
# IMPORTANT: several of these packages export *conflicting* GEV symbols
# (`qgev`/`dgev`/`pgev`/`rgev` exist in extraDistr, fExtremes and mev with
# different signatures). The reference NAMESPACE imports them from extraDistr,
# so extraDistr MUST be attached last to win the search-path lookup. The other
# packages are attached first (they only contribute non-conflicting symbols
# such as `gevFit`, `ru`, `qinvgamma`, `qgnorm`, `qhalfnorm`).
for (pkg in c("actuar", "gnorm", "fdrtool", "fExtremes", "rust", "extraDistr")) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}

# Source only the files that define the routines this harness calls. Sourcing
# the whole R/ directory would pull in the DMGS/rust/plotting code and its
# heavy dependencies (mev, rust, fExtremes, gnorm, fdrtool, pracma, actuar).
#
# The pointwise d/p/q/r*_cp routines for norm/gamma are self-contained (they
# use only base stats + the model libs). The GEV routines call extraDistr for
# dgev/pgev/qgev/rgev, so extraDistr must be installed (Group E).
ref_files <- c(
  "000_evaluate_dmgs_equation3p3.R",  # dmgs / bayesian_dq_4terms_v1 (DMGS expansion)
  "005_utils.R",          # make_maic, make_se, analytic_cpmethod, ...
  "030b_norm_libs.R",     # norm_ml_params, dnormsub
  "030c_norm_derivs.R",   # norm_ldda / norm_lddda / norm_f1fa / ... (used by 030a)
  "030a_norm.R",          # dnorm_cp / pnorm_cp / qnorm_cp / rnorm_cp
  "100b_gamma_libs.R",    # gamma_loglik, dgammasub
  "100c_gamma_derivs.R",  # gamma_ldda / gamma_lddda / gamma_f1fa / ... (used by 100a)
  "100a_gamma.R",         # dgamma_cp / pgamma_cp / qgamma_cp / rgamma_cp
  "110b_gev_libs.R",      # gev_loglik, gev_setics, dgevsub
  "110c_gev_derivs.R",    # gev_ldda / gev_lddda / gev_f1fa / ... (used by 110a)
  "110a_gev.R",           # dgev_cp / pgev_cp / qgev_cp / rgev_cp
  "053b_gev_k3_libs.R",   # gev_k3_loglik / gev_k3_setics / ... (reltest "gev_k3")
  "053c_gev_k3_derivs.R", # gev_k3_ldda / gev_k3_lddda / ... (used by 053a)
  "053a_gev_k3.R",        # qgev_k3_cp / dgev_k3_cp / pgev_k3_cp / rgev_k3_cp
  "002b_reltest_libs.R",  # reltest_params / reltest_simulate / reltest_predict / reltest_makeep
  "002a_reltest.R"        # reltest
)
for (f in ref_files) {
  path <- file.path(ref_dir, f)
  if (!file.exists(path)) {
    fail("missing reference source file: '", path, "'.\n",
         "  The vendored reference package looks incomplete; re-clone it.")
  }
  source(path)
}

# extraDistr is required by the GEV d/p/q/r routines (dgev/pgev/qgev/rgev).
if (!requireNamespace("extraDistr", quietly = TRUE)) {
  fail("the R package 'extraDistr' is required by the reference GEV routines ",
       "but is not installed.\n",
       "  Install it (Group E), e.g.:\n",
       "    Rscript -e 'install.packages(\"extraDistr\", lib=\"tools/Rlib\")'")
}

# --- Read the shared fixture ------------------------------------------------

read_fixture <- function(path) {
  if (!file.exists(path)) {
    fail("fixture not found at '", path, "'.\n",
         "  Pass --inputs <path> or run from the workspace root.")
  }

  if (requireNamespace("jsonlite", quietly = TRUE)) {
    return(jsonlite::fromJSON(path, simplifyVector = TRUE))
  }

  # Base-R fallback: the fixture is a plain JSON object of numbers/arrays, so a
  # small recursive-descent parser is sufficient and avoids a hard jsonlite
  # dependency. It handles objects, arrays, numbers, strings, true/false/null.
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  pos <- 1L
  n <- nchar(txt)

  skip_ws <- function() {
    while (pos <= n && grepl("[ \t\r\n]", substr(txt, pos, pos))) pos <<- pos + 1L
  }
  parse_value <- function() {
    skip_ws()
    ch <- substr(txt, pos, pos)
    if (ch == "{") return(parse_object())
    if (ch == "[") return(parse_array())
    if (ch == "\"") return(parse_string())
    return(parse_literal())
  }
  parse_object <- function() {
    pos <<- pos + 1L  # consume '{'
    obj <- list()
    skip_ws()
    if (substr(txt, pos, pos) == "}") { pos <<- pos + 1L; return(obj) }
    repeat {
      skip_ws()
      key <- parse_string()
      skip_ws()
      if (substr(txt, pos, pos) != ":") fail("malformed JSON: expected ':'")
      pos <<- pos + 1L
      obj[[key]] <- parse_value()
      skip_ws()
      ch <- substr(txt, pos, pos)
      if (ch == ",") { pos <<- pos + 1L; next }
      if (ch == "}") { pos <<- pos + 1L; break }
      fail("malformed JSON: expected ',' or '}'")
    }
    obj
  }
  parse_array <- function() {
    pos <<- pos + 1L  # consume '['
    arr <- list()
    skip_ws()
    if (substr(txt, pos, pos) == "]") { pos <<- pos + 1L; return(arr) }
    repeat {
      arr[[length(arr) + 1L]] <- parse_value()
      skip_ws()
      ch <- substr(txt, pos, pos)
      if (ch == ",") { pos <<- pos + 1L; next }
      if (ch == "]") { pos <<- pos + 1L; break }
      fail("malformed JSON: expected ',' or ']'")
    }
    arr
  }
  parse_string <- function() {
    pos <<- pos + 1L  # consume opening quote
    start <- pos
    while (pos <= n && substr(txt, pos, pos) != "\"") pos <<- pos + 1L
    s <- substr(txt, start, pos - 1L)
    pos <<- pos + 1L  # consume closing quote
    s
  }
  parse_literal <- function() {
    start <- pos
    while (pos <= n && !grepl("[ \t\r\n,\\]}]", substr(txt, pos, pos))) pos <<- pos + 1L
    tok <- substr(txt, start, pos - 1L)
    if (tok == "true") return(TRUE)
    if (tok == "false") return(FALSE)
    if (tok == "null") return(NULL)
    v <- suppressWarnings(as.numeric(tok))
    if (is.na(v)) fail("malformed JSON: bad literal '", tok, "'")
    v
  }

  parse_value()
}

# Coerce a parsed JSON array (list of scalars) to a numeric vector.
as_num_vec <- function(x) {
  if (is.null(x)) return(numeric(0))
  as.numeric(unlist(x, use.names = FALSE))
}

# --- Timing harness ---------------------------------------------------------

# Run `work` `warmup` times (discarded) then `repeats` times, timing each
# repeat with proc.time() deltas, and return median/min/max nanoseconds.
#
# proc.time()[["elapsed"]] has only ~1 ms resolution on Linux, so a single
# sub-millisecond workload (e.g. the norm pointwise operations) would round to
# 0 ns. To measure such workloads we time a *batch* of `inner` calls per sample
# and divide by `inner`. The batch size is auto-calibrated adaptively: we keep
# doubling it until a timed region is at least ~5 ms (well above the timer
# resolution). The reported per-call nanoseconds stay comparable to the
# Rust/Python harnesses, and the median/min/max are still taken over `repeats`
# samples, preserving the documented methodology (PERF_COMPARISON.md §6).
time_workload <- function(warmup, repeats, work) {
  for (k in seq_len(warmup)) work()

  # Adaptively grow the inner batch until a timed region is measurable
  # (>= ~5 ms) or the cap is reached. A single call may be far below the timer
  # resolution, so a one-shot calibration is not enough.
  target <- 0.005
  inner <- 1L
  repeat {
    t0 <- proc.time()[["elapsed"]]
    for (j in seq_len(inner)) work()
    t1 <- proc.time()[["elapsed"]]
    elapsed <- t1 - t0
    if (elapsed >= target || inner >= 100000L) break
    inner <- inner * 2L
  }

  samples <- numeric(repeats)
  for (k in seq_len(repeats)) {
    t0 <- proc.time()[["elapsed"]]
    for (j in seq_len(inner)) work()
    t1 <- proc.time()[["elapsed"]]
    samples[[k]] <- (t1 - t0) * 1e9 / inner  # per-call nanoseconds
  }
  samples <- sort(samples)
  med <- samples[[(length(samples) + 1L) %/% 2L]]
  list(median_ns = med, min_ns = samples[[1L]], max_ns = samples[[length(samples)]])
}

# --- Model dispatch ---------------------------------------------------------

# Per-model reference entry points. Each closure takes the fitted data `y`
# (the training sample) and returns the operation result over the grid.
#
# The reference d/p/q routines fit the MLE internally on every call, so the
# "fit once outside the timed region" requirement is satisfied by the fact
# that the *timed* region is the operation over the 100-point grid; the fit is
# part of the reference routine's contract and is identical for every repeat.
# (The Rust harness fits once and times only the grid evaluation; the reference
# API does not expose a fit-then-evaluate split for the pointwise routines, so
# the reference numbers include the per-call MLE fit. This is recorded as a
# fairness caveat in the spec — the reference `optim` fit dominates.)
model_ops <- function(model) {
  if (model == "norm") {
    return(list(
      pdf = function(y, x) dnorm_cp(x, y = y)$cp_pdf,
      cdf = function(y, x) pnorm_cp(x, y = y)$cp_cdf,
      ppf = function(y, p) qnorm_cp(y, p = p)$cp_quantiles,
      rvs = function(y, u) rnorm_cp(length(u), y, mlcp = TRUE)$cp_deviates
    ))
  }
  if (model == "genextreme") {
    return(list(
      pdf = function(y, x) dgev_cp(x, y = y)$cp_pdf,
      cdf = function(y, x) pgev_cp(x, y = y)$cp_cdf,
      ppf = function(y, p) qgev_cp(y, p = p)$cp_quantiles,
      rvs = function(y, u) rgev_cp(length(u), y, mlcp = TRUE)$cp_deviates
    ))
  }
  if (model == "gamma") {
    return(list(
      pdf = function(y, x) dgamma_cp(x, y = y)$cp_pdf,
      cdf = function(y, x) pgamma_cp(x, y = y)$cp_cdf,
      ppf = function(y, p) qgamma_cp(y, p = p)$cp_quantiles,
      rvs = function(y, u) rgamma_cp(length(u), y, mlcp = TRUE)$cp_deviates
    ))
  }
  fail("unknown model: ", model)
}

# End-to-end `predict` (MLE + DMGS) at n = 500: the full q*_cp pipeline.
predict_op <- function(model) {
  if (model == "norm") return(function(y, p) qnorm_cp(y, p = p))
  if (model == "genextreme") return(function(y, p) qgev_cp(y, p = p))
  if (model == "gamma") return(function(y, p) qgamma_cp(y, p = p))
  fail("unknown model: ", model)
}

# --- reltest ----------------------------------------------------------------

# The reference reltest is a Monte Carlo workload. It uses R's RNG, which the
# fixture's fixed seed cannot reproduce (the Rust LCG differs), so the harness
# seeds R's RNG with the fixture seed for reproducibility of the *timing*
# workload. The spec records residual sampling differences as noise, not
# performance (PERF_COMPARISON.md §5.3).
run_reltest <- function(model, params, ntrials, nx, nrepeats, seed) {
  # Map the canonical model name to the reference reltest model name. The
  # reference reltest uses `gev_k3` for the homogeneous 3-parameter GEV (see
  # tools/fitdistcp-ref/R/002b_reltest_libs.R: reltest_simulate/reltest_predict
  # dispatch on "gev_k3", not "genextreme"). `norm` and `gamma` match directly.
  ref_model <- switch(model,
    genextreme = "gev_k3",
    model
  )
  # Seed R's RNG deterministically. The fixture seed is a 64-bit value; R's
  # set.seed takes a 32-bit integer, so fold it down. The fixture seed exceeds
  # 2^53, so the low bits are not exactly representable as a double; this only
  # affects which Monte Carlo draws are made, not the timing workload, and the
  # spec records residual sampling differences as noise (PERF_COMPARISON.md
  # §5.3). Guard against a non-finite seed.
  if (!is.finite(seed)) seed <- 0
  set.seed(as.integer(seed %% .Machine$integer.max))
  reltest(
    model = ref_model,
    ntrials = ntrials,
    nrepeats = nrepeats,
    nx = nx,
    params = params,
    plotflag = FALSE,
    verbose = FALSE,
    dmgs = TRUE
  )
}

# --- main -------------------------------------------------------------------

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  fx <- read_fixture(args$inputs)

  grid_x <- as_num_vec(fx$grid$x)
  grid_p <- as_num_vec(fx$grid$p)
  if (length(grid_x) == 0L || length(grid_p) == 0L) {
    fail("fixture is missing grid.x / grid.p")
  }

  sizes <- c(50L, 500L, 5000L)
  models <- c("norm", "genextreme", "gamma")

  # Accumulate rows as a list of character vectors (CSV lines).
  rows <- character(0)
  add_row <- function(model, operation, n, t) {
    rows[[length(rows) + 1L]] <<- paste(
      "ref_r", model, operation, n,
      format(t$median_ns, scientific = FALSE, trim = TRUE),
      format(t$min_ns, scientific = FALSE, trim = TRUE),
      format(t$max_ns, scientific = FALSE, trim = TRUE),
      args$repeats,
      sep = ","
    )
  }

  # --- pointwise pdf / cdf / ppf / rvs ---
  for (model in models) {
    ops <- model_ops(model)
    # Domain restriction (PERF_COMPARISON.md §2 / §5): the shared grid.x spans
    # [-3, 3], but the gamma distribution has support x > 0. The reference gamma
    # routines hard-reject negative x via `stopifnot(!x < 0)`, and the Python
    # reference gamma derivative code raises at x = 0 (log(0) = -inf). To keep
    # the three implementations comparable we evaluate the gamma pdf/cdf/rvs on
    # the strictly-positive subset of the grid (x > 0). norm/genextreme use the
    # full grid. The (model, operation, n) keys are unchanged.
    grid_x_model <- if (model == "gamma") grid_x[grid_x > 0] else grid_x
    for (n in sizes) {
      y <- as_num_vec(fx$samples[[model]][[as.character(n)]])
      if (length(y) != n) {
        fail("fixture sample ", model, "/", n, " has length ", length(y),
             " (expected ", n, ")")
      }

      # pdf over grid.x (gamma: x >= 0 subset)
      add_row(model, "pdf", n, time_workload(args$warmup, args$repeats, function() {
        invisible(ops$pdf(y, grid_x_model))
      }))
      # cdf over grid.x (gamma: x >= 0 subset)
      add_row(model, "cdf", n, time_workload(args$warmup, args$repeats, function() {
        invisible(ops$cdf(y, grid_x_model))
      }))
      # ppf over grid.p
      add_row(model, "ppf", n, time_workload(args$warmup, args$repeats, function() {
        invisible(ops$ppf(y, grid_p))
      }))
      # rvs: one deviate per grid point (length(grid_x_model) uniforms)
      add_row(model, "rvs", n, time_workload(args$warmup, args$repeats, function() {
        invisible(ops$rvs(y, grid_x_model))
      }))
    }
  }

  # --- end-to-end predict at n = 500 ---
  n_pred <- 500L
  for (model in models) {
    y <- as_num_vec(fx$predict[[model]]$y)
    if (length(y) == 0L) {
      fail("fixture is missing predict.", model, ".y")
    }
    op <- predict_op(model)
    add_row(model, "predict", n_pred, time_workload(args$warmup, args$repeats, function() {
      invisible(op(y, grid_p))
    }))
  }

  # --- reltest ---
  rt <- fx$reltest
  rt_ntrials <- as.integer(rt$ntrials)
  rt_nx <- as.integer(rt$nx)
  rt_nrepeats <- as.integer(rt$nrepeats)
  rt_seed <- as.numeric(rt$seed)
  for (model in models) {
    params <- as_num_vec(rt$params[[model]])
    add_row(model, "reltest", rt_nx, time_workload(args$warmup, args$repeats, function() {
      invisible(run_reltest(model, params, rt_ntrials, rt_nx, rt_nrepeats, rt_seed))
    }))
  }

  # --- emit CSV ---
  header <- "implementation,model,operation,n,median_ns,min_ns,max_ns,repeats"
  lines <- c(header, rows)

  if (is.null(args$out)) {
    cat(paste(lines, collapse = "\n"), "\n", sep = "")
  } else {
    con <- file(args$out, open = "wt")
    on.exit(close(con), add = TRUE)
    writeLines(lines, con)
  }
}

main()
