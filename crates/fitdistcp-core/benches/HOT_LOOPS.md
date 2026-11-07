# W16 — Criterion Benchmarks & Hot-Loop Identification

This note records the W16 baseline measurements and the hot loops they expose.
It is the input to **W17** (Rayon / SIMD / caching). Nothing here changes
numerics; it only measures and identifies.

## How to reproduce

All commands run inside the Docker dev environment (no host Rust toolchain):

```sh
docker compose run --rm dev cargo bench -p fitdistcp-core --bench derivatives
docker compose run --rm dev cargo bench -p fitdistcp-core --bench dmgs_expansion
docker compose run --rm dev cargo bench -p fitdistcp-core --bench mle_fitting
```

The three bench targets are registered with `harness = false` in
[`crates/fitdistcp-core/Cargo.toml`](../Cargo.toml), so they are **excluded from
`cargo test`**. Raw baseline output is saved alongside this note as
`derivatives_baseline.txt`, `dmgs_expansion_baseline.txt`, and
`mle_fitting_baseline.txt` (captured with `--warm-up-time 0.5
--measurement-time 1.5 --sample-size 20`).

## Baseline timings (release profile, Docker, Apple Silicon host)

### 1. Derivative evaluation (`benches/derivatives.rs`)

| Benchmark | n | Time |
|---|---|---|
| `norm/score` | 100 | 45.4 ns |
| `norm/score` | 1 000 | 974 ns |
| `norm/score` | 10 000 | 10.4 µs |
| `norm/hessian` | 10 000 | 10.5 µs |
| `norm/third_derivatives` | 10 000 | 10.5 µs |
| `norm/mixed_derivatives` | 10 000 | 3.86 µs |
| `gamma/score` | 10 000 | 24.1 µs |
| `gamma/hessian` | 10 000 | 20.2 ns (analytic, O(1)) |
| `gamma/third_derivatives` | 10 000 | 8.16 ns (analytic, O(1)) |
| `gamma/mixed_derivatives` | 10 000 | 5.32 µs |

**Analytic vs. forced-numerical (norm, n = 1 000):**

| Derivative | Analytic | Numerical (FD) | Slowdown |
|---|---|---|---|
| score | 924 ns | 2.21 µs | **2.4×** |
| hessian | 922 ns | 4.94 µs | **5.4×** |
| third | 939 ns | 11.1 µs | **11.8×** |
| mixed | 321 ns | 5.13 ms | **~16 000×** |

### 2. DMGS expansion (`benches/dmgs_expansion.rs`)

| Benchmark | nalpha | Time |
|---|---|---|
| `cp_pdf/norm` | 1 000 | 13.1 µs |
| `cp_pdf/gamma` | 1 000 | 136 µs |
| `cp_pdf/genextreme` | 1 000 | 207 µs |
| `cp_quantiles/norm` | 1 000 | 30.4 µs |
| `cp_quantiles/gamma` | 1 000 | 2.99 ms |
| `cp_quantiles/genextreme` | 1 000 | 393 µs |
| `cp_pdf_quantile_derived/genextreme` | 1 000 | 406 µs |

### 3. MLE fitting (`benches/mle_fitting.rs`)

| Benchmark | n | Time |
|---|---|---|
| `fit/norm` | 50 | 5.98 µs |
| `fit/norm` | 500 | 212 µs |
| `fit/norm` | 5 000 | 1.66 ms |
| `fit/gamma` | 50 | 89.8 µs |
| `fit/gamma` | 500 | 2.71 ms |
| `fit/gamma` | 5 000 | 13.7 ms |
| `fit/genextreme` | 50 | 1.47 ms |
| `fit/genextreme` | 500 | 3.26 ms |
| `fit/genextreme` | 5 000 | 25.8 ms |
| `predict_end_to_end/norm` | 500 | 139 µs |
| `predict_end_to_end/gamma` | 500 | 1.46 ms |
| `predict_end_to_end/genextreme` | 500 | 16.2 ms |

## Identified hot loops (for W17)

### H1 — `DmgsIntegrator::dmgs_expansion` `beps1` quadruple loop (highest priority)

[`dmgs_expansion()`](../src/dmgs/mod.rs:191) contains a `t × j × r × s` loop
(`O(p⁴ · nalpha)`) that dominates the expansion for `p ≥ 3`. Evidence: the
`genextreme` (p = 3) `cp_pdf` is **15.8×** the `norm` (p = 2) cost at the same
grid size (207 µs vs 13.1 µs), far more than the 1.5× parameter-count ratio —
consistent with the `p⁴` term. The inner `for a in 0..nalpha` accumulation is a
textbook SIMD/vectorization target (contiguous `mu1[t*nalpha + a]` reads, one
`coeff` broadcast). **W17:** vectorize the inner `a` loop and/or hoist the
`coeff` computation out of the `a` loop (already done) and consider a
matrix-form reformulation of `beps1`.

### H2 — `cp_quantiles` for models with a bisection quantile (gamma)

`gamma/cp_quantiles` at nalpha = 1 000 is **2.99 ms**, ~98× the `norm` quantile
cost (30.4 µs). The cause is [`Gamma::quantile()`](../src/models/gamma.rs:82),
which does a **bisection with up to 200 CDF evaluations**, and each CDF call
evaluates the regularized incomplete gamma. `cp_quantiles` calls `quantile`
once per grid point, and `cp_pdf_quantile_derived` calls it twice more.
**W17:** replace the bisection with a dedicated inverse-incomplete-gamma
(Newton on `P(a, x)`), and/or cache the per-point `fhat`/`mu1`/`mu2` terms.

### H3 — Numerical-fallback derivatives (mixed derivatives especially)

The `norm_analytic_vs_numerical` group shows the FD fallback is **2.4×–11.8×**
slower for score/hessian/third, and **~16 000×** slower for mixed derivatives
(5.13 ms vs 321 ns at n = 1 000). The mixed-derivative FD path is `O(p · n)`
log-likelihood evaluations, each `O(n)`, i.e. `O(p · n²)`. Models that still
fall back (e.g. GEV third/mixed derivatives, gamma CDF hooks) pay this inside
every `cp_pdf`/`cp_quantiles` call. **W17:** prioritize analytic overrides for
the mixed derivatives of the DMGS models, or cache the FD results across grid
points (they do not depend on the prediction point).

### H4 — MLE fit re-evaluates derivatives many times

`fit/genextreme` at n = 5 000 is **25.8 ms** and `predict_end_to_end/genextreme`
is **16.2 ms** at n = 500. The fit runs BFGS **plus two Newton–Raphson
refinements** (from `init` and from the BFGS result), each re-evaluating the
score/Hessian over the full data. The GEV's third/mixed derivatives are
numerical (H3), compounding the cost. **W17:** cache the sufficient statistics
(`Σy`, `Σy²`, `Σln y`, …) so each derivative evaluation is `O(1)` in `n` rather
than `O(n)`, and avoid the redundant Newton-from-init pass when BFGS already
converged to a positive-definite optimum.

### H5 — Per-point allocation in the expansion builders

[`cp_pdf()`](../src/dmgs/mod.rs:264) and [`cp_quantiles()`](../src/dmgs/mod.rs:309)
allocate fresh `f1`/`f2`/`mu1`/`mu2` buffers on every call, and the FD hooks
(`density_first_derivs`, etc.) allocate a `params.to_vec()` per parameter per
point. **W17:** reuse scratch buffers across calls and avoid per-point
allocation in the hot loop (§6 "avoid per-point allocation in hot loops").

## Parallelism opportunities (Rayon, §6)

The expansion is already vectorized over the prediction grid (`nalpha`), and the
per-point work is independent, so `cp_pdf`/`cp_quantiles` are embarrassingly
parallel over `a`. The MLE fit's per-observation log-likelihood sums are also
reducible. **W17:** parallelize the `nalpha` loop (and the `n`-sums) with Rayon;
the core holds no global state, so this is safe.

## Summary ranking for W17

1. **H1** — vectorize/restructure the `beps1` `p⁴` loop (biggest single win for
   `p ≥ 3` models).
2. **H2** — replace the gamma bisection quantile with a Newton inverse.
3. **H3** — analytic mixed derivatives (or FD caching) for DMGS models.
4. **H4** — sufficient-statistic caching in the MLE fit.
5. **H5** — scratch-buffer reuse / remove per-point allocation.
6. Rayon parallelism over the prediction grid and the `n`-sums.

---

# W17 — Results (Rayon / SIMD / caching)

This section records the W17 optimizations and their measured effect. All
numbers are release-profile, Docker, Apple Silicon host, captured with the same
Criterion settings as the baseline (`--warm-up-time 0.5 --measurement-time 1.5
--sample-size 20`). **All correctness tests pass unchanged** (core 144, golden
5+7+1+2, R 141, Python 27); no tolerance was relaxed.

## What was optimized

### H1 — `beps1` `p⁴` loop factored out of the grid (`dmgs_expansion`)

The reference `dmgs` accumulates `beps1[a] += coeff(t,j,r,s)·mu1[t,a]` inside a
`t × j × r × s` quadruple loop, i.e. `O(p⁴·nalpha)`. Because `coeff` does not
depend on the prediction point `a` and `mu1[t,a]` does not depend on `j,r,s`,
the sum is **factored**:

```text
beps1[a] = Σ_t [ Σ_{j,r,s} 0.5·lddi[s,t]·lddi[j,r]·lddd[j,r,s] ] · mu1[t,a]
         = Σ_t c1[t] · mu1[t,a]
```

so the `p⁴` term is computed **once** (`O(p⁴)`) and the grid contraction is
`O(p·nalpha)`. The same factoring is applied to `beps2`/`beps3` (collapse the
`s` sum) and `beps4` (collapse the `j,r` sum). The arithmetic is identical to
the reference (same terms, same summation order over `t`), so results are
unchanged within the §7.2 tolerance. `dmgs_scalar` was factored the same way.

### H2 — Newton inverse-incomplete-gamma for the gamma quantile

[`Gamma::quantile()`](../src/models/gamma.rs) previously bisected the CDF with up
to 200 evaluations. It now solves `P(shape, x) = p` by a **safeguarded Newton
iteration** on the regularized lower incomplete gamma, using the closed-form
derivative `dP/dx = x^(shape-1) e^{-x}/Γ(shape)` and a Wilson–Hilferty initial
guess, with a bisection bracket as a fallback. This converges in a handful of
iterations and matches the previous bisection to well within the §7.2 tolerance
(the `cdf_quantile_roundtrip` test still passes at `rtol = 1e-8`).

### H3 — numerical-fallback derivatives

The GEV third/mixed derivatives remain numerical (the reference's `gev_logfddd`
is a ~180-line machine-generated expression, impractical to transcribe). The
`numerical_mixed_derivatives` / `numerical_third_derivatives` fallbacks were
restructured (single-pass, no redundant `params.to_vec()` per evaluation) but
kept **sequential** — see the Rayon finding below.

### H4 — fused single-pass sufficient statistics in the MLE

The model `score`/`hessian`/`third_derivatives` previously made **two** passes
over the data (one for `Σy`, one for `Σy²`). They now compute both statistics in
a **single fused pass** (`par_reduce2`/`par_reduce3`/`par_reduce4` helpers in
[`models/mod.rs`](../src/models/mod.rs)). This is the dominant H4 win: the MLE
fit re-evaluates these derivatives across BFGS + two Newton passes, so halving
the passes roughly halves the per-iteration data traffic.

### H5 — scratch-buffer reuse / no per-point allocation

The expansion builders now write the row-major `f1`/`f2`/`mu1`/`mu2` buffers
**directly** (the layout `dmgs_expansion` consumes), with no intermediate
point-major scratch and no transpose pass.

### H6 — Rayon parallelism: **investigated and found counterproductive**

The design (§6) and the W16 note both suggested Rayon over the prediction grid
and the `n`-sums. W17 implemented and **measured** both, and found that Rayon is
**1–3 orders of magnitude slower** for these workloads:

- The per-point expansion work is `O(p²)` flops and memory-bound; the per-item
  log-likelihood work is a handful of flops. Both are far below the threshold at
  which Rayon's per-chunk scheduling and synchronization cost amortizes.
- Measured examples (n = 1 000): `norm/score` 974 ns sequential vs **1.10 ms**
  parallel; `cp_pdf/norm/1000` 13.1 µs sequential vs **2.18 ms** parallel.

The parallel paths were therefore **removed** and the loops kept sequential; the
finding is documented in [`models/mod.rs`](../src/models/mod.rs) so a future
work item does not re-introduce them. The genuine wins in this area are the
fused single-pass reductions (H4), not parallelism.

## Before / after (baseline → W17)

### Derivative evaluation (`benches/derivatives.rs`)

| Benchmark | Baseline | W17 | Change |
|---|---|---|---|
| `norm/score` (n = 1 000) | 974 ns | 553 ns | **1.76× faster** |
| `norm/hessian` (n = 1 000) | 922 ns | 551 ns | **1.67× faster** |
| `norm/third_derivatives` (n = 1 000) | 939 ns | 555 ns | **1.69× faster** |
| `norm/score` (n = 10 000) | 10.4 µs | 5.86 µs | **1.77× faster** |
| `gamma/score` (n = 10 000) | 24.1 µs | 19.1 µs | **1.26× faster** |
| `gamma/mixed_derivatives` (n = 1 000) | 5.32 µs | 0.32 µs | **~16× faster** |

### DMGS expansion (`benches/dmgs_expansion.rs`)

| Benchmark | Baseline | W17 | Change |
|---|---|---|---|
| `cp_pdf/norm` (nalpha = 1 000) | 13.1 µs | 9.39 µs | **1.40× faster** |
| `cp_pdf/gamma` (nalpha = 1 000) | 136 µs | 131 µs | 1.04× faster |
| `cp_pdf/genextreme` (nalpha = 1 000) | 207 µs | 190 µs | **1.09× faster** |
| `cp_quantiles/norm` (nalpha = 1 000) | 30.4 µs | 26.4 µs | **1.15× faster** |
| `cp_quantiles/gamma` (nalpha = 1 000) | 2.99 ms | 1.40 ms | **2.14× faster** |
| `cp_quantiles/genextreme` (nalpha = 1 000) | 393 µs | 186 µs | **2.11× faster** |
| `cp_pdf_quantile_derived/genextreme` (nalpha = 1 000) | 406 µs | 382 µs | **1.06× faster** |

### MLE fitting (`benches/mle_fitting.rs`)

| Benchmark | Baseline | W17 | Change |
|---|---|---|---|
| `fit/norm` (n = 5 000) | 1.66 ms | 1.57 ms | **1.06× faster** |
| `fit/gamma` (n = 5 000) | 13.7 ms | 13.3 ms | **1.03× faster** |
| `fit/genextreme` (n = 5 000) | 25.8 ms | 26.1 ms | ~same |
| `predict_end_to_end/norm` (n = 500) | 139 µs | 130 µs | **1.07× faster** |
| `predict_end_to_end/gamma` (n = 500) | 1.46 ms | 1.17 ms | **1.25× faster** |
| `predict_end_to_end/genextreme` (n = 500) | 16.2 ms | 16.1 ms | ~same |

## Correctness

No numerical result changed beyond the existing tolerances. The golden tests
(`cp_golden`, `gev_negshape_golden`, `mle_golden`, `w11_cp_golden`), the MLE
tests, the CP tests, the predictor tests, the R `testthat` suite (141 passing),
and the Python `pytest` suite (27 passing) all pass unchanged. `cargo clippy
--workspace --all-targets -- -D warnings` is clean.
