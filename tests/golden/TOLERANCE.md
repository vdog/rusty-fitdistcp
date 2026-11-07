# Golden-test tolerance policy (W19)

This document is the **single source of truth** for the tolerance policy used by
the golden-fixture tests in `crates/fitdistcp-core/tests/`. It consolidates and
supersedes the per-harness notes that were previously scattered across the test
modules, and it is the authoritative reference for the CI wiring in W20.

The policy derives from the design document
[`plans/fitdistcp-rust-port-design.md`](../../plans/fitdistcp-rust-port-design.md)
§7.2 ("Golden-test tolerance policy (locked in)") and §10 (Phase 1f). Where a
harness uses a tolerance **looser** than §7.2, the rationale is recorded here and
in the harness module docs; no tolerance is ever loosened without a documented
reason.

---

## 1. Comparison rule

All floating-point comparisons use the standard mixed absolute/relative form:

```text
|actual - expected| <= atol + rtol * |expected|
```

implemented by the `assert_close` helper in each harness. The **expected** value
is the reference (R package) value, so the relative term is scaled by the
reference magnitude.

Discrete fields are compared with **exact equality** (`==` / `assert_eq!`).

---

## 2. Exact-match fields (discrete)

These are compared with exact equality — no tolerance:

| Field | Where | Notes |
|---|---|---|
| Parameter **count** | all harnesses | `assert_eq!(actual.len(), expected.len())` |
| `converged` / convergence flags | `mle_golden`, `w11_cp_golden` | boolean |
| `small_n` flag | `gev_negshape_golden` | set for `n < 10`, cleared at `n = 10` (§4.4) |
| `revert2ml` | `gev_negshape_golden` | boolean; DMGS vs. revert-to-MLE regime |
| `n` (sample size) | `gev_negshape_golden` | integer |
| method / parameter **names** | fixtures (`param_names`) | strings |
| `predictedparameter.len()`, `adjustedx.len()` | `w14_predictor` | integer lengths |

---

## 3. Floating-point tolerances

### 3.1 Baseline (§7.2, locked in)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `RTOL` / `ATOL` | `1e-8` | `1e-10` | Exact/analytic comparisons: closed-form MLEs (`norm`, `expon`, `lnorm`, `norm_p1`), log-likelihood, AIC, and analytic CP where the Rust and reference derivative paths match. |

This is the default. Any deviation below is a **documented relaxation** with a
stated rationale.

### 3.2 Optimizer-produced parameters

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `OPT_RTOL` / `OPT_ATOL` | `1e-4` | `1e-6` | **Parameters** of the DMGS models fitted by an optimizer (`weibull`, `gamma`, `gumbel`, `genextreme`, `genpareto`, `genextreme_p1`). |

**Rationale.** The reference MLEs for these models come from R's `optim`
(Nelder–Mead), while Rust uses BFGS/Newton. Both converge to the same optimum but
stop at slightly different points on a flat likelihood surface. The fixtures are
generated with a tightly-converged `optim` (`reltol = 1e-14`, see
[`tools/generate_golden.R`](../../tools/generate_golden.R)) so they represent the
true MLE. `1e-4` relative is far tighter than the optimizer's own convergence
tolerance and still catches any genuine model/parameterization bug. The
**log-likelihood** (the objective, which is well-determined at the optimum) is
compared at the full §7.2 tolerance.

### 3.3 Standard errors of the DMGS models

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `SE_RTOL` / `SE_ATOL` | `1e-5` | `1e-8` | Standard errors of `weibull`, `gamma`, `gumbel`, `genextreme`, `genpareto`, `genextreme_p1`. |

**Rationale.** As of **W9** both the reference and Rust use the same *analytic*
second derivatives, so the SEs agree to ~`1e-7`; the observed maximum relative
residual is `3.5e-6` (`genextreme_p1`, whose observed-information matrix is the
most ill-conditioned). `1e-5` is a **100× tightening** of the pre-W9 `1e-3` and
still catches any genuine SE bug (e.g. a wrong sign or a missing factor of `n`).

### 3.4 DMGS CP output (analytic-derivative models)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `DMGS_RTOL` / `DMGS_ATOL` | `1e-5` | `1e-10` | DMGS CP pdf/quantiles for `norm`, `expon`, `lnorm`, `gumbel`, `weibull` (Rust DMGS vs. reference DMGS). |

**Rationale.** As of **W9** the Rust path uses the same *analytic* derivatives as
the reference (`*_ldda`, `*_lddda`, `*_mu1fa`, `*_mu2fa`, `*_f1fa`, `*_f2fa`), so
the two agree to near machine precision. Observed maximum relative residuals:
`norm 2.8e-14`, `expon 4.8e-9`, `lnorm 2.6e-8`, `gumbel 1.2e-7`,
`weibull 1.2e-6` (the weibull is loosest because its `powf`-heavy density
derivatives carry more round-off). `1e-5` is a **100× tightening** of the pre-W9
`1e-3` and still comfortably covers the weibull round-off.

### 3.5 DMGS truncation cross-check (analytic CP)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `DMGS_TRUNC_RTOL` / `DMGS_TRUNC_ATOL` | `5e-2` | `1e-4` | Rust DMGS vs. the reference's **exact analytic** CP for `norm`, `expon`, `lnorm`. |
| `NORMP1_TRUNC_RTOL` / `NORMP1_TRUNC_ATOL` | `5e-2` | `1e-4` | Rust DMGS vs. the reference's **exact analytic** CP for `norm_p1`. |

**Rationale.** The DMGS expansion is a **first-order asymptotic approximation**
of the exact Bayesian prediction integral, so the two differ by the `O(1/n)`
truncation error (observed ~1% at `n = 30` for the homogeneous models, ~2.4% for
`norm_p1`). This is a documented, expected difference — the apples-to-apples
DMGS comparison (§3.4) uses the full §7.2 tolerance.

### 3.6 GEV / GEV-p1 CP (numerical third derivatives)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GEV_RTOL` / `GEV_ATOL` | `1e-3` | `1e-10` | CP pdf/quantiles for `genextreme` (negative and positive shape) and `genextreme_p1`. |
| `NEGSHAPE_RTOL` / `NEGSHAPE_ATOL` | `1e-3` | `1e-10` | Negative-shape GEV quantile-derived CP density. |
| `CPQ_RTOL` / `CPQ_ATOL` | `1e-3` | `1e-8` | Negative-shape GEV CP quantiles. |
| `SMALLN_RTOL` / `SMALLN_ATOL` | `1e-3` | `1e-8` | Small-n (`n = 8`) negative-shape GEV CP pdf/quantiles. |

**Rationale.** The Rust GEV keeps the **numerical** third-derivative fallback
(the reference's `gev_logfddd` is a ~180-line machine-generated expression), so
the DMGS bias-correction term carries finite-difference error. This is the
dominant residual source. Observed maximum relative residuals: negative-shape CP
pdf `2.4e-5` / CP quantiles `1.6e-4`; positive-shape CP pdf `8.5e-6` / CP
quantiles `4.0e-6`; `genextreme_p1` CP quantile `~5e-6`; quantile-derived density
`1.4e-4`. `1e-3` is a ~6× margin over the worst case and still catches any
formula/sign bug.

> **Note on §7.2's `1e-6` for the quantile-derived density.** The design §7.2
> specifies `rtol = 1e-6` for the negative-shape GEV quantile-derived density,
> which assumes the Rust and reference derivative paths match. They do **not**
> (numerical vs. analytic third derivatives), so the harness uses `1e-3`. This is
> the one place a §7.2 tolerance is relaxed, and the rationale is recorded here
> and in [`gev_negshape_golden.rs`](../crates/fitdistcp-core/tests/gev_negshape_golden.rs).

### 3.7 GPD CP (analytic derivatives)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GPD_RTOL` / `GPD_ATOL` | `1e-6` | `1e-10` | `genpareto` CP pdf/quantiles. |

**Rationale.** The Rust GPD uses the reference's **analytic** derivatives
(`gpd_k1_*`), so the two agree to near machine precision. Observed max relative
residuals: CP pdf `2.9e-7`, CP quantiles `2.1e-7`. `1e-6` is the §7.2
quantile-derived-density tolerance and gives a comfortable margin.

### 3.8 norm_p1 DMGS CP

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `NORMP1_DMGS_RTOL` / `NORMP1_DMGS_ATOL` | `1e-6` | `1e-10` | Rust DMGS CP vs. the reference's own DMGS CP at the **same parameters**. |

**Rationale.** The reference `qnorm_p1_cp` returns the **analytic** CP
(`qnorm_p1_formula`, a Student-t formula), not the DMGS expansion. The Rust port
implements the DMGS expansion, so the apples-to-apples comparison is against the
reference's own DMGS CP (`cp_quantiles_dmgs`) evaluated at the same parameters.
Observed residual `4.8e-7`.

### 3.9 Gamma CP (parameterization-dependent)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GAMMA_RTOL` / `GAMMA_ATOL` | `2e-1` | `1e-5` | `gamma` CP pdf. |

**Rationale.** The reference uses R's `dgamma(scale=)` parameterization and a
non-homogeneous "type 1" prior `lambdad_cp = c(-1/shape, -1/scale)`. The Rust
model uses `[shape, rate]`. The prior slope is correct (Rust `[-1/shape, -rate]`
== the reference type-1 prior), but the DMGS expansion is
**parameterization-dependent**: the first-order truncation is not invariant under
nonlinear reparameterization, so the two differ by ~9% at `n = 30` (confirmed by
[`tools/diag_gamma_cp.R`](../../tools/diag_gamma_cp.R)). The reference emits no CP
quantiles for the gamma (it uses the numerical `gamma_mu1f`/`gamma_mu2f`), so
only the CP pdf is compared.

### 3.10 norm_p1 analytic CP quantiles

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `RTOL_CP_Q` | `1e-6` | `1e-10` | Analytic `norm_p1` CP quantiles. |

**Rationale.** Limited by the `special` crate's `inc_beta` precision (documented
discrepancy). The reference values are also quoted to 12 significant figures, so
`1e-6` absorbs the reference's own rounding.

### 3.11 Reliability simulation (`reltest`)

| Assertion | Rule |
|---|---|
| CP vs. MLE aggregate MSE | `MSE_cp <= MSE_mle + mc_tolerance` |

**Rationale.** Per §7.2, the reliability simulation is a **statistical** test:
"CP mean squared error must be ≤ MLE MSE within Monte Carlo error over a fixed
seed; assert on the aggregate, not per-replicate." The harness runs a fixed-seed
Monte Carlo (`SEED = 0x5EED_1234_5678_9ABC`, `N_TRIALS = 400`, `NX = 50`) and
requires the aggregate CP MSE not to exceed the MLE MSE by more than the paired
Monte Carlo standard error of the difference. This is not a per-replicate
assertion.

---

## 4. Documented legitimate differences

These are **expected** differences between the Rust port and the reference
package, not bugs. Each is handled explicitly in the harness (by converting the
reference value or by comparing against a different reference quantity) rather
than by loosening a tolerance blindly.

| Difference | Models | Handling |
|---|---|---|
| **Gamma parameterization** | `gamma` | Reference uses `(shape, scale)`; Rust uses `(shape, rate)`. The MLE harness converts `rate = 1/scale` and the SE by the delta method (`SE(rate) = SE(scale)/scale²`). The CP harness compares at `GAMMA_RTOL` because the DMGS truncation is not reparameterization-invariant. |
| **GEV numerical third derivatives** | `genextreme`, `genextreme_p1` | Rust keeps the numerical `gev_logfddd` fallback; the reference uses a machine-generated analytic expression. Handled by the `GEV_RTOL` / `NEGSHAPE_RTOL` tolerances (§3.6). |
| **norm_p1 sigma convention** | `norm_p1` | Reference `norm_p1_mlparams` uses the **unbiased** residual SD (`n-1` denominator); Rust converges to the **true MLE** (`n` denominator). The harness converts sigma and its SE by `sqrt((n-1)/n)`. The reference's own `ml_value` also uses the unbiased sigma, so the fixture stores the true-MLE loglik/AIC. |
| **norm_p1 centering** | `norm_p1` | Reference reports the *uncentered* intercept from `lm(x~t)` but its `expinfmat` uses the *centered* predictor `t - mean(t)`. The harness runs Rust with `t0 = mean(t)` and converts the reference intercept to the centered one (`centered = intercept + slope·mean(t)`). |
| **`_p1` prediction point (W14)** | `norm_p1`, `genextreme_p1` | Rust centers the predictor by its mean and evaluates the DMGS expansion at `t0 - mean(t)`. The fixture emits the reference CP at the prediction point `t0 = t[0]` in the mean-centered predictor space, so the two coincide. |
| **`movexiawayfromzero`** | `genextreme` | The reference nudges `shape` to `±1e-7`; the Rust derivative hooks do the same (W10), so the `shape ≈ 0` limit is handled identically. |
| **DMGS truncation vs. exact CP** | `norm`, `expon`, `lnorm`, `norm_p1` | The DMGS expansion is a first-order approximation of the exact integral; handled by the `*_TRUNC_RTOL` cross-check tolerances (§3.5). |

---

## 5. Where the constants live

| Harness | File | Constants |
|---|---|---|
| MLE golden | [`crates/fitdistcp-core/tests/mle_golden.rs`](../crates/fitdistcp-core/tests/mle_golden.rs) | `RTOL`, `ATOL`, `OPT_RTOL`, `OPT_ATOL`, `SE_RTOL`, `SE_ATOL` |
| CP golden (analytic) | [`crates/fitdistcp-core/tests/cp_golden.rs`](../crates/fitdistcp-core/tests/cp_golden.rs) | `DMGS_RTOL`, `DMGS_ATOL`, `DMGS_TRUNC_RTOL`, `DMGS_TRUNC_ATOL`, `GAMMA_RTOL`, `GAMMA_ATOL` |
| GEV negative-shape | [`crates/fitdistcp-core/tests/gev_negshape_golden.rs`](../crates/fitdistcp-core/tests/gev_negshape_golden.rs) | `NEGSHAPE_RTOL`, `NEGSHAPE_ATOL`, `CPQ_RTOL`, `CPQ_ATOL`, `SMALLN_RTOL`, `SMALLN_ATOL` |
| W11 CP (GEV/GPD/p1) | [`crates/fitdistcp-core/tests/w11_cp_golden.rs`](../crates/fitdistcp-core/tests/w11_cp_golden.rs) | `GPD_RTOL`, `GPD_ATOL`, `GEV_RTOL`, `GEV_ATOL`, `NORMP1_DMGS_RTOL`, `NORMP1_DMGS_ATOL`, `NORMP1_TRUNC_RTOL`, `NORMP1_TRUNC_ATOL` |
| W14 predictor | [`crates/fitdistcp-core/tests/w14_predictor.rs`](../crates/fitdistcp-core/tests/w14_predictor.rs) | `RTOL`, `ATOL`, `RTOL_CP_Q` |
| Reliability sim | [`crates/fitdistcp-core/tests/reltest.rs`](../crates/fitdistcp-core/tests/reltest.rs) | `SEED`, `N_TRIALS`, `NX`, `ALPHAS` |

Each harness's module-level doc comment restates the relevant subset of this
policy so the rationale is visible at the point of use. This document is the
canonical, consolidated version.

---

## 6. CI-readiness guarantees

- **No network access at test time.** All fixtures are committed JSON files under
  [`tests/golden/`](.) and loaded via `std::fs::read_to_string` relative to
  `CARGO_MANIFEST_DIR`. The harnesses use a tiny in-tree JSON parser (no
  `serde_json` dependency) and never invoke R, Python, or the network.
- **No reference package at test time.** The reference R package is only needed
  to *regenerate* fixtures (see [`README.md`](README.md)); the tests themselves
  consume the committed JSON.
- **Deterministic.** Fixture datasets are fixed literals; the reliability
  simulation uses a fixed seed. Re-running the suite yields identical results.
- **Fail-closed.** A tolerance breach fails the test with the actual/expected
  values and the computed tolerance printed.
