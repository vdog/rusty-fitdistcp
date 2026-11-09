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
`weibull 1.18e-6` (the weibull is loosest because its `powf`-heavy density
derivatives carry more round-off; Phase 5 rewrote them in `ln_1p`/`exp_m1` form,
see §3.16). `1e-5` is a **100× tightening** of the pre-W9 `1e-3` and still
comfortably covers the weibull round-off.

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

### 3.9 Gamma CP (reference `(shape, scale)` parameterization)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GAMMA_RTOL` / `GAMMA_ATOL` | `1e-5` | `1e-10` | `gamma` CP pdf. |

**Rationale.** The reference uses R's `dgamma(scale=)` parameterization and a
non-homogeneous "type 1" prior `lambdad_cp = c(-1/shape, -1/scale)`. The DMGS
expansion is **parameterization-dependent**: the first-order truncation is not
invariant under nonlinear reparameterization. Before Phase 5 the Rust core
computed the expansion in `(shape, rate)`, which diverged from the reference by
~9% at `n = 30` (confirmed by [`tools/diag_gamma_cp.R`](../../tools/diag_gamma_cp.R)).
Phase 5 (Issue 5) computes the expansion in the reference's `(shape, scale)`
parameterization via [`GammaScale`](../../crates/fitdistcp-core/src/models/gamma.rs)
(converting `scale = 1/rate` at the boundary), so the residual now matches the
DMGS tolerance: the observed maximum relative residual is `2.17e-7`, a ~4×10⁵×
improvement over the pre-Phase-5 ~9%. `1e-5` is a 20000× tightening of the
pre-Phase-5 `2e-1`. The reference emits no CP quantiles for the gamma (it uses
the numerical `gamma_mu1f`/`gamma_mu2f`), so only the CP pdf is compared.

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

### 3.12 Analytic CP quantiles (`norm`, `lnorm`, `expon`) — Phase 1

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `RTOL` / `ATOL` | `1e-8` | `1e-10` | Exact analytic CP quantiles for `norm`, `lnorm`, `expon` (Rust analytic path vs. the reference `qnorm_cp` / `qlnorm_cp` / `qexp_cp`). |
| `DMGS_TRUNC_RTOL` / `DMGS_TRUNC_ATOL` | `5e-2` | `1e-4` | DMGS cross-check: the pre-Phase-1 DMGS CP quantiles vs. the reference's exact analytic CP. |

**Rationale.** Phase 1 (Issue 1) replaced the DMGS expansion with the reference's
**exact analytic** CP quantile formulas for these three models. The analytic path
is a closed-form match, so it uses the baseline §7.2 tolerance; the observed
maximum relative residual is `2.0e-15` (`lnorm`, `n = 10`), limited only by the
`student_t_quantile` bisection precision and the reference's own rounding. The
DMGS cross-check keeps the pre-Phase-1 path behind the `use_dmgs` flag and
compares it at the documented `O(1/n)` truncation tolerance: the `lnorm` DMGS
residual at `n = 10` is `3.4e-2` (the defect), versus `2.0e-15` for the analytic
path — a ~`1.6e13`× improvement. The harness
([`analytic_cp_golden.rs`](../crates/fitdistcp-core/tests/analytic_cp_golden.rs))
asserts both the tight analytic match and that the analytic residual is `< 1e-3`
of the DMGS residual, so a regression that silently re-routes `lnorm` back to
DMGS is caught.

### 3.13 GPD dollar-scale fixture — Phase 2

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GPD_RTOL` / `GPD_ATOL` | `1e-6` | `1e-10` | GPD CP quantiles / ML quantiles on the dollar-scale fixture (Rust `predict_gpd` vs. the reference `qgpd_k1_cp`). |
| `OPT_RTOL` / `OPT_ATOL` | `1e-4` | `1e-6` | GPD `ml_params` on the dollar-scale fixture (optimizer-produced, §3.2). |

**Rationale.** Phase 2 (Issue 2) fixed the GPD binding's hard-coded
`init = [1.0, 0.0]`, which on raw dollar-scale data (~10^3-10^5) started the
optimizer many orders of magnitude below the data's scale and silently returned
`NaN`. The binding now seeds the scale from `setics` = `[sd(x), 0]` (the
reference `gpd_k1_setics`). The fixture
([`gpd_dollar_golden.json`](gpd_dollar_golden.json), generated by
[`tools/generate_gpd_dollar_golden.R`](../../tools/generate_gpd_dollar_golden.R))
pins the reference `ml_params` (`[35098.898, -0.181885]`) and `cp_quantiles` on a
fixed 20-point dollar-scale dataset. The Rust GPD uses the reference's analytic
derivatives, so the CP quantiles match at the §3.7 `GPD_RTOL = 1e-6`; the
optimizer-produced `ml_params` use the §3.2 `OPT_RTOL = 1e-4`. The harness
([`gpd_dollar_golden.rs`](../crates/fitdistcp-core/tests/gpd_dollar_golden.rs))
also asserts the old `init = [1.0, 0.0]` is defective (non-finite or a scale
`< 1e2`), pinning the improvement so a regression that reverts the init is
caught.

### 3.14 GEV small-n CP support clamp — Phase 4

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `CPQ_RTOL` / `CPQ_ATOL` | `5e-3` | `1e-8` | GEV CP quantiles on the small-n (n = 20) fixture (Rust `predict_gev` vs. the reference `qgev_cp` after the reference `fixgevrange` clamp). |
| `OPT_RTOL` / `OPT_ATOL` | `1e-4` | `1e-6` | GEV `ml_params` on the small-n fixture (optimizer-produced, §3.2). |

**Rationale.** Phase 4 (Issue 4) fixed the GEV CP quantiles not being clamped to
the support bound: at small n the DMGS expansion can push a CP quantile outside
the GEV support (e.g. a negative 1-in-100 return level for a positive-support
fit). The core now applies the reference `fixgevrange` clamp (`005_utils.R`) to
the CP quantiles and falls back to the MLE quantile when a CP quantile still
leaves the support. The fixture
([`gev_smalln_golden.json`](gev_smalln_golden.json), generated by
[`tools/generate_gev_smalln_golden.R`](../../tools/generate_gev_smalln_golden.R))
pins the reference `ml_params`, `ml_max`, `revert2ml`, and `cp_quantiles` on two
fixed n = 20 datasets. The `gev_smalln_clamp` dataset is the defect case: its raw
DMGS CP quantile at `p = 0.99` (`2.1202`) exceeds `ml_max` (`2.0475`), so the
clamp is genuinely exercised (`cp_quantiles_raw[5] > ml_max` and
`cp_quantiles[5] == ml_max`). The harness
([`gev_smalln_golden.rs`](../crates/fitdistcp-core/tests/gev_smalln_golden.rs))
asserts the CP quantiles match the reference clamp, that every CP quantile is
finite and within the support, and that the 1-in-100 return level is finite,
within support, and non-negative.

The CP-quantile tolerance is `5e-3` rather than §3.6's `1e-3`: the
`gev_smalln_clamp` fixture has a larger-magnitude shape (`~ -0.56`) than the
`gev_negshape` fixture (`~ -0.30`), so the numerical third-derivative error in
the DMGS bias-correction term is larger. The observed maximum relative residual
is `3.2e-3`; `5e-3` is a ~1.6× margin that still catches any formula/sign bug.
This is a documented relaxation of §3.6 for this fixture.

### 3.15 Short-record robust ML optimiser — Phase 3

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `RTOL` / `ATOL` | `1e-8` | `1e-10` | Log-likelihood of the short-record (hail-region shape) GEV fits (Rust `fit` vs. the reference `optim`). |
| `OPT_RTOL` / `OPT_ATOL` | `1e-4` | `1e-6` | GEV `ml_params` on the short-record fixture (optimizer-produced, §3.2). |

**Rationale.** Phase 3 (Issue 3) fixed the hand-rolled BFGS stopping at a worse
fit on short records (return levels changed in 19-39% of hail regions; NaN in up
to 121 regions). The core `fit_with_config` now seeds from `setics`-style
heuristics in addition to the caller's `init`, adds a Nelder-Mead polish, and
keeps the best log-likelihood across the multi-start set. The fixture
([`short_record_golden.json`](short_record_golden.json), generated by
[`tools/generate_short_record_golden.R`](../../tools/generate_short_record_golden.R))
pins the reference `ml_params` and `loglik` on three fixed short (n = 10-15)
hail-region-shaped datasets. The log-likelihood is the objective, which is
well-determined at the optimum, so it is compared at the §7.2 tolerance; the
optimizer-produced `ml_params` use the §3.2 `OPT_RTOL = 1e-4`. The harness
([`short_record_golden.rs`](../crates/fitdistcp-core/tests/short_record_golden.rs))
also asserts the fit is **init-independent** (a deliberately poor init reaches
the same log-likelihood) and that no region returns NaN.

The multi-start selection rejects candidates in the degenerate region
(`valid_params`): a non-positive scale and a runaway shape (`|shape| >= 10`).
Without this guard the multi-start BFGS/Nelder-Mead can find the GEV/GPD
`scale -> 0`, `shape -> +inf` degeneracy (a spurious unbounded log-likelihood)
that the reference `optim` does not reach. The shape bound is deliberately
generous: the reference `optim` legitimately returns `|shape| >= 1` for some
short records (e.g. the symmetric 3-point dataset `[-1, 0, 1]` gives
`shape ~ -1.32`), so a tight `(-1, 1)` bound would diverge from the reference.

### 3.16 Weibull/gamma accuracy — Phase 5

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GAMMA_RTOL` / `GAMMA_ATOL` | `1e-5` | `1e-10` | `gamma` CP pdf (Rust `GammaScale` DMGS vs. the reference `(shape, scale)` DMGS). |
| `DMGS_RTOL` / `DMGS_ATOL` | `1e-5` | `1e-10` | `weibull` CP pdf (Rust DMGS vs. the reference DMGS). |

**Rationale.** Phase 5 (Issue 5) fixed two accuracy defects:

- **Gamma.** The DMGS expansion is parameterization-dependent. The reference
  computes it in `(shape, scale)`; the Rust core computed it in `(shape, rate)`,
  diverging by ~9% at `n = 30`. The core now computes the expansion in the
  reference's `(shape, scale)` parameterization via
  [`GammaScale`](../../crates/fitdistcp-core/src/models/gamma.rs) (converting
  `scale = 1/rate` at the boundary), so the residual drops to `2.17e-7` — a
  ~4×10⁵× improvement. `GAMMA_RTOL` is tightened from `2e-1` to `1e-5` (see
  §3.9).
- **Weibull.** The `powf`-heavy density/CDF derivative hooks were rewritten in
  `ln_1p`/`exp_m1`/`exp(k·ln z)` form to cut round-off. The residual drops from
  `1.2e-6` to `1.18e-6`. The residual is dominated by the **absolute** round-off
  of the DMGS correction term in the tail (where the density is tiny), not a
  formula error, so re-deriving the third-derivative tensor does not reduce it
  further (verified: the `powf` and `exp(k·ln)` third-derivative forms give the
  identical residual).

The harness ([`cp_golden.rs`](../crates/fitdistcp-core/tests/cp_golden.rs))
asserts both residuals directly in `golden_gamma_weibull_residuals`, pinning the
gamma residual below `GAMMA_RTOL` and the Weibull residual below the pre-Phase-5
`1.2e-6`.

### 3.17 `genextreme_p12` (two-predictor GEV) — Phase 6

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `RTOL` / `ATOL` | `1e-8` | `1e-10` | `genextreme_p12` MLE/CP quantiles and densities (the model is additive and reuses the existing GEV DMGS path). |

**Rationale.** Phase 6 (Issue 6) adds the 5-parameter GEV-with-two-predictors
model (`params = [beta0, beta1, beta2, scale, shape]`, location
`mu_i = beta0 + beta1·(t1_i - mean(t1))`, scale
`sigma_i = exp(beta2 + scale·(t2_i - mean(t2)))`). It is **additive**: it reuses
the existing GEV DMGS expansion, the `revert2ml` guard, and the
quantile-derived CP density, so it inherits the baseline §3.1 tolerance. The
model's derivatives use the numerical fallback (as `genextreme_p1` does), so the
`genextreme_p12` entry in the `analytic_vs_numerical` consistency test is
trivially satisfied.

The ported reference test
([`test_reference_port.py`](../crates/fitdistcp-py/tests/test_reference_port.py):
`test_reference_ppf_genextreme_p12_keys_and_finite`) asserts the reference
return-dict keys (`predictedparameter`, `adjustedx`, `revert2ml`, `ml_max`,
`ml_pdf`, `cp_pdf`) and that the 5-parameter fit and its MLE/CP
quantiles/densities are finite. As in the reference, `adjustedx` may contain
`±inf` when the fitted shape is negative and an observation lies beyond the
support (`pgev` saturates at 1 and `qgev(1)` is `+inf`), so the test asserts the
key, its length, and that the finite entries are finite rather than that every
entry is finite.

---

## 4. Documented legitimate differences

These are **expected** differences between the Rust port and the reference
package, not bugs. Each is handled explicitly in the harness (by converting the
reference value or by comparing against a different reference quantity) rather
than by loosening a tolerance blindly.

| Difference | Models | Handling |
|---|---|---|
| **Gamma parameterization** | `gamma` | Reference uses `(shape, scale)`; the core `Gamma` model uses `(shape, rate)`. The MLE harness converts `rate = 1/scale` and the SE by the delta method (`SE(rate) = SE(scale)/scale²`). As of Phase 5 the CP harness computes the DMGS expansion in the reference's `(shape, scale)` parameterization via `GammaScale` (converting at the boundary), so the CP pdf matches at `GAMMA_RTOL = 1e-5` (§3.9, §3.16). |
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
| Weibull/gamma accuracy (Phase 5) | [`crates/fitdistcp-core/tests/cp_golden.rs`](../crates/fitdistcp-core/tests/cp_golden.rs) | `GAMMA_RTOL`, `GAMMA_ATOL`, `DMGS_RTOL`, `DMGS_ATOL` |
| Analytic CP quantiles (Phase 1) | [`crates/fitdistcp-core/tests/analytic_cp_golden.rs`](../crates/fitdistcp-core/tests/analytic_cp_golden.rs) | `RTOL`, `ATOL`, `DMGS_TRUNC_RTOL`, `DMGS_TRUNC_ATOL` |
| GPD dollar-scale (Phase 2) | [`crates/fitdistcp-core/tests/gpd_dollar_golden.rs`](../crates/fitdistcp-core/tests/gpd_dollar_golden.rs) | `GPD_RTOL`, `GPD_ATOL`, `OPT_RTOL`, `OPT_ATOL` |
| GEV negative-shape | [`crates/fitdistcp-core/tests/gev_negshape_golden.rs`](../crates/fitdistcp-core/tests/gev_negshape_golden.rs) | `NEGSHAPE_RTOL`, `NEGSHAPE_ATOL`, `CPQ_RTOL`, `CPQ_ATOL`, `SMALLN_RTOL`, `SMALLN_ATOL` |
| GEV small-n clamp (Phase 4) | [`crates/fitdistcp-core/tests/gev_smalln_golden.rs`](../crates/fitdistcp-core/tests/gev_smalln_golden.rs) | `CPQ_RTOL`, `CPQ_ATOL`, `OPT_RTOL`, `OPT_ATOL` |
| Short-record optimiser (Phase 3) | [`crates/fitdistcp-core/tests/short_record_golden.rs`](../crates/fitdistcp-core/tests/short_record_golden.rs) | `RTOL`, `ATOL`, `OPT_RTOL`, `OPT_ATOL` |
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
