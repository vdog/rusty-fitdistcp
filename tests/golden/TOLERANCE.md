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

### 3.14 GEV small-n CP quantiles — Phase 4, **unclamped as of item A1**

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `CPQ_RTOL` / `CPQ_ATOL` | `5e-3` | `1e-8` | GEV CP quantiles on the small-n (n = 20) fixture (Rust `predict_gev` vs. the reference `qgev_cp`). |
| `OPT_RTOL` / `OPT_ATOL` | `1e-4` | `1e-6` | GEV `ml_params` on the small-n fixture (optimizer-produced, §3.2). |

**History.** Phase 4 (Issue 4) added a `fixgevrange` support clamp plus an
`ml_max`/`ml_min` per-point fallback to the GEV CP quantiles. **Item A1 of the
Issue 2 re-check removed both from the quantile path**: the reference `qgev_cp`
applies `fixgevrange` to the *data*, not to the CP quantiles, and it computes
`ml_max` only to decide `revert2ml` — it never caps the CP quantiles at it. The
Phase 4 clamp therefore biased the CP tail low (the reported 1-in-1000 return
level: reference `4.287` vs. clamped Rust `2.769`). The monotonicity guard (§7)
is **retained**.

**Fixture.** [`gev_smalln_golden.json`](gev_smalln_golden.json) (generated by
[`tools/generate_gev_smalln_golden.R`](../../tools/generate_gev_smalln_golden.R))
was **regenerated unclamped** and pins the reference `ml_params`, `ml_max`,
`revert2ml`, and `cp_quantiles` on two fixed n = 20 datasets. The
`gev_smalln_clamp` dataset is still the interesting case — its raw DMGS CP
quantile at `p = 0.99` (`2.1202`) exceeds `ml_max` (`2.0475`) — but the harness
([`gev_smalln_golden.rs`](../crates/fitdistcp-core/tests/gev_smalln_golden.rs))
now asserts the CP quantile **matches the reference unclamped value** and is
**not** equal to `ml_max` (`golden_gev_smalln_tail_is_not_clamped`), so a
reintroduced clamp fails the test. It also asserts every CP quantile is finite
and the 1-in-100 return level is finite and non-negative.

The CP-quantile tolerance is `5e-3` rather than §3.6's `1e-3`: the
`gev_smalln_clamp` fixture has a larger-magnitude shape (`~ -0.56`) than the
`gev_negshape` fixture (`~ -0.30`), so the numerical third-derivative error in
the DMGS bias-correction term is larger. The observed maximum relative residual
is `3.2e-3`; `5e-3` is a ~1.6× margin that still catches any formula/sign bug.
This is a documented relaxation of §3.6 for this fixture.

### 3.14a GEV CP tail — item A1/A2 (no `ml_max` clamp)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `CPQ_RTOL` / `CPQ_ATOL` | `1e-3` | `1e-8` | GEV CP quantiles on the tail fixture (Rust `predict_gev` vs. the reference `qgev_cp`), including `p = 0.999`. |
| `OPT_RTOL` / `OPT_ATOL` | `1e-4` | `1e-6` | GEV `ml_params` on the tail fixture (optimizer-produced, §3.2). |

**Rationale.** [`gev_tail_golden.json`](gev_tail_golden.json) (generator
[`tools/generate_gev_tail_golden.R`](../../tools/generate_gev_tail_golden.R))
pins a negative-shape GEV dataset whose raw DMGS CP quantile at `p = 0.99` and
`p = 0.999` exceeds `ml_max`. The harness
([`gev_cp_tail_golden.rs`](../crates/fitdistcp-core/tests/gev_cp_tail_golden.rs),
`golden_gev_cp_tail_is_not_clamped_to_ml_max`) asserts the 1-in-1000 CP quantile
**exceeds `ml_max`** and matches the reference within `CPQ_RTOL = 1e-3` (the
§3.6 numerical-third-derivative tolerance). This is the dedicated regression
test for item A: it fails on the pre-fix (clamped) code and passes post-fix.

### 3.14b GEV degenerate fits — item B1/B2 (NaN + `degenerate_fit` flag)

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `CPQ_RTOL` / `CPQ_ATOL` | `5e-3` | `1e-8` | CP quantiles of the **legitimate control** record (numerical third derivatives, §3.14). |
| `MAX_MEANINGFUL_CP` | — | — | Bound (`1e3`) on a meaningful 1-in-100 CP return level for the degenerate records. |

**Rationale.** On ~10-point records the GEV likelihood has a spurious maximum as
`scale -> 0` with `shape -> +inf`. The reference `qgev_cp` does **not** reject it
(`gev_checkmle` is commented out) and returns huge finite CP quantiles (up to
`~2e11`). The Rust multi-start used to find the same point (1-in-100 = `1.2e8` /
`2.0e9`; worst `2e13` in the hail corpus). Item B1 adds a **relative** guard in
the multi-start selection path
([`Derivatives::valid_params_for`](../crates/fitdistcp-core/src/models/mod.rs)):
a candidate is rejected when `scale <= 1e-3 * spread(y)` or
`|loc| > 1e3 * spread(y)`, where `spread(y) = max(sd(y), |mean(y)|, 1)`. Item B2
adds the [`FitResult::degenerate_fit`](../crates/fitdistcp-core/src/api.rs) flag:
if every candidate is degenerate, the CP output is **`NaN`** with
`degenerate_fit = true` (never a huge finite number).

**This is a deliberate Rust-only divergence.** The reference returns huge finite
values on these records; the Rust port returns `NaN` + a flag, which is strictly
safer and matches the report's intent. The flag is surfaced in both bindings
(Python `degenerate_fit`, R `degenerate_fit`).

**Fixture.** [`gev_degenerate_golden.json`](gev_degenerate_golden.json)
(generator
[`tools/generate_gev_degenerate_golden.R`](../../tools/generate_gev_degenerate_golden.R))
pins the reference output on the canonical degenerate records
(`tight_cluster`, `near_constant`, `constant`) plus a legitimate control. The
harness
([`gev_degenerate_golden.rs`](../crates/fitdistcp-core/tests/gev_degenerate_golden.rs))
asserts, for each degenerate record, that the Rust fit is **not** degenerate
(`degenerate_fit == false`), that its 1-in-100 CP quantile is finite and
`< MAX_MEANINGFUL_CP` (not the reference's `~2e11`), and that its parameters pass
the relative guard; the control must still match the reference at `CPQ_RTOL`.
The degenerate records are compared by **bound**, not by value, because the Rust
fit intentionally diverges from the reference's degenerate output.

### 3.14c GPD small-`n` near-−1 revert — item C1

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GPD_SMALLN_NEGSHAPE_TAIL_RTOL` / `_ATOL` | `5e-2` | `1e-6` | 1-in-100 CP quantile on the `n = 20` near-−1 GPD fixture (Rust `predict_gpd` vs. the reference `qgpd_k1_cp`). |
| `MAX_CP_OVER_DATA` | — | — | Bound (`1.1`) on the CP curve's maximum as a multiple of `max(y)`. |

**Rationale.** The reference `revert2ml = (|shape| >= 1) || (shape > 30)` does
not catch a shape just **above** −1 (e.g. `-0.987`), where the DMGS `1/fhat`
term blows up, the CP quantiles leave the support `y < -scale/shape`, and the
monotone guard flattens the curve far above the data (reported `n = 20`: `19.8`
vs. reference `2.06`). Item C1 widens the revert to `shape <= -0.98`
(`GPD_NEAR_MINUS_ONE_DELTA = 0.02`, calibrated in the plan §7 open question 3),
so the CP output equals the ML output. The reference `optim` never lands in
`(-1, -0.98)`, so this only affects the Rust multi-start.

**Fixture.** [`gpd_smalln_negshape_golden.json`](gpd_smalln_negshape_golden.json)
(generator
[`tools/generate_gpd_smalln_negshape_golden.R`](../../tools/generate_gpd_smalln_negshape_golden.R))
pins the reference CP quantiles on two fixed `n = 20` near-−1 records. The
harness
([`gpd_smalln_negshape_golden.rs`](../crates/fitdistcp-core/tests/gpd_smalln_negshape_golden.rs))
asserts the Rust fit **reverts** (`cp_quantiles == ml_quantiles`), the CP curve
is **not flat far above the data** (`max(cp) <= MAX_CP_OVER_DATA * max(y)`; the
pre-fix curve was `3.92×` / `1.45×`), the curve is monotone, and the 1-in-100
quantile matches the reference within `GPD_SMALLN_NEGSHAPE_TAIL_RTOL`. The tail
tolerance is loose because the two fits land at different (both valid) near-−1
shapes (Rust `~ -0.988` vs. reference `~ -1.46`); the low-`p` quantiles differ by
up to ~30% for the same reason, so only the tail is pinned tightly. The
regression assertions (reverted / not flat / monotone) are exact.

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

**Item B1 (Issue 2 re-check) tightened this guard** with a *relative* check
applied in the multi-start selection path
([`Derivatives::valid_params_for`](../crates/fitdistcp-core/src/models/mod.rs)):
a candidate is rejected when `scale <= 1e-3 * spread(y)` or
`|loc| > 1e3 * spread(y)`, where `spread(y) = max(sd(y), |mean(y)|, 1)`. The
shape bound is **unchanged** (`|shape| < 10`) — see §3.14b and the plan §7 open
question 1 for why a `maxxi = 1` bound was rejected.

### 3.16 Weibull/gamma accuracy — Phase 5

| Constant | rtol | atol | Applies to |
|---|---|---|---|
| `GAMMA_RTOL` / `GAMMA_ATOL` | `1e-5` | `1e-10` | `gamma` CP pdf (Rust `GammaScale` DMGS vs. the reference `(shape, scale)` DMGS). |
| `DMGS_RTOL` / `DMGS_ATOL` | `1e-5` | `1e-10` | `weibull` CP pdf/quantiles (Rust DMGS vs. the reference DMGS), including the `n = 10` `weibull_n10` case (item D1). |

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

### 3.16a Weibull small-`n` CP residual — item D1 (root cause: prior slope)

**Finding: the ~3% `n = 10` Weibull CP residual was an algorithmic difference
(the prior slope), not inherent `O(1/n)` truncation.** The reference
`qweibull_cp` uses `lambdad_rhp = c(-1/v1hat, -1/v2hat)`, i.e. the right-Haar
prior `π(k, σ) ∝ 1/(k σ)` for the `(shape, scale)` parameterization, whose
log-slope is `[-1/shape, -1/scale]`. The Rust `Weibull` `DmgsModel` used the
**default** scale-only slope `[0, -1/scale]`, dropping the `-1/shape` term.

**Evidence.** At the `n = 10` MLE (`shape = 1.5858`, `scale = 1.9044`), the
reference DMGS CP quantiles computed with the two priors differ by up to
**13.3%** at `p = 0.1` (and `7.1%` at `p = 0.25`, `3.1%` at `p = 0.5`), far
larger than the `O(1/n)` truncation itself (the DMGS correction is ~23% of the
ML quantile at `p = 0.1`). The Rust DMGS output matched the reference
**scale-only-prior** column to machine precision and the reference
**correct-prior** column to machine precision once the prior was corrected
(verified with [`tools/diag_w11_weibull_n10.R`](../../tools/diag_w11_weibull_n10.R)
and a temporary Rust diagnostic). The pre-fix golden generator
([`tools/generate_cp_golden.R`](../../tools/generate_cp_golden.R)) also used
`c(0, -1/v2)`, so the `n = 30` golden passed at `1.18e-6` only because **both**
sides used the same wrong prior — it masked the defect.

**Fix.** [`Weibull`](../../crates/fitdistcp-core/src/dmgs/models.rs) now
overrides `log_prior_slope` to return `[-1/shape, -1/scale]`, matching the
reference. The generator was corrected to the reference prior and regenerated
([`cp_golden.json`](cp_golden.json)); a new `weibull_n10` fixture pins the
`n = 10` case. With the corrected prior the Rust DMGS matches the reference at
the tight `DMGS_RTOL = 1e-5` (no loosened small-`n` tolerance is needed). The
harness test `golden_weibull_n10_prior_slope` also pins the prior slope itself,
so a regression that reverts the override is caught even if the fixture is
regenerated with the wrong prior.

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
| GEV small-n (Phase 4, unclamped per item A1) | [`crates/fitdistcp-core/tests/gev_smalln_golden.rs`](../crates/fitdistcp-core/tests/gev_smalln_golden.rs) | `CPQ_RTOL`, `CPQ_ATOL`, `OPT_RTOL`, `OPT_ATOL` |
| GEV CP tail (item A1/A2) | [`crates/fitdistcp-core/tests/gev_cp_tail_golden.rs`](../crates/fitdistcp-core/tests/gev_cp_tail_golden.rs) | `CPQ_RTOL`, `CPQ_ATOL`, `OPT_RTOL`, `OPT_ATOL`, `FDALPHA` |
| GEV degenerate fits (item B1/B2) | [`crates/fitdistcp-core/tests/gev_degenerate_golden.rs`](../crates/fitdistcp-core/tests/gev_degenerate_golden.rs) | `CPQ_RTOL`, `CPQ_ATOL`, `MAX_MEANINGFUL_CP`, `FDALPHA` |
| GPD small-n near-−1 (item C1) | [`crates/fitdistcp-core/tests/gpd_smalln_negshape_golden.rs`](../crates/fitdistcp-core/tests/gpd_smalln_negshape_golden.rs) | `GPD_SMALLN_NEGSHAPE_TAIL_RTOL`, `GPD_SMALLN_NEGSHAPE_TAIL_ATOL`, `MAX_CP_OVER_DATA` |
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

---

## 7. Monotonicity guard (Phase A, Issue 4)

This section documents a **deliberate Rust-only divergence** from the reference
`fitdistcp` package, introduced in Phase A of
[`plans/phase1-remaining-fixes-plan.md`](../../plans/phase1-remaining-fixes-plan.md).

### 7.1 What the guard does

[`enforce_monotone_quantiles()`](../crates/fitdistcp-core/src/models/numerical.rs)
performs a **running-max pass** over a CP quantile vector `q` (ordered by
increasing probability `p`): every element that would decrease relative to the
running maximum is raised to that maximum. It returns the monotone vector plus a
`clamped` flag that is `true` when at least one element was raised.

It is applied in the binding-facing CP paths only:

- [`predict_gev()`](../crates/fitdistcp-core/src/api.rs) — **after** the DMGS CP
  quantiles are computed. (As of item A1 there is **no** `fixgevrange` support
  clamp and **no** `ml_max`/`ml_min` fallback on the quantile path; the guard is
  the only post-processing step.)
- [`predict_gpd()`](../crates/fitdistcp-core/src/api.rs) — **after** the DMGS CP
  quantiles are computed (and only on the finite DMGS branch).

The `clamped` flag is surfaced to callers as the `cp_monotone_clamped` field on
[`FitResult`](../crates/fitdistcp-core/src/api.rs) and in both bindings' return
dicts (Python `cp_monotone_clamped`, R `cp_monotone_clamped`).

### 7.2 Why it is a divergence

The reference package does **not** sort or clamp its CP quantiles. Its
`optim`-based fit simply avoids the degenerate region where the first-order DMGS
expansion produces a non-monotone curve. The Rust multi-start optimizer can still
land near that region at small `n`, so the guard restores the non-decreasing
invariant that quantiles must satisfy by definition (e.g. a negative median or a
decreasing 1-in-100 return level is not a valid quantile curve).

### 7.3 How the golden cross-check still runs unguarded

The guard is applied **only** in the binding-facing `predict_gev` /
`predict_gpd` entry points. The golden harnesses that compare against the
reference values can therefore still exercise the **unguarded** DMGS path where
the reference values are the target:

- The generic [`predict()`](../crates/fitdistcp-core/src/api.rs) DMGS path does
  **not** apply the guard (`cp_monotone_clamped` is always `false` there).
- The analytic closed-form models (`norm`, `lnorm`, `expon`) and the `_p1` /
  `_p12` GEV paths do **not** apply the guard.
- A harness that needs the raw, unguarded CP quantiles should call the DMGS
  integrator directly (or the generic `predict`) rather than `predict_gev` /
  `predict_gpd`.

When a fixture records the reference's (unguarded) CP quantiles and the Rust
value is produced by `predict_gev` / `predict_gpd`, the harness must account for
the guard: either assert the Rust result is non-decreasing (the guard's
invariant) or compare on the unguarded path. The `cp_monotone_clamped` flag makes
this explicit — a `true` value means the returned CP quantiles were clamped and
will differ from the reference's raw values wherever the reference curve was
non-monotone.

### 7.4 Unit coverage

The running-max pass is unit-tested in
[`models/numerical.rs`](../crates/fitdistcp-core/src/models/numerical.rs) for the
three canonical cases: already-monotone input (flag `false`), a single inversion
(flag `true`, corrected), and all-equal input (flag `false`).

---

## 8. Calibration experiment and hail-region divergence (Phase C)

This section records the **post-fix** headline metrics from the Phase-1
numerical-issues report, reproduced by two fixed-seed harnesses added in Phase C
of [`plans/phase1-remaining-fixes-plan.md`](../../plans/phase1-remaining-fixes-plan.md).
Both harnesses are ordinary `cargo test` integration tests (no R/Python/network
at test time) and are deterministic for their fixed seeds.

### 8.1 Calibration experiment

Harness: [`crates/fitdistcp-core/tests/calibration_experiment.rs`](../crates/fitdistcp-core/tests/calibration_experiment.rs).

The harness draws `N_TRIALS = 200` training samples of size `n` from a **true**
GEV with shape `xi = 0.1` (a heavy upper tail), fits the GEV by MLE, forms the CP
**1-in-100** predictive quantile (`p = 0.99`), maps it back through the true CDF,
and reports the **exceedance rate ratio** = realized / nominal, where nominal is
`1 - p = 0.01`. A ratio of `1.0` is perfectly calibrated; `> 1` means the CP
1-in-100 quantile is too low (the true tail is exceeded more often than nominal).

| `n`  | pre-fix ratio (report) | post-fix ratio (measured) | realized rate |
|------|------------------------|---------------------------|---------------|
| 12   | 2.35x                  | **2.15x**                 | 0.02150       |
| 20   | 1.89x                  | **1.64x**                 | 0.01638       |
| 40   | 0.91x                  | **1.12x**                 | 0.01118       |

**Result.** At the small sample sizes where the defect was reported (`n = 12`,
`n = 20`) the post-fix ratio is **strictly closer to nominal** than the pre-fix
reference value (2.15x < 2.35x, 1.64x < 1.89x). At `n = 40` the ratio is near
nominal (1.12x).

**Known limitation (documented residual).** The post-fix ratios are **not**
within a tight band of nominal at small `n`: the CP 1-in-100 quantile is still
somewhat under-estimated at `n = 12` and `n = 20`. This is a **known limitation**
of the first-order DMGS expansion in the extreme tail at small `n` — the Phase A
monotonicity/support guard restores the *validity* of the quantile curve (monotone,
within support) but does not remove the expansion's small-`n` tail bias. The
harness therefore asserts a documented **ceiling** (`RATIO_CEILING = 3.0`) rather
than a tight tolerance, plus the strict small-`n` improvement over the pre-fix
values. The residual is Monte-Carlo-limited: the binomial standard error of the
realized rate at `p = 0.01` over 200 trials is `sqrt(0.01·0.99/200) ≈ 0.0070`,
i.e. a ratio standard error of `≈ 0.70`.

### 8.2 Hail-region divergence

Harness: [`crates/fitdistcp-core/tests/hail_region_divergence.rs`](../crates/fitdistcp-core/tests/hail_region_divergence.rs).

The real hail data is **not** in the repository, so the harness uses a
**synthetic surrogate** generator ([`synthetic_hail_regions`](../crates/fitdistcp-core/tests/hail_region_divergence.rs),
fixed seed) that reproduces the short-record, zero-inflated shape of the real
MESH / hail-report corpora: each region is a short (n = 10-40) series of annual
maxima with a large fraction of zero years, dropped before fitting (the reference
fits GEV on the **non-zero years** only). For each region the harness fits the
GEV, forms the CP 1-in-100 return level, and counts the CP-vs-MLE relative
divergences (`> 1%`, `> 10%`) and non-finite CP return levels.

| Corpus (synthetic surrogate, 2000 regions) | `> 1%` | `> 10%` | degenerate (item B2) | huge finite | non-finite |
|--------------------------------------------|--------|---------|----------------------|-------------|------------|
| measured (post-fix, items A–F)             | 1063 (57.2%) | 1017 | **0** | 192 | 9 |
| pre-fix reference (real MESH)              | 39.1%  | 794     | — | — | 121 |
| pre-fix reference (real hail reports)      | 18.9%  | 329     | — | — | 42  |

**Result (item B).** The **degenerate-fit count is zero** — the item-B1 relative
guard rejects the `scale -> 0` spurious maximum, so the multi-start finds a valid
interior candidate on every region and no fit is flagged `degenerate_fit`. This
is the item-B guarantee and the harness asserts it **exactly** (`== 0`).

**Result (items A and C analogues).** The remaining 192 **huge finite** and 9
**non-finite** CP return levels are **not** the item-B degeneracy:

- The 192 huge-finite regions all have **valid** parameters
  (`degenerate_fit == false`) and a **large positive shape** (≥ 3.2, median
  ~7.4) — the DMGS small-`n` blow-up, i.e. the GEV analogue of item C, which item
  B does not address.
- The 9 non-finite regions are GEV fits with shapes in `[-0.998, -0.967]` — the
  DMGS expansion being singular at a near-−1 shape, again item C's GEV analogue.
  **Item A unmasked these**: the removed `ml_max`/`fixgevrange` clamp had been
  masking them as finite values. This matches the reference, which does not clamp
  either.

Both counts are **pinned** in the harness (`KNOWN_LARGE_POSITIVE_SHAPE_HUGE = 192`,
`KNOWN_NEAR_MINUS_ONE_NONFINITE = 9`) so a regression that reintroduces the
item-B degeneracy (which would raise them) is caught. Wiring the shared
`near_minus_one_shape` predicate into the GEV `revert2ml` would remove the 9
non-finite regions; that is left as a documented follow-up (plan §7 open
question 3) so the GEV goldens are not disturbed.

**Known limitation (documented residual).** The synthetic-surrogate divergence
percentages are **not** directly comparable to the real-data pre-fix percentages:
the surrogate's short-record / zero-inflation parameters are a modelling choice,
not the real corpus, so the absolute counts differ. The harness therefore asserts
the **structural** invariants (degenerate count is zero; the huge-finite and
non-finite counts equal their pinned values; `> 10%` count does not exceed the
`> 1%` count; counts do not exceed the fitted-region count) and records the
counts as a baseline. **The real-data counts must be re-run by the reporter**
against the actual MESH / hail-report corpora to confirm the headline
improvement.

### 8.3 Verification commands

```text
docker compose run --rm dev cargo test -p fitdistcp-core --test calibration_experiment -- --nocapture
docker compose run --rm dev cargo test -p fitdistcp-core --test hail_region_divergence -- --nocapture
```
