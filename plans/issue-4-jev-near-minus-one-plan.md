# Plan: Issue #4 — GEV near-−1 DMGS singularity (9 hail regions)

Branch: `issue-4-jev-near-minus-one` (created from `main`).
Scope: **plan only** — no code in this document's authoring step.

---

## 0. Resumable status checklist

> **Resumption protocol (mandatory).**
> Every subtask MUST, before it finishes, (a) flip its own line below to `[x]`,
> (b) update the **Last updated** timestamp, and (c) update the **Current step**
> line. If a subtask is partially done, mark it `[-]` and append a short note in
> the "Notes" column. This file is the single source of truth for resuming an
> interrupted run — do not rely on conversation history.

**Last updated:** 2026-11-23T04:06Z (S9 complete: branch pushed, CI green, PR opened)
**Current step:** complete — all subtasks done

| # | Subtask | Status | Notes |
|---|---------|--------|-------|
| S1 | Calibrate GEV near-−1 threshold; record evidence in §3.3 | `[x]` | **GEV-specific `delta = 0.04` (`shape <= -0.96`)** — shared `0.02` left 2 residuals; onset record-dependent `[-0.9755, -0.900]`, nearest legit ref shape `-0.9140` |
| S2 | Wire guard into `GenExtreme::revert2ml` (+ `_p1`/`_p12` per §3.2 decision) | `[x]` | All three wired via `gev_near_minus_one_shape`; non-finite 9 → 0, huge finite stays 192 |
| S3 | Update `near_minus_one_shape` doc comment in `models/mod.rs` | `[x]` | Added `GEV_NEAR_MINUS_ONE_DELTA` + `gev_near_minus_one_shape` docs; no stale "follow-up" wording; `cargo doc` clean (only pre-existing `perf_timing` `rvs` warning) |
| S4 | Update/extend unit tests (`genextreme.rs`, `api.rs`, `_p1`/`_p12`) | `[x]` | Flipped `-0.99` assertion; added `revert2ml_fires_near_minus_one` + `_p1`/`_p12` parity tests; 214 lib tests pass |
| S5 | Add golden generator + fixture + harness for near-−1 GEV records | `[x]` | `tools/generate_gev_near_minus_one_golden.R` + `tests/golden/gev_near_minus_one_golden.json` + `crates/fitdistcp-core/tests/gev_near_minus_one_golden.rs`; 3 tests pass, fixture byte-identical on regen, harness fails without the guard |
| S6 | Tighten `hail_region_divergence` non-finite assertion (target 0) | `[x]` | `KNOWN_NEAR_MINUS_ONE_NONFINITE = 0`; measured: non-finite 0, huge finite 192 (unchanged), `> 1%` 1080 (58.1%), `> 10%` 1075 |
| S7 | Update `tests/golden/TOLERANCE.md` (§3.14c / §8.2) and `tests/golden/README.md` | `[x]` | Added §3.14d, §5 constants row, §8.2 post-Issue-#4 row + resolution; README inventory row |
| S8 | Full local verification in the dev container | `[x]` | All green: build, 214 lib + all harnesses, clippy `-D warnings`, fmt, bench `--no-run`; R 221 pass; Python 141 pass/1 skip (updated `test_gev_p1_accepts_predictor_args` for the reverted `_p1` fit) |
| S9 | Push branch, watch GitHub Actions green, open PR with `Closes #4` | `[x]` | Pushed `3b6d02e`; CI run green (Detect source changes, Build dev Docker image, Rust pinned, Rust stable, R testthat, Python maturin+pytest all `success`; perf comparison skipped as manual-only); PR opened with `Closes #4` |

---

## 1. Problem statement

In the hail-region corpus harness
([`hail_region_divergence.rs`](crates/fitdistcp-core/tests/hail_region_divergence.rs:306)),
**9 of 1859** fitted regions produce **non-finite** GEV CP quantiles. The harness
currently pins this at exactly 9 (`KNOWN_NEAR_MINUS_ONE_NONFINITE`).

These 9 are GEV fits whose ML shape lands in `[-0.998, -0.967]` (7 of 9 within
`-0.98`). The reference `qgev_cp` guard is `revert2ml = |shape| >= 1`, so a shape
just **above** −1 is *not* reverted, and the DMGS expansion is evaluated there.

This is the GEV analogue of the GPD defect fixed in #3 (item C1), which was
deliberately left out of scope in #3 so the GEV goldens were not disturbed.

## 2. Root-cause analysis

The DMGS bias-correction expansion contains a `1/fhat` term (the derivative of the
CDF w.r.t. the shape, `gev_mu2fa`-style). As `shape -> -1+`:

- `fhat` (the shape-derivative of the GEV CDF at the fitted quantile) tends to a
  value that makes `1/fhat` diverge;
- the CP quantiles leave the GEV support `y < loc - scale/shape` (finite upper
  endpoint for negative shape);
- the monotone guard then either flattens the curve far above the data or the
  arithmetic overflows to `NaN`/`inf`.

The reference `optim` never lands in the narrow `(-1, -0.98)` band on these
records, so the reference never exhibits the blow-up; the Rust multi-start does.
The GPD fix (#3) established exactly this mechanism and resolved it by widening
`revert2ml` to fire at `shape <= -0.98`, which sets the CP output equal to the
(finite) ML output.

**Where the guard is applied.** In
[`predict_gev`](crates/fitdistcp-core/src/api.rs:496) the flag
`let revert2ml = model.revert2ml(shape);` selects between the DMGS CP path and
`(ml_pdf.clone(), ml_quantiles.clone())`. Widening the predicate is therefore
sufficient — no change to `api.rs` control flow is needed.

## 3. Design decisions

### 3.1 Reuse `near_minus_one_shape()` vs. a GEV-specific predicate

**Decision: reuse the shared predicate**
[`near_minus_one_shape()`](crates/fitdistcp-core/src/models/mod.rs:229).

Rationale:

- The mechanism is identical (DMGS `1/fhat` blow-up as `shape -> -1+`), so a
  second predicate would duplicate logic and invite drift.
- The predicate was factored out in #3 explicitly for this reuse; its doc comment
  names the GEV follow-up.
- The constant name `GPD_NEAR_MINUS_ONE_DELTA` becomes slightly misleading once
  shared. **Decision:** keep the constant name (renaming touches the GPD fix and
  its docs for no behavioural gain) but update the doc comment to state it is
  shared by GPD and GEV, and record the GEV calibration alongside the GPD one.
  If S1 shows the GEV needs a *different* delta, introduce
  `GEV_NEAR_MINUS_ONE_DELTA` and a GEV-specific wrapper — see §3.3.

### 3.2 Apply to `_p1` / `_p12`?

**Decision: yes — wire all three** (`GenExtreme`, `GenExtremeP1`,
`GenExtremeP12`).

Rationale:

- [`genextreme_p1.rs:167`](crates/fitdistcp-core/src/models/genextreme_p1.rs:167)
  and [`genextreme_p12.rs:264`](crates/fitdistcp-core/src/models/genextreme_p12.rs:264)
  carry the identical `shape.abs() >= 1.0` guard and the identical DMGS
  expansion, so they have the identical singularity.
- The issue names only the plain GEV because that is what the hail harness
  exercises; leaving the variants unguarded would be a latent bug of the same
  class.
- The `_p1`/`_p12` goldens (`w11_cp_golden`, `mle_golden`) use shapes far from
  −1, so the widened guard should not fire on them — S8 verifies this.

If S8 shows a `_p1`/`_p12` golden regression, fall back to wiring only
`GenExtreme` and file a follow-up issue for the variants (document the decision
in §3.2 and in the PR body).

### 3.3 Threshold: reuse `0.02` or calibrate a GEV-specific delta?

**Decision: calibrate first (S1), then choose.** Default expectation is that
`0.02` (`shape <= -0.98`) is correct for GEV too, because the blow-up mechanism
and the reference's `optim` behaviour are the same. The calibration must confirm
it rather than assume it.

**Calibration procedure (S1):**

1. Reproduce the 9 affected regions from the synthetic surrogate
   (`synthetic_hail_regions(SEED, N_REGIONS)`, fixed seed) and dump, for each:
   the non-zero years `y`, the Rust `ml_params`, and the reference `ml_params` +
   `cp_quantiles` from `qgev_cp`.
2. For each affected record, hold `(loc, scale)` at the fitted values and sweep
   `shape` over `[-0.999, -0.90]` in fine steps (e.g. `1e-4`). Record, per shape:
   the raw DMGS CP quantile at `p = 0.99`, whether it is finite, and whether it
   exceeds the support bound `loc - scale/shape`.
3. Identify the **blow-up onset** (first shape, scanning from −0.90 downward, at
   which the CP quantile becomes non-finite or leaves the support) and the
   **nearest legitimate shape** (the closest shape above the band at which the
   reference `optim` lands on any of the 1859 records, or on a broader sweep of
   short records).
4. Choose `delta` so that `-1 + delta` sits **above** the onset and **below** the
   nearest legitimate shape, with headroom on both sides — mirroring the GPD
   result (`onset ≈ -0.9835`, nearest legitimate `-0.95`, chosen `-0.98`).
5. Record the numbers in §3.3 (below) and in the `mod.rs` doc comment.

**Calibration evidence (S1, measured 2026-11-23):**

| Quantity | Value |
|---|---|
| Affected records (count) | **9** (n = 3–8; regions 144, 349, 372, 750, 824, 901, 1334, 1373, 1647 of the fixed-seed surrogate) |
| Rust ML shape range | **`[-0.9983, -0.9669]`** (7 of 9 within `-0.98`; 2 in `(-0.98, -0.967]`) |
| Blow-up onset (shape) | **record-dependent, `[-0.9755, -0.900]`** — first shape (sweeping down from `-0.90` at `1e-4`) at which the raw DMGS CP `q99` exceeds the support bound `loc - scale/shape`. The two residuals that survive `-0.98` are at `-0.9669` and `-0.9767` |
| CP goes `NaN` | at the fitted shapes `-0.9669` (n=8) and `-0.9767` (n=5); the `1/fhat` term overflows |
| Reference `optim` shape on the 9 records | **8 of 9 `<= -1`** (revert: `-1.09` … `-1.39`); 1 at **`+0.0346`** — **none** in `(-1, -0.90)` |
| Nearest legitimate reference shape above −1 | **`-0.9140`** (20 000 short `n = 3–8` records; only 2 landed in `(-1, -0.90)`, at `-0.9342` and `-0.9140`; 8537 landed `<= -1`) |
| Chosen delta | **`0.04`** — new `GEV_NEAR_MINUS_ONE_DELTA` + `gev_near_minus_one_shape()` (GPD `0.02` left 2 residuals) |
| Chosen threshold | **`shape <= -0.96`** |

**Decision: introduce a GEV-specific delta (`0.04`, `shape <= -0.96`).** The
shared `0.02` (`shape <= -0.98`) was tried first and **left 2 residuals** (the
`-0.9669` and `-0.9767` records), because the GEV onset is record-dependent and
reaches further from −1 than the GPD's tight `(-0.988, -0.9835)` band. An
empirical sweep of the delta against the hail harness gave:

| delta | threshold | non-finite | `> 1%` | `> 10%` | huge finite |
|---|---|---|---|---|---|
| 0.02 | `-0.98` | 2 | 1085 | 1080 | 192 |
| 0.03 | `-0.97` | 1 | 1083 | 1078 | 192 |
| **0.04** | **`-0.96`** | **0** | **1080** | **1075** | **192** |
| 0.05 | `-0.95` | 0 | 1079 | 1074 | 192 |
| 0.07 | `-0.93` | 0 | 1078 | 1073 | 192 |

`0.04` is the **smallest** delta that drives the non-finite count to 0, and it
sits well below the nearest legitimate reference shape (`-0.9140`), leaving
headroom. Only **14 of 1859** corpus fits fall in `(-1, -0.96]`, all near-−1 (the
reference never lands there), so over-triggering is negligible. The GPD delta is
left untouched (item C1 calibration preserved); the two predicates are kept
separate per the §3.3 fallback.

Diagnostics used (temporary, not committed): a `#[ignore]`d Rust test
(`crates/fitdistcp-core/tests/zz_diag_gev_near_minus_one.rs`) for the record dump
and the shape sweep, and [`tools/diag_gev_near_minus_one.R`](tools/diag_gev_near_minus_one.R)
for the reference `optim` fits and the 20 000-record sweep.

**Fallback.** If the GEV onset is materially different from the GPD onset, add
`pub(crate) const GEV_NEAR_MINUS_ONE_DELTA: f64 = <calibrated>;` and a
`gev_near_minus_one_shape(shape)` wrapper in
[`models/mod.rs`](crates/fitdistcp-core/src/models/mod.rs:220), leaving the GPD
constant untouched. Document why the two differ.

### 3.4 Guard shape (exact predicate)

Mirror the GPD form so the two read identically:

```rust
pub fn revert2ml(&self, shape: f64) -> bool {
    shape.abs() >= 1.0 || crate::models::near_minus_one_shape(shape)
}
```

Note the GEV has **no** `shape > 30` clause (that is GPD-specific); do not add
one — the 192 large-positive-shape huge-finite regions are explicitly out of
scope (§5).

## 4. Subtasks, acceptance criteria, commands

All commands run **inside the dev container** (AGENTS.MD). Never on the host.

### S1 — Calibrate the GEV near-−1 threshold

- Implement the sweep as a throwaway diagnostic (a `#[test]` behind
  `--ignored`, or a `tools/diag_gev_near_minus_one.R` / small Rust bin) — do not
  commit a permanent diagnostic unless it is cheap and useful.
- Fill in the §3.3 evidence table.

**Acceptance:** evidence table complete; chosen delta justified with onset and
nearest-legitimate-shape numbers; decision recorded (reuse `0.02` or add a GEV
constant).

```sh
docker compose run --rm dev cargo test -p fitdistcp-core --test hail_region_divergence -- --nocapture
```

### S2 — Wire the guard into `revert2ml`

- [`genextreme.rs:224`](crates/fitdistcp-core/src/models/genextreme.rs:224):
  add `|| crate::models::near_minus_one_shape(shape)`.
- [`genextreme_p1.rs:167`](crates/fitdistcp-core/src/models/genextreme_p1.rs:167)
  and [`genextreme_p12.rs:264`](crates/fitdistcp-core/src/models/genextreme_p12.rs:264):
  same change (per §3.2).
- Update each doc comment: the reference guard is `|shape| >= 1`, but the Rust
  guard additionally reverts in the near-−1 band (Rust-only divergence, same
  rationale as GPD item C1).

**Acceptance:** the three predicates return `true` for `-0.99` and `false` for
`-0.5`; `cargo build --workspace` clean.

```sh
docker compose run --rm dev cargo build --workspace
```

### S3 — Update the `near_minus_one_shape` doc comment

- [`models/mod.rs:222-231`](crates/fitdistcp-core/src/models/mod.rs:222): remove
  the "not wired in here yet" sentence; state it is shared by GPD and GEV
  (`GenExtreme`, `GenExtremeP1`, `GenExtremeP12`).
- Extend the calibration section (§3.3 of this plan) with the GEV numbers, or
  add a short "GEV calibration" paragraph referencing this plan.

**Acceptance:** no stale "follow-up" wording remains; `cargo doc` builds without
broken intra-doc links.

```sh
docker compose run --rm dev cargo doc -p fitdistcp-core --no-deps
```

### S4 — Update/extend unit tests

- [`genextreme.rs:522`](crates/fitdistcp-core/src/models/genextreme.rs:522)
  `revert2ml_matches_reference_threshold`: the assertion
  `assert!(!m.revert2ml(-0.99))` **flips** to `assert!(m.revert2ml(-0.99))`.
  Add boundary cases: `-0.98` reverts, `-0.97` does not, `-0.5` does not.
  Rename or re-document the test so it no longer claims to match the reference
  threshold exactly (it now matches the reference *plus* the near-−1 widening).
- [`api.rs:1348`](crates/fitdistcp-core/src/api.rs:1348): keep
  `assert!(GenExtreme.revert2ml(-1.2))` and `assert!(!GenExtreme.revert2ml(-0.3))`;
  add a near-−1 case and, if cheap, a `predict_gev` assertion that a record whose
  fit lands near −1 returns `cp_quantiles == ml_quantiles`.
- Add equivalent `revert2ml` unit tests for `GenExtremeP1` / `GenExtremeP12` if
  they do not already exist.

**Acceptance:** all unit tests pass; the flipped assertion is intentional and
documented in the test comment.

```sh
docker compose run --rm dev cargo test -p fitdistcp-core --lib
```

### S5 — Golden generator + fixture + harness

Mirror the GPD item-C1 fixture
([`gpd_smalln_negshape_golden.json`](tests/golden/gpd_smalln_negshape_golden.json),
[`generate_gpd_smalln_negshape_golden.R`](tools/generate_gpd_smalln_negshape_golden.R),
[`gpd_smalln_negshape_golden.rs`](crates/fitdistcp-core/tests/gpd_smalln_negshape_golden.rs)).

- **Generator** `tools/generate_gev_near_minus_one_golden.R`:
  - `source()` the reference `000_evaluate_dmgs_equation3p3.R`, `005_utils.R`,
    `110b_gev_libs.R`, `110c_gev_derivs.R` (same set as
    [`generate_gev_degenerate_golden.R`](tools/generate_gev_degenerate_golden.R:61)).
  - Use **fixed literal** short records (n = 3–8, the regime the 9 regions come
    from) whose reference MLE shape is just above −1. Derive them from the S1
    dump so they are real reproductions, then freeze them as literals for
    byte-for-byte reproducibility.
  - Emit `source`, `generator`, `tolerance`, and per-record: `y`,
    `reference_ml_params`, `reference_revert2ml`, `reference_cp_quantiles`,
    `p_grid`.
- **Fixture** `tests/golden/gev_near_minus_one_golden.json` (committed).
- **Harness** `crates/fitdistcp-core/tests/gev_near_minus_one_golden.rs`:
  - reuse the minimal JSON parser pattern from the existing GEV harnesses;
  - assert the Rust fit **reverts** (`cp_quantiles == ml_quantiles`);
  - assert the CP quantiles are **finite** and **monotone**;
  - assert the CP curve is not flat far above the data
    (`max(cp) <= MAX_CP_OVER_DATA * max(y)`, `1.1`, as in the GPD harness);
  - assert the 1-in-100 quantile matches the reference within a documented
    tolerance (loose, like `GPD_SMALLN_NEGSHAPE_TAIL_RTOL = 5e-2`, because the
    two fits land at different near-−1 shapes).

**Acceptance:** fixture regenerates deterministically; harness passes; harness
fails if the guard is reverted (verify by temporarily removing the clause).

```sh
docker compose run --rm dev Rscript tools/generate_gev_near_minus_one_golden.R
docker compose run --rm dev cargo test -p fitdistcp-core --test gev_near_minus_one_golden
```

### S6 — Tighten the hail-region non-finite assertion

- [`hail_region_divergence.rs:306`](crates/fitdistcp-core/tests/hail_region_divergence.rs:306):
  change `KNOWN_NEAR_MINUS_ONE_NONFINITE` from `9` to `0` (target), or to the
  measured residual if any remains.
- If a residual remains, document **why** (e.g. a shape in the band that is also
  degenerate, or a non-finite arising from a different mechanism) in the test
  comment and in TOLERANCE.md §8.2 — do not silently pin a non-zero value.
- **Do not touch** `KNOWN_LARGE_POSITIVE_SHAPE_HUGE = 192`
  ([`hail_region_divergence.rs:288`](crates/fitdistcp-core/tests/hail_region_divergence.rs:288)).
- Update the module doc comment (it currently says the non-finite count "should
  be zero" from the Phase-3 guard — clarify the near-−1 mechanism).

**Acceptance:** harness passes with the tightened count; the 192 count is
unchanged; the `> 1%` / `> 10%` counts are re-recorded if they shift (they will
shift slightly, since 9 regions move from non-finite to reverted-ML and are no
longer counted as `> 10%` divergences — update the printed baseline and
TOLERANCE.md §8.2 table).

```sh
docker compose run --rm dev cargo test -p fitdistcp-core --test hail_region_divergence -- --nocapture
```

### S7 — Documentation

- [`tests/golden/TOLERANCE.md`](tests/golden/TOLERANCE.md:343): add a new
  subsection (e.g. **§3.14d GEV near-−1 revert — Issue #4**) modelled on §3.14c,
  with the tolerance constants, rationale, and fixture/harness links.
- §8.2: update the counts table (non-finite 9 → 0, huge finite stays 192,
  `> 1%` / `> 10%` re-recorded) and replace the "left as a documented follow-up"
  paragraph with the resolution.
- [`tests/golden/README.md`](tests/golden/README.md:20): add the new fixture row
  to the inventory table.
- Optionally add a one-line cross-reference in
  [`plans/issue-2-recheck-fixes-plan.md`](plans/issue-2-recheck-fixes-plan.md:529)
  §7 open question 3 pointing at this plan (do not rewrite history there).

**Acceptance:** docs match the implemented behaviour and the measured counts.

### S8 — Full local verification (dev container)

Run the full CI-equivalent suite:

```sh
docker compose run --rm dev cargo build --workspace
docker compose run --rm dev cargo test --workspace
docker compose run --rm dev cargo clippy --workspace --all-targets -- -D warnings
docker compose run --rm dev cargo fmt --check
docker compose run --rm dev cargo bench --workspace --no-run
```

Also confirm the existing GEV goldens are undisturbed:

```sh
docker compose run --rm dev cargo test -p fitdistcp-core --test gev_degenerate_golden
docker compose run --rm dev cargo test -p fitdistcp-core --test gev_negshape_golden
docker compose run --rm dev cargo test -p fitdistcp-core --test gev_smalln_golden
docker compose run --rm dev cargo test -p fitdistcp-core --test gev_cp_tail_golden
docker compose run --rm dev cargo test -p fitdistcp-core --test gev_monotone_golden
docker compose run --rm dev cargo test -p fitdistcp-core --test w11_cp_golden
docker compose run --rm dev cargo test -p fitdistcp-core --test mle_golden
```

**Acceptance:** all green; no golden fixture needed regeneration; `fmt --check`
clean.

**Results (S8, measured 2026-11-23, dev container):**

| Command | Result |
|---|---|
| `cargo build --workspace` | ✅ clean |
| `cargo test --workspace` | ✅ 214 lib + all harnesses pass (incl. new `gev_near_minus_one_golden` 3/3, `hail_region_divergence` 3/3) |
| `cargo clippy --workspace --all-targets -- -D warnings` | ✅ clean |
| `cargo fmt --check` | ✅ clean |
| `cargo bench --workspace --no-run` | ✅ all benches build |
| GEV goldens (`gev_degenerate`, `gev_negshape`, `gev_smalln`, `gev_cp_tail`, `gev_monotone`, `w11_cp`, `mle`) | ✅ all pass, **no fixture regeneration needed** |
| R testthat (`devtools::load_all` + `testthat::test_dir`) | ✅ 221 pass / 0 fail |
| Python (`maturin develop` + `pytest -q`) | ✅ 141 pass / 1 skip |

**Python test update.** `crates/fitdistcp-py/tests/test_api.py::test_gev_p1_accepts_predictor_args`
asserted `cp_pdf.shape == (1,)` (the CP-quantile-derived density at `p`). That
record's `_p1` fit lands at shape `-0.9997`, which now reverts (Issue #4), so
`cp_pdf` is the ML density at `y` (length `len(y)`). The test now asserts
`revert2ml is True` and `cp_pdf.shape == Y_GEV.shape`, with a comment explaining
the reverted path. This is the expected consequence of wiring `_p1` (§3.2), not a
regression.

### S9 — Push, CI, PR

- Push `issue-4-jev-near-minus-one`.
- Watch GitHub Actions until every job is terminal (Rust, R, Python). Treat
  `mergeable_state: unstable` as "still running".
- Open the PR via the GitHub MCP server with `Closes #4`, summarising: the
  calibration numbers, the `_p1`/`_p12` decision, the 9 → 0 count change, and the
  unchanged 192 count.

**Acceptance:** CI green; PR open with `Closes #4`.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Disturbing existing GEV goldens** (the reason #3 deferred this). | The widened guard only fires for `shape <= -0.98`; all existing GEV fixtures use shapes far from −1. S8 runs every GEV golden explicitly. If one shifts, investigate before regenerating — a shift means the fixture's shape is in the band and the fixture itself documents the old (buggy) behaviour. |
| **`-0.99` unit assertion flip** in [`genextreme.rs:528`](crates/fitdistcp-core/src/models/genextreme.rs:528). | Intentional; S4 flips it and re-documents the test as "reference threshold + near-−1 widening". |
| **The 192 large-positive-shape huge-finite count must stay 192.** | The GEV guard has no `shape > 30` clause and the near-−1 predicate cannot fire for positive shapes. S6 asserts 192 unchanged. |
| **`_p1`/`_p12` goldens regress.** | S8 runs `w11_cp_golden` and `mle_golden`; fallback per §3.2 is to wire only `GenExtreme` and file a follow-up. |
| **R/Python parity.** | The guard is Rust-only (the reference never lands in the band). If any golden is mirrored in `crates/fitdistcp-py/tests/test_golden_values.py` or `crates/fitdistcp-r/tests/testthat/`, check whether it pins a near-−1 GEV CP value; if so, update it and note the deliberate divergence. The new fixture is Rust-only (no R/Python mirror) unless S5 finds an existing mirrored pattern. |
| **Threshold over-triggering on legitimate fits.** | S1's "nearest legitimate shape" step bounds the delta from below; the GPD precedent (`-0.95` nearest legitimate vs `-0.98` threshold) gives headroom. |
| **Residual non-finite after the fix.** | S6 documents any residual with its mechanism rather than pinning silently. |

## 6. Out of scope

- The 192 large-positive-shape huge-finite regions (separate defect class).
- Any change to the reference-matching `|shape| >= 1` semantics beyond the
  near-−1 widening.
- Renaming `GPD_NEAR_MINUS_ONE_DELTA` (documented as shared instead).
