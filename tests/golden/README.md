# Golden fixtures (W19)

This directory holds the **committed** golden fixtures used by the Rust port's
golden tests, plus the consolidated tolerance policy.

- [`TOLERANCE.md`](TOLERANCE.md) — the single source of truth for the
  golden-test tolerance policy (consolidated from design §7.2 and the per-harness
  notes).
- `*.json` — the fixtures themselves (see the inventory below).

The fixtures are **committed to the repository** and loaded by the test harnesses
via `std::fs::read_to_string` relative to `CARGO_MANIFEST_DIR`. **No network
access and no reference package are required at test time** — the reference R
package is only needed to *regenerate* the fixtures.

---

## Fixture inventory

| Fixture | Generator | Harness | Coverage |
|---|---|---|---|
| [`mle_golden.json`](mle_golden.json) | [`tools/generate_golden.R`](../../tools/generate_golden.R) | [`crates/fitdistcp-core/tests/mle_golden.rs`](../../crates/fitdistcp-core/tests/mle_golden.rs) | MLE params/SE/AIC/loglik for all 10 Phase-1 models (`norm`, `expon`, `lnorm`, `weibull`, `gamma`, `gumbel`, `genextreme`, `genpareto`, `norm_p1`, `genextreme_p1`). |
| [`cp_golden.json`](cp_golden.json) | [`tools/generate_cp_golden.R`](../../tools/generate_cp_golden.R) | [`crates/fitdistcp-core/tests/cp_golden.rs`](../../crates/fitdistcp-core/tests/cp_golden.rs) | DMGS CP pdf/quantiles for the analytic models (`norm`, `expon`, `lnorm`, `gumbel`, `weibull`, `gamma`), plus the exact analytic CP for the cross-check. |
| [`gev_negshape_golden.json`](gev_negshape_golden.json) | [`tools/generate_gev_negshape_golden.R`](../../tools/generate_gev_negshape_golden.R) | [`crates/fitdistcp-core/tests/gev_negshape_golden.rs`](../../crates/fitdistcp-core/tests/gev_negshape_golden.rs) | Negative-shape GEV quantile-derived CP density, CP quantiles, `revert2ml`, `ml_max`, and the small-n (`n = 8`) case. |
| [`w11_cp_golden.json`](w11_cp_golden.json) | [`tools/generate_w11_cp_golden.R`](../../tools/generate_w11_cp_golden.R) | [`crates/fitdistcp-core/tests/w11_cp_golden.rs`](../../crates/fitdistcp-core/tests/w11_cp_golden.rs) | CP pdf/quantiles for the four models not covered by `cp_golden.rs`: `genextreme` (neg + pos shape), `genextreme_p1`, `genpareto`, `norm_p1`. |
| [`gpd_dollar_golden.json`](gpd_dollar_golden.json) | [`tools/generate_gpd_dollar_golden.R`](../../tools/generate_gpd_dollar_golden.R) | [`crates/fitdistcp-core/tests/gpd_dollar_golden.rs`](../../crates/fitdistcp-core/tests/gpd_dollar_golden.rs) | Phase 2 (Issue 2): GPD on raw dollar-scale data (~10^3-10^5) — `setics`-derived init, `ml_params`, and `cp_quantiles` vs. the reference `qgpd_k1_cp`. |

Additional non-fixture tests (no JSON): [`w14_predictor.rs`](../../crates/fitdistcp-core/tests/w14_predictor.rs)
(predictor semantics, reference values inlined) and
[`reltest.rs`](../../crates/fitdistcp-core/tests/reltest.rs) (fixed-seed
reliability simulation).

Each fixture's JSON header records its `source`, `generator`, and the baseline
`tolerance`:

```json
{
  "source": "stephenjewson/fitdistcp (R package, v0.2.3)",
  "generator": "tools/generate_golden.R",
  "tolerance": {"rtol": 1e-8, "atol": 1e-10},
  ...
}
```

---

## Reference package version

All fixtures were generated from the reference R package:

- **Package:** `fitdistcp` by Stephen Jewson
- **Version:** **0.2.3** (recorded in [`tools/fitdistcp-ref/DESCRIPTION`](../../tools/fitdistcp-ref/DESCRIPTION))
- **Source:** <https://github.com/stephenjewson/fitdistcp>

The reference package is cloned into `tools/fitdistcp-ref/` (gitignored; see
[`.gitignore`](../../.gitignore)). The generators `source()` the reference's own
R files directly, so the fixtures come from the reference code, not a
re-implementation.

---

## How to regenerate the fixtures

All generation runs **inside the Docker dev container** (no host R toolchain is
required). From the workspace root:

```sh
# 1. Build the dev image (once).
docker compose build

# 2. Clone the reference package at the pinned version (once).
docker compose run --rm dev bash -lc \
  'git clone --depth 1 --branch v0.2.3 \
     https://github.com/stephenjewson/fitdistcp.git tools/fitdistcp-ref'

# 3. Regenerate each fixture (order is independent).
docker compose run --rm dev Rscript tools/generate_golden.R
docker compose run --rm dev Rscript tools/generate_cp_golden.R
docker compose run --rm dev Rscript tools/generate_gev_negshape_golden.R
docker compose run --rm dev Rscript tools/generate_w11_cp_golden.R
```

Each generator writes its JSON into `tests/golden/` and prints the models it
emitted. The generators install `extraDistr` into a workspace-local library
(`tools/Rlib`, gitignored) so it survives across ephemeral container runs.

### Reproducibility guarantees

- **Fixed datasets.** Every generator embeds its datasets as **fixed literals**
  (no RNG dependence), so the fixtures are byte-for-byte reproducible on any
  machine.
- **Reference code, not a re-implementation.** The generators `source()` the
  reference package's own `*_libs.R` / `*_derivs.R` / `*_cp` routines.
- **Tightly-converged optimizer.** The MLE fixtures use `optim` with
  `reltol = 1e-14` so they represent the true MLE rather than the default
  Nelder–Mead stopping point (see [`TOLERANCE.md`](TOLERANCE.md) §3.2).
- **Pinned reference version.** Clone the `v0.2.3` tag (or the commit recorded in
  `tools/fitdistcp-ref/DESCRIPTION`) so a regeneration matches the committed
  fixtures.

### After regenerating

Run the full suite to confirm the fixtures still pass at the documented
tolerances:

```sh
docker compose run --rm dev cargo test --workspace
```

If a regeneration changes a fixture value beyond tolerance, that is a signal to
investigate (a reference-version change or a genuine numerics regression) — do
**not** silently loosen a tolerance. Any intentional relaxation must be recorded
in [`TOLERANCE.md`](TOLERANCE.md) with its rationale.

---

## CI-readiness (W19)

- Fixtures are committed; tests read them from disk with no network access.
- The harnesses use a tiny in-tree JSON parser (no `serde_json` dependency).
- The reference package and R are **not** needed to run the tests — only to
  regenerate fixtures.
- The reliability simulation is deterministic (fixed seed).
- CI wiring itself is W20; this directory is the CI-ready input.
