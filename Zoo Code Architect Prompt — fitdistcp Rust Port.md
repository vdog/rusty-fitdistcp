# Zoo Code Architect Mode — Prompt: Rust Port of `fitdistcp`

Use this prompt in **Zoo Code Architect mode**. It is written to produce a design document and phased plan _before_ any implementation code.

* * *

## Prompt

**Context:** Reimplement the R/Python package `fitdistcp` as a compiled library with R and Python bindings, optimized for maximum performance. Do NOT implement yet — Architect mode: produce a design document and a phased plan first.
Reference material: GitHub repo `stephenjewson/fitdistcp`; full model list, method signatures, theory and examples at `www.fitdistcp.info`. Core technique is **calibrating prior prediction** — objective Bayesian prediction that reduces reliability bias vs. maximum likelihood, using the right Haar prior for homogeneous models and simple priors for inhomogeneous ones. **Phase 1 scope: optimize the standard DMGS asymptotic-expansion integration path only.** Defer posterior sampling to a later phase.

### Language and bindings (decided — design around these)

*   Core in **Rust**. Justify the choice against C++ but treat Rust as fixed.
    
*   **R bindings via extendr** (crate `extendr-api`, package `rextendr`) — frictionless R↔Rust bindings requiring only Rust and R installed. Plan the `src/rust/` crate layout and auto-generated `R/extendr-wrappers.R` per the rextendr package vignette.
    
*   **Python bindings via PyO3 + maturin + rust-numpy**, the standard stack for Rust-based Python scientific extensions.
    
*   Keep the Rust core binding-agnostic; both bindings should be thin wrappers over one shared API so numerics are identical across R and Python.
    

### What to cover in the design

1.  **Model coverage for Phase 1.** A representative subset rather than all ~35 R models: e.g. `norm`, `norm_p1`, `genextreme` (GEV), `genextreme_p1`, `genpareto` (GPD), `gumbel`, `lnorm`, `weibull`, `gamma`, `expon`. Note which use analytic integration vs. DMGS, since Phase 1 targets DMGS.
    
2.  **Numerics.** Specify the DMGS asymptotic expansion and the derivatives it needs (prefer analytic — the project notes ~most have been switched from numerical to analytic — fall back to numerical); MLE fitting (BFGS/Newton); special functions (gamma, digamma, trigamma) and which Rust crates to use; and how to keep it accurate at small n, where DMGS degrades below n≈10.
    
3.  **API surface.** Match reference signatures and return shapes so it's drop-in: the R `d/p/q/r/t` + `t***_cp` convention and fields `ml_pdf`, `cp_pdf`, `ml_quantiles`, `cp_quantiles`, `ml_params`, `cp_params`, AIC/MAIC, standard errors (plus optional WAIC and log scores). Mirror Python `pdf/cdf/ppf/rvs` with optional `p`, `means`, `waicscores`, `logscores`, `y`, and predictor handling for `genextreme_p1` (`t`, `t0`, `n0`).
    
4.  **Performance strategy.** Identify hot loops in the DMGS path; propose vectorization/SIMD, parallelism via Rayon (over prediction grids, densities, quantile batches), memory layout for contiguous arrays, and caching of reused derivative/normalization quantities. Since DMGS is already ~1000x faster than MCMC for quantiles, establish where the remaining wins actually are rather than assuming raw speed is the only goal.
    
5.  **Correctness & validation.** Reproduce the package's R Examples 1–7 and Python Examples 1–5 as golden tests, reimplement the `reltest` reliability simulation to confirm CP beats MLE, set numerical tolerances against R/Python reference outputs, and cover the negative-shape GEV case where DMGS cannot return values beyond the MLE bound and the package instead derives density from quantiles.
    
6.  **Future-proofing for posterior sampling.** Design the integration interface so a later phase can plug in posterior sampling, and note that the reference uses Paul Northrop's `rust` package (Ratio-of-Uniforms Simulation with Transformation) — flag as an open question whether to reuse its algorithm from R, reimplement it in Rust, or interop via extendr, since it is an R package not a Rust crate.
    
7.  **Project structure.** Cargo workspace (core crate + two binding crates), CMake/`rextendr` and maturin build paths, test framework, CI (R CMD check + cibuildwheel), and packaging for GitHub-release distribution (no CRAN/PyPI publishing).
    

### Output format

A Markdown design doc: language justification, module/file breakdown, phased plan with milestones, a "risks and unknowns" section, and explicit open questions for me. Do not write implementation code until I approve the plan.

* * *

## Notes for you (not part of the prompt)

*   The biggest genuine unknown to watch for in the returned design is whether `extendr` **and** `PyO3` **can cleanly share one core crate**, since both bring their own FFI and build constraints. Read that section closely before approving.
    
*   Optional: add an "recommended default decisions" block (crate names, starting model list) if you want fewer open questions bounced back.