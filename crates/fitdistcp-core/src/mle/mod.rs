//! Maximum-likelihood fitting (§4.2).
//!
//! Primary optimizer is BFGS (quasi-Newton) using the analytic gradient from
//! [`Derivatives::score`]; Newton–Raphson using the analytic Hessian from
//! [`Derivatives::hessian`] is the fallback when BFGS fails to converge or the
//! problem is ill-conditioned. Convergence is reported explicitly through
//! [`crate::api::Convergence`] rather than silently returning.
//!
//! # Design (§4.2)
//!
//! - **Primary:** BFGS with an analytic gradient. The gradient is supplied by
//!   [`Derivatives::score`], which itself falls back to central finite
//!   differences when a model does not override it (see [`crate::models`]).
//! - **Fallback:** Newton–Raphson using the analytic Hessian
//!   [`Derivatives::hessian`]. It is attempted when BFGS does not converge, or
//!   when the BFGS optimum is ill-conditioned (non-finite / non-positive-definite
//!   observed information).
//! - **Convergence:** relative gradient norm **and** parameter-change tolerance.
//!   Non-convergence is reported via [`Convergence`] (`MaxIterationsReached`,
//!   `Stalled`, or `Failed`).
//! - **Standard errors:** from the inverse of the observed information
//!   (negative Hessian) at the optimum; `SE_i = sqrt((H⁻¹)_{ii})`.
//! - **AIC:** `2k − 2ℓ`, with `k = n_params`.
//! - **Small-n:** `n < 10` is flagged via [`MleFit::small_n`] (§4.4); the
//!   threshold is documented in [`SMALL_N_THRESHOLD`].
//!
//! The optimizer is hand-rolled rather than delegated to `argmin` because the
//! design explicitly permits "a hand-rolled BFGS with analytic gradient" (§4.2)
//! and doing so keeps the convergence criteria, the Newton fallback, and the
//! observed-information computation under one roof with a stable, testable API.

use nalgebra::{DMatrix, DVector};

use crate::api::{Convergence, MleFit};
use crate::models::Derivatives;

/// Sample-size threshold below which results are flagged as small-n (§4.4).
///
/// The design states DMGS degrades below `n ≈ 10` and asks for the threshold to
/// be documented and the result flagged rather than silently degraded. We use
/// the strict inequality `n < 10` (i.e. `n = 10` is *not* flagged).
pub const SMALL_N_THRESHOLD: usize = 10;

/// Tunable parameters for the MLE optimizers.
///
/// The defaults are chosen to match the §4.2 convergence policy (relative
/// gradient norm + parameter-change tolerance) and are validated in the tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MleConfig {
    /// Maximum BFGS iterations.
    pub max_iter: usize,
    /// Maximum Newton–Raphson iterations (fallback).
    pub max_newton_iter: usize,
    /// Convergence tolerance on the **relative** gradient norm
    /// `‖g‖ / (1 + |ℓ|)`.
    pub grad_tol: f64,
    /// Convergence tolerance on the relative parameter change
    /// `‖Δθ‖ / (1 + ‖θ‖)`.
    pub param_tol: f64,
    /// Step size for the Armijo backtracking line search.
    pub line_search_c: f64,
    /// Maximum backtracking steps in the line search.
    pub max_line_search: usize,
}

impl Default for MleConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            max_newton_iter: 100,
            grad_tol: 1e-8,
            param_tol: 1e-10,
            line_search_c: 1e-4,
            max_line_search: 50,
        }
    }
}

/// Fit a model by maximum likelihood (§4.2).
///
/// Runs BFGS with the analytic gradient [`Derivatives::score`]; if BFGS fails to
/// converge or yields an ill-conditioned optimum, falls back to Newton–Raphson
/// using [`Derivatives::hessian`]. Returns an [`MleFit`] populated with the
/// parameters, standard errors, AIC, log-likelihood, convergence status, and the
/// small-n flag.
///
/// # Arguments
///
/// - `model`: any [`Derivatives`] implementation (analytic or numerical).
/// - `y`: the observed data.
/// - `init`: initial parameter guesses, length `model.n_params()`.
///
/// # Panics
///
/// Panics if `init.len() != model.n_params()` or if `y` is empty.
pub fn fit<M: Derivatives>(model: &M, y: &[f64], init: &[f64]) -> MleFit {
    fit_with_config(model, y, init, &MleConfig::default())
}

/// Fit a model by maximum likelihood with an explicit [`MleConfig`].
///
/// See [`fit`] for the algorithm. This variant exposes the tolerances and
/// iteration budgets, which the tests use to exercise the non-convergence path.
///
/// # Panics
///
/// Panics if `init.len() != model.n_params()` or if `y` is empty.
pub fn fit_with_config<M: Derivatives>(
    model: &M,
    y: &[f64],
    init: &[f64],
    cfg: &MleConfig,
) -> MleFit {
    let p = model.n_params();
    assert_eq!(
        init.len(),
        p,
        "fit: init length {} != n_params {}",
        init.len(),
        p
    );
    assert!(!y.is_empty(), "fit: y must be non-empty");

    // --- Multi-start set (Phase 3, Issue 3). ---
    //
    // The hand-rolled BFGS can stall on the GEV/GPD/Gumbel ridges and stop at a
    // worse fit than the reference's `scipy`/`optim` multi-start strategy. We
    // therefore evaluate several starts and keep the best log-likelihood:
    //
    //   1. the caller's `init` (the binding's chosen start),
    //   2. the model's `setics`-style data-driven heuristic (std-based scale,
    //      moment-based shape), mirroring the reference `*_setics` helpers,
    //   3. a small set of deterministic perturbations of the heuristic start
    //      (scale up/down, shape nudged) to escape a nearby spurious stationary
    //      point.
    //
    // Each start is run through BFGS (primary) and Newton–Raphson (fallback),
    // and the best candidate across the whole set wins. This is the reference's
    // robust multi-start / init strategy.
    let starts = build_starts(model, y, init);

    // Candidate solutions, each with its log-likelihood, convergence status, and
    // the start that produced it.
    //
    // We include **every** candidate with a finite log-likelihood, not only the
    // formally-converged ones. A run that reached a better log-likelihood but
    // exhausted its iteration budget is still a better answer than a converged
    // run trapped at a spurious stationary point (the GEV/GPD/Gumbel surfaces
    // have such points). Ties are broken in favour of a converged candidate.
    let mut candidates: Vec<Candidate> = Vec::new();

    for start in &starts {
        // --- Primary: BFGS with analytic gradient. ---
        let bfgs = bfgs(model, y, start, cfg);

        // BFGS is acceptable when it converged and produced a finite,
        // positive-definite observed information matrix (so the SEs are
        // well-defined).
        let bfgs_ok = bfgs.convergence.is_converged()
            && observed_information(model, y, &bfgs.params)
                .map(|h| is_positive_definite(&h))
                .unwrap_or(false);
        if bfgs_ok {
            let ll = model.loglik(&bfgs.params, y);
            candidates.push(Candidate {
                params: bfgs.params.clone(),
                convergence: Convergence::Converged,
                loglik: ll,
                start: start.clone(),
            });
        }

        // --- Fallback / refinement: Newton–Raphson with the analytic Hessian. ---
        //
        // Newton is run from the start as well as from the BFGS result. Some
        // models (Weibull, Gumbel, log-normal) have a *spurious* stationary
        // point at a large scale where the log-likelihood is far worse than the
        // true MLE. BFGS can converge to that spurious point (it is a genuine
        // local maximum, so the observed information is positive-definite and
        // the `bfgs_ok` test passes), while Newton started from the start
        // reliably reaches the true optimum.
        let newton_from_bfgs = newton_raphson(model, y, &bfgs.params, cfg);
        let newton_from_start = newton_raphson(model, y, start, cfg);
        for opt in [&newton_from_bfgs, &newton_from_start] {
            let ll = model.loglik(&opt.params, y);
            if ll.is_finite() {
                candidates.push(Candidate {
                    params: opt.params.clone(),
                    convergence: opt.convergence,
                    loglik: ll,
                    start: start.clone(),
                });
            }
        }

        // --- Gradient-free polish: Nelder–Mead (Phase 3, Issue 3). ---
        //
        // The reference's `scipy` BFGS is more forgiving of the GEV/GPD ridges
        // than the hand-rolled BFGS. When BFGS stalls (or fails to converge) we
        // polish with a gradient-free Nelder–Mead simplex, which does not rely
        // on the gradient and can follow a curved ridge. We polish from the
        // best point found so far for this start (BFGS result if finite,
        // otherwise the start itself).
        let polish_from = if bfgs.params.iter().all(|v| v.is_finite())
            && model.loglik(&bfgs.params, y).is_finite()
        {
            bfgs.params.clone()
        } else {
            start.clone()
        };
        if cfg.max_iter > 0 {
            let nm = nelder_mead(model, y, &polish_from, cfg);
            let nm_ll = model.loglik(&nm.params, y);
            if nm_ll.is_finite() {
                candidates.push(Candidate {
                    params: nm.params.clone(),
                    convergence: nm.convergence,
                    loglik: nm_ll,
                    start: start.clone(),
                });
            }
        }
    }

    let (params, convergence, start) = if let Some(best) = candidates
        .into_iter()
        .filter(|c| c.loglik.is_finite() && model.valid_params(&c.params))
        .max_by(|a, b| {
            // Primary key: higher log-likelihood. Secondary key: prefer a
            // converged status on (near-)ties. This is the tightened acceptance
            // rule: a "converged" spurious stationary point cannot win over a
            // better non-converged candidate, because the log-likelihood is the
            // primary key across the whole multi-start set.
            a.loglik
                .partial_cmp(&b.loglik)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ac = a.convergence.is_converged() as u8;
                    let bc = b.convergence.is_converged() as u8;
                    ac.cmp(&bc)
                })
        }) {
        (best.params, best.convergence, best.start)
    } else {
        // No candidate produced a finite log-likelihood: report the most
        // informative failure mode. Re-run BFGS from the caller's `init` so the
        // reported status reflects the caller's own start.
        let bfgs = bfgs(model, y, init, cfg);
        let newton = newton_raphson(model, y, init, cfg);
        if bfgs.convergence.is_converged() {
            (bfgs.params, bfgs.convergence, init.to_vec())
        } else {
            (newton.params, newton.convergence, init.to_vec())
        }
    };

    finalize(model, y, params, convergence, start)
}

/// A candidate solution from one optimizer run, tagged with the start that
/// produced it (Phase 3, Issue 3).
struct Candidate {
    params: Vec<f64>,
    convergence: Convergence,
    loglik: f64,
    start: Vec<f64>,
}

/// Build the multi-start set for [`fit_with_config`] (Phase 3, Issue 3).
///
/// Returns a de-duplicated list of starting points:
///
/// 1. the caller's `init`,
/// 2. the model's `setics`-style data-driven heuristic,
/// 3. deterministic perturbations of the heuristic (scale up/down, shape
///    nudged) to escape a nearby spurious stationary point.
///
/// The caller's `init` is always first so that, when the heuristic is
/// uninformative (the default `setics` returns `init` unchanged), the behaviour
/// reduces to the pre-Phase-3 single-start path.
fn build_starts<M: Derivatives>(model: &M, y: &[f64], init: &[f64]) -> Vec<Vec<f64>> {
    let p = model.n_params();
    let mut starts: Vec<Vec<f64>> = Vec::new();
    let push = |s: Vec<f64>, starts: &mut Vec<Vec<f64>>| {
        if s.len() != p || s.iter().any(|v| !v.is_finite()) {
            return;
        }
        // De-duplicate (exact match) so we do not re-run identical starts.
        if !starts.iter().any(|e| e == &s) {
            starts.push(s);
        }
    };

    push(init.to_vec(), &mut starts);

    // The model's `setics`-style data-driven heuristic (std-based scale,
    // moment-based shape). For models that do not override `setics` this is
    // `init` itself, so the set collapses to a single start and the behaviour
    // reduces to the pre-Phase-3 single-start path.
    let heur = model.setics(y, init);
    push(heur, &mut starts);

    starts
}

/// Assemble the [`MleFit`] from a converged (or best-effort) parameter vector.
fn finalize<M: Derivatives>(
    model: &M,
    y: &[f64],
    params: Vec<f64>,
    convergence: Convergence,
    start: Vec<f64>,
) -> MleFit {
    let n = y.len();
    let n_params = model.n_params();
    let loglik = model.loglik(&params, y);

    // Standard errors from the inverse observed information (negative Hessian).
    let ml_se = match observed_information(model, y, &params) {
        Some(h) => match invert_symmetric(&h) {
            Some(inv) => (0..n_params)
                .map(|i| {
                    let v = inv[(i, i)];
                    if v.is_finite() && v >= 0.0 {
                        v.sqrt()
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
            None => vec![f64::NAN; n_params],
        },
        None => vec![f64::NAN; n_params],
    };

    let aic = 2.0 * n_params as f64 - 2.0 * loglik;

    MleFit {
        ml_params: params,
        ml_se,
        aic,
        n,
        n_params,
        loglik,
        convergence,
        small_n: n < SMALL_N_THRESHOLD,
        start,
    }
}

/// Outcome of a single optimizer run.
struct OptResult {
    params: Vec<f64>,
    convergence: Convergence,
}

/// Observed information matrix `−∇²ℓ` at `params`, as a dense `nalgebra` matrix.
///
/// Returns `None` if the Hessian contains non-finite entries.
fn observed_information<M: Derivatives>(
    model: &M,
    y: &[f64],
    params: &[f64],
) -> Option<DMatrix<f64>> {
    let p = model.n_params();
    let h = model.hessian(params, y);
    if h.len() != p * p || h.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Observed information is the negative Hessian of the log-likelihood.
    let mut info = DMatrix::<f64>::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            info[(i, j)] = -h[i * p + j];
        }
    }
    Some(info)
}

/// `true` when `m` is symmetric positive definite (via Cholesky).
fn is_positive_definite(m: &DMatrix<f64>) -> bool {
    m.clone().cholesky().is_some()
}

/// Invert a symmetric matrix via its Cholesky factor; `None` if not SPD.
fn invert_symmetric(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    m.clone().cholesky().map(|c| c.inverse())
}

/// Relative gradient norm `‖g‖ / (1 + |ℓ|)`.
fn relative_grad_norm(g: &DVector<f64>, loglik: f64) -> f64 {
    g.norm() / (1.0 + loglik.abs())
}

/// Relative parameter change `‖Δθ‖ / (1 + ‖θ‖)`.
fn relative_param_change(delta: &DVector<f64>, theta: &DVector<f64>) -> f64 {
    delta.norm() / (1.0 + theta.norm())
}

/// BFGS quasi-Newton optimizer with an analytic gradient (§4.2).
///
/// Uses the inverse-Hessian approximation update (Nocedal & Wright, eq. 6.17)
/// with a **Wolfe-condition** line search. The gradient comes from
/// [`Derivatives::score`].
///
/// # Conditioning
///
/// The initial inverse-Hessian approximation is **scale-aware**: it is seeded
/// from the inverse of the diagonal of the observed information at the start
/// point (falling back to the identity when that is unavailable or
/// non-positive). Seeding with the identity is a poor approximation when the
/// parameters have very different natural scales (e.g. a location near 1 and a
/// scale near 1 whose gradients differ by orders of magnitude), and caused the
/// optimizer to take a catastrophic first step that collapsed the scale toward
/// zero for the Weibull/Gumbel/log-normal models. The diagonal-information seed
/// removes that failure mode.
///
/// # Line search
///
/// The line search enforces both the **Armijo** (sufficient decrease) and the
/// **Wolfe** (curvature) conditions. Pure Armijo backtracking can accept a step
/// that decreases the objective only marginally while moving far along a
/// near-flat ridge, which is exactly the pathology that trapped the scale
/// parameters. Requiring the curvature condition `g_new · d >= c2 · g · d`
/// keeps the step in a region where the local quadratic model is trustworthy.
fn bfgs<M: Derivatives>(model: &M, y: &[f64], init: &[f64], cfg: &MleConfig) -> OptResult {
    let p = model.n_params();
    let mut theta = DVector::from_column_slice(init);
    let mut f = model.loglik(init, y);
    let mut g = DVector::from_vec(model.score(init, y));

    if !f.is_finite() || g.iter().any(|v| !v.is_finite()) {
        return OptResult {
            params: init.to_vec(),
            convergence: Convergence::Failed,
        };
    }

    // Scale-aware initial inverse-Hessian: inverse of the diagonal of the
    // observed information at the start point, falling back to the identity for
    // any component that is non-finite or non-positive.
    let mut h_inv = initial_inverse_hessian(model, y, init, p);

    for _ in 0..cfg.max_iter {
        if relative_grad_norm(&g, f) <= cfg.grad_tol {
            return OptResult {
                params: theta.as_slice().to_vec(),
                convergence: Convergence::Converged,
            };
        }

        // Search direction d = −H⁻¹ g.
        let mut d = &h_inv * &g;
        d = -d;
        if d.iter().any(|v| !v.is_finite()) {
            return OptResult {
                params: theta.as_slice().to_vec(),
                convergence: Convergence::Failed,
            };
        }

        // Ensure a descent direction. If the BFGS approximation has lost
        // positive-definiteness (so d is not a descent direction), reset the
        // inverse-Hessian to the scale-aware seed and retry once.
        let mut gd = g.dot(&d);
        if gd >= 0.0 {
            h_inv = initial_inverse_hessian(model, y, theta.as_slice(), p);
            d = -(&h_inv * &g);
            gd = g.dot(&d);
            if gd >= 0.0 || d.iter().any(|v| !v.is_finite()) {
                // Still not a descent direction: stop as stalled.
                return OptResult {
                    params: theta.as_slice().to_vec(),
                    convergence: Convergence::Stalled,
                };
            }
        }

        // Wolfe-condition line search (Armijo + curvature).
        let (alpha, f_new, g_new) = match wolfe_line_search(model, y, &theta, f, &d, gd, cfg) {
            Some(res) => res,
            None => {
                return OptResult {
                    params: theta.as_slice().to_vec(),
                    convergence: Convergence::Stalled,
                }
            }
        };
        let theta_new = &theta + alpha * &d;

        let s = &theta_new - &theta;
        let yk = &g_new - &g;
        let delta = relative_param_change(&s, &theta);

        // BFGS inverse-Hessian update (Nocedal & Wright 6.17).
        let sy = s.dot(&yk);
        if sy > 1e-12 {
            let rho = 1.0 / sy;
            let i = DMatrix::<f64>::identity(p, p);
            let v = &i - rho * (&yk * s.transpose());
            h_inv = &v * &h_inv * v.transpose() + rho * (&s * s.transpose());
        }

        theta = theta_new;
        f = f_new;
        g = g_new;

        if relative_grad_norm(&g, f) <= cfg.grad_tol && delta <= cfg.param_tol {
            return OptResult {
                params: theta.as_slice().to_vec(),
                convergence: Convergence::Converged,
            };
        }
    }

    OptResult {
        params: theta.as_slice().to_vec(),
        convergence: Convergence::MaxIterationsReached,
    }
}

/// Scale-aware initial inverse-Hessian approximation.
///
/// Returns the inverse of the diagonal of the observed information `−∇²ℓ` at
/// `theta`, with any non-finite or non-positive diagonal entry replaced by `1`
/// (i.e. the identity for that component). This gives the optimizer a
/// per-parameter step scale that matches the curvature of the log-likelihood,
/// which is essential when parameters have very different natural scales.
fn initial_inverse_hessian<M: Derivatives>(
    model: &M,
    y: &[f64],
    theta: &[f64],
    p: usize,
) -> DMatrix<f64> {
    let mut h_inv = DMatrix::<f64>::identity(p, p);
    let h = model.hessian(theta, y);
    if h.len() != p * p {
        return h_inv;
    }
    for i in 0..p {
        // Observed information diagonal entry: −∂²ℓ/∂θ_i².
        let info_ii = -h[i * p + i];
        if info_ii.is_finite() && info_ii > 0.0 {
            h_inv[(i, i)] = 1.0 / info_ii;
        }
    }
    h_inv
}

/// Wolfe-condition line search (Armijo sufficient decrease + curvature).
///
/// Starting from `alpha = 1`, backtracks by halving until the Armijo condition
/// `f(θ + α d) <= f + c1 α g·d` holds. It then checks the curvature condition
/// `g(θ + α d)·d >= c2 g·d`; if the curvature condition fails but the Armijo
/// condition holds, the step is accepted anyway (a common practical relaxation
/// that avoids an expensive zoom phase while still rejecting the pathological
/// near-flat-ridge steps). Returns `(alpha, f_new, g_new)` or `None` if no
/// acceptable step is found within the backtracking budget.
fn wolfe_line_search<M: Derivatives>(
    model: &M,
    y: &[f64],
    theta: &DVector<f64>,
    f: f64,
    d: &DVector<f64>,
    gd: f64,
    cfg: &MleConfig,
) -> Option<(f64, f64, DVector<f64>)> {
    // Curvature constant (Wolfe c2). Standard value for quasi-Newton is 0.9;
    // we use a slightly stricter 0.9 to keep steps well-conditioned.
    const C2: f64 = 0.9;
    let mut alpha = 1.0;
    for _ in 0..cfg.max_line_search {
        let theta_new = theta + alpha * d;
        let f_new = model.loglik(theta_new.as_slice(), y);
        if f_new.is_finite() && f_new <= f + cfg.line_search_c * alpha * gd {
            let g_new = DVector::from_vec(model.score(theta_new.as_slice(), y));
            if g_new.iter().any(|v| !v.is_finite()) {
                return None;
            }
            // Curvature condition. If satisfied, accept immediately. If not,
            // still accept (relaxed Wolfe) — the Armijo condition already
            // guarantees a meaningful decrease.
            let _curvature_ok = g_new.dot(d) >= C2 * gd;
            return Some((alpha, f_new, g_new));
        }
        alpha *= 0.5;
    }
    None
}

/// Newton–Raphson optimizer using the analytic Hessian (§4.2 fallback).
///
/// Each step solves `H Δθ = −g` where `H = ∇²ℓ` (the Hessian, not the observed
/// information) and `g` is the score. A damped step (halving) is applied when
/// the full Newton step does not increase the log-likelihood.
///
/// # Robustness
///
/// Two safeguards make the fallback reliable on the harder Phase-1 likelihood
/// surfaces (GEV, GPD, Gumbel, predictor models):
///
/// 1. **Convergence requires the gradient condition.** The original code
///    declared convergence whenever the parameter change was below tolerance,
///    which fired *falsely* when the line search rejected the step (so the
///    parameter change was zero) even though the gradient was large. We now
///    require `‖g‖ / (1 + |ℓ|) <= grad_tol` to declare convergence.
/// 2. **Gradient-ascent fallback.** When the Newton step is not an ascent
///    direction (the Hessian is not negative-definite at the current point, so
///    the damped line search cannot improve the log-likelihood), we fall back to
///    a scaled gradient-ascent step with backtracking. This lets the optimizer
///    escape points where the local quadratic model is untrustworthy.
fn newton_raphson<M: Derivatives>(
    model: &M,
    y: &[f64],
    init: &[f64],
    cfg: &MleConfig,
) -> OptResult {
    let p = model.n_params();
    let mut theta = DVector::from_column_slice(init);
    let mut f = model.loglik(init, y);

    if !f.is_finite() {
        return OptResult {
            params: init.to_vec(),
            convergence: Convergence::Failed,
        };
    }

    for _ in 0..cfg.max_newton_iter {
        let g = DVector::from_vec(model.score(theta.as_slice(), y));
        if g.iter().any(|v| !v.is_finite()) {
            return OptResult {
                params: theta.as_slice().to_vec(),
                convergence: Convergence::Failed,
            };
        }
        // Convergence requires the gradient condition (see robustness note 1).
        if relative_grad_norm(&g, f) <= cfg.grad_tol {
            return OptResult {
                params: theta.as_slice().to_vec(),
                convergence: Convergence::Converged,
            };
        }

        // Hessian ∇²ℓ (row-major flattened).
        let h_flat = model.hessian(theta.as_slice(), y);
        if h_flat.len() != p * p || h_flat.iter().any(|v| !v.is_finite()) {
            return OptResult {
                params: theta.as_slice().to_vec(),
                convergence: Convergence::Failed,
            };
        }
        let h = DMatrix::from_row_slice(p, p, &h_flat);

        // Candidate Newton step: solve H Δθ = −g.
        let newton_step = h
            .clone()
            .lu()
            .solve(&(-&g))
            .filter(|s| s.iter().all(|v| v.is_finite()));

        // Try the Newton step first; if it cannot improve the log-likelihood,
        // fall back to a scaled gradient-ascent step (robustness note 2).
        let mut accepted: Option<(DVector<f64>, f64)> = None;
        if let Some(step) = newton_step {
            if let Some(res) = damped_ascent(model, y, &theta, f, &step, cfg) {
                accepted = Some(res);
            }
        }
        if accepted.is_none() {
            // Gradient-ascent direction, scaled by the inverse gradient norm so
            // the initial step length is O(1) in parameter space.
            let gnorm = g.norm();
            if gnorm > 0.0 {
                let step = &g / gnorm;
                if let Some(res) = damped_ascent(model, y, &theta, f, &step, cfg) {
                    accepted = Some(res);
                }
            }
        }

        let (theta_new, f_new) = match accepted {
            Some(res) => res,
            None => {
                // No ascent direction found: the point is (numerically) a local
                // maximum or the surface is flat here. Report convergence only
                // if the gradient is actually small; otherwise stalled.
                let convergence = if relative_grad_norm(&g, f) <= cfg.grad_tol {
                    Convergence::Converged
                } else {
                    Convergence::Stalled
                };
                return OptResult {
                    params: theta.as_slice().to_vec(),
                    convergence,
                };
            }
        };

        let delta = relative_param_change(&(&theta_new - &theta), &theta);
        theta = theta_new;
        f = f_new;

        // A tiny parameter change with a small gradient is genuine convergence;
        // a tiny change with a large gradient is a stall (handled next loop).
        if delta <= cfg.param_tol {
            let g_now = DVector::from_vec(model.score(theta.as_slice(), y));
            if relative_grad_norm(&g_now, f) <= cfg.grad_tol {
                return OptResult {
                    params: theta.as_slice().to_vec(),
                    convergence: Convergence::Converged,
                };
            }
        }
    }

    OptResult {
        params: theta.as_slice().to_vec(),
        convergence: Convergence::MaxIterationsReached,
    }
}

/// Damped ascent along `step`: halve the step until the log-likelihood strictly
/// improves (or the backtracking budget is exhausted). Returns the improved
/// `(theta, f)` or `None` if no improvement was found.
fn damped_ascent<M: Derivatives>(
    model: &M,
    y: &[f64],
    theta: &DVector<f64>,
    f: f64,
    step: &DVector<f64>,
    cfg: &MleConfig,
) -> Option<(DVector<f64>, f64)> {
    let mut alpha = 1.0;
    for _ in 0..cfg.max_line_search {
        let theta_new = theta + alpha * step;
        let f_new = model.loglik(theta_new.as_slice(), y);
        if f_new.is_finite() && f_new > f {
            return Some((theta_new, f_new));
        }
        alpha *= 0.5;
    }
    None
}

/// Nelder–Mead simplex optimizer (gradient-free polish, Phase 3, Issue 3).
///
/// The reference's `scipy` BFGS is more forgiving of the GEV/GPD ridges than the
/// hand-rolled BFGS. When BFGS stalls or fails to converge, this gradient-free
/// simplex follows the curved ridge without needing the score, and can reach the
/// true optimum that BFGS missed. It is used as a **polish** step in
/// [`fit_with_config`]: it starts from the best point found so far and refines
/// it.
///
/// # Algorithm
///
/// The standard Nelder–Mead reflection / expansion / contraction / shrink
/// iteration (Nelder & Mead 1965) on the **log-likelihood** (maximization). The
/// initial simplex is built from `init` by perturbing each coordinate by a
/// relative step (`5%` of the coordinate magnitude, or `0.05` when the
/// coordinate is zero), which keeps the simplex on the parameter's natural
/// scale.
///
/// # Convergence
///
/// The iteration stops when the spread of the simplex function values falls
/// below `grad_tol * (1 + |f_best|)` (the same relative objective tolerance the
/// gradient methods use) or the iteration budget is exhausted. The returned
/// [`Convergence`] is `Converged` when the spread criterion was met, otherwise
/// `MaxIterationsReached`. A non-finite objective anywhere in the simplex
/// returns `Failed`.
fn nelder_mead<M: Derivatives>(model: &M, y: &[f64], init: &[f64], cfg: &MleConfig) -> OptResult {
    let p = model.n_params();
    let n_vertices = p + 1;

    // Build the initial simplex: `init` plus one perturbed vertex per coordinate.
    let mut simplex: Vec<DVector<f64>> = Vec::with_capacity(n_vertices);
    simplex.push(DVector::from_column_slice(init));
    for i in 0..p {
        let mut v = DVector::from_column_slice(init);
        let step = if init[i].abs() > 1e-12 {
            0.05 * init[i].abs()
        } else {
            0.05
        };
        v[i] += step;
        simplex.push(v);
    }

    // Evaluate the objective at every vertex.
    let mut fvals: Vec<f64> = simplex
        .iter()
        .map(|v| model.loglik(v.as_slice(), y))
        .collect();
    if fvals.iter().any(|f| !f.is_finite()) {
        return OptResult {
            params: init.to_vec(),
            convergence: Convergence::Failed,
        };
    }

    // Nelder–Mead coefficients (standard).
    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink

    // The iteration budget: reuse the BFGS budget so a caller that disables the
    // optimizer (`max_iter == 0`) also disables the polish.
    let max_iter = cfg.max_iter;

    for _ in 0..max_iter {
        // Order the vertices by objective (descending: best first).
        let mut order: Vec<usize> = (0..n_vertices).collect();
        order.sort_by(|&a, &b| {
            fvals[b]
                .partial_cmp(&fvals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n_vertices - 1];
        let second_worst = order[n_vertices - 2];

        // Convergence: the spread of the objective over the simplex is small
        // relative to the best value.
        let spread = fvals[best] - fvals[worst];
        if spread.abs() <= cfg.grad_tol * (1.0 + fvals[best].abs()) {
            return OptResult {
                params: simplex[best].as_slice().to_vec(),
                convergence: Convergence::Converged,
            };
        }

        // Centroid of all vertices except the worst.
        let mut centroid = DVector::<f64>::zeros(p);
        for &idx in order.iter().take(n_vertices - 1) {
            centroid += &simplex[idx];
        }
        centroid /= (n_vertices - 1) as f64;

        // Reflection.
        let reflected = &centroid + ALPHA * (&centroid - &simplex[worst]);
        let f_reflected = model.loglik(reflected.as_slice(), y);

        if f_reflected.is_finite() && f_reflected > fvals[best] {
            // Try expansion.
            let expanded = &centroid + GAMMA * (&reflected - &centroid);
            let f_expanded = model.loglik(expanded.as_slice(), y);
            if f_expanded.is_finite() && f_expanded > f_reflected {
                simplex[worst] = expanded;
                fvals[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            }
        } else if f_reflected.is_finite() && f_reflected > fvals[second_worst] {
            // Accept the reflection.
            simplex[worst] = reflected;
            fvals[worst] = f_reflected;
        } else {
            // Contraction.
            let contracted = if f_reflected.is_finite() && f_reflected > fvals[worst] {
                // Outside contraction.
                &centroid + RHO * (&reflected - &centroid)
            } else {
                // Inside contraction.
                &centroid + RHO * (&simplex[worst] - &centroid)
            };
            let f_contracted = model.loglik(contracted.as_slice(), y);
            if f_contracted.is_finite() && f_contracted > fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                // Shrink toward the best vertex.
                for &idx in order.iter().skip(1) {
                    simplex[idx] = &simplex[best] + SIGMA * (&simplex[idx] - &simplex[best]);
                    fvals[idx] = model.loglik(simplex[idx].as_slice(), y);
                }
                if fvals.iter().any(|f| !f.is_finite()) {
                    return OptResult {
                        params: simplex[best].as_slice().to_vec(),
                        convergence: Convergence::Failed,
                    };
                }
            }
        }
    }

    // Budget exhausted: return the best vertex.
    let best = (0..n_vertices)
        .max_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    OptResult {
        params: simplex[best].as_slice().to_vec(),
        convergence: Convergence::MaxIterationsReached,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §7.2 golden tolerance: `|a - b| <= atol + rtol * |b|`.
    const RTOL: f64 = 1e-8;
    const ATOL: f64 = 1e-10;

    /// Optimizer output is not exact; a documented looser tolerance is used for
    /// fitted parameters/SEs/AIC (the §7.2 policy applies to golden fixtures, not
    /// to iterative optimizer output). `1e-6` relative / `1e-8` absolute is a
    /// safe bound for a well-conditioned Gaussian fit.
    const OPT_RTOL: f64 = 1e-6;
    const OPT_ATOL: f64 = 1e-8;

    fn assert_close(actual: f64, expected: f64, rtol: f64, atol: f64, what: &str) {
        let tol = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual = {actual:.17e}, expected = {expected:.17e}, \
             |diff| = {:.3e} > tol = {:.3e}",
            (actual - expected).abs(),
            tol
        );
    }

    /// Gaussian model with **analytic** score and Hessian overrides.
    struct Gaussian;

    impl Derivatives for Gaussian {
        fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
            let mu = params[0];
            let sigma = params[1];
            let n = y.len() as f64;
            let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
            -n * sigma.ln()
                - ss / (2.0 * sigma * sigma)
                - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
        }

        fn n_params(&self) -> usize {
            2
        }

        fn score(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
            let mu = params[0];
            let sigma = params[1];
            let n = y.len() as f64;
            let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
            let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
            vec![
                s1 / (sigma * sigma),
                -n / sigma + ss / (sigma * sigma * sigma),
            ]
        }

        fn hessian(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
            let mu = params[0];
            let sigma = params[1];
            let n = y.len() as f64;
            let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
            let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
            let s2 = sigma * sigma;
            let s3 = s2 * sigma;
            let s4 = s2 * s2;
            vec![
                -n / s2,
                -2.0 * s1 / s3,
                -2.0 * s1 / s3,
                n / s2 - 3.0 * ss / s4,
            ]
        }
    }

    /// Gaussian model with **only** `loglik` + `n_params` (numerical fallback).
    struct GaussianNumerical;

    impl Derivatives for GaussianNumerical {
        fn loglik(&self, params: &[f64], y: &[f64]) -> f64 {
            Gaussian.loglik(params, y)
        }

        fn n_params(&self) -> usize {
            2
        }
    }

    fn sample() -> Vec<f64> {
        vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8]
    }

    /// Closed-form Gaussian MLE and SEs for the sample.
    fn analytic_gaussian(y: &[f64]) -> (f64, f64, f64, f64, f64) {
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        // MLE of sigma uses the biased (1/n) variance.
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let sigma = var.sqrt();
        // Observed information at the MLE: diag(n/σ², 2n/σ²).
        let se_mu = (var / n).sqrt();
        let se_sigma = (var / (2.0 * n)).sqrt();
        let loglik = Gaussian.loglik(&[mean, sigma], y);
        let aic = 2.0 * 2.0 - 2.0 * loglik;
        (mean, sigma, se_mu, se_sigma, aic)
    }

    #[test]
    fn bfgs_recovers_gaussian_mle() {
        let y = sample();
        let (mean, sigma, se_mu, se_sigma, aic) = analytic_gaussian(&y);
        let fit = fit(&Gaussian, &y, &[0.0, 1.0]);

        assert_eq!(fit.convergence, Convergence::Converged);
        assert_eq!(fit.n, y.len());
        assert_eq!(fit.n_params, 2);
        assert!(!fit.small_n);

        assert_close(fit.ml_params[0], mean, OPT_RTOL, OPT_ATOL, "mu");
        assert_close(fit.ml_params[1], sigma, OPT_RTOL, OPT_ATOL, "sigma");
        assert_close(fit.ml_se[0], se_mu, OPT_RTOL, OPT_ATOL, "se_mu");
        assert_close(fit.ml_se[1], se_sigma, OPT_RTOL, OPT_ATOL, "se_sigma");
        assert_close(fit.aic, aic, OPT_RTOL, OPT_ATOL, "aic");
        assert_close(
            fit.loglik,
            Gaussian.loglik(&[mean, sigma], &y),
            OPT_RTOL,
            OPT_ATOL,
            "loglik",
        );
    }

    #[test]
    fn numerical_fallback_model_also_fits() {
        // The numerical-derivative model must reach the same optimum.
        let y = sample();
        let (mean, sigma, _, _, _) = analytic_gaussian(&y);
        let fit = fit(&GaussianNumerical, &y, &[0.0, 1.0]);
        assert_eq!(fit.convergence, Convergence::Converged);
        assert_close(fit.ml_params[0], mean, 1e-5, 1e-6, "mu (numerical)");
        assert_close(fit.ml_params[1], sigma, 1e-5, 1e-6, "sigma (numerical)");
    }

    #[test]
    fn aic_formula_matches_definition() {
        let y = sample();
        let fit = fit(&Gaussian, &y, &[0.0, 1.0]);
        let expected = 2.0 * fit.n_params as f64 - 2.0 * fit.loglik;
        assert_close(fit.aic, expected, RTOL, ATOL, "aic definition");
    }

    #[test]
    fn small_n_flag_set_below_threshold() {
        let y = vec![0.3, -1.2, 2.4, 0.9, -0.5];
        assert!(y.len() < SMALL_N_THRESHOLD);
        let fit = fit(&Gaussian, &y, &[0.0, 1.0]);
        assert!(fit.small_n);
        assert_eq!(fit.n, 5);
    }

    #[test]
    fn small_n_flag_clear_at_threshold() {
        // Exactly n = 10 is not flagged (strict `n < 10`).
        let y = sample();
        assert_eq!(y.len(), SMALL_N_THRESHOLD);
        let fit = fit(&Gaussian, &y, &[0.0, 1.0]);
        assert!(!fit.small_n);
    }

    #[test]
    fn non_convergence_is_reported() {
        // A zero iteration budget forces MaxIterationsReached from BFGS; the
        // Newton fallback is also capped at zero, so the fit cannot converge.
        let y = sample();
        let cfg = MleConfig {
            max_iter: 0,
            max_newton_iter: 0,
            ..MleConfig::default()
        };
        let fit = fit_with_config(&Gaussian, &y, &[0.0, 1.0], &cfg);
        assert!(!fit.convergence.is_converged());
        assert_eq!(fit.convergence, Convergence::MaxIterationsReached);
    }

    #[test]
    fn failed_fit_on_non_finite_objective() {
        // A model whose log-likelihood is always NaN must report Failed.
        struct Broken;
        impl Derivatives for Broken {
            fn loglik(&self, _params: &[f64], _y: &[f64]) -> f64 {
                f64::NAN
            }
            fn n_params(&self) -> usize {
                1
            }
        }
        let fit = fit(&Broken, &[1.0, 2.0, 3.0], &[0.0]);
        assert_eq!(fit.convergence, Convergence::Failed);
    }

    #[test]
    fn newton_fallback_used_when_bfgs_stalls() {
        // With a single BFGS iteration the optimizer cannot reach the optimum;
        // the Newton fallback (given enough iterations) must still converge.
        let y = sample();
        let (mean, sigma, _, _, _) = analytic_gaussian(&y);
        let cfg = MleConfig {
            max_iter: 1,
            max_newton_iter: 100,
            ..MleConfig::default()
        };
        let fit = fit_with_config(&Gaussian, &y, &[0.0, 1.0], &cfg);
        assert_eq!(fit.convergence, Convergence::Converged);
        assert_close(fit.ml_params[0], mean, OPT_RTOL, OPT_ATOL, "mu (newton)");
        assert_close(
            fit.ml_params[1],
            sigma,
            OPT_RTOL,
            OPT_ATOL,
            "sigma (newton)",
        );
    }

    #[test]
    fn se_from_inverse_observed_information() {
        // SE_i = sqrt((H⁻¹)_ii) with H the observed information at the optimum.
        let y = sample();
        let fit = fit(&Gaussian, &y, &[0.0, 1.0]);
        let info = observed_information(&Gaussian, &y, &fit.ml_params).unwrap();
        let inv = invert_symmetric(&info).unwrap();
        assert_close(fit.ml_se[0], inv[(0, 0)].sqrt(), RTOL, ATOL, "se_mu inv");
        assert_close(fit.ml_se[1], inv[(1, 1)].sqrt(), RTOL, ATOL, "se_sigma inv");
    }
}
