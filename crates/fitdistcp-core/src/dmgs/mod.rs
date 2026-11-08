//! DMGS asymptotic-expansion integration path (§4.1, §8).
//!
//! This module owns the calibrated-prior (CP) predictive density construction
//! and the [`Integrator`] trait. The trait is the seam that lets a later phase
//! plug in posterior sampling (§8) without touching model code.
//!
//! # What DMGS computes
//!
//! The reference `fitdistcp` package approximates the Bayesian prediction
//! integral with the asymptotic expansion of Datta, Mukerjee, Ghosh & Sweeting
//! (2000) — "DMGS". The expansion corrects the MLE-based predictive density with
//! a first-order bias term and the right-Haar-prior (RHP) terms. The core
//! quantity is the vector `beps` of equation 3.3 of the reference
//! (`bayesian_dq_4terms_v1` / `dmgs` in
//! `tools/fitdistcp-ref/R/000_evaluate_dmgs_equation3p3.R`):
//!
//! ```text
//! beps1 = Σ_{t,j,r,s} 0.5 · lddi[s,t] · lddi[j,r] · lddd[j,r,s] · mu1[t]   (bias correction)
//! beps2 = -Σ_{t,s} lddi[s,t] · (2·pidopi[s]) · mu1[t]                       (Haar prior, part 1)
//! beps3 = -Σ_{t,s} lddi[s,t] · (-pidopi[s]) · mu1[t]                        (Haar prior, part 2)
//! beps4 = -Σ_{j,r} 0.5 · lddi[j,r] · mu2[j,r]                               (mu2 term)
//! beps  = beps1 + beps2 + beps3 + beps4
//! ```
//!
//! where
//!
//! - `ldd` is the **normalized** observed information `(1/n) Σ_a ∂²ℓ/∂θ_i∂θ_j`,
//!   `lddi = ldd⁻¹`,
//! - `lddd` is the **normalized** third-derivative tensor
//!   `(1/n) Σ_a ∂³ℓ/∂θ_i∂θ_j∂θ_k`,
//! - `mu1[t] = -∂F/∂θ_t` and `mu2[j,r] = -∂²F/∂θ_j∂θ_r` are the (negated)
//!   derivatives of the model CDF evaluated at the prediction point,
//! - `pidopi` is the slope of the log prior (`lambdad_rhp`), which for the
//!   homogeneous models is the right-Haar prior `π(σ) ∝ 1/σ` (so the only
//!   non-zero entry is the scale parameter, `-1/σ`).
//!
//! The CP predictive density and quantiles are then
//!
//! ```text
//! cp_pdf(x)       = ml_pdf(x) + dmgs(lddi, lddd, f1(x), pidopi, f2(x)) / n
//! cp_quantile(p)  = ml_quantile(p) + dmgs(lddi, lddd, mu1(p), pidopi, mu2(p)) / (n · fhat)
//! ```
//!
//! with `f1 = ∂f/∂θ`, `f2 = ∂²f/∂θ∂θ` the derivatives of the **density**, and
//! `fhat = f(ml_quantile(p))` the MLE density at the quantile.
//!
//! # Vectorization over prediction points
//!
//! In the reference, `mu1` is a `p × nalpha` matrix (parameter × prediction
//! point) and `mu2` a `p × p × nalpha` array; the sums over the parameter
//! indices collapse the parameter dimension and leave one `beps` value **per
//! prediction point**. [`DmgsIntegrator::dmgs_expansion`] reproduces this: it
//! takes `mu1` as a row-major `p × nalpha` matrix and `mu2` as a row-major
//! `p × p × nalpha` array, and returns a vector of length `nalpha`. This is the
//! key structural detail — a naive per-parameter sum gives the wrong answer.
//!
//! # Analytic vs. numerical derivatives (§4.1)
//!
//! The expansion needs `ldd`, `lddd`, and the density/CDF derivatives. The
//! [`Derivatives`] trait supplies `hessian` and `third_derivatives` (analytic
//! where a model overrides them, numerical otherwise). The density and CDF
//! derivatives `f1`, `f2`, `mu1`, `mu2` are computed here by central finite
//! differences of the model's own `pdf`/`cdf`, which is exactly what the
//! reference does (its `*_f1fa`/`*_mu1fa` helpers are `Vectorize`d analytic
//! derivatives; the finite-difference equivalent is used here so the expansion
//! works for every model behind one interface). This keeps the analytic-vs-
//! numerical switch per-model and testable, as the design requires.
//!
//! # Reference agreement
//!
//! The formulas above are transcribed directly from the reference R source. The
//! one place the design document is silent and the reference is authoritative is
//! the **sign convention** of `ldd`: the reference's `ldd` is the second
//! derivative of the log-likelihood (negative-definite at the MLE), and `lddi`
//! is its inverse; the `dmgs` routine's comments ("this has to be negative
//! because ldd is the opposite sign to c") encode the resulting signs. We follow
//! the reference exactly.

use nalgebra::{DMatrix, DVector};

use crate::models::{Derivatives, FiniteDiffConfig};

pub mod models;

/// Integration strategy for the CP predictive density (§8).
///
/// Implementors turn a fitted model plus a prediction grid into the CP
/// predictive density. The Phase 1 implementation is the DMGS asymptotic
/// expansion ([`DmgsIntegrator`]); a later phase can add posterior sampling
/// behind the same trait.
pub trait Integrator {
    /// Integrate the CP predictive density over the prediction points `x`.
    ///
    /// `params` are the fitted parameters, `y` the observed data, and `x` the
    /// points at which the predictive density is evaluated. Returns one density
    /// value per element of `x`.
    fn integrate(&self, params: &[f64], y: &[f64], x: &[f64]) -> Vec<f64>;
}

/// The DMGS asymptotic-expansion integrator (§4.1).
///
/// Wraps a model implementing [`Derivatives`] and computes the CP predictive
/// density via the DMGS expansion. The model must additionally expose its
/// density and CDF through the [`DmgsModel`] trait so the expansion can form the
/// `f1`/`f2`/`mu1`/`mu2` terms.
#[derive(Debug, Clone, Copy)]
pub struct DmgsIntegrator<M> {
    model: M,
    fd: FiniteDiffConfig,
}

impl<M: DmgsModel + Sync> DmgsIntegrator<M> {
    /// Create a DMGS integrator over `model` with the default finite-difference
    /// configuration.
    pub fn new(model: M) -> Self {
        Self {
            model,
            fd: FiniteDiffConfig::default(),
        }
    }

    /// Create a DMGS integrator with an explicit finite-difference
    /// configuration (used by the tests to validate the expansion).
    pub fn with_fd_config(model: M, fd: FiniteDiffConfig) -> Self {
        Self { model, fd }
    }

    /// The wrapped model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// The finite-difference configuration.
    pub fn fd_config(&self) -> &FiniteDiffConfig {
        &self.fd
    }

    /// The slope of the log prior `pidopi` (the reference `lambdad_rhp`).
    ///
    /// For the homogeneous Phase-1 models this is the right-Haar prior
    /// `π(σ) ∝ 1/σ`, whose log-slope is `-1/σ` in the scale component and `0`
    /// elsewhere. Models with a different prior override
    /// [`DmgsModel::log_prior_slope`].
    pub fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        self.model.log_prior_slope(params)
    }

    /// The **normalized** observed information `ldd = (1/n) Σ_a ∂²ℓ/∂θ_i∂θ_j`.
    ///
    /// This is the reference's `ldd` (the second derivative of the *normalized*
    /// log-likelihood). It is negative-definite at the MLE.
    pub fn normalized_ldd(&self, params: &[f64], y: &[f64]) -> DMatrix<f64> {
        let p = self.model.n_params();
        let n = y.len() as f64;
        let h = self.model.hessian(params, y);
        let mut ldd = DMatrix::<f64>::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                ldd[(i, j)] = h[i * p + j] / n;
            }
        }
        ldd
    }

    /// The **normalized** third-derivative tensor
    /// `lddd = (1/n) Σ_a ∂³ℓ/∂θ_i∂θ_j∂θ_k`, flattened row-major.
    pub fn normalized_lddd(&self, params: &[f64], y: &[f64]) -> Vec<f64> {
        let n = y.len() as f64;
        self.model
            .third_derivatives(params, y)
            .into_iter()
            .map(|v| v / n)
            .collect()
    }

    /// Evaluate the DMGS expansion `beps` (equation 3.3), vectorized over the
    /// prediction points.
    ///
    /// This is a direct transcription of the reference `dmgs` routine. In the
    /// reference, `mu1` is a `p × nalpha` matrix (parameter × prediction point)
    /// and `mu2` is a `p × p × nalpha` array; the sums over the parameter
    /// indices `t, j, r, s` collapse the parameter dimension and leave one
    /// `beps` value **per prediction point**. We therefore take:
    ///
    /// - `mu1` as a row-major `p × nalpha` matrix (`mu1[t * nalpha + a]`),
    /// - `mu2` as a row-major `p × p × nalpha` array
    ///   (`mu2[(j * p + r) * nalpha + a]`),
    ///
    /// and return a vector of length `nalpha`.
    #[allow(clippy::too_many_arguments)]
    pub fn dmgs_expansion(
        &self,
        lddi: &DMatrix<f64>,
        lddd: &[f64],
        mu1: &[f64],
        pidopi: &[f64],
        mu2: &[f64],
        p: usize,
        nalpha: usize,
    ) -> Vec<f64> {
        // pidopi1 = 2 * pidopi ; pidopi2 = -pidopi (reference `dmgs`).
        let pidopi1: Vec<f64> = pidopi.iter().map(|v| 2.0 * v).collect();
        let pidopi2: Vec<f64> = pidopi.iter().map(|v| -v).collect();

        // --- W17 (H1): factor the parameter sums out of the prediction grid. ---
        //
        // The reference `dmgs` accumulates `beps1[a] += coeff(t,j,r,s) * mu1[t,a]`
        // inside a `t × j × r × s` quadruple loop, i.e. `O(p⁴ · nalpha)` work.
        // But `coeff` does not depend on the prediction point `a`, and `mu1[t,a]`
        // does not depend on `j, r, s`. We can therefore **factor** the sum:
        //
        //   beps1[a] = Σ_t [ Σ_{j,r,s} 0.5·lddi[s,t]·lddi[j,r]·lddd[j,r,s] ] · mu1[t,a]
        //            = Σ_t c1[t] · mu1[t,a]
        //
        // which is `O(p⁴)` to form the `p` coefficients `c1[t]` plus `O(p·nalpha)`
        // to contract them against `mu1`. For `p = 3` this replaces 81·nalpha
        // multiply-adds per point with 3, and the `p⁴` term no longer scales with
        // the grid size. The same factoring applies to `beps2`/`beps3` (collapse
        // the `s` sum into a per-`t` coefficient) and `beps4` (collapse the
        // `j, r` sum into a per-`(j,r)` coefficient).
        //
        // The arithmetic is identical to the reference (same terms, same order of
        // summation over `t`), so the result is bit-for-bit the same up to
        // floating-point reassociation; the golden tests confirm it stays within
        // the §7.2 tolerance.

        // c1[t] = Σ_{j,r,s} 0.5 · lddi[s,t] · lddi[j,r] · lddd[j,r,s].
        let mut c1 = vec![0.0; p];
        for t in 0..p {
            let mut acc = 0.0;
            for j in 0..p {
                for r in 0..p {
                    let lddi_jr = lddi[(j, r)];
                    let base = j * p * p + r * p;
                    for s in 0..p {
                        acc += 0.5 * lddi[(s, t)] * lddi_jr * lddd[base + s];
                    }
                }
            }
            c1[t] = acc;
        }

        // c2[t] = Σ_s lddi[s,t] · pidopi1[s] ; c3[t] = Σ_s lddi[s,t] · pidopi2[s].
        let mut c2 = vec![0.0; p];
        let mut c3 = vec![0.0; p];
        for t in 0..p {
            let mut a2 = 0.0;
            let mut a3 = 0.0;
            for s in 0..p {
                let l = lddi[(s, t)];
                a2 += l * pidopi1[s];
                a3 += l * pidopi2[s];
            }
            c2[t] = a2;
            c3[t] = a3;
        }

        // c4[j,r] = 0.5 · lddi[j,r] (row-major p × p).
        let mut c4 = vec![0.0; p * p];
        for j in 0..p {
            for r in 0..p {
                c4[j * p + r] = 0.5 * lddi[(j, r)];
            }
        }

        // Contract the coefficients against the prediction grid. The per-point
        // work is `O(p²)` (the `mu2` term dominates) and memory-bound, so this
        // loop is kept sequential: the W17 benchmark sweep showed Rayon's
        // per-point overhead dominates for the expansion grid. The inputs keep
        // the reference's row-major `p × nalpha` / `p × p × nalpha` layout
        // (`mu1[t*nalpha + a]`, `mu2[(j*p+r)*nalpha + a]`), so the public
        // contract of this method is unchanged.
        let mut out = vec![0.0; nalpha];
        for (a, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            // beps1 + beps2 + beps3: all contract against mu1[t, a].
            for t in 0..p {
                acc += (c1[t] - c2[t] - c3[t]) * mu1[t * nalpha + a];
            }
            // beps4: -Σ_{j,r} 0.5 lddi[j,r] mu2[j,r,a].
            for jr in 0..p * p {
                acc -= c4[jr] * mu2[jr * nalpha + a];
            }
            *o = acc;
        }
        out
    }

    /// The CP predictive density at the points `x` (the reference `rh_pdf`).
    ///
    /// `cp_pdf(x) = ml_pdf(x) + dmgs(lddi, lddd, f1(x), pidopi, f2(x)) / n`,
    /// clamped at zero (the reference applies `pmax(., 0)`). The expansion is
    /// vectorized over the prediction points `x`.
    pub fn cp_pdf(&self, params: &[f64], y: &[f64], x: &[f64]) -> Vec<f64> {
        let p = self.model.n_params();
        let n = y.len() as f64;
        let nalpha = x.len();
        let ldd = self.normalized_ldd(params, y);
        let lddi = match invert(&ldd) {
            Some(inv) => inv,
            None => return vec![f64::NAN; nalpha],
        };
        let lddd = self.normalized_lddd(params, y);
        let pidopi = self.log_prior_slope(params);

        // Build the p × nalpha `f1` matrix and the p × p × nalpha `f2` array.
        // The density derivatives come from the model's `DmgsModel` hooks
        // (analytic where overridden, finite differences otherwise).
        //
        // W17 (H5): the buffers are built **row-major directly** (the layout
        // `dmgs_expansion` consumes), so there is no transpose pass. The
        // per-point work is tiny and memory-bound, so this loop is kept
        // sequential — the W17 benchmark sweep showed Rayon's per-point overhead
        // dominates for the expansion grid (see [`PARALLEL_MIN_NALPHA`]).
        let model = &self.model;
        let mut f1 = vec![0.0; p * nalpha];
        let mut f2 = vec![0.0; p * p * nalpha];
        let mut ml_pdf = vec![0.0; nalpha];
        for (a, &xi) in x.iter().enumerate() {
            ml_pdf[a] = model.pdf(xi, params);
            let d1 = model.density_first_derivs(params, xi);
            let d2 = model.density_second_derivs(params, xi);
            for t in 0..p {
                f1[t * nalpha + a] = d1[t];
            }
            for jr in 0..p * p {
                f2[jr * nalpha + a] = d2[jr];
            }
        }

        let dq = self.dmgs_expansion(&lddi, &lddd, &f1, &pidopi, &f2, p, nalpha);
        (0..nalpha)
            .map(|a| (ml_pdf[a] + dq[a] / n).max(0.0))
            .collect()
    }

    /// The CP predictive quantiles at the probabilities `p` (the reference
    /// `rh_quantiles`).
    ///
    /// `cp_quantile(p) = ml_quantile(p) + dmgs(lddi, lddd, mu1(p), pidopi,
    /// mu2(p)) / (n · fhat)`, with `mu1 = -∂F/∂θ`, `mu2 = -∂²F/∂θ∂θ` evaluated
    /// at the MLE quantile and `fhat` the MLE density there. The expansion is
    /// vectorized over the probabilities.
    pub fn cp_quantiles(&self, params: &[f64], y: &[f64], p: &[f64]) -> Vec<f64> {
        let dim = self.model.n_params();
        let n = y.len() as f64;
        let nalpha = p.len();
        let ldd = self.normalized_ldd(params, y);
        let lddi = match invert(&ldd) {
            Some(inv) => inv,
            None => return vec![f64::NAN; nalpha],
        };
        let lddd = self.normalized_lddd(params, y);
        let pidopi = self.log_prior_slope(params);

        // Build the dim × nalpha `mu1` matrix and the dim × dim × nalpha `mu2`
        // array. The reference defines `mu1 = -∂F/∂θ` and `mu2 = -∂²F/∂θ²`
        // (see `*_mu1fa`/`*_mu2fa`, which return `-vf(...)`), so we negate the
        // raw CDF derivatives computed here.
        //
        // W17 (H5): built **row-major directly** (the layout `dmgs_expansion`
        // consumes), so there is no transpose pass. The per-point work is tiny
        // and memory-bound, so this loop is kept sequential (see
        // [`PARALLEL_MIN_NALPHA`]).
        let model = &self.model;
        let mut mu1 = vec![0.0; dim * nalpha];
        let mut mu2 = vec![0.0; dim * dim * nalpha];
        let mut ml_q = vec![0.0; nalpha];
        let mut fhat = vec![0.0; nalpha];
        for (a, &pi) in p.iter().enumerate() {
            let q = model.quantile(pi, params);
            ml_q[a] = q;
            fhat[a] = model.pdf(q, params);
            let d1 = model.cdf_first_derivs(params, q);
            let d2 = model.cdf_second_derivs(params, q);
            for t in 0..dim {
                mu1[t * nalpha + a] = -d1[t];
            }
            for jr in 0..dim * dim {
                mu2[jr * nalpha + a] = -d2[jr];
            }
        }

        let dq = self.dmgs_expansion(&lddi, &lddd, &mu1, &pidopi, &mu2, dim, nalpha);
        (0..nalpha)
            .map(|a| ml_q[a] + dq[a] / (n * fhat[a]))
            .collect()
    }

    /// The MLE and CP predictive means (reference `*_means`).
    ///
    /// Returns `(ml_mean, cp_mean)` where `ml_mean = E[Y; θ]` and
    /// `cp_mean = ml_mean + dmgs_scalar(lddi, lddd, meand1, pidopi, meand2, p)/n`
    /// (the reference `*_means` construction). The mean derivatives come from
    /// [`DmgsModel::mean_derivs`] (analytic where overridden).
    pub fn cp_mean(&self, params: &[f64], y: &[f64]) -> (f64, f64) {
        let p = self.model.n_params();
        let n = y.len() as f64;
        let (ml_mean, meand1, meand2) = self.model.mean_derivs(params);
        let ldd = self.normalized_ldd(params, y);
        let lddi = match invert(&ldd) {
            Some(inv) => inv,
            None => return (ml_mean, f64::NAN),
        };
        let lddd = self.normalized_lddd(params, y);
        let pidopi = self.log_prior_slope(params);
        let dmean = dmgs_scalar(&lddi, &lddd, &meand1, &pidopi, &meand2, p);
        (ml_mean, ml_mean + dmean / n)
    }

    /// The CP predictive density **derived from the CP quantile function**
    /// (reference `qgev_cp`'s `rh_flat_pdf`).
    ///
    /// For the negative-shape GEV the reference does **not** evaluate the direct
    /// density formula at the CP quantile. Instead it forms the CP density as the
    /// reciprocal slope of the CP quantile function, evaluated by a central
    /// difference in the tail probability `alpha = 1 - p`:
    ///
    /// ```text
    /// dalpha        = min(fdalpha·alpha, fdalpha·(1-alpha))     (fdalpha = 0.01)
    /// alpham        = alpha - dalpha,  alphap = alpha + dalpha
    /// q_m           = cp_quantile(1 - alpham)
    /// q_p           = cp_quantile(1 - alphap)
    /// cp_pdf        = -(alphap - alpham) / (q_p - q_m)
    /// ```
    ///
    /// This is numerically stable where the direct density is not: for a negative
    /// shape the CP quantile can approach the finite upper endpoint `ml_max`, and
    /// the direct density formula (which involves `t^(-1/shape - 1)`) loses
    /// precision there, whereas the quantile slope remains well-conditioned. The
    /// reference uses this path for **all** GEV shapes when `pdf=TRUE`; we expose
    /// it as a method so the GEV CP density can use it (see
    /// [`crate::api::predict`]).
    ///
    /// `alpha` are the tail probabilities `1 - p` at which the density is wanted
    /// (one per prediction point). `fdalpha` is the relative step (the reference
    /// default is `0.01`).
    pub fn cp_pdf_quantile_derived(
        &self,
        params: &[f64],
        y: &[f64],
        alpha: &[f64],
        fdalpha: f64,
    ) -> Vec<f64> {
        let nalpha = alpha.len();
        // Build the shifted tail-probability grids alpham / alphap.
        let mut alpham = vec![0.0; nalpha];
        let mut alphap = vec![0.0; nalpha];
        for (a, &al) in alpha.iter().enumerate() {
            let dalpha = (fdalpha * al).min(fdalpha * (1.0 - al));
            alpham[a] = al - dalpha;
            alphap[a] = al + dalpha;
        }
        // CP quantiles at the shifted probabilities. The reference evaluates the
        // quantile at probability `1 - alpha`, so we pass `1 - alpham` / `1 - alphap`.
        let pm: Vec<f64> = alpham.iter().map(|&a| 1.0 - a).collect();
        let pp: Vec<f64> = alphap.iter().map(|&a| 1.0 - a).collect();
        let qm = self.cp_quantiles(params, y, &pm);
        let qp = self.cp_quantiles(params, y, &pp);
        (0..nalpha)
            .map(|a| {
                let dq = qp[a] - qm[a];
                if dq == 0.0 || !dq.is_finite() {
                    f64::NAN
                } else {
                    -(alphap[a] - alpham[a]) / dq
                }
            })
            .collect()
    }
}

impl<M: DmgsModel + Sync> Integrator for DmgsIntegrator<M> {
    fn integrate(&self, params: &[f64], y: &[f64], x: &[f64]) -> Vec<f64> {
        self.cp_pdf(params, y, x)
    }
}

/// A model that can be integrated by the DMGS expansion.
///
/// Extends [`Derivatives`] with the density, CDF, and quantile primitives the
/// expansion needs, plus the log-prior slope. Every Phase-1 model implements
/// this; the blanket-style default for [`DmgsModel::log_prior_slope`] is the
/// right-Haar prior `π(σ) ∝ 1/σ` (non-zero only in the scale component).
///
/// # Analytic density/CDF derivatives (W9)
///
/// The DMGS expansion needs the first and second derivatives of the **density**
/// (`f1`, `f2`) and of the **CDF** (`mu1`, `mu2`) with respect to the
/// parameters, evaluated at a single prediction point. The reference package
/// supplies these analytically via its `*_f1fa`/`*_f2fa`/`*_mu1fa`/`*_mu2fa`
/// helpers. This trait exposes the same four quantities as
/// [`DmgsModel::density_first_derivs`], [`DmgsModel::density_second_derivs`],
/// [`DmgsModel::cdf_first_derivs`], and [`DmgsModel::cdf_second_derivs`].
///
/// Each has a **numerical default** (central finite differences of
/// [`DmgsModel::pdf`]/[`DmgsModel::cdf`]) so a model that does not override them
/// still works; a model with an analytic form overrides them, which is what
/// closes the W8 finite-difference residual. The defaults take the
/// finite-difference configuration from [`DmgsModel::dmgs_fd_config`].
pub trait DmgsModel: Derivatives {
    /// The density `f(x; params)` at a single point.
    fn pdf(&self, x: f64, params: &[f64]) -> f64;

    /// The CDF `F(x; params)` at a single point.
    fn cdf(&self, x: f64, params: &[f64]) -> f64;

    /// The quantile `F^{-1}(p; params)`.
    fn quantile(&self, p: f64, params: &[f64]) -> f64;

    /// The slope of the log prior `pidopi` (the reference `lambdad_rhp`).
    ///
    /// Defaults to the right-Haar prior `π(σ) ∝ 1/σ`: the vector is zero except
    /// for the **scale** component, which is `-1/σ`. Models whose scale is not
    /// the second parameter (or that use a different prior) override this.
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        let p = self.n_params();
        let mut v = vec![0.0; p];
        // Right-Haar prior for the homogeneous Phase-1 models: the scale is the
        // second parameter, and d/dσ ln(1/σ) = -1/σ.
        if p >= 2 {
            v[1] = -1.0 / params[1];
        }
        v
    }

    /// The model mean `E[Y; θ]` and its first/second derivatives w.r.t. the
    /// parameters (reference `*_means`).
    ///
    /// Returns `(ml_mean, meand1, meand2)` where `ml_mean = E[Y; θ]`,
    /// `meand1[i] = ∂E[Y]/∂θ_i` (length `p`), and `meand2[i*p + j] =
    /// ∂²E[Y]/∂θ_i∂θ_j` (row-major `p × p`). The CP predictive mean is then
    /// `cp_mean = ml_mean + dmgs_scalar(lddi, lddd, meand1, pidopi, meand2, p)/n`
    /// (reference `*_means`).
    ///
    /// The default computes `ml_mean` by numerical integration of the density
    /// and the derivatives by central finite differences of that integral. A
    /// model with a closed-form mean overrides this with the reference's
    /// analytic expression.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let p = self.n_params();
        let cfg = self.dmgs_fd_config();
        let mean_at = |q: &[f64]| self.numerical_mean(q);
        let ml_mean = mean_at(params);
        let mut meand1 = vec![0.0; p];
        for i in 0..p {
            let h = cfg.step_for(params[i]);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[i] += h;
            pm[i] -= h;
            meand1[i] = (mean_at(&pp) - mean_at(&pm)) / (2.0 * h);
        }
        let mut meand2 = vec![0.0; p * p];
        for i in 0..p {
            let hi = cfg.step_for_2(params[i]);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[i] += hi;
            pm[i] -= hi;
            meand2[i * p + i] = (mean_at(&pp) - 2.0 * ml_mean + mean_at(&pm)) / (hi * hi);
            for j in (i + 1)..p {
                let hj = cfg.step_for_2(params[j]);
                let eval = |di: f64, dj: f64| {
                    let mut q = params.to_vec();
                    q[i] += di;
                    q[j] += dj;
                    mean_at(&q)
                };
                let val = (eval(hi, hj) - eval(hi, -hj) - eval(-hi, hj) + eval(-hi, -hj))
                    / (4.0 * hi * hj);
                meand2[i * p + j] = val;
                meand2[j * p + i] = val;
            }
        }
        (ml_mean, meand1, meand2)
    }

    /// Numerical mean `∫ x f(x; θ) dx` used by the default [`DmgsModel::mean_derivs`].
    ///
    /// Integrates over a wide, model-agnostic bracket using the trapezoid rule.
    /// Models with a closed-form mean override [`DmgsModel::mean_derivs`] and
    /// never call this.
    fn numerical_mean(&self, params: &[f64]) -> f64 {
        // A wide bracket centred on the quantile range; the density decays to
        // zero well inside it for the Phase-1 models.
        let lo = self.quantile(1e-9, params);
        let hi = self.quantile(1.0 - 1e-9, params);
        if !lo.is_finite() || !hi.is_finite() || hi <= lo {
            return f64::NAN;
        }
        let n = 2000usize;
        let h = (hi - lo) / n as f64;
        let mut acc = 0.0;
        for k in 0..=n {
            let x = lo + k as f64 * h;
            let w = if k == 0 || k == n { 0.5 } else { 1.0 };
            acc += w * x * self.pdf(x, params);
        }
        acc * h
    }

    /// Finite-difference configuration used by the **numerical defaults** of the
    /// density/CDF derivative hooks below.
    ///
    /// Defaults to [`FiniteDiffConfig::default`]; a model may override it to
    /// tune the step for parameters on unusual scales.
    fn dmgs_fd_config(&self) -> FiniteDiffConfig {
        FiniteDiffConfig::default()
    }

    /// First derivatives of the **density** w.r.t. each parameter, at `x`.
    ///
    /// Returns a `p`-vector `∂f/∂θ_i`. The reference calls this `*_f1fa`.
    /// Defaults to central finite differences of [`DmgsModel::pdf`]; models with
    /// an analytic form override it.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let p = params.len();
        let cfg = self.dmgs_fd_config();
        let mut out = vec![0.0; p];
        for i in 0..p {
            let h = cfg.step_for(params[i]);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[i] += h;
            pm[i] -= h;
            out[i] = (self.pdf(x, &pp) - self.pdf(x, &pm)) / (2.0 * h);
        }
        out
    }

    /// Second derivatives of the **density** w.r.t. each parameter pair, at `x`.
    ///
    /// Returns a row-major `p × p` matrix `∂²f/∂θ_i∂θ_j`. The reference calls
    /// this `*_f2fa`. Defaults to central finite differences of
    /// [`DmgsModel::pdf`]; models with an analytic form override it.
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let p = params.len();
        let cfg = self.dmgs_fd_config();
        let mut out = vec![0.0; p * p];
        let f0 = self.pdf(x, params);
        for i in 0..p {
            let hi = cfg.step_for_2(params[i]);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[i] += hi;
            pm[i] -= hi;
            out[i * p + i] = (self.pdf(x, &pp) - 2.0 * f0 + self.pdf(x, &pm)) / (hi * hi);
            for j in (i + 1)..p {
                let hj = cfg.step_for_2(params[j]);
                let eval = |di: f64, dj: f64| {
                    let mut q = params.to_vec();
                    q[i] += di;
                    q[j] += dj;
                    self.pdf(x, &q)
                };
                let val = (eval(hi, hj) - eval(hi, -hj) - eval(-hi, hj) + eval(-hi, -hj))
                    / (4.0 * hi * hj);
                out[i * p + j] = val;
                out[j * p + i] = val;
            }
        }
        out
    }

    /// First derivatives of the **CDF** w.r.t. each parameter, at `x`.
    ///
    /// Returns a `p`-vector `∂F/∂θ_i`. The reference calls this `*_p1fa` and
    /// negates it to form `mu1 = -∂F/∂θ`. Defaults to central finite differences
    /// of [`DmgsModel::cdf`]; models with an analytic form override it.
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let p = params.len();
        let cfg = self.dmgs_fd_config();
        let mut out = vec![0.0; p];
        for i in 0..p {
            let h = cfg.step_for(params[i]);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[i] += h;
            pm[i] -= h;
            out[i] = (self.cdf(x, &pp) - self.cdf(x, &pm)) / (2.0 * h);
        }
        out
    }

    /// Second derivatives of the **CDF** w.r.t. each parameter pair, at `x`.
    ///
    /// Returns a row-major `p × p` matrix `∂²F/∂θ_i∂θ_j`. The reference calls
    /// this `*_p2fa` and negates it to form `mu2 = -∂²F/∂θ²`. Defaults to
    /// central finite differences of [`DmgsModel::cdf`]; models with an analytic
    /// form override it.
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let p = params.len();
        let cfg = self.dmgs_fd_config();
        let mut out = vec![0.0; p * p];
        let f0 = self.cdf(x, params);
        for i in 0..p {
            let hi = cfg.step_for_2(params[i]);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[i] += hi;
            pm[i] -= hi;
            out[i * p + i] = (self.cdf(x, &pp) - 2.0 * f0 + self.cdf(x, &pm)) / (hi * hi);
            for j in (i + 1)..p {
                let hj = cfg.step_for_2(params[j]);
                let eval = |di: f64, dj: f64| {
                    let mut q = params.to_vec();
                    q[i] += di;
                    q[j] += dj;
                    self.cdf(x, &q)
                };
                let val = (eval(hi, hj) - eval(hi, -hj) - eval(-hi, hj) + eval(-hi, -hj))
                    / (4.0 * hi * hj);
                out[i * p + j] = val;
                out[j * p + i] = val;
            }
        }
        out
    }
}

/// Invert a square matrix, returning `None` if it is singular or non-finite.
fn invert(m: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let inv = m.clone().try_inverse()?;
    if inv.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(inv)
}

/// Convenience: the DMGS expansion for a model, given the fitted parameters and
/// data, evaluated at the density points `x`.
///
/// This is the free-function form of [`DmgsIntegrator::cp_pdf`], provided so
/// callers can use the expansion without naming the integrator type. The model
/// is taken by value (Phase-1 models are zero-sized `Copy` types).
pub fn dmgs_cp_pdf<M: DmgsModel + Sync>(
    model: M,
    params: &[f64],
    y: &[f64],
    x: &[f64],
) -> Vec<f64> {
    DmgsIntegrator::new(model).cp_pdf(params, y, x)
}

/// Convenience: the DMGS CP quantiles for a model at probabilities `p`.
pub fn dmgs_cp_quantiles<M: DmgsModel + Sync>(
    model: M,
    params: &[f64],
    y: &[f64],
    p: &[f64],
) -> Vec<f64> {
    DmgsIntegrator::new(model).cp_quantiles(params, y, p)
}

/// The DMGS expansion applied to a scalar-valued function of the parameters.
///
/// This is the low-level entry point used by the WAIC/log-score machinery and
/// by tests: given the normalized `ldd`/`lddd`, the log-prior slope, and the
/// first/second derivatives of a scalar function `g(θ)` (as `g1` and `g2`), it
/// returns the scalar DMGS correction `Σ_i beps_i`.
///
/// It mirrors the reference `dmgs(lddi, lddd, mu1, pidopi, mu2, dim)` where
/// `mu1` is a 1-column matrix (so the result is a scalar).
pub fn dmgs_scalar(
    lddi: &DMatrix<f64>,
    lddd: &[f64],
    g1: &[f64],
    pidopi: &[f64],
    g2: &[f64],
    p: usize,
) -> f64 {
    let pidopi1: Vec<f64> = pidopi.iter().map(|v| 2.0 * v).collect();
    let pidopi2: Vec<f64> = pidopi.iter().map(|v| -v).collect();

    // W17 (H1): factor the parameter sums exactly as in `dmgs_expansion`. The
    // scalar case is `O(p⁴)` in the `beps1` term; collapsing the `j, r, s` sums
    // into a per-`t` coefficient keeps the same arithmetic with the same
    // summation order over `t`.
    let mut beps1 = 0.0;
    for t in 0..p {
        let mut c1 = 0.0;
        for j in 0..p {
            for r in 0..p {
                let lddi_jr = lddi[(j, r)];
                let base = j * p * p + r * p;
                for s in 0..p {
                    c1 += 0.5 * lddi[(s, t)] * lddi_jr * lddd[base + s];
                }
            }
        }
        beps1 += c1 * g1[t];
    }

    let mut beps2 = 0.0;
    let mut beps3 = 0.0;
    for t in 0..p {
        let mut c2 = 0.0;
        let mut c3 = 0.0;
        for s in 0..p {
            let l = lddi[(s, t)];
            c2 += l * pidopi1[s];
            c3 += l * pidopi2[s];
        }
        beps2 -= c2 * g1[t];
        beps3 -= c3 * g1[t];
    }

    let mut beps4 = 0.0;
    for j in 0..p {
        for r in 0..p {
            beps4 -= 0.5 * lddi[(j, r)] * g2[j * p + r];
        }
    }
    beps1 + beps2 + beps3 + beps4
}

/// Build a `DVector` from a slice (helper for tests and callers).
pub fn dvector(v: &[f64]) -> DVector<f64> {
    DVector::from_column_slice(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Bring the Phase-1 models (whose `DmgsModel` impls live in
    // `crate::dmgs::models`) into scope.
    use crate::models::{Exponential, LogNormal, Normal};

    /// §7.2 golden tolerance: `|a - b| <= atol + rtol * |b|`.
    const RTOL: f64 = 1e-8;
    const ATOL: f64 = 1e-10;

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

    fn sample() -> Vec<f64> {
        vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8]
    }

    fn positive_sample() -> Vec<f64> {
        vec![0.5, 1.2, 0.3, 2.1, 0.9, 1.7, 0.4, 1.1, 0.8, 1.5]
    }

    // The `DmgsModel` impls for all ten Phase-1 models live in
    // [`crate::dmgs::models`]; the tests below use them via the re-exports.

    // --- Expansion structure ------------------------------------------------

    #[test]
    fn dmgs_expansion_is_zero_for_zero_mu() {
        // With mu1 = 0 and mu2 = 0 the expansion must vanish.
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y = sample();
        let params = [0.4, 1.3];
        let ldd = integ.normalized_ldd(&params, &y);
        let lddi = invert(&ldd).unwrap();
        let lddd = integ.normalized_lddd(&params, &y);
        let pidopi = integ.log_prior_slope(&params);
        let beps = integ.dmgs_expansion(&lddi, &lddd, &[0.0, 0.0], &pidopi, &[0.0; 4], 2, 1);
        for (i, b) in beps.iter().enumerate() {
            assert!(b.abs() < 1e-14, "beps[{i}] = {b} should be 0");
        }
    }

    #[test]
    fn dmgs_scalar_matches_expansion_sum() {
        // The scalar helper must equal the sum of the vector expansion.
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y = sample();
        let params = [0.4, 1.3];
        let ldd = integ.normalized_ldd(&params, &y);
        let lddi = invert(&ldd).unwrap();
        let lddd = integ.normalized_lddd(&params, &y);
        let pidopi = integ.log_prior_slope(&params);
        let mu1 = [0.1, -0.2];
        let mu2 = [0.3, 0.05, 0.05, -0.4];
        let vec_sum: f64 = integ
            .dmgs_expansion(&lddi, &lddd, &mu1, &pidopi, &mu2, 2, 1)
            .iter()
            .sum();
        let scalar = dmgs_scalar(&lddi, &lddd, &mu1, &pidopi, &mu2, 2);
        assert_close(vec_sum, scalar, RTOL, ATOL, "expansion sum vs scalar");
    }

    #[test]
    fn normalized_ldd_is_negative_definite_at_mle() {
        // The normalized observed information is negative-definite at the MLE.
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y = sample();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let ldd = integ.normalized_ldd(&params, &y);
        // Analytic normalized ldd: diag(-1/sigma^2, -2/sigma^2).
        assert_close(ldd[(0, 0)], -1.0 / var, RTOL, ATOL, "ldd[0,0]");
        assert_close(ldd[(1, 1)], -2.0 / var, RTOL, ATOL, "ldd[1,1]");
        assert_close(ldd[(0, 1)], 0.0, RTOL, ATOL, "ldd[0,1]");
    }

    #[test]
    fn right_haar_prior_slope_is_minus_one_over_scale() {
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let slope = integ.log_prior_slope(&[0.0, 2.0]);
        assert_eq!(slope.len(), 2);
        assert_close(slope[0], 0.0, RTOL, ATOL, "slope[mu]");
        assert_close(slope[1], -0.5, RTOL, ATOL, "slope[sigma]");
    }

    #[test]
    fn exponential_prior_slope_is_minus_one_over_rate() {
        let m = Exponential;
        let integ = DmgsIntegrator::new(m);
        let slope = integ.log_prior_slope(&[2.0]);
        assert_eq!(slope.len(), 1);
        assert_close(slope[0], -0.5, RTOL, ATOL, "slope[rate]");
    }

    // --- CP density / quantiles --------------------------------------------

    #[test]
    fn cp_pdf_is_finite_and_positive() {
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y = sample();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let cp = integ.cp_pdf(&params, &y, &x);
        assert_eq!(cp.len(), x.len());
        for (i, v) in cp.iter().enumerate() {
            assert!(v.is_finite(), "cp_pdf[{i}] = {v} not finite");
            assert!(*v >= 0.0, "cp_pdf[{i}] = {v} negative");
        }
    }

    #[test]
    fn cp_pdf_close_to_ml_pdf_for_large_n() {
        // As n grows the DMGS correction is O(1/n), so cp_pdf -> ml_pdf.
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        // A large, well-behaved sample.
        let y: Vec<f64> = (0..2000)
            .map(|i| {
                let t = i as f64 / 2000.0;
                // Deterministic pseudo-sample around 0 with sd ~ 1.
                (t * 12.0 - 6.0) * 0.5
            })
            .collect();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let x = vec![-1.0, 0.0, 1.0];
        let cp = integ.cp_pdf(&params, &y, &x);
        for (i, &xi) in x.iter().enumerate() {
            let ml = m.pdf(xi, params[0], params[1]);
            // The correction is O(1/n) ~ 5e-4; allow a generous relative bound.
            assert_close(cp[i], ml, 1e-2, 1e-4, &format!("cp_pdf[{i}] -> ml_pdf"));
        }
    }

    #[test]
    fn cp_quantiles_are_monotone() {
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y = sample();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let p = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let q = integ.cp_quantiles(&params, &y, &p);
        assert_eq!(q.len(), p.len());
        for i in 1..q.len() {
            assert!(q[i] > q[i - 1], "cp_quantiles not monotone at {i}");
        }
    }

    #[test]
    fn cp_quantiles_close_to_ml_quantiles_for_large_n() {
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y: Vec<f64> = (0..2000)
            .map(|i| {
                let t = i as f64 / 2000.0;
                (t * 12.0 - 6.0) * 0.5
            })
            .collect();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let p = vec![0.1, 0.5, 0.9];
        let cp = integ.cp_quantiles(&params, &y, &p);
        for (i, &pi) in p.iter().enumerate() {
            let ml = m.quantile(pi, params[0], params[1]);
            assert_close(cp[i], ml, 1e-2, 1e-3, &format!("cp_q[{i}] -> ml_q"));
        }
    }

    #[test]
    fn integrator_trait_delegates_to_cp_pdf() {
        let m = Normal;
        let integ = DmgsIntegrator::new(m);
        let y = sample();
        let params = [0.4, 1.3];
        let x = vec![-1.0, 0.0, 1.0];
        let via_trait = Integrator::integrate(&integ, &params, &y, &x);
        let direct = integ.cp_pdf(&params, &y, &x);
        assert_eq!(via_trait, direct);
    }

    #[test]
    fn free_functions_match_methods() {
        let m = Normal;
        let y = sample();
        let params = [0.4, 1.3];
        let x = vec![-1.0, 0.0, 1.0];
        let p = vec![0.25, 0.5, 0.75];
        let integ = DmgsIntegrator::new(m);
        assert_eq!(
            dmgs_cp_pdf(m, &params, &y, &x),
            integ.cp_pdf(&params, &y, &x)
        );
        assert_eq!(
            dmgs_cp_quantiles(m, &params, &y, &p),
            integ.cp_quantiles(&params, &y, &p)
        );
    }

    #[test]
    fn lognormal_cp_pdf_is_finite() {
        let m = LogNormal;
        let integ = DmgsIntegrator::new(m);
        let y = positive_sample();
        let n = y.len() as f64;
        let logs: Vec<f64> = y.iter().map(|&yi| yi.ln()).collect();
        let mean = logs.iter().sum::<f64>() / n;
        let var = logs.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let x = vec![0.5, 1.0, 2.0];
        let cp = integ.cp_pdf(&params, &y, &x);
        for (i, v) in cp.iter().enumerate() {
            assert!(v.is_finite() && *v >= 0.0, "lnorm cp_pdf[{i}] = {v}");
        }
    }

    #[test]
    fn exponential_cp_pdf_is_finite() {
        let m = Exponential;
        let integ = DmgsIntegrator::new(m);
        let y = positive_sample();
        let rate = y.len() as f64 / y.iter().sum::<f64>();
        let x = vec![0.5, 1.0, 2.0];
        let cp = integ.cp_pdf(&[rate], &y, &x);
        for (i, v) in cp.iter().enumerate() {
            assert!(v.is_finite() && *v >= 0.0, "expon cp_pdf[{i}] = {v}");
        }
    }
}
