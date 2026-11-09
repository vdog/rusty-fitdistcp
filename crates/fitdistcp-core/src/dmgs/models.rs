//! [`DmgsModel`] implementations for the ten Phase-1 models (§3, §4.1).
//!
//! The DMGS expansion needs, for each model, the density `f(x; θ)`, the CDF
//! `F(x; θ)`, the quantile `F⁻¹(p; θ)`, and the log-prior slope `pidopi`. This
//! module wires the model primitives defined in [`crate::models`] into the
//! [`DmgsModel`] trait so the expansion in [`crate::dmgs`] can be invoked
//! generically over any of them.
//!
//! # Parameterizations
//!
//! The parameter vectors match the model modules exactly:
//!
//! | Model            | `params`                       | Scale index | Prior slope `pidopi` |
//! |------------------|--------------------------------|-------------|---------------------|
//! | `norm`           | `[mu, sigma]`                  | 1           | `[0, -1/σ]`         |
//! | `norm_p1`        | `[beta0, beta1, sigma]`        | 2           | `[0, 0, -1/σ]`      |
//! | `genextreme`     | `[loc, scale, shape]`          | 1           | `[0, -1/σ, 0]`      |
//! | `genextreme_p1`  | `[beta0, beta1, scale, shape]` | 2           | `[0, 0, -1/σ, 0]`   |
//! | `genpareto`      | `[scale, shape]`               | 0           | `[-1/σ, 0]`         |
//! | `gumbel`         | `[loc, scale]`                 | 1           | `[0, -1/σ]`         |
//! | `lnorm`          | `[meanlog, sdlog]`             | 1           | `[0, -1/σ]`         |
//! | `weibull`        | `[shape, scale]`               | 1           | `[0, -1/σ]`         |
//! | `gamma`          | `[shape, rate]`                | 1           | `[0, -1/rate]`      |
//! | `expon`          | `[rate]`                       | 0           | `[-1/rate]`         |
//!
//! The prior slope is the right-Haar prior `π(σ) ∝ 1/σ` for the homogeneous
//! models, so `pidopi` is `-1/σ` in the **scale** component and `0` elsewhere.
//! For the gamma the natural scale is `1/rate`, so the slope is `-1/rate`; for
//! the exponential (rate parameterization) it is `-1/rate`. These match the
//! reference `lambdad_rhp` vectors (`c(0, -1/v2hat)` for the two-parameter
//! models, `-1/v1hat` for the exponential).
//!
//! # Predictor models (`_p1`)
//!
//! The density/CDF/quantile of the `_p1` models depend on the predictor. The
//! DMGS expansion evaluates them at a single **prediction point** `t0` (the
//! reference `tresid0`), which the binding resolves from the user's `t0`/`n0`
//! via `maket0` (W14). The model centers the predictor by its mean
//! (`t_i - mean(t)`), and the prediction point is centered the same way
//! (`t0 - mean(t)`); the `DmgsModel` hooks below therefore evaluate at
//! [`NormalP1::centered_prediction`] / [`GenExtremeP1::centered_prediction`].
//! When the predictor is absent the model degenerates to a constant location.

use crate::dmgs::DmgsModel;
use crate::models::{
    Exponential, Gamma, GammaScale, GenExtreme, GenExtremeP1, GenExtremeP12, GenPareto, Gumbel,
    LogNormal, Normal, NormalP1, Weibull,
};

/// The standard-normal density `φ(z) = exp(-z²/2)/√(2π)`.
///
/// Used by the analytic density/CDF derivative hooks (the reference's `dnorm`).
#[inline]
fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Numerical first derivatives of a model's density, used as a fallback in the
/// shape ≈ 0 limit where the analytic GEV/GPD forms are singular.
fn numerical_density_first<M: DmgsModel>(m: &M, params: &[f64], x: f64) -> Vec<f64> {
    let p = params.len();
    let cfg = m.dmgs_fd_config();
    let mut out = vec![0.0; p];
    for i in 0..p {
        let h = cfg.step_for(params[i]);
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[i] += h;
        pm[i] -= h;
        out[i] = (m.pdf(x, &pp) - m.pdf(x, &pm)) / (2.0 * h);
    }
    out
}

/// Numerical second derivatives of a model's density (row-major `p × p`).
fn numerical_density_second<M: DmgsModel>(m: &M, params: &[f64], x: f64) -> Vec<f64> {
    let p = params.len();
    let cfg = m.dmgs_fd_config();
    let mut out = vec![0.0; p * p];
    let f0 = m.pdf(x, params);
    for i in 0..p {
        let hi = cfg.step_for_2(params[i]);
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[i] += hi;
        pm[i] -= hi;
        out[i * p + i] = (m.pdf(x, &pp) - 2.0 * f0 + m.pdf(x, &pm)) / (hi * hi);
        for j in (i + 1)..p {
            let hj = cfg.step_for_2(params[j]);
            let eval = |di: f64, dj: f64| {
                let mut q = params.to_vec();
                q[i] += di;
                q[j] += dj;
                m.pdf(x, &q)
            };
            let val =
                (eval(hi, hj) - eval(hi, -hj) - eval(-hi, hj) + eval(-hi, -hj)) / (4.0 * hi * hj);
            out[i * p + j] = val;
            out[j * p + i] = val;
        }
    }
    out
}

/// Numerical first derivatives of a model's CDF.
fn numerical_cdf_first<M: DmgsModel>(m: &M, params: &[f64], x: f64) -> Vec<f64> {
    let p = params.len();
    let cfg = m.dmgs_fd_config();
    let mut out = vec![0.0; p];
    for i in 0..p {
        let h = cfg.step_for(params[i]);
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[i] += h;
        pm[i] -= h;
        out[i] = (m.cdf(x, &pp) - m.cdf(x, &pm)) / (2.0 * h);
    }
    out
}

/// Numerical second derivatives of a model's CDF (row-major `p × p`).
fn numerical_cdf_second<M: DmgsModel>(m: &M, params: &[f64], x: f64) -> Vec<f64> {
    let p = params.len();
    let cfg = m.dmgs_fd_config();
    let mut out = vec![0.0; p * p];
    let f0 = m.cdf(x, params);
    for i in 0..p {
        let hi = cfg.step_for_2(params[i]);
        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        pp[i] += hi;
        pm[i] -= hi;
        out[i * p + i] = (m.cdf(x, &pp) - 2.0 * f0 + m.cdf(x, &pm)) / (hi * hi);
        for j in (i + 1)..p {
            let hj = cfg.step_for_2(params[j]);
            let eval = |di: f64, dj: f64| {
                let mut q = params.to_vec();
                q[i] += di;
                q[j] += dj;
                m.cdf(x, &q)
            };
            let val =
                (eval(hi, hj) - eval(hi, -hj) - eval(-hi, hj) + eval(-hi, -hj)) / (4.0 * hi * hj);
            out[i * p + j] = val;
            out[j * p + i] = val;
        }
    }
    out
}

// --- norm -------------------------------------------------------------------

impl DmgsModel for Normal {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        Normal::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        Normal::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        Normal::quantile(self, p, params[0], params[1])
    }

    /// Analytic density first derivatives (reference `norm_fd`).
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = v2 * v2;
        let e2 = x - v1;
        let e3 = e2 * e2;
        let e4 = 2.0 * e1;
        let e7 = (-(e3 / e4)).exp();
        let e8 = (2.0 * std::f64::consts::PI).sqrt();
        vec![
            e7 * e2 / (v2 * v2 * v2 * e8),
            (4.0 * (e3 / (e4 * e4)) - 1.0 / e1) * e7 / e8,
        ]
    }

    /// Analytic density second derivatives (reference `norm_fdd`).
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = v2 * v2;
        let e2 = x - v1;
        let e3 = 2.0 * e1;
        let e4 = e2 * e2;
        let e5 = e3 * e3;
        let e7 = (2.0 * std::f64::consts::PI).sqrt();
        let e9 = (-(e4 / e3)).exp();
        let e10 = v2 * v2 * v2;
        let e14 = 4.0 * (e4 / e5) - 1.0 / e1;
        let e15 = e10 * e7;
        // Row-major 2x2: [v1v1, v1v2; v2v1, v2v2].
        let d11 = (e4 / e1 - 1.0) * e9 / e15;
        let d12 = (e14 / e1 - 8.0 / e5) * e9 * e2 / e7;
        let d21 = (4.0 * (e4 / (e1 * e5 * e7)) - 3.0 * (e1 * e7 / (e15 * e15))) * e9 * e2;
        let d22 = (2.0 / e10 + v2 * (4.0 * e14 - 64.0 * (e1 / e5)) * e4 / e5) * e9 / e7;
        vec![d11, d12, d21, d22]
    }

    /// Analytic CDF first derivatives (reference `norm_pd`).
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x - v1;
        let e3 = standard_normal_pdf(e1 / v2);
        vec![-(e3 / v2), -(e3 * e1 / (v2 * v2))]
    }

    /// Analytic CDF second derivatives (reference `norm_pdd`).
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x - v1;
        let e2 = v2 * v2;
        let e4 = standard_normal_pdf(e1 / v2);
        let e6 = e1 * e1 / e2;
        let e7 = -((e6 - 1.0) * e4 / e2);
        let e8 = v2 * v2 * v2;
        let d11 = -(e4 * e1 / e8);
        let d22 = -((e6 - 2.0) * e4 * e1 / e8);
        vec![d11, e7, e7, d22]
    }

    /// Analytic mean derivatives (reference `norm.py`): `ml_mean = mu`,
    /// `∂/∂mu = 1`, `∂/∂sigma = 0`, all second derivatives zero.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let mu = params[0];
        (mu, vec![1.0, 0.0], vec![0.0, 0.0, 0.0, 0.0])
    }
}

// --- norm_p1 ----------------------------------------------------------------

impl DmgsModel for NormalP1 {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        // Evaluate at the prediction point `t0` (the reference `tresid0`), not at
        // observation 0: the DMGS expansion is reported at the prediction point.
        let mu = self.location_prediction(params[0], params[1]);
        let z = (x - mu) / params[2];
        (-0.5 * z * z - params[2].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()).exp()
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        let mu = self.location_prediction(params[0], params[1]);
        let z = (x - mu) / (params[2] * std::f64::consts::SQRT_2);
        0.5 * crate::special::erfc(-z)
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        let mu = self.location_prediction(params[0], params[1]);
        mu + params[2] * crate::models::norm::standard_normal_quantile(p)
    }
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        // Right-Haar prior on the scale (sigma, index 2).
        vec![0.0, 0.0, -1.0 / params[2]]
    }

    /// Analytic density first derivatives (reference `norm_p1_fd`).
    ///
    /// The predictor `t` is the centered value at the prediction point.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = params[2];
        let t = self.centered_prediction();
        let e3 = x - (t * v2 + v1);
        let e4 = v3 * v3;
        let e5 = e3 * e3;
        let e6 = 2.0 * e4;
        let e9 = (-(e5 / e6)).exp();
        let e10 = (2.0 * std::f64::consts::PI).sqrt();
        let e12 = v3 * v3 * v3 * e10;
        vec![
            e9 * e3 / e12,
            t * e9 * e3 / e12,
            (4.0 * (e5 / (e6 * e6)) - 1.0 / e4) * e9 / e10,
        ]
    }

    /// Analytic density second derivatives (reference `norm_p1_fdd`).
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = params[2];
        let t = self.centered_prediction();
        let e1 = v3 * v3;
        let e4 = x - (t * v2 + v1);
        let e5 = e4 * e4;
        let e6 = 2.0 * e1;
        let e8 = (2.0 * std::f64::consts::PI).sqrt();
        let e10 = e6 * e6;
        let e11 = (-(e5 / e6)).exp();
        let e12 = v3 * v3 * v3;
        let e13 = e12 * e8;
        let e15 = e5 / e1 - 1.0;
        let e19 = 4.0 * (e5 / e10) - 1.0 / e1;
        let e21 = e19 / e1 - 8.0 / e10;
        let e26 = 4.0 * (e5 / (e1 * e10 * e8)) - 3.0 * (e1 * e8 / (e13 * e13));
        let e29 = t * e15 * e11 / e13;
        // Row-major 3x3: [b0, b1, sigma].
        vec![
            e15 * e11 / e13,
            e29,
            e21 * e11 * e4 / e8,
            e29,
            t * t * e15 * e11 / e13,
            t * e21 * e11 * e4 / e8,
            e26 * e11 * e4,
            t * e26 * e11 * e4,
            (2.0 / e12 + v3 * (4.0 * e19 - 64.0 * (e1 / e10)) * e5 / e10) * e11 / e8,
        ]
    }

    /// Analytic CDF first derivatives (reference `norm_p1_pd`).
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = params[2];
        let t = self.centered_prediction();
        let e3 = x - (t * v2 + v1);
        let e5 = standard_normal_pdf(e3 / v3);
        vec![-(e5 / v3), -(t * e5 / v3), -(e5 * e3 / (v3 * v3))]
    }

    /// Analytic CDF second derivatives (reference `norm_p1_pdd`).
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = params[2];
        let t = self.centered_prediction();
        let e3 = x - (t * v2 + v1);
        let e5 = standard_normal_pdf(e3 / v3);
        let e6 = v3 * v3;
        let e8 = e3 * e3 / e6;
        let e9 = v3 * v3 * v3;
        let e10 = e8 - 1.0;
        let e11 = -(e10 * e5 / e6);
        let e12 = -(t * e10 * e5 / e6);
        let e13 = -(t * e5 * e3 / e9);
        // Row-major 3x3: [b0, b1, sigma].
        vec![
            -(e5 * e3 / e9),
            e13,
            e11,
            e13,
            -(t * t * e5 * e3 / e9),
            e12,
            e11,
            e12,
            -((e8 - 2.0) * e5 * e3 / e9),
        ]
    }

    /// Analytic mean derivatives (reference `norm_p1`): the predictive mean is
    /// the location at the prediction point `muhat0 = beta0 + beta1·t0c`, so
    /// `∂/∂beta0 = 1`, `∂/∂beta1 = t0c`, `∂/∂sigma = 0`, all second derivatives
    /// zero.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let t = self.centered_prediction();
        let muhat0 = params[0] + params[1] * t;
        (muhat0, vec![1.0, t, 0.0], vec![0.0; 9])
    }
}

// --- genextreme (GEV) -------------------------------------------------------

impl DmgsModel for GenExtreme {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        GenExtreme::pdf(self, x, params[0], params[1], params[2])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        GenExtreme::cdf(self, x, params[0], params[1], params[2])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        GenExtreme::quantile(self, p, params[0], params[1], params[2])
    }

    /// Analytic density first derivatives (reference `gev_fd`).
    ///
    /// The reference `gev_f1fa` applies `movexiawayfromzero` to the shape before
    /// evaluating `gev_fd`, so the `1/shape` singularity is avoided by nudging
    /// `shape` to `±1e-7` rather than by switching to the Gumbel limit. We follow
    /// the reference exactly (W10).
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = crate::models::genextreme::movexiawayfromzero(params[2]);
        let e1 = x - v1;
        let e2 = 1.0 / v3;
        let e4 = v3 * e1 / v2;
        let e5 = 1.0 + e4;
        let e6 = 1.0 + e2;
        let e7 = e5.powf(e6);
        let e9 = e5.powf(e2 + 2.0);
        let e10 = (-e5.powf(-e2)).exp();
        let e13 = e4.ln_1p();
        let e14 = v2 * v2;
        let e17 = v3 * e6 / e9 - 1.0 / e5.powf(2.0 * e6);
        vec![
            e10 * e17 / e14,
            (e17 * e1 / v2 - 1.0 / e7) * e10 / e14,
            ((e13 / (v3 * e7) - (e13 / (v3 * e5.powf(e2)) - e1 / (v2 * e7)) / e7) / v3
                - e6 * e1 / (v2 * e9))
                * e10
                / v2,
        ]
    }

    /// Analytic density second derivatives (reference `gev_fdd`).
    ///
    /// The reference `gev_f2fa` applies `movexiawayfromzero` to the shape before
    /// evaluating `gev_fdd` (W10).
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = crate::models::genextreme::movexiawayfromzero(params[2]);
        let e1 = x - v1;
        let e2 = 1.0 / v3;
        let e4 = v3 * e1 / v2;
        let e5 = 1.0 + e4;
        let e6 = 1.0 + e2;
        let e7 = e5.powf(e6);
        let e8 = e2 + 2.0;
        let e9 = e4.ln_1p();
        let e10 = e5.powf(e8);
        let e11 = 2.0 * e6;
        let e12 = e5.powf(e2);
        let e13 = v3 * e12;
        let e14 = e5.powf(e11);
        let e15 = v2 * e7;
        let e16 = v3 * e6;
        let e18 = 1.0 / e14;
        let e20 = e9 / e13 - e1 / e15;
        let e21 = v3 * e7;
        let e22 = v3 * v3;
        let e24 = (-e5.powf(-e2)).exp();
        let e25 = v2 * e10;
        let e27 = e16 / e10 - e18;
        let e28 = 1.0 / e7;
        let e29 = 1.0 / e10;
        let e30 = 2.0 * e8;
        let e31 = e20 / e7;
        let e32 = e9 / e21;
        let e33 = v2 * v2;
        let e37 = e6 * e12 * e1 / v2 - e7 * e9 / e22;
        let e39 = e6 * e1 / e25;
        let e42 = e5.powf(e2 - e11);
        let e43 = (e32 - e31) / v3;
        let e46 = e27 * e1 / v2 - e28;
        let e48 = v2 * v2 * v2;
        let e51 = v3 * e5.powf(e6 - e30) * e8 - 2.0 / e5.powf(1.0 + e11);
        let e55 = e7 * e8 * e1 / v2 - e10 * e9 / e22;
        let e56 = e5.powf(e2 - 1.0);
        let e57 = e43 - e39;
        let e58 = e15 * e15;
        let e59 = e25 * e25;
        let e60 = e21 * e21;
        let e61 = e13 * e13;
        let e62 = e37 / e14;
        let e80 = e18 + v3 * (e51 * e1 / v2 - (e42 + e29)) * e6 - e46 / e7;
        let e85 = e16 * e12 * e1;
        let e87 = e16 * e51 - e27 / e7;
        let e89 = e16 * (e13 * e9 / e60 - e42 * e20) - e29;
        let e91 = e21 * e8 * e1;
        let e94 = v3 * e56 * e9 / e61;
        let d11 = e24 * e87 / e48;
        let d12 = e80 * e24 / e48;
        let d13 = ((e89 / v2 - ((e28 + e94 - e28) / v2 - e85 / e58) / e7) / v3
            - (e57 / e15 + e6 * (e91 / e59 - 1.0 / e25)))
            * e24
            / v2;
        let d22 = (e80 * e1 / v2 - 2.0 * e46) * e24 / e48;
        let d23 = (((e10 - e91 / v2) * e6 / e59
            + (e89 / e33 - ((e7 - e85 / v2) / e58 + (e94 - e28) / e33) / e7) / v3
            - e57 / (e33 * e7))
            * e1
            - e57 / v2)
            * e24
            / v2;
        let d33 = (((e62 + e39 - e43) * e20 + (e31 + e1 / e25 - e32) / v3
            - ((e7 + v3 * e37) * e9 / e60
                + ((1.0 / (v2 * v3 * e7) + v2 * e37 / e58) * e1
                    - (e56 * e1 / v2 + e12 - e12 * e9 / v3) * e9 / e61)
                    / e7))
            / v3
            + (1.0 / (v2 * e22 * e10) + v2 * e55 * e6 / e59) * e1)
            * e24
            / v2;
        // Row-major 3x3: [loc, scale, shape].
        vec![
            d11, d12, d13, // loc row
            d12, d22, d23, // scale row
            d13, d23, d33, // shape row
        ]
    }

    /// Analytic CDF first derivatives (reference `gev_pd`).
    ///
    /// The reference `gev_mu1fa` applies `movexiawayfromzero` to the shape before
    /// evaluating `gev_pd` (W10).
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = crate::models::genextreme::movexiawayfromzero(params[2]);
        let e1 = x - v1;
        let e3 = v3 * e1 / v2;
        let e4 = 1.0 + e3;
        let e5 = 1.0 / v3;
        let e7 = e4.powf(1.0 + e5);
        let e8 = (-e4.powf(-e5)).exp();
        let e9 = v2 * e7;
        vec![
            -(e8 / e9),
            -(e8 * e1 / (v2 * v2 * e7)),
            -(e8 * (e3.ln_1p() / (v3 * e4.powf(e5)) - e1 / e9) / v3),
        ]
    }

    /// Analytic CDF second derivatives (reference `gev_pdd`).
    ///
    /// The reference `gev_mu2fa` applies `movexiawayfromzero` to the shape before
    /// evaluating `gev_pdd` (W10).
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let v3 = crate::models::genextreme::movexiawayfromzero(params[2]);
        let e1 = x - v1;
        let e3 = v3 * e1 / v2;
        let e4 = 1.0 / v3;
        let e5 = 1.0 + e3;
        let e6 = 1.0 + e4;
        let e7 = e5.powf(e6);
        let e8 = e5.powf(e4);
        let e9 = v2 * e7;
        let e10 = e3.ln_1p();
        let e12 = (-e5.powf(-e4)).exp();
        let e13 = v3 * e8;
        let e14 = v2 * v2;
        let e15 = e1 / e9;
        let e16 = e9 * e9;
        let e17 = e10 / e13;
        let e18 = e17 - e15;
        let e20 = v3 * e6 * e8;
        let e21 = e14 * e7;
        let e22 = e20 * e1;
        let e26 = e6 * e8 * e1 / v2 - e7 * e10 / (v3 * v3);
        let e27 = e5.powf(e4 - 1.0);
        let e28 = e5.powf(2.0 * e6);
        let e29 = e21 * e21;
        let e30 = e13 * e13;
        let e31 = 1.0 / e7;
        let e32 = v2 * v3;
        let e33 = (e7 - e22 / v2) / e16;
        let e35 = e18 / e7 + e31;
        let e37 = v2 * e26 / e16;
        let e38 = e32 * e7;
        let e41 = v3 * e27 * e10 / e30;
        let d11 = -(e12 * (e20 / e16 - 1.0 / (e14 * e28)));
        let d12 = -(e12 * (e32 * e6 * e8 * e1 / e29 - (e15 + 1.0) / e21));
        let d13 = -(((e31 + e41 - e35) / v2 - e22 / e16) * e12 / v3);
        let d22 = ((2.0 * e9 - e22) / e29 + e1 / (v2 * v2 * v2 * v2 * e28)) * e12 * e1;
        let d23 = -((e33 + (e41 - e35) / e14) * e12 * e1 / v3);
        let d33 = -(((1.0 / e38 + e37) * e1
            - ((e27 * e1 / v2 + e8 - e8 * e10 / v3) * e10 / e30 + (1.0 + e17 - e15) * e18 / v3))
            * e12
            / v3);
        vec![
            d11, d12, d13, // loc row
            d12, d22, d23, // scale row
            d13, d23, d33, // shape row
        ]
    }

    /// Analytic mean derivatives (reference `genextreme_libs.py::gev_means`).
    ///
    /// For `shape == 0` (Gumbel limit): `ml_mean = mu + sigma·γ`,
    /// `∂/∂mu = 1`, `∂/∂sigma = γ`, `∂/∂shape = 0`, all second derivatives zero.
    ///
    /// For `shape != 0`, with `g0 = Γ(1-ξ)`, `g1 = g0·ψ(1-ξ)`,
    /// `g2 = (ψ₁(1-ξ)·g0² + g1²)/g0`:
    /// `ml_mean = mu + sigma·(g0-1)/ξ`,
    /// `∂/∂mu = 1`, `∂/∂sigma = (g0-1)/ξ`,
    /// `∂/∂ξ = (1 - g0 - ξ·g1)·sigma/ξ²`,
    /// `∂²/∂sigma∂ξ = (1 - g0 - ξ·g1)/ξ²`,
    /// `∂²/∂ξ² = sigma·(-2 + 2g0 + 2ξ·g1 + ξ²·g2)/ξ³`.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        use crate::special::{digamma, gamma, trigamma};
        const EULER: f64 = 0.577_215_664_901_532_9;
        let mu = params[0];
        let sigma = params[1];
        let xi = params[2];
        let mut meand1 = vec![0.0; 3];
        let mut meand2 = vec![0.0; 9];
        let ml_mean;
        if xi == 0.0 {
            ml_mean = mu + sigma * EULER;
            meand1[0] = 1.0;
            meand1[1] = EULER;
        } else {
            let g0 = gamma(1.0 - xi);
            let g1 = g0 * digamma(1.0 - xi);
            let g2 = (trigamma(1.0 - xi) * g0 * g0 + g1 * g1) / g0;
            ml_mean = mu + sigma * (g0 - 1.0) / xi;
            meand1[0] = 1.0;
            meand1[1] = (g0 - 1.0) / xi;
            meand1[2] = (1.0 - g0 - xi * g1) * sigma / (xi * xi);
            // meand2[1,2] = meand2[2,1] = (1 - g0 - xi*g1)/xi^2
            let d12 = (1.0 - g0 - xi * g1) / (xi * xi);
            meand2[5] = d12; // (1, 2)
            meand2[7] = d12; // (2, 1)
            meand2[8] = sigma * (-2.0 + 2.0 * g0 + 2.0 * xi * g1 + xi * xi * g2) / (xi * xi * xi);
            // (2, 2)
        }
        (ml_mean, meand1, meand2)
    }
}

// --- genextreme_p1 ----------------------------------------------------------

impl DmgsModel for GenExtremeP1 {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        // Evaluate at the prediction point `t0` (the reference `tresid0`).
        let loc = self.location_prediction(params[0], params[1]);
        let scale = params[2];
        let shape = params[3];
        let z = (x - loc) / scale;
        if shape.abs() < 1e-12 {
            return (-z - (-z).exp() - scale.ln()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return 0.0;
        }
        (-scale.ln() - (1.0 / shape + 1.0) * t.ln() - t.powf(-1.0 / shape)).exp()
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        let loc = self.location_prediction(params[0], params[1]);
        let scale = params[2];
        let shape = params[3];
        let z = (x - loc) / scale;
        if shape.abs() < 1e-12 {
            return (-(-z).exp()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return if shape > 0.0 { 0.0 } else { 1.0 };
        }
        (-t.powf(-1.0 / shape)).exp()
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        let loc = self.location_prediction(params[0], params[1]);
        let scale = params[2];
        let shape = params[3];
        if shape.abs() < 1e-12 {
            return loc - scale * (-p.ln()).ln();
        }
        loc + (scale / shape) * ((-p.ln()).powf(-shape) - 1.0)
    }
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        // Right-Haar prior on the scale (index 2).
        vec![0.0, 0.0, -1.0 / params[2], 0.0]
    }

    /// Analytic mean derivatives (reference `genextreme_p1_libs.py::gev_p1_means`).
    ///
    /// `ml_mean = ymn + slope·t0 + sigma·(g0-1)/xi` (or `+ sigma·γ` for xi=0),
    /// where `t0` is the centered prediction point. The derivative array mirrors
    /// the reference with the location/slope/scale/shape ordering.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        use crate::special::{digamma, gamma, trigamma};
        const EULER: f64 = 0.577_215_664_901_532_9;
        let ymn = params[0];
        let slope = params[1];
        let sigma = params[2];
        let xi = params[3];
        let t0 = self.centered_prediction();
        let mut meand1 = vec![0.0; 4];
        let mut meand2 = vec![0.0; 16];
        let ml_mean;
        if xi == 0.0 {
            ml_mean = ymn + slope * t0 + sigma * EULER;
            meand1[0] = 1.0;
            meand1[1] = t0;
            meand1[2] = EULER;
        } else {
            let g0 = gamma(1.0 - xi);
            let g1 = g0 * digamma(1.0 - xi);
            let g2 = (trigamma(1.0 - xi) * g0 * g0 + g1 * g1) / g0;
            ml_mean = ymn + slope * t0 + sigma * (g0 - 1.0) / xi;
            meand1[0] = 1.0;
            meand1[1] = t0;
            meand1[2] = (g0 - 1.0) / xi;
            meand1[3] = (1.0 - g0 - xi * g1) * sigma / (xi * xi);
            // meand2[2,3] = meand2[3,2] = (1 - g0 - xi*g1)/xi^2
            let d23 = (1.0 - g0 - xi * g1) / (xi * xi);
            meand2[11] = d23; // (2, 3)
            meand2[14] = d23; // (3, 2)
            meand2[15] = sigma * (-2.0 + 2.0 * g0 + 2.0 * xi * g1 + xi * xi * g2) / (xi * xi * xi);
            // (3, 3)
        }
        (ml_mean, meand1, meand2)
    }
}

// --- genextreme_p12 ---------------------------------------------------------

impl DmgsModel for GenExtremeP12 {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        // Evaluate at the prediction points `t01`/`t02` (the reference
        // `tresid0`), not at observation 0: the DMGS expansion is reported at
        // the prediction point.
        let loc = self.location_prediction(params[0], params[1]);
        let sigma = self.scale_prediction(params[2], params[3]);
        let shape = params[4];
        let z = (x - loc) / sigma;
        if shape.abs() < 1e-12 {
            return (-z - (-z).exp() - sigma.ln()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return 0.0;
        }
        (-sigma.ln() - (1.0 / shape + 1.0) * t.ln() - t.powf(-1.0 / shape)).exp()
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        let loc = self.location_prediction(params[0], params[1]);
        let sigma = self.scale_prediction(params[2], params[3]);
        let shape = params[4];
        let z = (x - loc) / sigma;
        if shape.abs() < 1e-12 {
            return (-(-z).exp()).exp();
        }
        let t = 1.0 + shape * z;
        if t <= 0.0 {
            return if shape > 0.0 { 0.0 } else { 1.0 };
        }
        (-t.powf(-1.0 / shape)).exp()
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        let loc = self.location_prediction(params[0], params[1]);
        let sigma = self.scale_prediction(params[2], params[3]);
        let shape = params[4];
        if shape.abs() < 1e-12 {
            return loc - sigma * (-p.ln()).ln();
        }
        loc + (sigma / shape) * ((-p.ln()).powf(-shape) - 1.0)
    }
    fn log_prior_slope(&self, _params: &[f64]) -> Vec<f64> {
        // Right-Haar prior on the scale. The scale is `exp(beta2 + scale*t2)`, so
        // the log-slope w.r.t. the log-scale intercept `beta2` (index 2) is `-1`
        // and w.r.t. the log-scale slope `scale` (index 3) is `-t2` at the
        // prediction point. The reference `lambdad_rhp` for the `_p12` model is
        // the right-Haar prior on the (predictor-dependent) scale; we mirror the
        // `_p1` convention with the log-scale intercept carrying `-1`.
        vec![0.0, 0.0, -1.0, 0.0, 0.0]
    }

    /// Analytic mean derivatives (reference `genextreme_p12_libs.py::gev_p12_means`).
    ///
    /// `ml_mean = ymn + slope·t01 + sigma·(g0-1)/xi` (or `+ sigma·γ` for xi=0),
    /// where `sigma = exp(sigma1 + sigma2·t02)` and `t01`/`t02` are the centered
    /// prediction points. The derivative array mirrors the reference with the
    /// location/slope/log-scale-intercept/log-scale-slope/shape ordering.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        use crate::special::{digamma, gamma, trigamma};
        const EULER: f64 = 0.577_215_664_901_532_9;
        let ymn = params[0];
        let slope = params[1];
        let sigma1 = params[2];
        let sigma2 = params[3];
        let xi = params[4];
        let t01 = self.centered_prediction1();
        let t02 = self.centered_prediction2();
        let sigma = (sigma1 + sigma2 * t02).exp();
        let mut meand1 = vec![0.0; 5];
        let mut meand2 = vec![0.0; 25];
        let ml_mean;
        if xi == 0.0 {
            ml_mean = ymn + slope * t01 + sigma * EULER;
            meand1[0] = 1.0;
            meand1[1] = t01;
            // d/d(sigma1) = sigma·γ, d/d(sigma2) = sigma·γ·t02.
            meand1[2] = sigma * EULER;
            meand1[3] = sigma * EULER * t02;
        } else {
            let g0 = gamma(1.0 - xi);
            let g1 = g0 * digamma(1.0 - xi);
            let g2 = (trigamma(1.0 - xi) * g0 * g0 + g1 * g1) / g0;
            let dsigma = (g0 - 1.0) / xi;
            ml_mean = ymn + slope * t01 + sigma * dsigma;
            meand1[0] = 1.0;
            meand1[1] = t01;
            meand1[2] = sigma * dsigma;
            meand1[3] = sigma * dsigma * t02;
            meand1[4] = (1.0 - g0 - xi * g1) * sigma / (xi * xi);
            // Second derivatives. Indices are row-major 5x5 over
            // [ymn, slope, sigma1, sigma2, xi].
            // d²/d(sigma1)d(xi) = sigma·(1 - g0 - xi·g1)/xi²
            let d2_s1_xi = sigma * (1.0 - g0 - xi * g1) / (xi * xi);
            meand2[2 * 5 + 4] = d2_s1_xi;
            meand2[4 * 5 + 2] = d2_s1_xi;
            // d²/d(sigma2)d(xi) = t02 · d2_s1_xi
            meand2[3 * 5 + 4] = t02 * d2_s1_xi;
            meand2[4 * 5 + 3] = t02 * d2_s1_xi;
            // d²/d(xi)² = sigma·(-2 + 2g0 + 2ξ·g1 + ξ²·g2)/ξ³
            meand2[4 * 5 + 4] =
                sigma * (-2.0 + 2.0 * g0 + 2.0 * xi * g1 + xi * xi * g2) / (xi * xi * xi);
        }
        (ml_mean, meand1, meand2)
    }
}

// --- genpareto (GPD) --------------------------------------------------------

impl DmgsModel for GenPareto {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        GenPareto::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        GenPareto::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        GenPareto::quantile(self, p, params[0], params[1])
    }
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        // The GPD scale is the first parameter; right-Haar prior pi(sigma) ∝ 1/sigma.
        vec![-1.0 / params[0], 0.0]
    }

    /// Analytic density first derivatives (reference `gpd_k1_fd`, location 0).
    ///
    /// Returns `[∂f/∂scale, ∂f/∂shape]`.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v2 = params[0];
        let v3 = params[1];
        if v3.abs() < 1e-12 {
            return numerical_density_first(self, params, x);
        }
        let e1 = 1.0 + v3;
        let e2 = x;
        let e3 = e1 / v3;
        let e5 = v3 * e2 / v2;
        let e6 = 1.0 + e5;
        let e8 = e6.powf(e3);
        let e10 = e1 * e2 / (v2 * e6.powf(e3 + 1.0));
        vec![
            (e10 - 1.0 / e8) / (v2 * v2),
            -(((1.0 - e3) * e5.ln_1p() / e8 + e10) / (v2 * v3)),
        ]
    }

    /// Analytic density second derivatives (reference `gpd_k1_fdd`, location 0).
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v2 = params[0];
        let v3 = params[1];
        if v3.abs() < 1e-12 {
            return numerical_density_second(self, params, x);
        }
        let e1 = x;
        let e2 = 1.0 + v3;
        let e3 = e2 / v3;
        let e5 = v3 * e1 / v2;
        let e6 = 1.0 + e5;
        let e7 = e3 + 1.0;
        let e8 = e6.powf(e7);
        let e9 = e6.powf(e3);
        let e10 = e5.ln_1p();
        let e11 = v2 * e8;
        let e12 = 1.0 - e3;
        let e13 = e11 * e11;
        let e14 = v2 * v3;
        let e15 = v2 * v2;
        let e17 = e2 * e1 / e11;
        let e24 = e12 * e9 * e10 + e2 * e6.powf(e3 - 1.0) * e1 / v2;
        let e27 = e12 * e10 / e9 + e17;
        let e28 = e8 - v3 * e7 * e9 * e1 / v2;
        let e30 = (1.0 / e11 - v2 * (e7 * e9 * e1 / v2 + e12 * e8 * e10 / v3) * e2 / e13) * e1;
        let e31 = e14 * e14;
        let e32 = v3 * e6.powf(2.0 * e3);
        let d11 =
            -(((e28 / e13 + 1.0 / (e15 * e8)) * e2 * e1 + 2.0 * ((e17 - 1.0 / e9) / v2)) / e15);
        let d12 =
            -(((e2 * e10 / e8 - v3 / e8) * e12 / e15 - e28 * e2 / e13) * e1 / e14 - v3 * e27 / e31);
        let d21 = (e24 / e32 + e30) / e15;
        let d22 = -((((e1 / (v2 * e6) - e10 / v3) / e9 - e24 * e10 / e32) * e12 + e30) / e14
            - v2 * e27 / e31);
        vec![d11, d12, d21, d22]
    }

    /// Analytic CDF first derivatives (reference `gpd_k1_pd`, location 0).
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v2 = params[0];
        let v3 = params[1];
        if v3.abs() < 1e-12 {
            return numerical_cdf_first(self, params, x);
        }
        let e1 = x;
        let e3 = v3 * e1 / v2;
        let e4 = 1.0 + e3;
        let e5 = 1.0 / v3;
        let e6 = e4.powf(1.0 + e5);
        vec![
            -(e1 / (v2 * v2 * e6)),
            -((e3.ln_1p() / (v3 * e4.powf(e5)) - e1 / (v2 * e6)) / v3),
        ]
    }

    /// Analytic CDF second derivatives (reference `gpd_k1_pdd`, location 0).
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v2 = params[0];
        let v3 = params[1];
        if v3.abs() < 1e-12 {
            return numerical_cdf_second(self, params, x);
        }
        let e1 = x;
        let e3 = v3 * e1 / v2;
        let e4 = 1.0 / v3;
        let e5 = 1.0 + e3;
        let e6 = 1.0 + e4;
        let e7 = e5.powf(e6);
        let e8 = e5.powf(e4);
        let e9 = e3.ln_1p();
        let e10 = v2 * e7;
        let e11 = v2 * v2;
        let e12 = v3 * e8;
        let e16 = e6 * e8 * e1 / v2 - e7 * e9 / (v3 * v3);
        let e17 = e5.powf(e4 - 1.0);
        let e18 = e10 * e10;
        let e19 = (e11 * e7) * (e11 * e7);
        let e20 = e12 * e12;
        let e23 = v3 * e6 * e8 * e1;
        let d11 = (2.0 * e10 - e23) * e1 / e19;
        let d12 = -(((e7 - e23 / v2) / e18 + (v3 * e17 * e9 / e20 - 1.0 / e7) / e11) * e1 / v3);
        let d21 = e11 * e16 * e1 / e19;
        let d22 = -(((1.0 / (v2 * v3 * e7) + v2 * e16 / e18) * e1
            - ((e17 * e1 / v2 + e8 - e8 * e9 / v3) * e9 / e20 + (e9 / e12 - e1 / e10) / v3))
            / v3);
        vec![d11, d12, d21, d22]
    }

    /// Analytic mean derivatives (reference `genpareto_libs.py::gpd_k1_means`,
    /// location `kloc = 0`).
    ///
    /// `ml_mean = sigma/(1-xi)`, `∂/∂sigma = 1/(1-xi)`,
    /// `∂/∂xi = sigma/(1-xi)²`, `∂²/∂sigma∂xi = 1/(1-xi)²`,
    /// `∂²/∂xi² = -2·sigma/(1-xi)³`.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let sigma = params[0];
        let xi = params[1];
        let onemxi = 1.0 - xi;
        let onemxisq = onemxi * onemxi;
        let onemxicu = onemxisq * onemxi;
        let ml_mean = sigma / onemxi;
        let meand1 = vec![1.0 / onemxi, sigma / onemxisq];
        let mut meand2 = vec![0.0; 4];
        meand2[1] = 1.0 / onemxisq;
        meand2[2] = 1.0 / onemxisq;
        meand2[3] = -2.0 * sigma / onemxicu;
        (ml_mean, meand1, meand2)
    }
}

// --- gumbel -----------------------------------------------------------------

impl DmgsModel for Gumbel {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        Gumbel::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        Gumbel::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        Gumbel::quantile(self, p, params[0], params[1])
    }

    /// Analytic density first derivatives (reference `gumbel_fd`).
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x - v1;
        let e2 = e1 / v2;
        let e4 = (-e2).exp();
        let e6 = (-(e2 + e4)).exp();
        let e7 = e4 - 1.0;
        let e8 = v2 * v2;
        vec![-(e6 * e7 / e8), -((e7 * e1 / v2 + 1.0) * e6 / e8)]
    }

    /// Analytic density second derivatives (reference `gumbel_fdd`).
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x - v1;
        let e2 = e1 / v2;
        let e4 = (-e2).exp();
        let e5 = e4 - 1.0;
        let e7 = (-(e2 + e4)).exp();
        let e8 = v2 * v2 * v2;
        let e11 = e5 * e1 / v2 + 1.0;
        let e15 = (e2 - 1.0) * e4 + 1.0 - e11 * e5;
        let e17 = e4 - e5 * e5;
        let d11 = -(e7 * e17 / e8);
        let d12 = -(e15 * e7 / e8);
        let d21 = -((e17 * e1 / v2 - 2.0 * e5) * e7 / e8);
        let d22 = -((e15 * e1 / v2 - 2.0 * e11) * e7 / e8);
        vec![d11, d12, d21, d22]
    }

    /// Analytic CDF first derivatives (reference `gumbel_pd`).
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x - v1;
        let e3 = (-(e1 / v2)).exp();
        let e5 = e3 * (-e3).exp();
        vec![-(e5 / v2), -(e5 * e1 / (v2 * v2))]
    }

    /// Analytic CDF second derivatives (reference `gumbel_pdd`).
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x - v1;
        let e3 = (-(e1 / v2)).exp();
        let e5 = 1.0 - e3;
        let e6 = (-e3).exp();
        let e8 = e5 * e1 / v2;
        let e9 = v2 * v2;
        let e10 = -((e8 - 1.0) * e3 * e6 / e9);
        let d11 = -(e5 * e3 * e6 / e9);
        let d22 = -((e8 - 2.0) * e3 * e6 * e1 / (v2 * v2 * v2));
        vec![d11, e10, e10, d22]
    }

    /// Analytic mean derivatives (reference `gumbel_libs.py::gumbel_means`):
    /// `ml_mean = loc + scale·γ`, `∂/∂loc = 1`, `∂/∂scale = γ`, all second
    /// derivatives zero.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        const EULER: f64 = 0.577_215_664_901_532_9;
        let loc = params[0];
        let scale = params[1];
        (loc + scale * EULER, vec![1.0, EULER], vec![0.0; 4])
    }
}

// --- lnorm ------------------------------------------------------------------

impl DmgsModel for LogNormal {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        LogNormal::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        LogNormal::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        LogNormal::quantile(self, p, params[0], params[1])
    }

    /// Analytic density first derivatives (reference `lnorm_fd`).
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e2 = x.ln() - v1;
        let e3 = v2 * v2;
        let e4 = e2 * e2;
        let e5 = 2.0 * e3;
        let e8 = (-(e4 / e5)).exp();
        let e9 = (2.0 * std::f64::consts::PI).sqrt();
        vec![
            e8 * e2 / (v2 * v2 * v2 * x * e9),
            (4.0 * (e4 / (e5 * e5)) - 1.0 / e3) * e8 / (x * e9),
        ]
    }

    /// Analytic density second derivatives (reference `lnorm_fdd`).
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = v2 * v2;
        let e3 = x.ln() - v1;
        let e4 = 2.0 * e1;
        let e5 = e3 * e3;
        let e6 = e4 * e4;
        let e8 = (2.0 * std::f64::consts::PI).sqrt();
        let e10 = (-(e5 / e4)).exp();
        let e11 = v2 * v2 * v2;
        let e15 = 4.0 * (e5 / e6) - 1.0 / e1;
        let e16 = e1 * x;
        let e18 = e11 * x * e8;
        let e19 = x * e8;
        let d11 = (e5 / e1 - 1.0) * e10 / e18;
        let d12 = (e15 / e1 - 8.0 / e6) * e10 * e3 / e19;
        let d21 = (4.0 * (e5 / (e16 * e6 * e8)) - 3.0 * (e16 * e8 / (e18 * e18))) * e10 * e3;
        let d22 = (2.0 / e11 + v2 * (4.0 * e15 - 64.0 * (e1 / e6)) * e5 / e6) * e10 / e19;
        vec![d11, d12, d21, d22]
    }

    /// Analytic CDF first derivatives (reference `lnorm_pd`).
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e2 = x.ln() - v1;
        let e4 = standard_normal_pdf(e2 / v2);
        vec![-(e4 / v2), -(e4 * e2 / (v2 * v2))]
    }

    /// Analytic CDF second derivatives (reference `lnorm_pdd`).
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e2 = x.ln() - v1;
        let e3 = v2 * v2;
        let e5 = standard_normal_pdf(e2 / v2);
        let e7 = e2 * e2 / e3;
        let e8 = -((e7 - 1.0) * e5 / e3);
        let e9 = v2 * v2 * v2;
        let d11 = -(e5 * e2 / e9);
        let d22 = -((e7 - 2.0) * e5 * e2 / e9);
        vec![d11, e8, e8, d22]
    }

    /// Analytic mean derivatives (reference `lnorm.py`):
    /// `ml_mean = exp(mu + sigma²/2)`. The reference reports
    /// `rh_mean = "no analytic expression"`; we return the MLE mean and let the
    /// DMGS correction supply the CP mean (the reference's `cp_mean` is a
    /// placeholder string, so the numeric value here is the MLE mean plus the
    /// DMGS correction, which is the natural numeric analogue).
    ///
    /// With `m = exp(mu + sigma²/2)`: `∂/∂mu = m`, `∂/∂sigma = m·sigma`,
    /// `∂²/∂mu² = m`, `∂²/∂mu∂sigma = m·sigma`, `∂²/∂sigma² = m·(1+sigma²)`.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let mu = params[0];
        let sigma = params[1];
        let s2 = sigma * sigma;
        let m = (mu + 0.5 * s2).exp();
        let meand1 = vec![m, m * sigma];
        let mut meand2 = vec![0.0; 4];
        meand2[0] = m;
        meand2[1] = m * sigma;
        meand2[2] = m * sigma;
        meand2[3] = m * (1.0 + s2);
        (m, meand1, meand2)
    }
}

// --- weibull ----------------------------------------------------------------

impl DmgsModel for Weibull {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        Weibull::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        Weibull::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        Weibull::quantile(self, p, params[0], params[1])
    }

    /// Analytic density first derivatives (reference `weibull_fd`).
    ///
    /// Phase 5 (Issue 5): all powers of `e1 = x/v2` are derived from a single
    /// `ln_e1` via `exp(k · ln_e1)`, cutting the `powf` round-off.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x / v2;
        let ln_e1 = e1.ln();
        let e2 = v1 - 1.0;
        let e3 = (e2 * ln_e1).exp();
        let e5 = (-(v1 * ln_e1).exp()).exp();
        vec![
            (e3 + v1 * (e3 - ((2.0 * v1 - 1.0) * ln_e1).exp()) * (x.ln() - v2.ln())) * e5 / v2,
            v1 * e5
                * (x * (v1 * ((2.0 * e2) * ln_e1).exp() - e2 * ((v1 - 2.0) * ln_e1).exp()) / v2
                    - e3)
                / (v2 * v2),
        ]
    }

    /// Analytic density second derivatives (reference `weibull_fdd`).
    ///
    /// Phase 5 (Issue 5): all powers of `e1 = x/v2` are derived from a single
    /// `ln_e1` via `exp(k · ln_e1)`, cutting the `powf` round-off.
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x / v2;
        let ln_e1 = e1.ln();
        let e2 = v1 - 1.0;
        let e3 = (e2 * ln_e1).exp();
        let e6 = x.ln() - v2.ln();
        let e7 = v1 - 2.0;
        let e8 = (e7 * ln_e1).exp();
        let e9 = 2.0 * v1;
        let e10 = 2.0 * e2;
        let e11 = e9 - 1.0;
        let e12 = (e10 * ln_e1).exp();
        let e13 = (e11 * ln_e1).exp();
        let e14 = (v1 * ln_e1).exp();
        let e15 = e2 * e8;
        let e17 = (-e14).exp();
        let e18 = v1 * e12;
        let e19 = e3 - e13;
        let e23 = x * (e18 - e15) / v2 - e3;
        let e24 = e3 + v1 * e19 * e6;
        let e25 = v2 * v2;
        let d11 = (2.0 * e3 + v1 * (e3 - 2.0 * e13) * e6 - (e24 * e14 + e13)) * e17 * e6 / v2;
        let d12 = ((1.0 - v1 * e6 * e14) * e23
            + v1 * (x * ((2.0 * e18 - e15) * e6 + e12 - e8) / v2 - e6 * e3))
            * e17
            / e25;
        let d21 = e17
            * (v1
                * (e13
                    + x * ((e11 * ((e9 - 2.0) * ln_e1).exp() - e15) * e6 + e24 * e3) / v2
                    - (e19 * e6 + e3))
                - (e3 + x * e2 * e8 / v2))
            / e25;
        let d22 = v1
            * e17
            * (x * ((2.0 * e8
                + x * (e7 * ((v1 - 3.0) * ln_e1).exp()
                    - 2.0 * (v1 * ((e10 - 1.0) * ln_e1).exp()))
                    / v2)
                * e2
                + v1 * (e23 * e3 - e12))
                / v2
                - 2.0 * e23)
            / (v2 * v2 * v2);
        vec![d11, d12, d21, d22]
    }

    /// Analytic CDF first derivatives (reference `weibull_pd`).
    ///
    /// Phase 5 (Issue 5): powers of `e1 = x/v2` are derived from a single
    /// `ln_e1` via `exp(k · ln_e1)`, cutting the `powf` round-off.
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x / v2;
        let ln_e1 = e1.ln();
        let e2 = (v1 * ln_e1).exp();
        let e4 = (-e2).exp();
        vec![
            e4 * (x.ln() - v2.ln()) * e2,
            -(v1 * x * e4 * ((v1 - 1.0) * ln_e1).exp() / (v2 * v2)),
        ]
    }

    /// Analytic CDF second derivatives (reference `weibull_pdd`).
    ///
    /// Phase 5 (Issue 5): powers of `e1 = x/v2` are derived from a single
    /// `ln_e1` via `exp(k · ln_e1)`, cutting the `powf` round-off.
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let v2 = params[1];
        let e1 = x / v2;
        let ln_e1 = e1.ln();
        let e2 = (v1 * ln_e1).exp();
        let e3 = v1 - 1.0;
        let e4 = (e3 * ln_e1).exp();
        let e7 = x.ln() - v2.ln();
        let e9 = (-e2).exp();
        let e10 = v1 * x;
        let e11 = v1 * e7;
        let e14 = e10 * e7 * e4 / v2;
        let d11 = (e2 - ((2.0 * v1) * ln_e1).exp()) * e9 * e7 * e7;
        let d12 = -(x * ((1.0 - e11 * e2) * e4 + e11 * e4) * e9 / (v2 * v2));
        let d21 = ((e14 - 1.0) * e2 - e14) * e9 / v2;
        let d22 = -(e10
            * e9
            * (x * (v1 * ((2.0 * e3) * ln_e1).exp() - e3 * ((v1 - 2.0) * ln_e1).exp()) / v2
                - 2.0 * e4)
            / (v2 * v2 * v2));
        vec![d11, d12, d21, d22]
    }

    /// Analytic mean derivatives (reference `weibull_libs.py::weibull_means`).
    ///
    /// With `f1 = 1 + 1/v1` (v1 = shape, v2 = scale):
    /// `ml_mean = v2·Γ(f1)`,
    /// `∂/∂v1 = -v2·Γ(f1)·ψ(f1)/v1²`, `∂/∂v2 = Γ(f1)`,
    /// `∂²/∂v1² = v2·(Γ(f1)·ψ₁(f1)/v1² + 2·Γ(f1)·ψ(f1)/v1³)`,
    /// `∂²/∂v1∂v2 = -Γ(f1)·ψ(f1)/v1²`, `∂²/∂v2² = 0`.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        use crate::special::{digamma, gamma, trigamma};
        let v1 = params[0];
        let v2 = params[1];
        let f1 = 1.0 + 1.0 / v1;
        let iv12 = 1.0 / (v1 * v1);
        let iv13 = 1.0 / (v1 * v1 * v1);
        let g = gamma(f1);
        let dg = digamma(f1);
        let tg = trigamma(f1);
        let ml_mean = v2 * g;
        let meand1 = vec![-v2 * g * dg * iv12, g];
        let mut meand2 = vec![0.0; 4];
        meand2[0] = v2 * (g * tg * iv12 + 2.0 * g * dg * iv13);
        let d01 = -g * dg * iv12;
        meand2[1] = d01;
        meand2[2] = d01;
        meand2[3] = 0.0;
        (ml_mean, meand1, meand2)
    }
}

// --- gamma ------------------------------------------------------------------

impl DmgsModel for Gamma {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        Gamma::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        Gamma::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        Gamma::quantile(self, p, params[0], params[1])
    }
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        // The gamma is a NON-homogeneous model: the reference uses an ad-hoc
        // "type 1" prior (the default in `qgamma_cp`), whose log-slope is
        // `lambdad_cp = c(-1/shape, -1/scale)` in the reference's
        // (shape, scale) parameterization. In the Rust (shape, rate)
        // parameterization `scale = 1/rate`, so `-1/scale = -rate` and the
        // slope is `[-1/shape, -rate]`.
        //
        // NOTE: this differs from the right-Haar prior `c(0, -1/scale)` that
        // the homogeneous models use. The reference's own comment records that
        // the double prior ("type 1") "worked better" than the single
        // right-Haar prior ("type 2") for the gamma, so we follow the
        // reference default.
        vec![-1.0 / params[0], -params[1]]
    }

    /// Analytic density first derivatives (reference `gamma_fd`).
    ///
    /// The reference parameterizes by `(shape, scale)`; here `v2` is the
    /// **rate**, so the reference's `scale` is `1/rate`. The reference's
    /// `gamma_fd` is expressed in `(shape, scale)`; we transcribe it with
    /// `scale = 1/rate` and apply the chain rule for the rate component.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let rate = params[1];
        // Reference (shape, scale) form with scale = 1/rate.
        let scale = 1.0 / rate;
        let (d_shape, d_scale) = gamma_density_first_scale(x, v1, scale);
        // Chain rule: d/d(rate) = d/d(scale) * d(scale)/d(rate) = -1/rate^2.
        vec![d_shape, d_scale * (-1.0 / (rate * rate))]
    }

    /// Analytic density second derivatives (reference `gamma_fdd`).
    ///
    /// Transcribed in `(shape, scale)` with `scale = 1/rate`, then converted to
    /// the `(shape, rate)` parameterization by the chain rule. Because
    /// `scale = 1/rate` is nonlinear, the rate-rate entry carries a second-order
    /// term `∂f/∂scale · ∂²scale/∂rate² = d_scale · 2/rate³` in addition to
    /// `∂²f/∂scale² · (∂scale/∂rate)²`.
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let rate = params[1];
        let scale = 1.0 / rate;
        let (d_ss, d_sl, d_ll) = gamma_density_second_scale(x, v1, scale);
        // ∂f/∂scale in the (shape, scale) parameterization.
        let d_scale = gamma_density_first_scale(x, v1, scale).1;
        // Jacobian J = d(shape, scale)/d(shape, rate) = [[1, 0], [0, -1/rate^2]].
        let j = -1.0 / (rate * rate);
        // H_rate = J^T H_scale J + (∂f/∂scale) · Hess(scale).
        let h11 = d_ss;
        let h12 = d_sl * j;
        let h22 = d_ll * j * j + d_scale * (2.0 / (rate * rate * rate));
        vec![h11, h12, h12, h22]
    }

    /// Analytic mean derivatives (reference `gamma_libs.py::gamma_means`).
    ///
    /// The reference uses `(shape, scale)` with `ml_mean = shape·scale`; the
    /// core uses `(shape, rate)` with `scale = 1/rate`, so `ml_mean = shape/rate`.
    /// In the rate parameterization:
    /// `∂/∂shape = 1/rate`, `∂/∂rate = -shape/rate²`,
    /// `∂²/∂shape∂rate = -1/rate²`, `∂²/∂rate² = 2·shape/rate³`.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let shape = params[0];
        let rate = params[1];
        let r2 = rate * rate;
        let r3 = r2 * rate;
        let ml_mean = shape / rate;
        let meand1 = vec![1.0 / rate, -shape / r2];
        let mut meand2 = vec![0.0; 4];
        let d01 = -1.0 / r2;
        meand2[1] = d01;
        meand2[2] = d01;
        meand2[3] = 2.0 * shape / r3;
        (ml_mean, meand1, meand2)
    }
}

// --- gamma (reference (shape, scale) parameterization) ----------------------

impl DmgsModel for GammaScale {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        GammaScale::pdf(self, x, params[0], params[1])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        GammaScale::cdf(self, x, params[0], params[1])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        GammaScale::quantile(self, p, params[0], params[1])
    }
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        // The reference gamma uses the NON-homogeneous ad-hoc "type 1" prior
        // `lambdad_cp = c(-1/shape, -1/scale)` (the default in `qgamma_cp`).
        // In the `(shape, scale)` parameterization this is exactly
        // `[-1/shape, -1/scale]`.
        vec![-1.0 / params[0], -1.0 / params[1]]
    }

    /// Analytic density first derivatives (reference `gamma_fd`).
    ///
    /// Returns `[∂f/∂shape, ∂f/∂scale]` in the `(shape, scale)` parameterization.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let (d_shape, d_scale) = gamma_density_first_scale(x, params[0], params[1]);
        vec![d_shape, d_scale]
    }

    /// Analytic density second derivatives (reference `gamma_fdd`).
    ///
    /// Returns the row-major `2 × 2` `[∂²f/∂shape², ∂²f/∂shape∂scale;
    /// ∂²f/∂scale∂shape, ∂²f/∂scale²]`.
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let (d_ss, d_sl, d_ll) = gamma_density_second_scale(x, params[0], params[1]);
        vec![d_ss, d_sl, d_sl, d_ll]
    }

    /// Analytic mean derivatives (reference `gamma_libs.py::gamma_means`).
    ///
    /// In `(shape, scale)`: `ml_mean = shape·scale`, `∂/∂shape = scale`,
    /// `∂/∂scale = shape`, `∂²/∂shape∂scale = 1`, all other second derivatives 0.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let shape = params[0];
        let scale = params[1];
        let ml_mean = shape * scale;
        let meand1 = vec![scale, shape];
        let mut meand2 = vec![0.0; 4];
        meand2[1] = 1.0;
        meand2[2] = 1.0;
        (ml_mean, meand1, meand2)
    }
}

/// The reference `gamma_fd` in `(shape, scale)` parameterization.
///
/// Returns `(∂f/∂shape, ∂f/∂scale)`.
fn gamma_density_first_scale(x: f64, v1: f64, v2: f64) -> (f64, f64) {
    let e2 = v2.powf(v1);
    let e3 = x.powf(v1 - 1.0);
    let e5 = (-(x / v2)).exp();
    let e6 = crate::special::gamma(v1);
    let d_shape =
        e5 * (e3 * x.ln() / e2 - e3 * (crate::special::digamma(v1) / e2 + v2.ln() / e2)) / e6;
    let d_scale = e5 * (x.powf(v1) / v2.powf(2.0 + v1) - v1 * e3 / v2.powf(1.0 + v1)) / e6;
    (d_shape, d_scale)
}

/// The reference `gamma_fdd` in `(shape, scale)` parameterization.
///
/// Returns `(∂²f/∂shape², ∂²f/∂shape∂scale, ∂²f/∂scale²)`.
fn gamma_density_second_scale(x: f64, v1: f64, v2: f64) -> (f64, f64, f64) {
    let e1 = v1 - 1.0;
    let e2 = x.powf(e1);
    let e3 = v2.powf(v1);
    let e4 = 1.0 + v1;
    let e5 = v2.ln();
    let e6 = 2.0 + v1;
    let e7 = crate::special::digamma(v1);
    let e8 = x.ln();
    let e9 = v2.powf(e4);
    let e12 = e7 / e3 + e5 / e3;
    let e13 = (-(x / v2)).exp();
    let e14 = crate::special::gamma(v1);
    let e15 = x.powf(v1);
    let e17 = v1 * e2;
    let e18 = v2.powf(e6);
    let e19 = e2 * e12;
    let e20 = x.powf(v1 - 2.0);
    let e21 = 2.0 * e4;
    let e22 = 2.0 * e6;
    let e24 = v2 * v2;
    let e26 = e2 * e8 / e3;
    let e28 = e15 / e18 - e17 / e9;
    let d11 = e13
        * (e8 * (e26 - (e19 + e2 * e5 / e3))
            - (e7 * (e26 - e19) + e2 * (crate::special::trigamma(v1) / e3 - e12 * e5)))
        / e14;
    let d12 = e13
        * (e5 * (v1 * v2.powf(e4 - e21) * e2 - v2.powf(e6 - e22) * e15) + e15 * e8 / e18
            - ((e17 * e8 + e2) / e9 + e7 * e28))
        / e14;
    let d22 =
        x * (e28 / e24 + v1 * v2.powf(v1 - e21) * e20 * e4 - v2.powf(e4 - e22) * e2 * e6) * e13
            / e14;
    (d11, d12, d22)
}

// --- expon ------------------------------------------------------------------

impl DmgsModel for Exponential {
    fn pdf(&self, x: f64, params: &[f64]) -> f64 {
        Exponential::pdf(self, x, params[0])
    }
    fn cdf(&self, x: f64, params: &[f64]) -> f64 {
        Exponential::cdf(self, x, params[0])
    }
    fn quantile(&self, p: f64, params: &[f64]) -> f64 {
        Exponential::quantile(self, p, params[0])
    }
    fn log_prior_slope(&self, params: &[f64]) -> Vec<f64> {
        // Right-Haar prior pi(lambda) ∝ 1/lambda => d/dlambda ln(1/lambda)
        // = -1/lambda. The exponential has a single (rate) parameter.
        vec![-1.0 / params[0]]
    }

    /// Analytic density first derivative (reference `exp_fd`):
    /// `∂f/∂rate = (1 - rate x) exp(-rate x)`.
    fn density_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let e1 = v1 * x;
        vec![(1.0 - e1) * (-e1).exp()]
    }

    /// Analytic density second derivative (reference `exp_fdd`):
    /// `∂²f/∂rate² = -(x (2 - rate x) exp(-rate x))`.
    fn density_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        let e1 = v1 * x;
        vec![-(x * (2.0 - e1) * (-e1).exp())]
    }

    /// Analytic CDF first derivative (reference `exp_pd`):
    /// `∂F/∂rate = x exp(-rate x)`.
    fn cdf_first_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        vec![x * (-(v1 * x)).exp()]
    }

    /// Analytic CDF second derivative (reference `exp_pdd`):
    /// `∂²F/∂rate² = -(x² exp(-rate x))`.
    fn cdf_second_derivs(&self, params: &[f64], x: f64) -> Vec<f64> {
        let v1 = params[0];
        vec![-(x * x * (-(v1 * x)).exp())]
    }

    /// Analytic mean derivatives (reference `expon`): the exponential mean is
    /// the scale `1/rate`, so `ml_mean = 1/rate`, `∂/∂rate = -1/rate²`,
    /// `∂²/∂rate² = 2/rate³`.
    fn mean_derivs(&self, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let rate = params[0];
        let r2 = rate * rate;
        let r3 = r2 * rate;
        (1.0 / rate, vec![-1.0 / r2], vec![2.0 / r3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmgs::{dmgs_cp_pdf, dmgs_cp_quantiles, DmgsIntegrator, Integrator};

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

    fn general_sample() -> Vec<f64> {
        vec![
            9.13, 11.42, 8.77, 10.55, 12.31, 9.88, 7.94, 10.02, 11.77, 9.41, 10.93, 8.21, 12.05,
            10.66, 9.02, 11.18, 8.55, 10.34, 9.71, 11.90, 7.63, 10.81, 9.29, 12.44, 8.98, 10.17,
            11.53, 9.60, 10.72, 8.40,
        ]
    }

    fn positive_sample() -> Vec<f64> {
        vec![
            0.42, 1.87, 0.95, 3.21, 0.61, 2.44, 1.13, 0.78, 4.02, 1.55, 0.33, 2.91, 1.02, 0.69,
            3.77, 1.34, 0.51, 2.18, 0.88, 1.96, 0.27, 3.45, 1.21, 0.74, 2.63, 1.08, 0.46, 4.31,
            1.72, 0.93,
        ]
    }

    #[test]
    fn all_models_produce_finite_cp_pdf() {
        // Every Phase-1 model must produce a finite, non-negative CP density.
        let yg = general_sample();
        let yp = positive_sample();

        // (name, data, params, prediction points)
        type Case = (&'static str, Vec<f64>, Vec<f64>, Vec<f64>);
        let cases: Vec<Case> = vec![
            ("norm", yg.clone(), vec![10.0, 2.0], vec![9.0, 10.0, 11.0]),
            ("gumbel", yg.clone(), vec![10.0, 2.0], vec![9.0, 10.0, 11.0]),
            (
                "genextreme",
                yg.clone(),
                vec![10.0, 2.0, 0.0],
                vec![9.0, 10.0, 11.0],
            ),
            ("lnorm", yp.clone(), vec![0.0, 1.0], vec![0.5, 1.0, 2.0]),
            ("weibull", yp.clone(), vec![1.0, 1.0], vec![0.5, 1.0, 2.0]),
            ("gamma", yp.clone(), vec![1.0, 1.0], vec![0.5, 1.0, 2.0]),
            ("expon", yp.clone(), vec![1.0], vec![0.5, 1.0, 2.0]),
            ("genpareto", yp.clone(), vec![1.0, 0.0], vec![0.5, 1.0, 2.0]),
        ];

        for (name, y, params, x) in cases {
            let cp = match name {
                "norm" => dmgs_cp_pdf(Normal, &params, &y, &x),
                "gumbel" => dmgs_cp_pdf(Gumbel, &params, &y, &x),
                "genextreme" => dmgs_cp_pdf(GenExtreme, &params, &y, &x),
                "lnorm" => dmgs_cp_pdf(LogNormal, &params, &y, &x),
                "weibull" => dmgs_cp_pdf(Weibull, &params, &y, &x),
                "gamma" => dmgs_cp_pdf(Gamma, &params, &y, &x),
                "expon" => dmgs_cp_pdf(Exponential, &params, &y, &x),
                "genpareto" => dmgs_cp_pdf(GenPareto, &params, &y, &x),
                _ => unreachable!(),
            };
            assert_eq!(cp.len(), x.len(), "{name}: length");
            for (i, v) in cp.iter().enumerate() {
                assert!(v.is_finite(), "{name}: cp_pdf[{i}] = {v} not finite");
                assert!(*v >= 0.0, "{name}: cp_pdf[{i}] = {v} negative");
            }
        }
    }

    #[test]
    fn predictor_models_produce_finite_cp_pdf() {
        let yg = general_sample();
        let t: Vec<f64> = (0..yg.len()).map(|i| i as f64 / 10.0).collect();
        let t_mean = t.iter().sum::<f64>() / t.len() as f64;

        let norm_p1 = NormalP1::new(t.clone(), t_mean, 1.0);
        let cp = dmgs_cp_pdf(norm_p1, &[10.0, 0.0, 2.0], &yg, &[9.0, 10.0, 11.0]);
        for (i, v) in cp.iter().enumerate() {
            assert!(v.is_finite() && *v >= 0.0, "norm_p1 cp_pdf[{i}] = {v}");
        }

        let gev_p1 = GenExtremeP1::new(t, t_mean, 1.0);
        let cp = dmgs_cp_pdf(gev_p1, &[10.0, 0.0, 2.0, 0.0], &yg, &[9.0, 10.0, 11.0]);
        for (i, v) in cp.iter().enumerate() {
            assert!(
                v.is_finite() && *v >= 0.0,
                "genextreme_p1 cp_pdf[{i}] = {v}"
            );
        }
    }

    #[test]
    fn prior_slopes_match_reference_conventions() {
        // norm: [0, -1/sigma]
        let s = Normal.log_prior_slope(&[0.0, 2.0]);
        assert_close(s[0], 0.0, RTOL, ATOL, "norm slope[0]");
        assert_close(s[1], -0.5, RTOL, ATOL, "norm slope[1]");
        // gumbel: [0, -1/scale]
        let s = Gumbel.log_prior_slope(&[0.0, 4.0]);
        assert_close(s[1], -0.25, RTOL, ATOL, "gumbel slope[1]");
        // genpareto: [-1/scale, 0]
        let s = GenPareto.log_prior_slope(&[2.0, 0.0]);
        assert_close(s[0], -0.5, RTOL, ATOL, "gpd slope[0]");
        assert_close(s[1], 0.0, RTOL, ATOL, "gpd slope[1]");
        // expon: [-1/rate]
        let s = Exponential.log_prior_slope(&[2.0]);
        assert_close(s[0], -0.5, RTOL, ATOL, "expon slope[0]");
        // gamma: reference "type 1" prior c(-1/shape, -1/scale) in the
        // (shape, rate) parameterization => [-1/shape, -rate].
        let s = Gamma.log_prior_slope(&[2.0, 4.0]);
        assert_close(s[0], -0.5, RTOL, ATOL, "gamma slope[0]");
        assert_close(s[1], -4.0, RTOL, ATOL, "gamma slope[1]");
        // norm_p1: [0, 0, -1/sigma]
        let s = NormalP1::default().log_prior_slope(&[0.0, 0.0, 2.0]);
        assert_close(s[2], -0.5, RTOL, ATOL, "norm_p1 slope[2]");
        // genextreme_p1: [0, 0, -1/scale, 0]
        let s = GenExtremeP1::default().log_prior_slope(&[0.0, 0.0, 4.0, 0.0]);
        assert_close(s[2], -0.25, RTOL, ATOL, "gev_p1 slope[2]");
    }

    #[test]
    fn cp_quantiles_are_monotone_for_all_models() {
        // The DMGS correction is a first-order asymptotic term; at the MLE it is
        // well-behaved and the CP quantiles are monotone over the central
        // probability range. (In the far tail the first-order correction can
        // become non-monotone for poorly-fitting parameters — the reference
        // documents that DMGS degrades in the extreme tail — so we evaluate at
        // the fitted MLE, which is the realistic use case.)
        use crate::mle::fit;

        let yg = general_sample();
        let yp = positive_sample();
        let p = vec![0.1, 0.25, 0.5, 0.75, 0.9];

        let check = |name: &str, q: Vec<f64>| {
            assert_eq!(q.len(), p.len(), "{name}: length");
            for i in 1..q.len() {
                assert!(q[i] > q[i - 1], "{name}: cp_quantiles not monotone at {i}");
            }
        };

        let norm = fit(&Normal, &yg, &[10.0, 2.0]);
        check("norm", dmgs_cp_quantiles(Normal, &norm.ml_params, &yg, &p));

        let gumbel = fit(&Gumbel, &yg, &[10.0, 2.0]);
        check(
            "gumbel",
            dmgs_cp_quantiles(Gumbel, &gumbel.ml_params, &yg, &p),
        );

        let gev = fit(&GenExtreme, &yg, &[10.0, 2.0, 0.0]);
        check(
            "genextreme",
            dmgs_cp_quantiles(GenExtreme, &gev.ml_params, &yg, &p),
        );

        let lnorm = fit(&LogNormal, &yp, &[0.0, 1.0]);
        check(
            "lnorm",
            dmgs_cp_quantiles(LogNormal, &lnorm.ml_params, &yp, &p),
        );

        let weibull = fit(&Weibull, &yp, &[1.0, 1.0]);
        check(
            "weibull",
            dmgs_cp_quantiles(Weibull, &weibull.ml_params, &yp, &p),
        );

        let gamma = fit(&Gamma, &yp, &[1.0, 1.0]);
        check("gamma", dmgs_cp_quantiles(Gamma, &gamma.ml_params, &yp, &p));

        let expon = fit(&Exponential, &yp, &[1.0]);
        check(
            "expon",
            dmgs_cp_quantiles(Exponential, &expon.ml_params, &yp, &p),
        );

        let gpd = fit(&GenPareto, &yp, &[1.0, 0.0]);
        check(
            "genpareto",
            dmgs_cp_quantiles(GenPareto, &gpd.ml_params, &yp, &p),
        );
    }

    #[test]
    fn integrator_is_generic_over_models() {
        // The `Integrator` trait must be usable generically.
        fn integrate_any<M: DmgsModel + Sync>(
            m: M,
            params: &[f64],
            y: &[f64],
            x: &[f64],
        ) -> Vec<f64> {
            DmgsIntegrator::new(m).integrate(params, y, x)
        }
        let y = general_sample();
        let a = integrate_any(Normal, &[10.0, 2.0], &y, &[9.0, 10.0]);
        let b = dmgs_cp_pdf(Normal, &[10.0, 2.0], &y, &[9.0, 10.0]);
        assert_eq!(a, b);
    }
}
