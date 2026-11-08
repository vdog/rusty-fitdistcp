//! WAIC (Watanabe–Akaike Information Criterion) for the CP models.
//!
//! This is a faithful port of the reference `utils.py::make_waic`. The reference
//! computes two WAIC variants from the DMGS expansion of the predictive density:
//!
//! - `waic1` uses the **difference** penalty `pwaic1 = 2·(Σlog f1 − Σlog f2)`;
//! - `waic2` uses the **variance** penalty `pwaic2 = Σ(log f1² + dq/n − log f2²)`.
//!
//! Both share the same `lppd` (log pointwise predictive density) term
//! `Σ log(max(fhat + dq/n, ε))`.
//!
//! # Reference transcription
//!
//! The reference `make_waic(x, fhatx, lddi, lddd, f1f, lambdad, f2f, dim)`:
//!
//! ```text
//! nx = len(x)
//! lfhatx = log(fhatx)
//! fhatxsq = fhatx²
//! lfhatxof = lfhatx / fhatx
//! omlfhatx = 1 - lfhatx
//! outerf1f[j,k,i] = f1f[j,i] · f1f[k,i]
//!
//! # lppd
//! dq = dmgs(lddi, lddd, f1f, lambdad, f2f, dim) / nx
//! dqq = max(fhatx + dq, ε)
//! lppd = Σ log(dqq)
//!
//! # pwaic1 (difference)
//! t1 = f1f / fhatx
//! t2 = f2f / fhatx - outerf1f / fhatxsq
//! logf2 = lfhatx + dmgs(lddi, lddd, t1, lambdad, t2, dim) / nx
//! pwaic1 = 2·(Σ log(dqq) - Σ logf2)
//!
//! # pwaic2 (variance)
//! t1 = 2·lfhatxof · f1f
//! t2inner = f2f·lfhatx + outerf1f·omlfhatx / fhatx
//! t2 = 2·t2inner / fhatx
//! dq = dmgs(lddi, lddd, t1, lambdad, t2, dim)
//! pwaic2 = Σ(lfhatx² + dq/nx - logf2²)
//!
//! waic1 = [lppd, -pwaic1, lppd - pwaic1]
//! waic2 = [lppd, -pwaic2, lppd - pwaic2]
//! ```
//!
//! # Vectorization
//!
//! The reference `dmgs` is called with `mu1` a `dim × nx` matrix and `mu2` a
//! `dim × dim × nx` array, returning one value **per prediction point** (here,
//! per observation). We reuse [`crate::dmgs::DmgsIntegrator::dmgs_expansion`],
//! which has exactly that contract (row-major `dim × nx` / `dim × dim × nx`).

use nalgebra::DMatrix;

use crate::dmgs::{dmgs_scalar, DmgsIntegrator, DmgsModel};

/// The two WAIC variants, each a 3-vector `[lppd, -pwaic, lppd - pwaic]`.
///
/// Mirrors the reference `make_waic` return dict `{'waic1': ..., 'waic2': ...}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waic {
    /// WAIC variant 1 (difference penalty).
    pub waic1: [f64; 3],
    /// WAIC variant 2 (variance penalty).
    pub waic2: [f64; 3],
}

/// Compute the WAIC variants for a model (reference `utils.py::make_waic`).
///
/// # Arguments
///
/// - `model`: the fitted model (supplies the density derivatives).
/// - `params`: the fitted parameters.
/// - `x`: the data points at which the WAIC is evaluated (the reference `x`).
/// - `lddi`: the inverse normalized observed information `ldd⁻¹`.
/// - `lddd`: the normalized third-derivative tensor (row-major `p³`).
/// - `lambdad`: the log-prior slope `pidopi` (the reference `lambdad_rhp`).
///
/// The density `fhatx` and its first/second parameter derivatives `f1f`/`f2f`
/// are computed from the model's [`DmgsModel`] hooks, exactly as the reference
/// `*_waic` helpers do.
pub fn make_waic<M: DmgsModel + Copy + Sync>(
    model: &M,
    params: &[f64],
    x: &[f64],
    lddi: &DMatrix<f64>,
    lddd: &[f64],
    lambdad: &[f64],
) -> Waic {
    let dim = model.n_params();
    let nx = x.len();
    let nxf = nx as f64;
    let eps = f64::EPSILON;

    // fhatx, f1f (dim × nx), f2f (dim × dim × nx), row-major.
    let mut fhatx = vec![0.0; nx];
    let mut f1f = vec![0.0; dim * nx];
    let mut f2f = vec![0.0; dim * dim * nx];
    for (i, &xi) in x.iter().enumerate() {
        fhatx[i] = model.pdf(xi, params);
        let d1 = model.density_first_derivs(params, xi);
        let d2 = model.density_second_derivs(params, xi);
        for j in 0..dim {
            f1f[j * nx + i] = d1[j];
        }
        for jk in 0..dim * dim {
            f2f[jk * nx + i] = d2[jk];
        }
    }

    let lfhatx: Vec<f64> = fhatx.iter().map(|&f| f.ln()).collect();
    let fhatxsq: Vec<f64> = fhatx.iter().map(|&f| f * f).collect();
    let lfhatxof: Vec<f64> = (0..nx).map(|i| lfhatx[i] / fhatx[i]).collect();
    let omlfhatx: Vec<f64> = lfhatx.iter().map(|&l| 1.0 - l).collect();

    // outerf1f[j,k,i] = f1f[j,i] · f1f[k,i].
    let mut outerf1f = vec![0.0; dim * dim * nx];
    for i in 0..nx {
        for j in 0..dim {
            for k in 0..dim {
                outerf1f[(j * dim + k) * nx + i] = f1f[j * nx + i] * f1f[k * nx + i];
            }
        }
    }

    // --- lppd: dq = dmgs(lddi, lddd, f1f, lambdad, f2f, dim) / nx. ---
    let integ = DmgsIntegrator::new(*model);
    let dq_lppd = integ.dmgs_expansion(lddi, lddd, &f1f, lambdad, &f2f, dim, nx);
    let mut logf1_rhp = vec![0.0; nx];
    for i in 0..nx {
        let dqq = (fhatx[i] + dq_lppd[i] / nxf).max(eps);
        logf1_rhp[i] = dqq.ln();
    }
    let sumlogf1_rhp: f64 = logf1_rhp.iter().sum();
    let lppd_rhp = sumlogf1_rhp;

    // --- pwaic1 (difference). ---
    let mut t1 = vec![0.0; dim * nx];
    let mut t2 = vec![0.0; dim * dim * nx];
    for i in 0..nx {
        for j in 0..dim {
            t1[j * nx + i] = f1f[j * nx + i] / fhatx[i];
        }
        for jk in 0..dim * dim {
            t2[jk * nx + i] = f2f[jk * nx + i] / fhatx[i] - outerf1f[jk * nx + i] / fhatxsq[i];
        }
    }
    let dq2 = integ.dmgs_expansion(lddi, lddd, &t1, lambdad, &t2, dim, nx);
    let mut logf2_rhp = vec![0.0; nx];
    for i in 0..nx {
        logf2_rhp[i] = lfhatx[i] + dq2[i] / nxf;
    }
    let sumlogf2_rhp: f64 = logf2_rhp.iter().sum();
    let pwaic1_rhp = 2.0 * (sumlogf1_rhp - sumlogf2_rhp);

    // --- pwaic2 (variance). ---
    let mut t1v = vec![0.0; dim * nx];
    let mut t2v = vec![0.0; dim * dim * nx];
    for i in 0..nx {
        for j in 0..dim {
            t1v[j * nx + i] = 2.0 * lfhatxof[i] * f1f[j * nx + i];
        }
        for jk in 0..dim * dim {
            let t2inner = f2f[jk * nx + i] * lfhatx[i]
                + outerf1f[jk * nx + i] * omlfhatx[i] / fhatx[i];
            t2v[jk * nx + i] = 2.0 * t2inner / fhatx[i];
        }
    }
    let dq_v = integ.dmgs_expansion(lddi, lddd, &t1v, lambdad, &t2v, dim, nx);
    let mut sumvlogf = 0.0;
    for i in 0..nx {
        sumvlogf += lfhatx[i] * lfhatx[i] + dq_v[i] / nxf - logf2_rhp[i] * logf2_rhp[i];
    }
    let pwaic2_rhp = sumvlogf;

    let mut waic1 = [0.0; 3];
    waic1[0] = lppd_rhp;
    waic1[1] = -pwaic1_rhp;
    waic1[2] = waic1[0] + waic1[1];
    let mut waic2 = [0.0; 3];
    waic2[0] = lppd_rhp;
    waic2[1] = -pwaic2_rhp;
    waic2[2] = waic2[0] + waic2[1];

    Waic { waic1, waic2 }
}

/// The scalar DMGS correction for a scalar function of the parameters.
///
/// Re-exported for callers that need the low-level primitive (the reference
/// `dmgs` with a 1-column `mu1`).
pub fn waic_dmgs_scalar(
    lddi: &DMatrix<f64>,
    lddd: &[f64],
    g1: &[f64],
    pidopi: &[f64],
    g2: &[f64],
    p: usize,
) -> f64 {
    dmgs_scalar(lddi, lddd, g1, pidopi, g2, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmgs::DmgsIntegrator;
    use crate::models::Normal;

    fn sample() -> Vec<f64> {
        vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8]
    }

    #[test]
    fn waic_is_finite_for_normal() {
        let m = Normal;
        let y = sample();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let integ = DmgsIntegrator::new(m);
        let ldd = integ.normalized_ldd(&params, &y);
        let lddi = ldd.clone().try_inverse().unwrap();
        let lddd = integ.normalized_lddd(&params, &y);
        let lambdad = integ.log_prior_slope(&params);
        let w = make_waic(&m, &params, &y, &lddi, &lddd, &lambdad);
        for v in w.waic1.iter().chain(w.waic2.iter()) {
            assert!(v.is_finite(), "waic entry not finite: {v}");
        }
        // The third entry is the sum of the first two.
        assert!((w.waic1[2] - (w.waic1[0] + w.waic1[1])).abs() < 1e-12);
        assert!((w.waic2[2] - (w.waic2[0] + w.waic2[1])).abs() < 1e-12);
    }

    #[test]
    fn waic_lppd_close_to_loglik_for_large_n() {
        // For large n the DMGS correction is O(1/n), so lppd -> the in-sample
        // log-likelihood at the MLE.
        let m = Normal;
        let y: Vec<f64> = (0..500).map(|i| (i as f64 / 500.0) * 6.0 - 3.0).collect();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let integ = DmgsIntegrator::new(m);
        let ldd = integ.normalized_ldd(&params, &y);
        let lddi = ldd.clone().try_inverse().unwrap();
        let lddd = integ.normalized_lddd(&params, &y);
        let lambdad = integ.log_prior_slope(&params);
        let w = make_waic(&m, &params, &y, &lddi, &lddd, &lambdad);
        let loglik: f64 = y
            .iter()
            .map(|&yi| {
                let z = (yi - mean) / params[1];
                -0.5 * z * z - params[1].ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            })
            .sum();
        // The lppd is `Σ log(fhat + dq/n)`; each per-point correction is O(1/n)
        // but the sum over n points is O(1), so the total deviation from the
        // in-sample log-likelihood is bounded by a small constant (not O(1/n)).
        assert!(
            (w.waic1[0] - loglik).abs() < 1.0,
            "lppd {} vs loglik {}",
            w.waic1[0],
            loglik
        );
    }

    #[test]
    fn waic_matches_manual_reference_transcription() {
        // Reproduce the reference `make_waic` loop explicitly for the normal and
        // compare to the ported implementation.
        let m = Normal;
        let y = sample();
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n;
        let params = [mean, var.sqrt()];
        let integ = DmgsIntegrator::new(m);
        let ldd = integ.normalized_ldd(&params, &y);
        let lddi = ldd.clone().try_inverse().unwrap();
        let lddd = integ.normalized_lddd(&params, &y);
        let lambdad = integ.log_prior_slope(&params);
        let w = make_waic(&m, &params, &y, &lddi, &lddd, &lambdad);

        // Manual transcription.
        let dim = 2usize;
        let nx = y.len();
        let nxf = nx as f64;
        let eps = f64::EPSILON;
        let mut fhatx = vec![0.0; nx];
        let mut f1f = vec![0.0; dim * nx];
        let mut f2f = vec![0.0; dim * dim * nx];
        for (i, &xi) in y.iter().enumerate() {
            fhatx[i] = DmgsModel::pdf(&m, xi, &params);
            let d1 = m.density_first_derivs(&params, xi);
            let d2 = m.density_second_derivs(&params, xi);
            for j in 0..dim {
                f1f[j * nx + i] = d1[j];
            }
            for jk in 0..dim * dim {
                f2f[jk * nx + i] = d2[jk];
            }
        }
        let lfhatx: Vec<f64> = fhatx.iter().map(|&f| f.ln()).collect();
        let fhatxsq: Vec<f64> = fhatx.iter().map(|&f| f * f).collect();
        let lfhatxof: Vec<f64> = (0..nx).map(|i| lfhatx[i] / fhatx[i]).collect();
        let omlfhatx: Vec<f64> = lfhatx.iter().map(|&l| 1.0 - l).collect();
        let mut outerf1f = vec![0.0; dim * dim * nx];
        for i in 0..nx {
            for j in 0..dim {
                for k in 0..dim {
                    outerf1f[(j * dim + k) * nx + i] = f1f[j * nx + i] * f1f[k * nx + i];
                }
            }
        }
        let dq_lppd = integ.dmgs_expansion(&lddi, &lddd, &f1f, &lambdad, &f2f, dim, nx);
        let mut logf1 = vec![0.0; nx];
        for i in 0..nx {
            logf1[i] = (fhatx[i] + dq_lppd[i] / nxf).max(eps).ln();
        }
        let sumlogf1: f64 = logf1.iter().sum();
        let mut t1 = vec![0.0; dim * nx];
        let mut t2 = vec![0.0; dim * dim * nx];
        for i in 0..nx {
            for j in 0..dim {
                t1[j * nx + i] = f1f[j * nx + i] / fhatx[i];
            }
            for jk in 0..dim * dim {
                t2[jk * nx + i] = f2f[jk * nx + i] / fhatx[i] - outerf1f[jk * nx + i] / fhatxsq[i];
            }
        }
        let dq2 = integ.dmgs_expansion(&lddi, &lddd, &t1, &lambdad, &t2, dim, nx);
        let mut logf2 = vec![0.0; nx];
        for i in 0..nx {
            logf2[i] = lfhatx[i] + dq2[i] / nxf;
        }
        let sumlogf2: f64 = logf2.iter().sum();
        let pwaic1 = 2.0 * (sumlogf1 - sumlogf2);
        let mut t1v = vec![0.0; dim * nx];
        let mut t2v = vec![0.0; dim * dim * nx];
        for i in 0..nx {
            for j in 0..dim {
                t1v[j * nx + i] = 2.0 * lfhatxof[i] * f1f[j * nx + i];
            }
            for jk in 0..dim * dim {
                let inner =
                    f2f[jk * nx + i] * lfhatx[i] + outerf1f[jk * nx + i] * omlfhatx[i] / fhatx[i];
                t2v[jk * nx + i] = 2.0 * inner / fhatx[i];
            }
        }
        let dqv = integ.dmgs_expansion(&lddi, &lddd, &t1v, &lambdad, &t2v, dim, nx);
        let mut sumv = 0.0;
        for i in 0..nx {
            sumv += lfhatx[i] * lfhatx[i] + dqv[i] / nxf - logf2[i] * logf2[i];
        }
        let pwaic2 = sumv;

        assert!((w.waic1[0] - sumlogf1).abs() < 1e-10);
        assert!((w.waic1[1] - (-pwaic1)).abs() < 1e-10);
        assert!((w.waic2[1] - (-pwaic2)).abs() < 1e-10);
    }
}
