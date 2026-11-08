//! Leave-one-out out-of-sample log-scores (reference `*_logscores`).
//!
//! The reference `fitdistcp` computes, for each model, the MLE and CP
//! out-of-sample log-scores by **leave-one-out** cross-validation:
//!
//! ```text
//! for i in 0..nx:
//!     x1 = x without x[i]
//!     dd = <model>sub(x1, x[i])          # fit on x1, evaluate density at x[i]
//!     ml_oos_logscore += log(dd.ml_pdf)
//!     cp_oos_logscore += log(max(dd.cp_pdf, eps))
//! ```
//!
//! This is a faithful port. The MLE density is the fitted model's density at the
//! held-out point; the CP density is the DMGS CP predictive density at the
//! held-out point (the reference `rh_pdf` / `cp_pdf`).
//!
//! # Reference agreement
//!
//! The reference `norm_logscores` uses the **analytic** normal CP density
//! (`dnormsub`), not the DMGS expansion. For the normal we therefore use the
//! analytic Student-t CP density (matching `norm_libs.py::dnormsub`); for the
//! DMGS models we use [`crate::dmgs::DmgsIntegrator::cp_pdf`], which is the
//! reference `rh_pdf`.

use crate::dmgs::{DmgsIntegrator, DmgsModel};
use crate::mle::fit;

/// The MLE and CP leave-one-out out-of-sample log-scores.
///
/// Mirrors the reference `*_logscores` return dict
/// `{'ml_oos_logscore': ..., 'cp_oos_logscore': ...}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogScores {
    /// MLE out-of-sample log-score (`ml_oos_logscore`).
    pub ml_oos_logscore: f64,
    /// CP out-of-sample log-score (`cp_oos_logscore`).
    pub cp_oos_logscore: f64,
}

/// Compute the leave-one-out MLE and CP log-scores for a DMGS model.
///
/// For each held-out observation `x[i]` the model is refit on the remaining
/// `n - 1` points (starting from `init`), and the MLE density and DMGS CP
/// density are evaluated at `x[i]`. The scores are the sums of their logs.
///
/// `init` is the initial parameter guess for each refit (length
/// `model.n_params()`).
///
/// # Panics
///
/// Panics if `init.len() != model.n_params()` or if `x` is empty.
pub fn logscores<M: DmgsModel + Copy + Sync>(model: M, x: &[f64], init: &[f64]) -> LogScores {
    let nx = x.len();
    assert!(nx > 0, "logscores: x must be non-empty");
    let eps = f64::EPSILON;
    let integ = DmgsIntegrator::new(model);
    let mut ml_oos = 0.0;
    let mut cp_oos = 0.0;
    for i in 0..nx {
        // x1 = x without x[i].
        let mut x1: Vec<f64> = Vec::with_capacity(nx - 1);
        x1.extend_from_slice(&x[..i]);
        x1.extend_from_slice(&x[i + 1..]);
        if x1.is_empty() {
            continue;
        }
        let mle = fit(&model, &x1, init);
        let ml_pdf = model.pdf(x[i], &mle.ml_params);
        ml_oos += ml_pdf.max(eps).ln();
        let cp_pdf = integ.cp_pdf(&mle.ml_params, &x1, &[x[i]])[0];
        cp_oos += cp_pdf.max(eps).ln();
    }
    LogScores {
        ml_oos_logscore: ml_oos,
        cp_oos_logscore: cp_oos,
    }
}

/// Compute the leave-one-out MLE and CP log-scores for the **normal** model
/// using the reference's **analytic** CP density (reference
/// `norm_libs.py::norm_logscores` / `dnormsub`).
///
/// The reference normal CP density is the Student-t predictive
/// `t((y - mu)/sg1, df = n - 1) / sg1` with
/// `sg1 = sigma_ml·√(n/(n-1))·√((n+1)/n)`, where `sigma_ml` is the MLE sigma
/// (`√((n-1)/n)·sd(x)`). This is the analytic path the reference uses for the
/// normal, distinct from the DMGS expansion.
///
/// # Panics
///
/// Panics if `x` is empty.
pub fn norm_logscores(x: &[f64]) -> LogScores {
    use crate::special::student_t_pdf;

    let nx = x.len();
    assert!(nx > 0, "norm_logscores: x must be non-empty");
    let eps = f64::EPSILON;
    let mut ml_oos = 0.0;
    let mut cp_oos = 0.0;
    for i in 0..nx {
        let mut x1: Vec<f64> = Vec::with_capacity(nx - 1);
        x1.extend_from_slice(&x[..i]);
        x1.extend_from_slice(&x[i + 1..]);
        let n1 = x1.len();
        if n1 < 2 {
            continue;
        }
        let n1f = n1 as f64;
        // MLE params (reference norm_ml_params): mean and sqrt((n-1)/n)*sd(ddof=1).
        let mean = x1.iter().sum::<f64>() / n1f;
        let var1 = x1.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n1f - 1.0);
        let sd1 = var1.sqrt();
        let sigma_ml = ((n1f - 1.0) / n1f).sqrt() * sd1;
        // MLE density at the held-out point.
        let z = (x[i] - mean) / sigma_ml;
        let ml_pdf = (-0.5 * z * z).exp() / (sigma_ml * (2.0 * std::f64::consts::PI).sqrt());
        ml_oos += ml_pdf.max(eps).ln();
        // Analytic CP density (reference dnormsub rh_pdf).
        let sgu = sigma_ml * (n1f / (n1f - 1.0)).sqrt();
        let sg1 = sgu * ((n1f + 1.0) / n1f).sqrt();
        let yd = (x[i] - mean) / sg1;
        let rh_pdf = (student_t_pdf(yd, n1f - 1.0) / sg1).max(0.0);
        cp_oos += rh_pdf.max(eps).ln();
    }
    LogScores {
        ml_oos_logscore: ml_oos,
        cp_oos_logscore: cp_oos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Gumbel, Normal};

    fn sample() -> Vec<f64> {
        vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8]
    }

    #[test]
    fn norm_logscores_are_finite() {
        let x = sample();
        let ls = norm_logscores(&x);
        assert!(ls.ml_oos_logscore.is_finite());
        assert!(ls.cp_oos_logscore.is_finite());
    }

    #[test]
    fn norm_logscores_match_manual_leave_one_out() {
        // Reproduce the reference loop explicitly for the normal.
        use crate::special::student_t_pdf;
        let x = sample();
        let nx = x.len();
        let mut ml = 0.0;
        let mut cp = 0.0;
        for i in 0..nx {
            let mut x1: Vec<f64> = Vec::new();
            x1.extend_from_slice(&x[..i]);
            x1.extend_from_slice(&x[i + 1..]);
            let n1 = x1.len() as f64;
            let mean = x1.iter().sum::<f64>() / n1;
            let var1 = x1.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n1 - 1.0);
            let sd1 = var1.sqrt();
            let sigma_ml = ((n1 - 1.0) / n1).sqrt() * sd1;
            let z = (x[i] - mean) / sigma_ml;
            let ml_pdf = (-0.5 * z * z).exp() / (sigma_ml * (2.0 * std::f64::consts::PI).sqrt());
            ml += ml_pdf.ln();
            let sgu = sigma_ml * (n1 / (n1 - 1.0)).sqrt();
            let sg1 = sgu * ((n1 + 1.0) / n1).sqrt();
            let yd = (x[i] - mean) / sg1;
            let rh_pdf = (student_t_pdf(yd, n1 - 1.0) / sg1).max(0.0);
            cp += rh_pdf.ln();
        }
        let ls = norm_logscores(&x);
        assert!((ls.ml_oos_logscore - ml).abs() < 1e-10);
        assert!((ls.cp_oos_logscore - cp).abs() < 1e-10);
    }

    #[test]
    fn dmgs_logscores_are_finite_for_gumbel() {
        let x = sample();
        let ls = logscores(Gumbel, &x, &[0.0, 1.0]);
        assert!(ls.ml_oos_logscore.is_finite(), "ml {}", ls.ml_oos_logscore);
        assert!(ls.cp_oos_logscore.is_finite(), "cp {}", ls.cp_oos_logscore);
    }

    #[test]
    fn dmgs_logscores_ml_matches_manual_for_normal() {
        // The generic DMGS path's MLE score must match the manual MLE
        // leave-one-out for the normal (the CP differs: DMGS vs analytic).
        let x = sample();
        let ls = logscores(Normal, &x, &[0.0, 1.0]);
        let ls_norm = norm_logscores(&x);
        assert!(
            (ls.ml_oos_logscore - ls_norm.ml_oos_logscore).abs() < 1e-6,
            "ml {} vs {}",
            ls.ml_oos_logscore,
            ls_norm.ml_oos_logscore
        );
    }
}
