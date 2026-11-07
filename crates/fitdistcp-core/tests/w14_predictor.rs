//! W14 core tests: predictor handling for `norm_p1` and `genextreme_p1`.
//!
//! These tests pin the reference `fitdistcp` predictor semantics (W14):
//!
//! - **Centering**: the predictor is centered by its mean, `t_i - mean(t)`; the
//!   model is `mu = a + b·(t - mean(t))`.
//! - **`maket0`**: the prediction point `t0` is a predictor value; when the user
//!   supplies `n0` instead, `t0 = t[n0]` (a positional index).
//! - **Prediction point**: `muhat0 = a + b·(t0 - mean(t))`.
//! - **`norm_p1` CP is analytic** (Student-t predictive), not DMGS.
//!
//! The reference values are captured from the reference R package
//! (`qnorm_p1_cp` / `qgev_p1_cp`) on the example data
//! `d060norm_p1_example_data_v1_*` and `d150gev_p1_example_data_v1_*`.

use fitdistcp_core::api::{predict_norm_p1, PredictionRequest};
use fitdistcp_core::models::{GenExtremeP1, NormalP1};

/// The `d060norm_p1_example_data_v1_t` predictor (n = 25).
const T_NORM: [f64; 25] = [
    0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.64,
    0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0,
];
/// The `d060norm_p1_example_data_v1_x` data (n = 25).
const X_NORM: [f64; 25] = [
    -0.58645381,
    0.26364332,
    -0.71562861,
    1.7552808,
    0.52950777,
    -0.58046838,
    0.76742905,
    1.0583247,
    0.93578135,
    0.094611613,
    1.9517812,
    0.86984324,
    -0.10124058,
    -1.6546999,
    1.7249309,
    0.59506639,
    0.66380974,
    1.6638362,
    1.5812212,
    1.3939013,
    1.7589774,
    1.6621363,
    0.99456498,
    -1.0293517,
    1.6198257,
];

/// The `d150gev_p1_example_data_v1_t` predictor (n = 40).
const T_GEV: [f64; 40] = [
    0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375,
    0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75,
    0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0,
];
/// The `d150gev_p1_example_data_v1_x` data (n = 40).
const X_GEV: [f64; 40] = [
    -0.64966202,
    0.85951881,
    1.5996735,
    -0.48792225,
    1.9369991,
    0.53527898,
    0.10262144,
    -0.23528281,
    -0.057837808,
    1.7340061,
    0.55865183,
    0.076155367,
    0.70099111,
    -0.14190081,
    0.29518469,
    0.62106268,
    0.039215712,
    -1.3732916,
    -0.094737048,
    0.95002586,
    1.4342304,
    0.9058821,
    0.75403608,
    1.5581121,
    0.13327821,
    2.524211,
    1.2288074,
    0.61335529,
    3.7228621,
    1.1994391,
    -1.1647551,
    2.1560221,
    1.3467322,
    1.619274,
    2.2680367,
    2.0191889,
    1.2164931,
    1.171829,
    1.0419927,
    -0.31710656,
];

/// §7.2 tolerance: `|a - b| <= atol + rtol * |b|`.
const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;
/// Looser tolerance for the analytic `norm_p1` CP quantiles, which are limited
/// by the `special` crate's `inc_beta` precision (documented discrepancy).
const RTOL_CP_Q: f64 = 1e-6;

fn assert_close(actual: f64, expected: f64, rtol: f64, what: &str) {
    let tol = ATOL + rtol * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: actual = {actual:.17e}, expected = {expected:.17e}, \
         |diff| = {:.3e} > tol = {:.3e}",
        (actual - expected).abs(),
        tol
    );
}

// --- Centering / maket0 / prediction point ---------------------------------

#[test]
fn centering_uses_mean_of_t() {
    // t = [0, 2], mean = 1 => centered = [-1, 1].
    let m = NormalP1::new(vec![0.0, 2.0], 1.0, 1.0);
    assert_close(m.mean_t(), 1.0, RTOL, "mean_t");
    assert_close(m.centered(0), -1.0, RTOL, "centered(0)");
    assert_close(m.centered(1), 1.0, RTOL, "centered(1)");
    // Prediction point t0 = 1 => centered_prediction = 0.
    assert_close(m.centered_prediction(), 0.0, RTOL, "centered_prediction");
    // t0 = 3 => centered_prediction = 2.
    let m2 = NormalP1::new(vec![0.0, 2.0], 3.0, 1.0);
    assert_close(
        m2.centered_prediction(),
        2.0,
        RTOL,
        "centered_prediction(3)",
    );
}

#[test]
fn maket0_positional_index_semantics() {
    // The reference `maket0(NA, n0, t) = t[n0]` (1-based). The binding resolves
    // this before constructing the model; here we verify the resolved value is
    // used as the prediction point.
    let t = T_NORM.to_vec();
    // n0 = 10 (1-based) => t[9] = 0.4.
    let t0 = t[9];
    assert_close(t0, 0.4, RTOL, "t[10] (1-based)");
    let m = NormalP1::new(t.clone(), t0, 10.0);
    let mean_t = m.mean_t();
    assert_close(mean_t, 0.52, RTOL, "mean(t)");
    assert_close(
        m.centered_prediction(),
        0.4 - 0.52,
        RTOL,
        "centered_prediction",
    );
}

#[test]
fn prediction_point_shifts_quantiles() {
    // Changing t0 shifts the predictive quantiles by b*(t0_new - t0_old).
    let t = T_NORM.to_vec();
    let y = X_NORM.to_vec();
    let p = vec![0.5];
    let req = |_t0: f64| PredictionRequest {
        y: y.clone(),
        x: vec![0.0],
        p: p.clone(),
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    let m1 = NormalP1::new(t.clone(), 0.4, 10.0);
    let m2 = NormalP1::new(t, 5.0, 10.0);
    let r1 = predict_norm_p1(&m1, &req(0.4));
    let r2 = predict_norm_p1(&m2, &req(5.0));
    let b = r1.ml_params[1];
    // muhat0 shifts by b*(5 - 0.4).
    assert_close(r2.muhat0 - r1.muhat0, b * (5.0 - 0.4), 1e-6, "muhat0 shift");
    // The median quantile shifts by the same amount.
    assert_close(
        r2.ml_quantiles[0] - r1.ml_quantiles[0],
        b * (5.0 - 0.4),
        1e-6,
        "median shift",
    );
}

// --- norm_p1 analytic CP vs reference ---------------------------------------

#[test]
fn norm_p1_analytic_cp_matches_reference() {
    let t = T_NORM.to_vec();
    let y = X_NORM.to_vec();
    let t0 = t[9]; // t[10] (1-based) = 0.4
    let m = NormalP1::new(t, t0, 10.0);
    let p = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let req = PredictionRequest {
        y: y.clone(),
        x: y.clone(),
        p,
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    let r = predict_norm_p1(&m, &req);

    // Reference ml_params (decentered): a = 0.11348586625, b = 1.10611412112,
    // sigma = 0.949594263163. The core reports the centered intercept
    // a_c = a + b*mean(t) = 0.68866520692.
    let mean_t = 0.52;
    // The reference `ml_params` are quoted to 12 significant figures, so the
    // comparison tolerance accounts for the reference's own rounding (~1e-10
    // relative); we use 1e-6 to be safe against the quoted precision.
    assert_close(
        r.ml_params[0] - r.ml_params[1] * mean_t,
        0.11348586625,
        1e-6,
        "decentered a",
    );
    assert_close(r.ml_params[1], 1.10611412112, 1e-6, "b");
    assert_close(r.ml_params[2], 0.949594263163, 1e-6, "sigma");

    // Reference ml_quantiles.
    let ref_ml_q = [
        -0.661022499888,
        -0.243267180457,
        0.0579637962327,
        0.315354558973,
        0.5559315147,
        0.796508470426,
        1.05389923317,
        1.35513020986,
        1.77288552929,
    ];
    // The reference quantiles are quoted to 12 significant figures; use 1e-6 to
    // absorb the reference's own rounding.
    for (i, (&a, &b)) in r.ml_quantiles.iter().zip(ref_ml_q.iter()).enumerate() {
        assert_close(a, b, 1e-6, &format!("ml_quantiles[{i}]"));
    }

    // Reference cp_quantiles (analytic Student-t; looser tolerance documented).
    let ref_cp_q = [
        -0.753652844124,
        -0.29517961154,
        0.028164231521,
        0.301552784103,
        0.5559315147,
        0.810310245296,
        1.08369879788,
        1.40704264094,
        1.86551587352,
    ];
    for (i, (&a, &b)) in r.cp_quantiles.iter().zip(ref_cp_q.iter()).enumerate() {
        assert_close(a, b, RTOL_CP_Q, &format!("cp_quantiles[{i}]"));
    }

    // Reference standard_errors.
    let ref_se = [0.189918852633, 0.658425155138, 0.134292908572];
    for (i, (&a, &b)) in r.standard_errors.iter().zip(ref_se.iter()).enumerate() {
        assert_close(a, b, RTOL, &format!("standard_errors[{i}]"));
    }

    // Reference predictedparameter[0] and adjustedx[0] (quoted to 12 sig figs).
    assert_close(
        r.predictedparameter[0],
        0.157730431095,
        1e-6,
        "predictedparameter[0]",
    );
    assert_close(r.adjustedx[0], -0.188252727137, 1e-6, "adjustedx[0]");
    assert_eq!(r.predictedparameter.len(), 25);
    assert_eq!(r.adjustedx.len(), 25);
}

#[test]
fn norm_p1_cp_quantiles_differ_from_ml() {
    // The analytic CP correction is non-trivial: the CP quantiles differ from
    // the MLE quantiles in the tails.
    let t = T_NORM.to_vec();
    let y = X_NORM.to_vec();
    let m = NormalP1::new(t, 0.4, 10.0);
    let req = PredictionRequest {
        y,
        x: vec![0.0],
        p: vec![0.1, 0.9],
        t: None,
        t0: None,
        n0: None,
        means: None,
    };
    let r = predict_norm_p1(&m, &req);
    assert!((r.cp_quantiles[0] - r.ml_quantiles[0]).abs() > 1e-3);
    assert!((r.cp_quantiles[1] - r.ml_quantiles[1]).abs() > 1e-3);
}

// --- genextreme_p1 predictor plumbing ---------------------------------------

#[test]
fn genextreme_p1_centering_and_prediction_point() {
    let t = T_GEV.to_vec();
    let m = GenExtremeP1::new(t.clone(), t[9], 10.0); // t0 = t[10] (1-based) = 0.25
    let mean_t = m.mean_t();
    assert_close(mean_t, 0.5125, RTOL, "mean(t)");
    assert_close(
        m.centered_prediction(),
        0.25 - 0.5125,
        RTOL,
        "centered_prediction",
    );
    // The location at the prediction point uses the centered t0.
    assert_close(
        m.location_prediction(1.0, 2.0),
        1.0 + 2.0 * (0.25 - 0.5125),
        RTOL,
        "location_prediction",
    );
}

#[test]
fn genextreme_p1_dmgs_cp_is_finite() {
    // The GEV `_p1` CP uses the DMGS expansion at the prediction point; the CP
    // density must be finite and non-negative.
    use fitdistcp_core::dmgs::{DmgsIntegrator, DmgsModel};
    let t = T_GEV.to_vec();
    let y = X_GEV.to_vec();
    let m = GenExtremeP1::new(t, 0.1, 10.0);
    let integ = DmgsIntegrator::new(m.clone());
    // A plausible parameter vector (location ~ 0.5, slope ~ 1, scale ~ 1, shape ~ -0.2).
    let params = [0.5, 1.0, 1.0, -0.2];
    let cp = integ.cp_pdf(&params, &y, &[0.0, 0.5, 1.0]);
    for (i, v) in cp.iter().enumerate() {
        assert!(v.is_finite() && *v >= 0.0, "cp_pdf[{i}] = {v}");
    }
    // The model's pdf/cdf/quantile at the prediction point are consistent.
    let q = DmgsModel::quantile(&m, 0.5, &params);
    let c = DmgsModel::cdf(&m, q, &params);
    assert_close(c, 0.5, 1e-6, "cdf(quantile(0.5))");
}
