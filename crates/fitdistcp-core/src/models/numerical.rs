//! Generic finite-difference derivative layer (§4.1).
//!
//! This module provides the **numerical fallback** half of the analytic-first /
//! numerical-fallback design. Given nothing but a log-likelihood function
//! `ℓ(θ; y)`, it computes the score, the observed information (Hessian), the
//! third-derivative tensor, and the model-specific mixed derivatives required by
//! the DMGS Haar-prior term.
//!
//! # Why central differences
//!
//! All derivatives use **central** finite differences, which are second-order
//! accurate (`O(h²)`) and avoid the `O(h)` bias of forward differences. The
//! default step is chosen as `h = EPS^(1/3) * max(1, |θ_i|)`, the standard
//! balance between truncation error (`O(h²)`) and round-off error (`O(ε/h)`) for
//! a first derivative. For higher-order stencils the same step is reused; the
//! resulting accuracy is documented per function and validated in the tests
//! against a known analytic Gaussian case.
//!
//! # Accuracy
//!
//! Central differences are *not* exact. The tests in this module compare the
//! fallback against the closed-form Gaussian derivatives and use a documented
//! looser tolerance than the §7.2 golden policy (`rtol = 1e-8`, `atol = 1e-10`),
//! because finite differences carry an irreducible truncation/round-off error.
//! See [`crate::models::numerical::tests`] for the exact tolerances used.

/// Default relative step size for a **first** derivative (central difference).
///
/// `EPS^(1/3)` is the classic optimum for a central first difference: it
/// balances the `O(h²)` truncation error against the `O(ε/h)` round-off error.
/// For `f64`, `ε ≈ 2.22e-16`, so `h ≈ 6.06e-6`.
pub const DEFAULT_REL_STEP: f64 = 6.055_454_5e-6;

/// Default relative step size for a **second** derivative (central difference).
///
/// A central second difference has `O(h²)` truncation but `O(ε/h²)` round-off,
/// so the optimal step is `ε^(1/4) ≈ 1.22e-4` — noticeably larger than the
/// first-derivative step. Using the first-derivative step here would let
/// round-off dominate (observed error `~1e-5`).
pub const DEFAULT_REL_STEP_2: f64 = 1.220_703_1e-4;

/// Default relative step size for a **third** derivative (central difference).
///
/// A central third difference has `O(h²)` truncation but `O(ε/h³)` round-off,
/// so the optimal step is `ε^(1/5) ≈ 7.4e-4`. Using a smaller step here makes
/// round-off catastrophic (observed error `~1e1`).
pub const DEFAULT_REL_STEP_3: f64 = 7.4e-4;

/// Absolute floor on the finite-difference step, so that parameters near zero
/// still receive a meaningful perturbation.
pub const DEFAULT_ABS_STEP: f64 = 1e-6;

/// Configuration for the finite-difference stencils.
///
/// The step for parameter `i` at derivative order `k` is
/// `rel_step_k * max(1, |θ_i|) + abs_step`, which keeps the perturbation
/// scale-appropriate for both small and large parameters. Separate relative
/// steps are used per derivative order because the round-off/truncation balance
/// depends on the order (see [`DEFAULT_REL_STEP`], [`DEFAULT_REL_STEP_2`],
/// [`DEFAULT_REL_STEP_3`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDiffConfig {
    /// Relative step multiplier for first derivatives (default
    /// [`DEFAULT_REL_STEP`]).
    pub rel_step: f64,
    /// Relative step multiplier for second derivatives (default
    /// [`DEFAULT_REL_STEP_2`]).
    pub rel_step_2: f64,
    /// Relative step multiplier for third derivatives (default
    /// [`DEFAULT_REL_STEP_3`]).
    pub rel_step_3: f64,
    /// Absolute step floor (default [`DEFAULT_ABS_STEP`]).
    pub abs_step: f64,
}

impl Default for FiniteDiffConfig {
    fn default() -> Self {
        Self {
            rel_step: DEFAULT_REL_STEP,
            rel_step_2: DEFAULT_REL_STEP_2,
            rel_step_3: DEFAULT_REL_STEP_3,
            abs_step: DEFAULT_ABS_STEP,
        }
    }
}

impl FiniteDiffConfig {
    /// Create a configuration with explicit step parameters.
    ///
    /// `rel_step`, `rel_step_2`, and `rel_step_3` are the relative multipliers
    /// for first, second, and third derivatives respectively; `abs_step` is the
    /// absolute floor shared by all orders.
    pub fn new(rel_step: f64, rel_step_2: f64, rel_step_3: f64, abs_step: f64) -> Self {
        Self {
            rel_step,
            rel_step_2,
            rel_step_3,
            abs_step,
        }
    }

    /// The perturbation `h_i` used for a first derivative w.r.t. parameter `i`.
    #[inline]
    pub fn step_for(&self, theta_i: f64) -> f64 {
        self.rel_step * theta_i.abs().max(1.0) + self.abs_step
    }

    /// The perturbation `h_i` used for a second derivative w.r.t. parameter `i`.
    #[inline]
    pub fn step_for_2(&self, theta_i: f64) -> f64 {
        self.rel_step_2 * theta_i.abs().max(1.0) + self.abs_step
    }

    /// The perturbation `h_i` used for a third derivative w.r.t. parameter `i`.
    #[inline]
    pub fn step_for_3(&self, theta_i: f64) -> f64 {
        self.rel_step_3 * theta_i.abs().max(1.0) + self.abs_step
    }
}

/// Evaluate `f` at `params` with parameter `i` shifted by `delta`.
#[inline]
fn shifted<F>(f: &F, params: &[f64], i: usize, delta: f64) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut p = params.to_vec();
    p[i] += delta;
    f(&p)
}

/// Evaluate `f` at `params` with parameters `i` and `j` shifted by `di` and `dj`.
#[inline]
fn shifted2<F>(f: &F, params: &[f64], i: usize, di: f64, j: usize, dj: f64) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut p = params.to_vec();
    p[i] += di;
    p[j] += dj;
    f(&p)
}

/// Evaluate `f` at `params` with each `(index, delta)` in `shifts` applied.
///
/// Used by the third-derivative stencils, which perturb up to three parameters
/// at once.
#[inline]
fn shifted_many<F>(f: &F, params: &[f64], shifts: &[(usize, f64)]) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut p = params.to_vec();
    for &(idx, delta) in shifts {
        p[idx] += delta;
    }
    f(&p)
}

/// Numerical score (gradient of the log-likelihood) via central differences.
///
/// For each parameter `i`:
///
/// ```text
/// ∂ℓ/∂θ_i ≈ [ℓ(θ + h_i e_i) - ℓ(θ - h_i e_i)] / (2 h_i)
/// ```
///
/// Returns one value per parameter. The error is `O(h²)` in truncation plus
/// `O(ε/h)` round-off.
pub fn numerical_score<F>(f: &F, params: &[f64], cfg: &FiniteDiffConfig) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = params.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        let h = cfg.step_for(params[i]);
        let fp = shifted(f, params, i, h);
        let fm = shifted(f, params, i, -h);
        out[i] = (fp - fm) / (2.0 * h);
    }
    out
}

/// Numerical Hessian (observed information) via central second differences.
///
/// Diagonal entries use the standard three-point second difference:
///
/// ```text
/// ∂²ℓ/∂θ_i² ≈ [ℓ(θ + h e_i) - 2 ℓ(θ) + ℓ(θ - h e_i)] / h²
/// ```
///
/// Off-diagonal entries use the four-point central mixed difference:
///
/// ```text
/// ∂²ℓ/∂θ_i∂θ_j ≈ [ℓ(+h_i,+h_j) - ℓ(+h_i,-h_j) - ℓ(-h_i,+h_j) + ℓ(-h_i,-h_j)] / (4 h_i h_j)
/// ```
///
/// Returns a row-major `n_params × n_params` matrix flattened into a `Vec`.
/// The result is symmetric by construction (the mixed stencil is symmetric in
/// `i` and `j`).
pub fn numerical_hessian<F>(f: &F, params: &[f64], cfg: &FiniteDiffConfig) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = params.len();
    let mut out = vec![0.0; n * n];
    let f0 = f(params);
    for i in 0..n {
        let hi = cfg.step_for_2(params[i]);
        // Diagonal.
        let fp = shifted(f, params, i, hi);
        let fm = shifted(f, params, i, -hi);
        out[i * n + i] = (fp - 2.0 * f0 + fm) / (hi * hi);
        // Off-diagonal (upper triangle), mirrored to the lower triangle.
        for j in (i + 1)..n {
            let hj = cfg.step_for_2(params[j]);
            let fpp = shifted2(f, params, i, hi, j, hj);
            let fpm = shifted2(f, params, i, hi, j, -hj);
            let fmp = shifted2(f, params, i, -hi, j, hj);
            let fmm = shifted2(f, params, i, -hi, j, -hj);
            let val = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            out[i * n + j] = val;
            out[j * n + i] = val;
        }
    }
    out
}

/// Numerical third-derivative tensor via central differences.
///
/// The tensor is symmetric under permutation of its three indices. Only the
/// distinct index combinations are evaluated and then mirrored into all
/// permutations, which keeps the cost down and guarantees exact symmetry.
///
/// - **All-distinct** `i < j < k`: the eight-point central stencil
///   `[f(+++) - f(++-) - f(+-+) + f(+--) - f(-++) + f(-+-) + f(--+) - f(---)] / (8 h_i h_j h_k)`.
/// - **Two equal** `i = j < k`: `∂³ℓ/∂θ_i²∂θ_k` via the four-point stencil
///   `[f(+h_i,+h_k) - 2 f(+h_k) + f(-h_i,+h_k) - f(+h_i,-h_k) + 2 f(-h_k) - f(-h_i,-h_k)] / (2 h_i² h_k)`.
/// - **All equal** `i = j = k`: `∂³ℓ/∂θ_i³` via the four-point stencil
///   `[f(+2h) - 2 f(+h) + 2 f(-h) - f(-2h)] / (2 h³)`.
///
/// Returns the flattened `n_params × n_params × n_params` tensor in row-major
/// order (index `(i, j, k)` at `i*n² + j*n + k`).
pub fn numerical_third_derivatives<F>(f: &F, params: &[f64], cfg: &FiniteDiffConfig) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let n = params.len();
    let mut out = vec![0.0; n * n * n];
    let h: Vec<f64> = (0..n).map(|i| cfg.step_for_3(params[i])).collect();

    // W17 (H3/H6): evaluate the distinct `(i ≤ j ≤ k)` combinations in parallel
    // (each is a handful of `O(n)` log-likelihood evaluations), then scatter the
    // results into all index permutations. The combinations are independent, so
    // this is data-race free.
    let combos: Vec<(usize, usize, usize)> = {
        let mut v = Vec::new();
        for i in 0..n {
            for j in i..n {
                for k in j..n {
                    v.push((i, j, k));
                }
            }
        }
        v
    };
    // W17: only parallelize when there are enough combinations to amortize the
    // Rayon overhead (p >= 3 gives 10+ combos; p = 2 gives only 4).
    let compute = |&(i, j, k): &(usize, usize, usize)| -> f64 {
        {
            if i == j && j == k {
                // ∂³ℓ/∂θ_i³.
                let hi = h[i];
                let f2p = shifted(f, params, i, 2.0 * hi);
                let fp = shifted(f, params, i, hi);
                let fm = shifted(f, params, i, -hi);
                let f2m = shifted(f, params, i, -2.0 * hi);
                (f2p - 2.0 * fp + 2.0 * fm - f2m) / (2.0 * hi * hi * hi)
            } else if i == j {
                // ∂³ℓ/∂θ_i²∂θ_k with i == j < k.
                let hi = h[i];
                let hk = h[k];
                let fpp = shifted2(f, params, i, hi, k, hk);
                let fp = shifted(f, params, k, hk);
                let fmp = shifted2(f, params, i, -hi, k, hk);
                let fpm = shifted2(f, params, i, hi, k, -hk);
                let fm = shifted(f, params, k, -hk);
                let fmm = shifted2(f, params, i, -hi, k, -hk);
                (fpp - 2.0 * fp + fmp - fpm + 2.0 * fm - fmm) / (2.0 * hi * hi * hk)
            } else if j == k {
                // ∂³ℓ/∂θ_i∂θ_j² with i < j == k.
                let hi = h[i];
                let hj = h[j];
                let fpp = shifted2(f, params, i, hi, j, hj);
                let fp = shifted(f, params, i, hi);
                let fpm = shifted2(f, params, i, hi, j, -hj);
                let fmp = shifted2(f, params, i, -hi, j, hj);
                let fm = shifted(f, params, i, -hi);
                let fmm = shifted2(f, params, i, -hi, j, -hj);
                (fpp - 2.0 * fp + fpm - fmp + 2.0 * fm - fmm) / (2.0 * hi * hj * hj)
            } else {
                // All distinct: i < j < k.
                let hi = h[i];
                let hj = h[j];
                let hk = h[k];
                let fppp = shifted_many(f, params, &[(i, hi), (j, hj), (k, hk)]);
                let fppm = shifted_many(f, params, &[(i, hi), (j, hj), (k, -hk)]);
                let fpmp = shifted_many(f, params, &[(i, hi), (j, -hj), (k, hk)]);
                let fpmm = shifted_many(f, params, &[(i, hi), (j, -hj), (k, -hk)]);
                let fmpp = shifted_many(f, params, &[(i, -hi), (j, hj), (k, hk)]);
                let fmpm = shifted_many(f, params, &[(i, -hi), (j, hj), (k, -hk)]);
                let fmmp = shifted_many(f, params, &[(i, -hi), (j, -hj), (k, hk)]);
                let fmmm = shifted_many(f, params, &[(i, -hi), (j, -hj), (k, -hk)]);
                (fppp - fppm - fpmp + fpmm - fmpp + fmpm + fmmp - fmmm) / (8.0 * hi * hj * hk)
            }
        }
    };
    // W17: the number of distinct `(i ≤ j ≤ k)` combinations is `O(p³/6)` — only
    // 10 for `p = 3` — so the Rayon scheduling overhead exceeds the (already
    // parallelized) inner log-likelihood work. This loop is therefore kept
    // sequential; the parallelism lives in the model `loglik`/`score`/`hessian`
    // reductions instead.
    let vals: Vec<f64> = combos.iter().map(compute).collect();

    // Scatter each value into every permutation of its indices.
    for (&(i, j, k), &val) in combos.iter().zip(vals.iter()) {
        for &(a, b, c) in &[
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ] {
            out[a * n * n + b * n + c] = val;
        }
    }
    out
}

/// Numerical mixed derivatives `∂²ℓ/∂θ_i∂y_a` for the DMGS Haar-prior term.
///
/// The Haar-prior correction (§4.1) needs derivatives of the log-likelihood
/// with respect to each parameter **and** each observation. This computes the
/// `n_params × n_obs` matrix of mixed partials
/// `∂²ℓ/∂θ_i∂y_a` by central differences in both arguments:
///
/// ```text
/// ∂²ℓ/∂θ_i∂y_a ≈ [ℓ(θ+h_i, y+h_a) - ℓ(θ+h_i, y-h_a)
///                 - ℓ(θ-h_i, y+h_a) + ℓ(θ-h_i, y-h_a)] / (4 h_i h_a)
/// ```
///
/// Returns a row-major `n_params × n_obs` matrix flattened into a `Vec`.
pub fn numerical_mixed_derivatives<F>(
    f: &F,
    params: &[f64],
    y: &[f64],
    cfg: &FiniteDiffConfig,
) -> Vec<f64>
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    let np = params.len();
    let ny = y.len();
    // W17 (H3): the `np × ny` mixed partials are independent. This is the
    // `O(p·n²)` path the HOT_LOOPS note flags as ~16000× slower than the
    // analytic form. The W17 benchmark sweep found Rayon's per-item overhead
    // dominates here (the per-item work is a handful of `O(n)` log-likelihood
    // evaluations, but the outer `np × ny` grid is small), so the loop is kept
    // sequential; the parallelism lives in the model `loglik` reductions.
    let mut out = vec![0.0; np * ny];
    for i in 0..np {
        let hi = cfg.step_for_2(params[i]);
        for a in 0..ny {
            let ha = cfg.step_for_2(y[a]);
            let eval = |dp: f64, dy: f64| {
                let mut p = params.to_vec();
                let mut yy = y.to_vec();
                p[i] += dp;
                yy[a] += dy;
                f(&p, &yy)
            };
            let fpp = eval(hi, ha);
            let fpm = eval(hi, -ha);
            let fmp = eval(-hi, ha);
            let fmm = eval(-hi, -ha);
            out[i * ny + a] = (fpp - fpm - fmp + fmm) / (4.0 * hi * ha);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite differences are not exact, so the fallback is validated against a
    /// documented looser tolerance than the §7.2 golden policy (`rtol = 1e-8`,
    /// `atol = 1e-10`). Central differences are `O(h²)` accurate with
    /// order-appropriate steps (`h ≈ 6e-6` for first, `1.2e-4` for second,
    /// `7.4e-4` for third derivatives), giving truncation errors around
    /// `1e-10`–`1e-8`; round-off adds a few ULPs. `1e-6` relative / `1e-8`
    /// absolute is a safe, well-justified bound for the stencils used here.
    const FD_RTOL: f64 = 1e-6;
    const FD_ATOL: f64 = 1e-8;

    fn assert_close_fd(actual: f64, expected: f64, what: &str) {
        let tol = FD_ATOL + FD_RTOL * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual = {actual:.17e}, expected = {expected:.17e}, \
             |diff| = {:.3e} > tol = {:.3e}",
            (actual - expected).abs(),
            tol
        );
    }

    /// A Gaussian log-likelihood with known analytic derivatives.
    ///
    /// `ℓ(μ, σ; y) = -n ln σ - (1/2σ²) Σ (y_i - μ)² - (n/2) ln(2π)`
    ///
    /// Analytic derivatives (used as the reference):
    /// - `∂ℓ/∂μ = Σ (y_i - μ) / σ²`
    /// - `∂ℓ/∂σ = -n/σ + Σ (y_i - μ)² / σ³`
    /// - `∂²ℓ/∂μ² = -n/σ²`
    /// - `∂²ℓ/∂μ∂σ = -2 Σ (y_i - μ) / σ³`
    /// - `∂²ℓ/∂σ² = n/σ² - 3 Σ (y_i - μ)² / σ⁴`
    /// - `∂³ℓ/∂μ³ = 0`
    /// - `∂³ℓ/∂μ²∂σ = 2n/σ³`
    /// - `∂³ℓ/∂μ∂σ² = 6 Σ (y_i - μ) / σ⁴`
    /// - `∂³ℓ/∂σ³ = -2n/σ³ + 12 Σ (y_i - μ)² / σ⁵`
    fn gaussian_loglik(params: &[f64], y: &[f64]) -> f64 {
        let mu = params[0];
        let sigma = params[1];
        let n = y.len() as f64;
        let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
        -n * sigma.ln() - ss / (2.0 * sigma * sigma) - 0.5 * n * (2.0 * std::f64::consts::PI).ln()
    }

    fn gaussian_score(mu: f64, sigma: f64, y: &[f64]) -> Vec<f64> {
        let n = y.len() as f64;
        let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
        let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
        vec![
            s1 / (sigma * sigma),
            -n / sigma + ss / (sigma * sigma * sigma),
        ]
    }

    fn gaussian_hessian(mu: f64, sigma: f64, y: &[f64]) -> Vec<f64> {
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

    fn gaussian_third(mu: f64, sigma: f64, y: &[f64]) -> Vec<f64> {
        let n = y.len() as f64;
        let s1: f64 = y.iter().map(|&yi| yi - mu).sum();
        let ss: f64 = y.iter().map(|&yi| (yi - mu).powi(2)).sum();
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        let s4 = s2 * s2;
        let s5 = s4 * sigma;
        // Index (i, j, k) at i*4 + j*2 + k for n = 2.
        let d_mmm = 0.0;
        let d_mms = 2.0 * n / s3;
        let d_mss = 6.0 * s1 / s4;
        let d_sss = -2.0 * n / s3 + 12.0 * ss / s5;
        vec![
            // (0,0,0), (0,0,1)
            d_mmm, d_mms, // (0,1,0), (0,1,1)
            d_mms, d_mss, // (1,0,0), (1,0,1)
            d_mms, d_mss, // (1,1,0), (1,1,1)
            d_mss, d_sss,
        ]
    }

    fn sample() -> Vec<f64> {
        vec![0.3, -1.2, 2.4, 0.9, -0.5, 1.7, -2.1, 0.0, 1.1, -0.8]
    }

    #[test]
    fn numerical_score_matches_gaussian_analytic() {
        let y = sample();
        let params = [0.4, 1.3];
        let cfg = FiniteDiffConfig::default();
        let got = numerical_score(&|p: &[f64]| gaussian_loglik(p, &y), &params, &cfg);
        let want = gaussian_score(params[0], params[1], &y);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_close_fd(*g, *w, &format!("score[{i}]"));
        }
    }

    #[test]
    fn numerical_hessian_matches_gaussian_analytic() {
        let y = sample();
        let params = [0.4, 1.3];
        let cfg = FiniteDiffConfig::default();
        let got = numerical_hessian(&|p: &[f64]| gaussian_loglik(p, &y), &params, &cfg);
        let want = gaussian_hessian(params[0], params[1], &y);
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_close_fd(*g, *w, &format!("hessian[{idx}]"));
        }
        // Symmetry.
        assert_close_fd(got[1], got[2], "hessian symmetry");
    }

    #[test]
    fn numerical_third_derivatives_match_gaussian_analytic() {
        let y = sample();
        let params = [0.4, 1.3];
        let cfg = FiniteDiffConfig::default();
        let got = numerical_third_derivatives(&|p: &[f64]| gaussian_loglik(p, &y), &params, &cfg);
        let want = gaussian_third(params[0], params[1], &y);
        // The third difference has `O(ε/h³)` round-off. With `h ≈ 7.4e-4` and
        // `|ℓ| ~ 10`, that is `~5e-6` absolute. Entries whose analytic value is
        // exactly zero (`∂³ℓ/∂μ³`) are therefore round-off-limited and need a
        // dedicated absolute tolerance. Non-zero entries carry a larger
        // truncation error than first/second differences, so a looser relative
        // tolerance (`1e-4`) is used and documented here.
        const THIRD_ATOL: f64 = 1e-4;
        const THIRD_RTOL: f64 = 1e-4;
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = THIRD_ATOL + THIRD_RTOL * w.abs();
            assert!(
                (g - w).abs() <= tol,
                "third[{idx}]: actual = {g:.17e}, expected = {w:.17e}, \
                 |diff| = {:.3e} > tol = {:.3e}",
                (g - w).abs(),
                tol
            );
        }
    }

    #[test]
    fn numerical_mixed_derivatives_match_gaussian_analytic() {
        // ∂ℓ/∂μ = Σ(y_i - μ)/σ²  =>  ∂²ℓ/∂μ∂y_a = +1/σ²  (constant in a).
        // ∂ℓ/∂σ = -n/σ + Σ(y_i - μ)²/σ³  =>  ∂²ℓ/∂σ∂y_a = +2(y_a - μ)/σ³.
        let y = sample();
        let params = [0.4, 1.3];
        let mu = params[0];
        let sigma = params[1];
        let cfg = FiniteDiffConfig::default();
        let got = numerical_mixed_derivatives(
            &|p: &[f64], yy: &[f64]| gaussian_loglik(p, yy),
            &params,
            &y,
            &cfg,
        );
        let ny = y.len();
        for (a, &ya) in y.iter().enumerate() {
            let want_mu = 1.0 / (sigma * sigma);
            let want_sigma = 2.0 * (ya - mu) / (sigma * sigma * sigma);
            assert_close_fd(got[a], want_mu, &format!("mixed[mu][{a}]"));
            assert_close_fd(got[ny + a], want_sigma, &format!("mixed[sigma][{a}]"));
        }
    }

    #[test]
    fn step_scales_with_parameter_magnitude() {
        let cfg = FiniteDiffConfig::default();
        // Small parameter: step is dominated by the absolute floor.
        assert!(cfg.step_for(0.0) >= DEFAULT_ABS_STEP);
        // Large parameter: step grows with |theta|.
        assert!(cfg.step_for(1.0e6) > cfg.step_for(1.0));
    }
}
