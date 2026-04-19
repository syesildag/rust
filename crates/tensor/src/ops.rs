//! Differentiable tensor operations.
//!
//! Every public function computes a forward result and — when any input
//! `requires_grad` — attaches a `GradFn` so that `Tensor::backward()`
//! can propagate gradients through it.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::identity_op)]

use std::sync::Arc;

use crate::tensor_impl::{GradFn, Tensor};

// ── Internal CPU math helpers ──────────────────────────────────────��──────────

/// C = A @ B where A is [m×k] and B is [k×n].
fn matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Transpose a 2-D matrix [m×n] → [n×m].
fn transpose_2d(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Broadcast-add bias [n] to matrix [m×n].
fn add_bias(x: &[f32], bias: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] += bias[j];
        }
    }
    out
}

// ── add ───────────────────────────────────────────────────────────────────────

struct AddBackward {
    a: Tensor,
    b: Tensor,
}
impl GradFn for AddBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.a.requires_grad() {
            self.a.accumulate_grad(g);
        }
        if self.b.requires_grad() {
            self.b.accumulate_grad(g);
        }
    }
}

/// Element-wise addition of two tensors.
///
/// Both tensors must have identical shapes. Output shape equals input shape.
#[must_use]
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(
        a.shape(),
        b.shape(),
        "add: shape mismatch {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x + y)
        .collect();
    let shape = a.shape().to_vec();
    if a.requires_grad() || b.requires_grad() {
        Tensor::from_op(
            data,
            &shape,
            Arc::new(AddBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
    }
}

// ── sub ───────────────────────────────────────────────────────────────────────

struct SubBackward {
    a: Tensor,
    b: Tensor,
}
impl GradFn for SubBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.a.requires_grad() {
            self.a.accumulate_grad(g);
        }
        if self.b.requires_grad() {
            let neg: Vec<f32> = g.iter().map(|v| -v).collect();
            self.b.accumulate_grad(&neg);
        }
    }
}

/// Element-wise subtraction of two tensors.
///
/// Both tensors must have identical shapes. Output shape equals input shape.
#[must_use]
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape());
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| x - y)
        .collect();
    let shape = a.shape().to_vec();
    if a.requires_grad() || b.requires_grad() {
        Tensor::from_op(
            data,
            &shape,
            Arc::new(SubBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
    }
}

// ── mul_scalar ────────────────────────────────────────────────────────────────

struct MulScalarBackward {
    input: Tensor,
    scalar: f32,
}
impl GradFn for MulScalarBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let d: Vec<f32> = g.iter().map(|v| v * self.scalar).collect();
            self.input.accumulate_grad(&d);
        }
    }
}

/// Multiplies all elements of `x` by the `f32` scalar `s`.
///
/// Output shape equals input shape.
#[must_use]
pub fn mul_scalar(x: &Tensor, s: f32) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|v| v * s).collect();
    let shape = x.shape().to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &shape,
            Arc::new(MulScalarBackward {
                input: x.clone(),
                scalar: s,
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
    }
}

// ── matmul ────────────────────────────────────────────────────────────────────

struct MatMulBackward {
    a: Tensor,
    b: Tensor,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}
impl GradFn for MatMulBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, g: &[f32]) {
        // g: [M, N],  A: [M, K],  B: [K, N]
        let m = self.a_shape[0];
        let k = self.a_shape[1];
        let n = self.b_shape[1];
        if self.a.requires_grad() {
            let b_data = self.b.data();
            let bt = transpose_2d(&b_data, k, n); // [N, K]
            let da = matmul_2d(g, &bt, m, n, k); // [M,N]@[N,K]=[M,K]
            self.a.accumulate_grad(&da);
        }
        if self.b.requires_grad() {
            let a_data = self.a.data();
            let at = transpose_2d(&a_data, m, k); // [K, M]
            let db = matmul_2d(&at, g, k, m, n); // [K,M]@[M,N]=[K,N]
            self.b.accumulate_grad(&db);
        }
    }
}

/// 2-D matrix multiply: `[M, K] @ [K, N] → [M, N]`.
///
/// # Panics
/// Panics if `a` and `b` are not 2-D, or their inner dimensions don't match.
#[must_use]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let sa = a.shape();
    let sb = b.shape();
    assert_eq!(sa.len(), 2, "matmul: a must be 2-D, got {sa:?}");
    assert_eq!(sb.len(), 2, "matmul: b must be 2-D, got {sb:?}");
    assert_eq!(
        sa[1], sb[0],
        "matmul: inner dims must match: {sa:?} vs {sb:?}"
    );
    let (m, k, n) = (sa[0], sa[1], sb[1]);
    let data = matmul_2d(&a.data(), &b.data(), m, k, n);
    if a.requires_grad() || b.requires_grad() {
        Tensor::from_op(
            data,
            &[m, n],
            Arc::new(MatMulBackward {
                a: a.clone(),
                b: b.clone(),
                a_shape: sa.to_vec(),
                b_shape: sb.to_vec(),
            }),
        )
    } else {
        Tensor::from_vec(data, &[m, n])
    }
}

// ── linear (fused matmul + bias) ──────────────────────────────────────────────

struct LinearBackward {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    in_shape: Vec<usize>,
    w_shape: Vec<usize>,
}
impl GradFn for LinearBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.weight.clone(), self.bias.clone()]
    }
    fn backward(&self, g: &[f32]) {
        // output = input @ weight.T + bias
        // input: [S, in],  weight: [out, in],  bias: [out],  g: [S, out]
        let s = self.in_shape[0];
        let in_f = self.in_shape[1];
        let out_f = self.w_shape[0];
        if self.input.requires_grad() {
            // d_input = g @ weight  ([S,out] @ [out,in] = [S,in])
            let d = matmul_2d(g, &self.weight.data(), s, out_f, in_f);
            self.input.accumulate_grad(&d);
        }
        if self.weight.requires_grad() {
            // d_weight = g.T @ input  ([out,S] @ [S,in] = [out,in])
            let gt = transpose_2d(g, s, out_f);
            let d = matmul_2d(&gt, &self.input.data(), out_f, s, in_f);
            self.weight.accumulate_grad(&d);
        }
        if self.bias.requires_grad() {
            // d_bias = sum over rows of g  → [out]
            let mut d = vec![0.0f32; out_f];
            for i in 0..s {
                for j in 0..out_f {
                    d[j] += g[i * out_f + j];
                }
            }
            self.bias.accumulate_grad(&d);
        }
    }
}

/// Fused linear transformation: `input @ weight.T + bias`.
///
/// - `input`:  `[S, in_features]`
/// - `weight`: `[out_features, in_features]`
/// - `bias`:   `[out_features]`
/// - Output:   `[S, out_features]`
///
/// # Panics
/// Panics if `input` or `weight` is not 2-D, or their feature dimensions don't match.
#[must_use]
pub fn linear(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    let si = input.shape();
    let sw = weight.shape();
    assert_eq!(si.len(), 2);
    assert_eq!(sw.len(), 2);
    assert_eq!(si[1], sw[1], "linear: feature mismatch");
    let (s, in_f, out_f) = (si[0], si[1], sw[0]);
    let raw = matmul_2d(
        &input.data(),
        &transpose_2d(&weight.data(), out_f, in_f),
        s,
        in_f,
        out_f,
    );
    let data = add_bias(&raw, &bias.data(), s, out_f);
    let needs_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    if needs_grad {
        Tensor::from_op(
            data,
            &[s, out_f],
            Arc::new(LinearBackward {
                input: input.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                in_shape: si.to_vec(),
                w_shape: sw.to_vec(),
            }),
        )
    } else {
        Tensor::from_vec(data, &[s, out_f])
    }
}

// ── relu ──────────────────────────────────────────────────────────────────────

struct ReluBackward {
    mask: Vec<bool>,
    input: Tensor,
}
impl GradFn for ReluBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let d: Vec<f32> = g
                .iter()
                .zip(self.mask.iter())
                .map(|(gi, &m)| if m { *gi } else { 0.0 })
                .collect();
            self.input.accumulate_grad(&d);
        }
    }
}

/// Rectified linear unit: `max(0, x)`.
///
/// Element-wise; output shape equals input shape.
#[must_use]
pub fn relu(x: &Tensor) -> Tensor {
    let src = x.data();
    let mask: Vec<bool> = src.iter().map(|&v| v > 0.0).collect();
    let data: Vec<f32> = src.iter().map(|&v| v.max(0.0)).collect();
    let shape = x.shape().to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &shape,
            Arc::new(ReluBackward {
                mask,
                input: x.clone(),
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
    }
}

// ── gelu ────────────────────────────────────────────────────────────────────��─

#[inline]
fn gelu_fwd(v: f32) -> f32 {
    const S: f32 = 0.797_884_6; // sqrt(2/π)
    const C: f32 = 0.044_715;
    let inner = S * (v + C * v * v * v);
    0.5 * v * (1.0 + inner.tanh())
}

#[inline]
fn gelu_bwd(v: f32, g: f32) -> f32 {
    const S: f32 = 0.797_884_6;
    const C: f32 = 0.044_715;
    let inner = S * (v + C * v.powi(3));
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let deriv = 0.5 * (1.0 + t) + 0.5 * v * sech2 * S * (1.0 + 3.0 * C * v * v);
    g * deriv
}

struct GeluBackward {
    input: Tensor,
    saved_input: Vec<f32>,
}
impl GradFn for GeluBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let d: Vec<f32> = g
                .iter()
                .zip(self.saved_input.iter())
                .map(|(&gi, &xi)| gelu_bwd(xi, gi))
                .collect();
            self.input.accumulate_grad(&d);
        }
    }
}

/// Gaussian Error Linear Unit (GELU) activation.
///
/// Uses the tanh approximation: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
/// Element-wise; output shape equals input shape.
#[must_use]
pub fn gelu(x: &Tensor) -> Tensor {
    let src = x.data();
    let data: Vec<f32> = src.iter().map(|&v| gelu_fwd(v)).collect();
    let shape = x.shape().to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &shape,
            Arc::new(GeluBackward {
                input: x.clone(),
                saved_input: src,
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
    }
}

// ── tanh ──────────────────────────────────────────────────────────────────────

struct TanhBackward {
    input: Tensor,
    output: Vec<f32>,
}
impl GradFn for TanhBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let d: Vec<f32> = g
                .iter()
                .zip(self.output.iter())
                .map(|(&gi, &yi)| gi * (1.0 - yi * yi))
                .collect();
            self.input.accumulate_grad(&d);
        }
    }
}

/// Element-wise hyperbolic tangent.
///
/// Element-wise; output shape equals input shape.
#[must_use]
pub fn tanh(x: &Tensor) -> Tensor {
    let src = x.data();
    let data: Vec<f32> = src.iter().map(|&v| v.tanh()).collect();
    let shape = x.shape().to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data.clone(),
            &shape,
            Arc::new(TanhBackward {
                input: x.clone(),
                output: data,
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
    }
}

// ── softmax ───────────────────────────────────────────────────────────────────

/// Applies numerically stable row-wise softmax to a 2-D tensor `[S, D]`.
fn softmax_rows(x: &[f32], s: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; s * d];
    for i in 0..s {
        let row = &x[i * d..(i + 1) * d];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            out[i * d + j] = e / sum;
        }
    }
    out
}

struct SoftmaxBackward {
    input: Tensor,
    output: Vec<f32>,
    s: usize,
    d: usize,
}
impl GradFn for SoftmaxBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if !self.input.requires_grad() {
            return;
        }
        // dL/dx[i] = y[i] * (g[i] - sum_j(g[j]*y[j]))
        let mut d = vec![0.0f32; self.s * self.d];
        for i in 0..self.s {
            let row_y = &self.output[i * self.d..(i + 1) * self.d];
            let row_g = &g[i * self.d..(i + 1) * self.d];
            let dot: f32 = row_y.iter().zip(row_g.iter()).map(|(y, g)| y * g).sum();
            for j in 0..self.d {
                d[i * self.d + j] = row_y[j] * (row_g[j] - dot);
            }
        }
        self.input.accumulate_grad(&d);
    }
}

/// Row-wise softmax on a 2-D tensor `[S, D]`.
///
/// Applies numerically stable softmax independently to each row.
/// Each row of the output sums to 1 along the last dimension.
/// Output shape equals input shape `[S, D]`.
///
/// # Panics
/// Panics if `x` is not 2-D.
#[must_use]
pub fn softmax(x: &Tensor) -> Tensor {
    let s_x = x.shape();
    assert_eq!(s_x.len(), 2, "softmax: expected 2-D tensor, got {s_x:?}");
    let (s, d) = (s_x[0], s_x[1]);
    let src = x.data();
    let data = softmax_rows(&src, s, d);
    if x.requires_grad() {
        Tensor::from_op(
            data.clone(),
            &[s, d],
            Arc::new(SoftmaxBackward {
                input: x.clone(),
                output: data,
                s,
                d,
            }),
        )
    } else {
        Tensor::from_vec(data, &[s, d])
    }
}

// ── layer_norm ────────────────────────────────────────────────────────────────

struct LayerNormBackward {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    saved_x: Vec<f32>,
    saved_mean: Vec<f32>,
    saved_inv_std: Vec<f32>,
    s: usize,
    d: usize,
}
impl GradFn for LayerNormBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.gamma.clone(), self.beta.clone()]
    }
    fn backward(&self, g: &[f32]) {
        let (s, d) = (self.s, self.d);
        let gamma_data = self.gamma.data();
        let nd = d as f32;
        let mut d_input = vec![0.0f32; s * d];
        let mut d_gamma = vec![0.0f32; d];
        let mut d_beta = vec![0.0f32; d];
        for i in 0..s {
            let mean = self.saved_mean[i];
            let inv_std = self.saved_inv_std[i];
            let x_row = &self.saved_x[i * d..(i + 1) * d];
            let g_row = &g[i * d..(i + 1) * d];
            // x_hat = (x - mean) * inv_std
            let x_hat: Vec<f32> = x_row.iter().map(|&v| (v - mean) * inv_std).collect();
            // d_gamma += g * x_hat  (sum over samples)
            for j in 0..d {
                d_gamma[j] += g_row[j] * x_hat[j];
            }
            // d_beta  += g          (sum over samples)
            for j in 0..d {
                d_beta[j] += g_row[j];
            }
            if self.input.requires_grad() {
                // dx = gamma * inv_std * (g - mean(g) - x_hat * mean(g*x_hat)) / N
                let dx_hat: Vec<f32> = g_row
                    .iter()
                    .zip(gamma_data.iter())
                    .map(|(gi, &gam)| gi * gam)
                    .collect();
                let mean_dx_hat: f32 = dx_hat.iter().sum::<f32>() / nd;
                let mean_dx_hat_xhat: f32 = dx_hat
                    .iter()
                    .zip(x_hat.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / nd;
                for j in 0..d {
                    d_input[i * d + j] =
                        inv_std * (dx_hat[j] - mean_dx_hat - x_hat[j] * mean_dx_hat_xhat);
                }
            }
        }
        if self.input.requires_grad() {
            self.input.accumulate_grad(&d_input);
        }
        if self.gamma.requires_grad() {
            self.gamma.accumulate_grad(&d_gamma);
        }
        if self.beta.requires_grad() {
            self.beta.accumulate_grad(&d_beta);
        }
    }
}

/// Layer normalisation over the last dimension.
///
/// Normalises `x` along dimension `D`, then scales by `gamma` and shifts by `beta`.
/// `x`: `[S, D]`, `gamma`: `[D]`, `beta`: `[D]` → `[S, D]`.
///
/// # Panics
/// Panics if `x` is not 2-D.
#[must_use]
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
    let sx = x.shape();
    assert_eq!(sx.len(), 2, "layer_norm: expected 2-D input");
    let (s, d) = (sx[0], sx[1]);
    let src = x.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();
    let mut data = vec![0.0f32; s * d];
    let mut saved_mean = vec![0.0f32; s];
    let mut saved_inv_std = vec![0.0f32; s];
    let nd = d as f32;
    for i in 0..s {
        let row = &src[i * d..(i + 1) * d];
        let mean: f32 = row.iter().sum::<f32>() / nd;
        let var: f32 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / nd;
        let inv_std = 1.0 / (var + eps).sqrt();
        saved_mean[i] = mean;
        saved_inv_std[i] = inv_std;
        for j in 0..d {
            data[i * d + j] = gamma_data[j] * (row[j] - mean) * inv_std + beta_data[j];
        }
    }
    let needs_grad = x.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    if needs_grad {
        Tensor::from_op(
            data,
            &[s, d],
            Arc::new(LayerNormBackward {
                input: x.clone(),
                gamma: gamma.clone(),
                beta: beta.clone(),
                saved_x: src,
                saved_mean,
                saved_inv_std,
                s,
                d,
            }),
        )
    } else {
        Tensor::from_vec(data, &[s, d])
    }
}

// ── cat ───────────────────────────────────────────────────────────────────────

struct CatBackward {
    inputs: Vec<Tensor>,
    sizes: Vec<usize>,
}
impl GradFn for CatBackward {
    fn inputs(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }
    fn backward(&self, g: &[f32]) {
        let mut offset = 0;
        for (t, &sz) in self.inputs.iter().zip(self.sizes.iter()) {
            if t.requires_grad() {
                t.accumulate_grad(&g[offset..offset + sz]);
            }
            offset += sz;
        }
    }
}

/// Concatenates 2-D tensors along axis 0 (rows).
///
/// All tensors must have the same number of columns.
/// Input tensors of shapes `[r₁, C]`, `[r₂, C]`, … → output `[r₁+r₂+…, C]`.
///
/// # Panics
/// Panics if `tensors` is empty or column counts differ.
#[must_use]
pub fn cat(tensors: &[&Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "cat: empty input list");
    let ncols: usize = tensors[0].shape()[1];
    let mut data = Vec::new();
    let mut sizes = Vec::new();
    let mut total_rows = 0usize;
    for t in tensors {
        assert_eq!(t.shape()[1], ncols, "cat: column mismatch");
        let d = t.data();
        sizes.push(d.len());
        data.extend_from_slice(&d);
        total_rows += t.shape()[0];
    }
    let needs_grad = tensors.iter().any(|t| t.requires_grad());
    let inputs: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    if needs_grad {
        Tensor::from_op(
            data,
            &[total_rows, ncols],
            Arc::new(CatBackward { inputs, sizes }),
        )
    } else {
        Tensor::from_vec(data, &[total_rows, ncols])
    }
}

// ── cat_cols ──────────────────────────────────────────────────────────────────

struct CatColsBackward {
    inputs: Vec<Tensor>,
    col_sizes: Vec<usize>,
    nrows: usize,
}
impl GradFn for CatColsBackward {
    fn inputs(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }
    fn backward(&self, g: &[f32]) {
        let ncols_total: usize = self.col_sizes.iter().sum();
        let mut col_offset = 0;
        for (t, &ncols) in self.inputs.iter().zip(self.col_sizes.iter()) {
            if t.requires_grad() {
                let mut d = vec![0.0f32; self.nrows * ncols];
                for i in 0..self.nrows {
                    for j in 0..ncols {
                        d[i * ncols + j] = g[i * ncols_total + col_offset + j];
                    }
                }
                t.accumulate_grad(&d);
            }
            col_offset += ncols;
        }
    }
}

/// Concatenates 2-D tensors along axis 1 (columns).
///
/// All tensors must have the same number of rows.
/// Input tensors of shapes `[R, c₁]`, `[R, c₂]`, … → output `[R, c₁+c₂+…]`.
///
/// # Panics
/// Panics if `tensors` is empty or row counts differ.
#[must_use]
pub fn cat_cols(tensors: &[&Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "cat_cols: empty input list");
    let nrows = tensors[0].shape()[0];
    let col_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[1]).collect();
    let ncols_total: usize = col_sizes.iter().sum();
    let mut data = vec![0.0f32; nrows * ncols_total];
    let mut col_offset = 0;
    for (t, &ncols) in tensors.iter().zip(col_sizes.iter()) {
        assert_eq!(t.shape()[0], nrows, "cat_cols: row count mismatch");
        let td = t.data();
        for i in 0..nrows {
            for j in 0..ncols {
                data[i * ncols_total + col_offset + j] = td[i * ncols + j];
            }
        }
        col_offset += ncols;
    }
    let needs_grad = tensors.iter().any(|t| t.requires_grad());
    let inputs: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    if needs_grad {
        Tensor::from_op(
            data,
            &[nrows, ncols_total],
            Arc::new(CatColsBackward {
                inputs,
                col_sizes,
                nrows,
            }),
        )
    } else {
        Tensor::from_vec(data, &[nrows, ncols_total])
    }
}

// ── select_row ────────────────────────────────────────────────────────────────

struct SelectRowBackward {
    input: Tensor,
    row_idx: usize,
    s: usize,
    d: usize,
}
impl GradFn for SelectRowBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let mut d = vec![0.0f32; self.s * self.d];
            d[self.row_idx * self.d..(self.row_idx + 1) * self.d].copy_from_slice(g);
            self.input.accumulate_grad(&d);
        }
    }
}

/// Extracts row `i` from a 2-D tensor, preserving gradient flow.
///
/// Input `[S, D]`, index `i` (must be `< S`) → output `[1, D]`.
///
/// # Panics
/// Panics if `x` is not 2-D or `i` is out of bounds.
#[must_use]
pub fn select_row(x: &Tensor, i: usize) -> Tensor {
    let s = x.shape();
    assert_eq!(s.len(), 2);
    let (rows, d) = (s[0], s[1]);
    let data = x.data()[i * d..(i + 1) * d].to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &[1, d],
            Arc::new(SelectRowBackward {
                input: x.clone(),
                row_idx: i,
                s: rows,
                d,
            }),
        )
    } else {
        Tensor::from_vec(data, &[1, d])
    }
}

// ── conv2d ────────────────────────────────────────────────────────────────────

/// im2col: unfold input `[N, C_in, H, W]` into columns `[N, H_out*W_out, C_in*kH*kW]`.
fn im2col(
    input: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    pad: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = h + 2 * pad - kh + 1;
    let w_out = w + 2 * pad - kw + 1;
    let col_rows = h_out * w_out;
    let col_cols = c_in * kh * kw;
    let mut cols = vec![0.0f32; n * col_rows * col_cols];
    let h_i = h as isize;
    let w_i = w as isize;
    let pad_i = pad as isize;
    for ni in 0..n {
        for ci in 0..c_in {
            for ki in 0..kh {
                for kj in 0..kw {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            let src_h = hi as isize + ki as isize - pad_i;
                            let src_w = wi as isize + kj as isize - pad_i;
                            let col_row = hi * w_out + wi;
                            let col_col = ci * kh * kw + ki * kw + kj;
                            let val = if src_h >= 0 && src_h < h_i && src_w >= 0 && src_w < w_i {
                                input[ni * c_in * h * w
                                    + ci * h * w
                                    + src_h as usize * w
                                    + src_w as usize]
                            } else {
                                0.0
                            };
                            cols[ni * col_rows * col_cols + col_row * col_cols + col_col] = val;
                        }
                    }
                }
            }
        }
    }
    (cols, h_out, w_out)
}

/// col2im: accumulate column gradients back into input gradient `[N, C_in, H, W]`.
fn col2im_add(
    cols: &[f32],
    d_input: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    pad: usize,
    h_out: usize,
    w_out: usize,
) {
    let col_cols = c_in * kh * kw;
    let h_i = h as isize;
    let w_i = w as isize;
    let pad_i = pad as isize;
    for ni in 0..n {
        for ci in 0..c_in {
            for ki in 0..kh {
                for kj in 0..kw {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            let src_h = hi as isize + ki as isize - pad_i;
                            let src_w = wi as isize + kj as isize - pad_i;
                            if src_h >= 0 && src_h < h_i && src_w >= 0 && src_w < w_i {
                                let col_row = hi * w_out + wi;
                                let col_col = ci * kh * kw + ki * kw + kj;
                                d_input[ni * c_in * h * w
                                    + ci * h * w
                                    + src_h as usize * w
                                    + src_w as usize] += cols
                                    [ni * h_out * w_out * col_cols + col_row * col_cols + col_col];
                            }
                        }
                    }
                }
            }
        }
    }
}

struct Conv2dBackward {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    saved_cols: Vec<f32>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    pad: usize,
    h_out: usize,
    w_out: usize,
}
impl GradFn for Conv2dBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.weight.clone(), self.bias.clone()]
    }
    fn backward(&self, g: &[f32]) {
        // g: [N, C_out, H_out, W_out]  →  reshape to [N, H_out*W_out, C_out]
        let (n, c_out, h_out, w_out) = (self.n, self.c_out, self.h_out, self.w_out);
        let col_rows = h_out * w_out;
        let col_cols = self.c_in * self.kh * self.kw;
        // Reorder g from [N, C_out, H_out, W_out] to [N, H_out*W_out, C_out]
        let mut g_mat = vec![0.0f32; n * col_rows * c_out];
        for ni in 0..n {
            for co in 0..c_out {
                for hi in 0..h_out {
                    for wi in 0..w_out {
                        g_mat[ni * col_rows * c_out + (hi * w_out + wi) * c_out + co] =
                            g[ni * c_out * h_out * w_out + co * h_out * w_out + hi * w_out + wi];
                    }
                }
            }
        }
        // weight: [C_out, C_in*kH*kW]
        let w_data = self.weight.data();
        if self.weight.requires_grad() {
            // d_weight = sum_N( cols_N.T @ g_mat_N )  →  [C_in*kH*kW, C_out]
            let mut dw = vec![0.0f32; col_cols * c_out];
            for ni in 0..n {
                let cols_n =
                    &self.saved_cols[ni * col_rows * col_cols..(ni + 1) * col_rows * col_cols];
                let gmat_n = &g_mat[ni * col_rows * c_out..(ni + 1) * col_rows * c_out];
                let ct = transpose_2d(cols_n, col_rows, col_cols);
                let part = matmul_2d(&ct, gmat_n, col_cols, col_rows, c_out);
                for (dw_i, p) in dw.iter_mut().zip(part.iter()) {
                    *dw_i += p;
                }
            }
            // d_weight has shape [C_in*kH*kW, C_out]; transpose to [C_out, C_in*kH*kW]
            let dw_t = transpose_2d(&dw, col_cols, c_out);
            self.weight.accumulate_grad(&dw_t);
        }
        if self.bias.requires_grad() {
            // d_bias[co] = sum over N, H_out, W_out of g[N, co, H_out, W_out]
            let mut db = vec![0.0f32; c_out];
            for ni in 0..n {
                for co in 0..c_out {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            db[co] += g
                                [ni * c_out * h_out * w_out + co * h_out * w_out + hi * w_out + wi];
                        }
                    }
                }
            }
            self.bias.accumulate_grad(&db);
        }
        if self.input.requires_grad() {
            // d_cols = g_mat @ weight  ([N, HW, C_out] @ [C_out, col_cols] = [N, HW, col_cols])
            let mut d_cols = vec![0.0f32; n * col_rows * col_cols];
            for ni in 0..n {
                let gmat_n = &g_mat[ni * col_rows * c_out..(ni + 1) * col_rows * c_out];
                let dc = matmul_2d(gmat_n, &w_data, col_rows, c_out, col_cols);
                d_cols[ni * col_rows * col_cols..(ni + 1) * col_rows * col_cols]
                    .copy_from_slice(&dc);
            }
            let mut d_input = vec![0.0f32; n * self.c_in * self.h * self.w];
            col2im_add(
                &d_cols,
                &mut d_input,
                n,
                self.c_in,
                self.h,
                self.w,
                self.kh,
                self.kw,
                self.pad,
                h_out,
                w_out,
            );
            self.input.accumulate_grad(&d_input);
        }
    }
}

/// 2-D convolution with optional zero-padding.
///
/// - `input`:   `[N, C_in, H, W]`
/// - `weight`:  `[C_out, C_in, kH, kW]`
/// - `bias`:    `[C_out]`
/// - Output:    `[N, C_out, H_out, W_out]` where
///   `H_out = H + 2*padding - kH + 1` and `W_out = W + 2*padding - kW + 1`.
///
/// # Panics
/// Panics if `input` is not 4-D, `weight` is not 4-D, or channel counts are inconsistent.
#[must_use]
pub fn conv2d(input: &Tensor, weight: &Tensor, bias: &Tensor, padding: usize) -> Tensor {
    let si = input.shape();
    let sw = weight.shape();
    assert_eq!(si.len(), 4, "conv2d: input must be 4-D [N,C_in,H,W]");
    assert_eq!(sw.len(), 4, "conv2d: weight must be 4-D [C_out,C_in,kH,kW]");
    let (n, c_in, h, w) = (si[0], si[1], si[2], si[3]);
    let (c_out, kh, kw) = (sw[0], sw[2], sw[3]);
    assert_eq!(sw[1], c_in, "conv2d: channel mismatch");
    let input_data = input.data();
    let weight_data = weight.data();
    let bias_data = bias.data();
    let (cols, h_out, w_out) = im2col(&input_data, n, c_in, h, w, kh, kw, padding);
    // weight_mat: [C_out, C_in*kH*kW]
    let col_cols = c_in * kh * kw;
    let col_rows = h_out * w_out;
    // output: [N, C_out, H_out, W_out]
    let mut data = vec![0.0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        let cols_n = &cols[ni * col_rows * col_cols..(ni + 1) * col_rows * col_cols];
        // [H_out*W_out, col_cols] @ [col_cols, C_out] = [H_out*W_out, C_out]
        let wt = transpose_2d(&weight_data, c_out, col_cols);
        let out_mat = matmul_2d(cols_n, &wt, col_rows, col_cols, c_out);
        // Reorder [H_out*W_out, C_out] → [C_out, H_out, W_out]
        for co in 0..c_out {
            for hi in 0..h_out {
                for wi in 0..w_out {
                    data[ni * c_out * h_out * w_out + co * h_out * w_out + hi * w_out + wi] =
                        out_mat[(hi * w_out + wi) * c_out + co] + bias_data[co];
                }
            }
        }
    }
    let needs_grad = input.requires_grad() || weight.requires_grad() || bias.requires_grad();
    if needs_grad {
        Tensor::from_op(
            data,
            &[n, c_out, h_out, w_out],
            Arc::new(Conv2dBackward {
                input: input.clone(),
                weight: weight.clone(),
                bias: bias.clone(),
                saved_cols: cols,
                n,
                c_in,
                h,
                w,
                c_out,
                kh,
                kw,
                pad: padding,
                h_out,
                w_out,
            }),
        )
    } else {
        Tensor::from_vec(data, &[n, c_out, h_out, w_out])
    }
}

// ── batch_norm_2d ─────────────────────────────────────────────────────────────

struct BatchNorm2dBackward {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    saved_x_hat: Vec<f32>,
    saved_inv_std: Vec<f32>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}
impl GradFn for BatchNorm2dBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.gamma.clone(), self.beta.clone()]
    }
    fn backward(&self, g: &[f32]) {
        let (n, c, h, w) = (self.n, self.c, self.h, self.w);
        let m = (n * h * w) as f32;
        let gamma_data = self.gamma.data();
        let mut d_input = vec![0.0f32; n * c * h * w];
        let mut d_gamma = vec![0.0f32; c];
        let mut d_beta = vec![0.0f32; c];
        for ci in 0..c {
            let inv_std = self.saved_inv_std[ci];
            // Collect x_hat and g values for this channel
            let mut x_hat_c = vec![0.0f32; n * h * w];
            let mut g_c = vec![0.0f32; n * h * w];
            for ni in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = ni * c * h * w + ci * h * w + hi * w + wi;
                        let flat = ni * h * w + hi * w + wi;
                        x_hat_c[flat] = self.saved_x_hat[idx];
                        g_c[flat] = g[idx];
                    }
                }
            }
            // d_gamma, d_beta
            d_gamma[ci] = g_c.iter().zip(x_hat_c.iter()).map(|(a, b)| a * b).sum();
            d_beta[ci] = g_c.iter().sum();
            if self.input.requires_grad() {
                // Standard BN backward
                let dx_hat: Vec<f32> = g_c.iter().map(|&gi| gi * gamma_data[ci]).collect();
                let sum_dx_hat: f32 = dx_hat.iter().sum();
                let sum_dx_hat_xhat: f32 =
                    dx_hat.iter().zip(x_hat_c.iter()).map(|(a, b)| a * b).sum();
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let flat = ni * h * w + hi * w + wi;
                            let idx = ni * c * h * w + ci * h * w + hi * w + wi;
                            d_input[idx] = inv_std
                                * (dx_hat[flat]
                                    - sum_dx_hat / m
                                    - x_hat_c[flat] * sum_dx_hat_xhat / m);
                        }
                    }
                }
            }
        }
        if self.input.requires_grad() {
            self.input.accumulate_grad(&d_input);
        }
        if self.gamma.requires_grad() {
            self.gamma.accumulate_grad(&d_gamma);
        }
        if self.beta.requires_grad() {
            self.beta.accumulate_grad(&d_beta);
        }
    }
}

/// Batch normalisation for 4-D tensors.
///
/// Normalises each channel `C` over the `N`, `H`, `W` dimensions, then scales
/// by `gamma` and shifts by `beta`.
///
/// - `input`:  `[N, C, H, W]`
/// - `gamma`:  `[C]`
/// - `beta`:   `[C]`
/// - Output:   `[N, C, H, W]` (same shape as input)
///
/// # Panics
/// Panics if `input` is not 4-D.
#[must_use]
pub fn batch_norm_2d(input: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
    let si = input.shape();
    assert_eq!(si.len(), 4, "batch_norm_2d: expected [N,C,H,W]");
    let (n, c, h, w) = (si[0], si[1], si[2], si[3]);
    let src = input.data();
    let gamma_data = gamma.data();
    let beta_data = beta.data();
    let m = (n * h * w) as f32;
    let mut data = vec![0.0f32; n * c * h * w];
    let mut saved_x_hat = vec![0.0f32; n * c * h * w];
    let mut saved_inv_std = vec![0.0f32; c];
    for ci in 0..c {
        let mut sum = 0.0f32;
        let mut sq = 0.0f32;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let v = src[ni * c * h * w + ci * h * w + hi * w + wi];
                    sum += v;
                    sq += v * v;
                }
            }
        }
        let mean = sum / m;
        let var = sq / m - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();
        saved_inv_std[ci] = inv_std;
        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let idx = ni * c * h * w + ci * h * w + hi * w + wi;
                    let x_hat = (src[idx] - mean) * inv_std;
                    saved_x_hat[idx] = x_hat;
                    data[idx] = gamma_data[ci] * x_hat + beta_data[ci];
                }
            }
        }
    }
    let needs_grad = input.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    if needs_grad {
        Tensor::from_op(
            data,
            &[n, c, h, w],
            Arc::new(BatchNorm2dBackward {
                input: input.clone(),
                gamma: gamma.clone(),
                beta: beta.clone(),
                saved_x_hat,
                saved_inv_std,
                n,
                c,
                h,
                w,
            }),
        )
    } else {
        Tensor::from_vec(data, &[n, c, h, w])
    }
}

// ── mse_loss ──────────────────────────────────────────────────────────────────

struct MseLossBackward {
    pred: Tensor,
    target: f32,
    n: usize,
}
impl GradFn for MseLossBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.pred.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.pred.requires_grad() {
            let pred_data = self.pred.data();
            let scale = 2.0 * g[0] / self.n as f32;
            let d: Vec<f32> = pred_data
                .iter()
                .map(|&v| scale * (v - self.target))
                .collect();
            self.pred.accumulate_grad(&d);
        }
    }
}

/// Mean-squared error loss between a predicted tensor and a scalar target.
///
/// Computes `mean((pred - target)²)` over all elements of `pred`.
/// Returns a scalar `Tensor` with shape `[1]`.
#[must_use]
pub fn mse_loss(pred: &Tensor, target: f32) -> Tensor {
    let data = pred.data();
    let n = data.len();
    let loss: f32 = data.iter().map(|&v| (v - target).powi(2)).sum::<f32>() / n as f32;
    if pred.requires_grad() {
        Tensor::from_op(
            vec![loss],
            &[1],
            Arc::new(MseLossBackward {
                pred: pred.clone(),
                target,
                n,
            }),
        )
    } else {
        Tensor::from_vec(vec![loss], &[1])
    }
}

// ── sum / mean ────────────────────────────────────────────────────────────────

struct SumBackward {
    input: Tensor,
}
impl GradFn for SumBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let d = vec![g[0]; self.input.numel()];
            self.input.accumulate_grad(&d);
        }
    }
}

/// Sums all elements of `x`, returning a scalar tensor with shape `[1]`.
#[must_use]
pub fn sum(x: &Tensor) -> Tensor {
    let s: f32 = x.data().iter().sum();
    if x.requires_grad() {
        Tensor::from_op(vec![s], &[1], Arc::new(SumBackward { input: x.clone() }))
    } else {
        Tensor::from_vec(vec![s], &[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_shape() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], &[3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        // [1,2,3]@[7,9,11] = 1*7+2*9+3*11=58
        assert!((c.data()[0] - 58.0).abs() < 1e-5);
    }

    #[test]
    fn relu_zero_boundary() {
        let x = Tensor::from_vec(vec![-1., 0., 1.], &[1, 3]);
        let y = relu(&x);
        assert_eq!(y.data(), vec![0., 0., 1.]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::from_vec(vec![1., 2., 3., 1., 1., 1.], &[2, 3]);
        let y = softmax(&x);
        let s1: f32 = y.data()[0..3].iter().sum();
        let s2: f32 = y.data()[3..6].iter().sum();
        assert!((s1 - 1.0).abs() < 1e-5);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn conv2d_output_shape() {
        let x = Tensor::from_vec(vec![1.0; 1 * 3 * 8 * 8], &[1, 3, 8, 8]);
        let w = Tensor::from_vec(vec![0.1; 16 * 3 * 3 * 3], &[16, 3, 3, 3]);
        let b = Tensor::zeros(&[16]);
        let y = conv2d(&x, &w, &b, 1);
        assert_eq!(y.shape(), &[1, 16, 8, 8]);
    }
}
