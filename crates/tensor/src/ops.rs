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

use rand::Rng;
use rayon::prelude::*;

use crate::{
    gpu,
    tensor_impl::{GradFn, Tensor},
};

// ── Internal CPU math helpers ──────────────────────────────────────��──────────

/// C = A @ B where A is [m×k] and B is [k×n].
///
/// Uses `ikj` loop order for sequential memory access on both B and C rows,
/// and parallelises over output rows with rayon.
fn cpu_matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for l in 0..k {
            let a_il = a[i * k + l];
            for j in 0..n {
                row[j] += a_il * b[l * n + j];
            }
        }
    });
    c
}

/// C = A @ B^T where A is [m×k] and B is [n×k] (B rows are the "right" vectors).
///
/// Avoids materialising the transpose: each output element is a row-dot-row product,
/// giving sequential memory access on both A and B rows.
fn cpu_matmul_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut s = 0.0f32;
            for l in 0..k {
                s += a_row[l] * b_row[l];
            }
            row[j] = s;
        }
    });
    c
}

/// Metal GPU dispatch has ~50–100 µs of fixed overhead (command encoding, submit,
/// poll-to-completion, staging buffer map). At fp32 peak ~10 TFLOPs on Apple Silicon
/// that means ≥ 1 M FLOPs are needed to break even. The old threshold of 8 192 FLOPs
/// caused the GPU path to be *slower* than CPU for almost every call in practice.
const GPU_MATMUL_FLOP_THRESHOLD: usize = 1_048_576; // 1 M

/// Element-wise / reduction GPU threshold (bytes rather than FLOPs).
/// 256 K f32 elements = 1 MiB; below this the rayon CPU path is faster.
const GPU_EW_ELEM_THRESHOLD: usize = 262_144; // 256 K

/// Dispatching wrapper: uses GPU when the FLOP count justifies transfer overhead.
fn matmul_2d(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    if m * k * n > GPU_MATMUL_FLOP_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            return g.matmul(a, b, m, k, n);
        }
    }
    cpu_matmul_2d(a, b, m, k, n)
}

/// A @ B^T dispatch — avoids materialising the transpose on the CPU path.
///
/// For the GPU path the shader still needs a contiguous `[k×n]` layout, so the
/// transpose is materialised only when a GPU is actually used.
fn matmul_2d_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    if m * k * n > GPU_MATMUL_FLOP_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            let bt = transpose_2d(b, n, k); // [k×n]
            return g.matmul(a, &bt, m, k, n);
        }
    }
    cpu_matmul_nt(a, b, m, k, n)
}

/// Transpose a 2-D matrix [m×n] → [n×m].
fn transpose_2d(a: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    if m * n > 65_536 {
        // Parallel: each output row (one column of input) is independent.
        out.par_chunks_mut(m).enumerate().for_each(|(j, row)| {
            for i in 0..m {
                row[i] = a[i * n + j];
            }
        });
    } else {
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = a[i * n + j];
            }
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
    let ad = a.data();
    let bd = b.data();
    let data: Vec<f32> = if ad.len() > GPU_EW_ELEM_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            g.elementwise(&ad, &bd, 2)
        } else {
            ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect()
        }
    } else {
        ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect()
    };
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
    let ad = a.data();
    let bd = b.data();
    let data: Vec<f32> = if ad.len() > GPU_EW_ELEM_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            g.elementwise(&ad, &bd, 3)
        } else {
            ad.iter().zip(bd.iter()).map(|(x, y)| x - y).collect()
        }
    } else {
        ad.iter().zip(bd.iter()).map(|(x, y)| x - y).collect()
    };
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
    /// A's data saved at forward time — avoids an `RwLock` read + Vec clone during backward.
    a_data: Vec<f32>,
    /// B's data saved at forward time.
    b_data: Vec<f32>,
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
            let bt = transpose_2d(&self.b_data, k, n); // [N, K]
            let da = matmul_2d(g, &bt, m, n, k); // [M,N]@[N,K]=[M,K]
            self.a.accumulate_grad(&da);
        }
        if self.b.requires_grad() {
            let at = transpose_2d(&self.a_data, m, k); // [K, M]
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
    let a_data = a.data();
    let b_data = b.data();
    let data = matmul_2d(&a_data, &b_data, m, k, n);
    if a.requires_grad() || b.requires_grad() {
        Tensor::from_op(
            data,
            &[m, n],
            Arc::new(MatMulBackward {
                a: a.clone(),
                b: b.clone(),
                a_data,
                b_data,
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
    /// Input activations saved at forward time — avoids `RwLock` reads during backward.
    input_data: Vec<f32>,
    /// Weight data saved at forward time.
    weight_data: Vec<f32>,
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
            let d = matmul_2d(g, &self.weight_data, s, out_f, in_f);
            self.input.accumulate_grad(&d);
        }
        if self.weight.requires_grad() {
            // d_weight = g.T @ input  ([out,S] @ [S,in] = [out,in])
            let gt = transpose_2d(g, s, out_f);
            let d = matmul_2d(&gt, &self.input_data, out_f, s, in_f);
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
/// Gradients flow to `input`, `weight`, and `bias` during the backward pass.
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
    let input_data = input.data();
    let weight_data = weight.data();
    // matmul_2d_nt computes input @ weight^T without materialising the transpose on CPU.
    let raw = matmul_2d_nt(&input_data, &weight_data, s, in_f, out_f);
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
                input_data,
                weight_data,
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
    let data: Vec<f32> = if src.len() > GPU_EW_ELEM_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            g.elementwise(&src, &[], 0)
        } else {
            src.iter().map(|&v| v.max(0.0)).collect()
        }
    } else {
        src.iter().map(|&v| v.max(0.0)).collect()
    };
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
    let data: Vec<f32> = if src.len() > GPU_EW_ELEM_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            g.elementwise(&src, &[], 1)
        } else {
            src.iter().map(|&v| gelu_fwd(v)).collect()
        }
    } else {
        src.iter().map(|&v| gelu_fwd(v)).collect()
    };
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
/// Output shape equals input shape.
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
    let mut data = softmax_rows(&src, s, d);
    if s * d > GPU_EW_ELEM_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            data = g.softmax(&src, s, d);
        }
    }
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
    // Optionally replace the CPU-computed output with GPU output.
    if s * d > GPU_EW_ELEM_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            data = g.layer_norm(&src, &gamma_data, &beta_data, eps, s, d);
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

// ── slice_cols ────────────────────────────────────────────────────────────────

struct SliceColsBackward {
    input: Tensor,
    start: usize,
    rows: usize,
    ncols: usize,
    total_cols: usize,
}
impl GradFn for SliceColsBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            self.input
                .accumulate_grad_cols(g, self.rows, self.start, self.ncols, self.total_cols);
        }
    }
}

/// Slices columns `[start, end)` from a 2-D tensor: `[M, N] → [M, end-start]`.
///
/// # Panics
/// Panics if the tensor is not 2-D or the range is out of bounds.
#[must_use]
pub fn slice_cols(x: &Tensor, start: usize, end: usize) -> Tensor {
    let shape = x.shape();
    assert_eq!(
        shape.len(),
        2,
        "slice_cols: expected 2-D tensor, got {shape:?}"
    );
    let (rows, total_cols) = (shape[0], shape[1]);
    assert!(
        end <= total_cols && start < end,
        "slice_cols: invalid range {start}..{end} for {total_cols} cols"
    );
    let ncols = end - start;
    let src = x.data();
    let mut out = Vec::with_capacity(rows * ncols);
    for i in 0..rows {
        out.extend_from_slice(&src[i * total_cols + start..i * total_cols + end]);
    }
    if x.requires_grad() {
        Tensor::from_op(
            out,
            &[rows, ncols],
            Arc::new(SliceColsBackward {
                input: x.clone(),
                start,
                rows,
                ncols,
                total_cols,
            }),
        )
    } else {
        Tensor::from_vec(out, &[rows, ncols])
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
    let item = col_rows * col_cols;
    let mut cols = vec![0.0f32; n * item];
    // Parallel over batch items — each writes to a non-overlapping slice.
    cols.par_chunks_mut(item)
        .enumerate()
        .for_each(|(ni, chunk)| {
            let h_i = h as isize;
            let w_i = w as isize;
            let pad_i = pad as isize;
            for ci in 0..c_in {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let col_col = ci * kh * kw + ki * kw + kj;
                        for hi in 0..h_out {
                            let src_h = hi as isize + ki as isize - pad_i;
                            for wi in 0..w_out {
                                let src_w = wi as isize + kj as isize - pad_i;
                                let val = if src_h >= 0 && src_h < h_i && src_w >= 0 && src_w < w_i
                                {
                                    input[ni * c_in * h * w
                                        + ci * h * w
                                        + src_h as usize * w
                                        + src_w as usize]
                                } else {
                                    0.0
                                };
                                chunk[(hi * w_out + wi) * col_cols + col_col] = val;
                            }
                        }
                    }
                }
            }
        });
    (cols, h_out, w_out)
}

/// col2im: maps column gradients back to input gradient `[N, C_in, H, W]`.
///
/// Uses a gather pattern (each output pixel independently accumulates its kH×kW
/// contributions from `cols`) so the computation is parallel with no write conflicts.
fn col2im(
    cols: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    pad: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<f32> {
    let col_cols = c_in * kh * kw;
    let item = c_in * h * w;
    let mut d_input = vec![0.0f32; n * item];
    // Parallel over batch items — each writes to a non-overlapping slice.
    d_input
        .par_chunks_mut(item)
        .enumerate()
        .for_each(|(ni, chunk)| {
            let pad_i = pad as isize;
            let h_out_i = h_out as isize;
            let w_out_i = w_out as isize;
            let cols_n = &cols[ni * h_out * w_out * col_cols..];
            for ci in 0..c_in {
                for src_h in 0..h {
                    for src_w in 0..w {
                        let mut acc = 0.0f32;
                        for ki in 0..kh {
                            let hi = src_h as isize + pad_i - ki as isize;
                            if hi < 0 || hi >= h_out_i {
                                continue;
                            }
                            for kj in 0..kw {
                                let wi = src_w as isize + pad_i - kj as isize;
                                if wi < 0 || wi >= w_out_i {
                                    continue;
                                }
                                let col_row = hi as usize * w_out + wi as usize;
                                let col_col = ci * kh * kw + ki * kw + kj;
                                acc += cols_n[col_row * col_cols + col_col];
                            }
                        }
                        chunk[ci * h * w + src_h * w + src_w] = acc;
                    }
                }
            }
        });
    d_input
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
        // Parallel over batch items — each writes to a non-overlapping slice.
        g_mat
            .par_chunks_mut(col_rows * c_out)
            .enumerate()
            .for_each(|(ni, chunk)| {
                for co in 0..c_out {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            chunk[(hi * w_out + wi) * c_out + co] = g
                                [ni * c_out * h_out * w_out + co * h_out * w_out + hi * w_out + wi];
                        }
                    }
                }
            });
        // weight: [C_out, C_in*kH*kW]
        let w_data = self.weight.data();
        if self.weight.requires_grad() {
            // d_weight = cols.T @ g_mat  →  [col_cols, N*HW].T @ [N*HW, C_out] …
            // cols: [N*HW, col_cols],  g_mat: [N*HW, C_out]
            // cols.T: [col_cols, N*HW]
            let cols_t = transpose_2d(&self.saved_cols, n * col_rows, col_cols);
            let dw = matmul_2d(&cols_t, &g_mat, col_cols, n * col_rows, c_out);
            // dw: [col_cols, c_out] → transpose to [c_out, col_cols]
            let dw_t = transpose_2d(&dw, col_cols, c_out);
            self.weight.accumulate_grad(&dw_t);
        }
        if self.bias.requires_grad() {
            // d_bias[co] = sum over N, H_out, W_out of g[N, co, H_out, W_out]
            // Parallel over channels — each co accumulates independently.
            let mut db = vec![0.0f32; c_out];
            db.par_iter_mut().enumerate().for_each(|(co, db_co)| {
                for ni in 0..n {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            *db_co += g
                                [ni * c_out * h_out * w_out + co * h_out * w_out + hi * w_out + wi];
                        }
                    }
                }
            });
            self.bias.accumulate_grad(&db);
        }
        if self.input.requires_grad() {
            // d_cols = g_mat @ weight  ([N*HW, C_out] @ [C_out, col_cols] = [N*HW, col_cols])
            let d_cols = matmul_2d(&g_mat, &w_data, n * col_rows, c_out, col_cols);
            let d_input = col2im(
                &d_cols, n, self.c_in, self.h, self.w, self.kh, self.kw, self.pad, h_out, w_out,
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
    let col_cols = c_in * kh * kw;
    let col_rows = h_out * w_out;
    // Single matmul over the full batch:
    // [N*HW, col_cols] @ [col_cols, C_out] = [N*HW, C_out]
    let wt = transpose_2d(&weight_data, c_out, col_cols);
    let out_mat = matmul_2d(&cols, &wt, n * col_rows, col_cols, c_out);
    // Reorder [N, H_out*W_out, C_out] → [N, C_out, H_out, W_out] and add bias.
    // Reorder [N, H_out*W_out, C_out] → [N, C_out, H_out, W_out] and add bias.
    // Parallel over batch items — each writes to a non-overlapping slice.
    let mut data = vec![0.0f32; n * c_out * h_out * w_out];
    data.par_chunks_mut(c_out * h_out * w_out)
        .enumerate()
        .for_each(|(ni, chunk)| {
            for co in 0..c_out {
                for hi in 0..h_out {
                    for wi in 0..w_out {
                        chunk[co * h_out * w_out + hi * w_out + wi] = out_mat
                            [ni * col_rows * c_out + (hi * w_out + wi) * c_out + co]
                            + bias_data[co];
                    }
                }
            }
        });
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

// ── stack ─────────────────────────────────────────────────────────────────────

struct StackBackward {
    inputs: Vec<Tensor>,
    item_size: usize,
}
impl GradFn for StackBackward {
    fn inputs(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }
    fn backward(&self, g: &[f32]) {
        for (i, t) in self.inputs.iter().enumerate() {
            if t.requires_grad() {
                t.accumulate_grad(&g[i * self.item_size..(i + 1) * self.item_size]);
            }
        }
    }
}

/// Stacks equal-shaped tensors into a new leading batch dimension.
///
/// B tensors of shape `[...]` → `[B, ...]`.
///
/// # Panics
/// Panics if the input list is empty or shapes differ.
#[must_use]
pub fn stack(tensors: &[&Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "stack: empty input");
    let item_shape = tensors[0].shape().to_vec();
    let item_size: usize = item_shape.iter().product();
    let b = tensors.len();
    let mut out_shape = vec![b];
    out_shape.extend_from_slice(&item_shape);
    let mut data = Vec::with_capacity(b * item_size);
    for t in tensors {
        assert_eq!(t.shape(), item_shape.as_slice(), "stack: shape mismatch");
        data.extend_from_slice(&t.data());
    }
    let needs_grad = tensors.iter().any(|t| t.requires_grad());
    let inputs: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    if needs_grad {
        Tensor::from_op(
            data,
            &out_shape,
            Arc::new(StackBackward { inputs, item_size }),
        )
    } else {
        Tensor::from_vec(data, &out_shape)
    }
}

// ── transpose_last_two ────────────────────────────────────────────────────────

struct TransposeLastTwoBackward {
    input: Tensor,
    b: usize,
    m: usize,
    n: usize,
}
impl GradFn for TransposeLastTwoBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let (b, m, n) = (self.b, self.m, self.n);
            // grad is [B, N, M]; rotate back to [B, M, N]
            let mut d = vec![0.0f32; b * m * n];
            for bi in 0..b {
                for i in 0..m {
                    for j in 0..n {
                        d[bi * m * n + i * n + j] = g[bi * n * m + j * m + i];
                    }
                }
            }
            self.input.accumulate_grad(&d);
        }
    }
}

/// Transposes the last two dimensions of a 3-D tensor: `[B, M, N] → [B, N, M]`.
///
/// # Panics
/// Panics if the tensor is not 3-D.
#[must_use]
pub fn transpose_last_two(x: &Tensor) -> Tensor {
    let s = x.shape();
    assert_eq!(
        s.len(),
        3,
        "transpose_last_two: expected 3-D tensor, got {s:?}"
    );
    let (b, m, n) = (s[0], s[1], s[2]);
    let src = x.data();
    let mut out = vec![0.0f32; b * m * n];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                out[bi * n * m + j * m + i] = src[bi * m * n + i * n + j];
            }
        }
    }
    if x.requires_grad() {
        Tensor::from_op(
            out,
            &[b, n, m],
            Arc::new(TransposeLastTwoBackward {
                input: x.clone(),
                b,
                m,
                n,
            }),
        )
    } else {
        Tensor::from_vec(out, &[b, n, m])
    }
}

// ── matmul_batched ────────────────────────────────────────────────────────────

struct MatMulBatchedBackward {
    a: Tensor,
    b: Tensor,
    /// A's data saved at forward time.
    a_data: Vec<f32>,
    /// B's data saved at forward time.
    b_data: Vec<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
}
impl GradFn for MatMulBatchedBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
    fn backward(&self, g: &[f32]) {
        let (batch, m, k, n) = (self.batch, self.m, self.k, self.n);
        let mut da = vec![0.0f32; batch * m * k];
        let mut db = vec![0.0f32; batch * k * n];
        for bi in 0..batch {
            let g_b = &g[bi * m * n..(bi + 1) * m * n];
            if self.a.requires_grad() {
                let b_b = &self.b_data[bi * k * n..(bi + 1) * k * n];
                let bt = transpose_2d(b_b, k, n);
                let d = cpu_matmul_2d(g_b, &bt, m, n, k);
                da[bi * m * k..(bi + 1) * m * k].copy_from_slice(&d);
            }
            if self.b.requires_grad() {
                let a_b = &self.a_data[bi * m * k..(bi + 1) * m * k];
                let at = transpose_2d(a_b, m, k);
                let d = cpu_matmul_2d(&at, g_b, k, m, n);
                db[bi * k * n..(bi + 1) * k * n].copy_from_slice(&d);
            }
        }
        if self.a.requires_grad() {
            self.a.accumulate_grad(&da);
        }
        if self.b.requires_grad() {
            self.b.accumulate_grad(&db);
        }
    }
}

/// Batched matrix multiply: `[B, M, K] × [B, K, N] → [B, M, N]`.
///
/// # Panics
/// Panics if `a` or `b` are not 3-D, or their dimensions are inconsistent.
#[must_use]
pub fn matmul_batched(a: &Tensor, b: &Tensor) -> Tensor {
    let sa = a.shape();
    let sb = b.shape();
    assert_eq!(sa.len(), 3, "matmul_batched: a must be 3-D, got {sa:?}");
    assert_eq!(sb.len(), 3, "matmul_batched: b must be 3-D, got {sb:?}");
    assert_eq!(
        sa[0], sb[0],
        "matmul_batched: batch mismatch {sa:?} vs {sb:?}"
    );
    assert_eq!(
        sa[2], sb[1],
        "matmul_batched: inner dim mismatch {sa:?} vs {sb:?}"
    );
    let (batch, m, k, n) = (sa[0], sa[1], sa[2], sb[2]);
    let a_data = a.data();
    let b_data = b.data();
    let data = if batch * m * k * n > GPU_MATMUL_FLOP_THRESHOLD {
        if let Some(g) = gpu::global_gpu() {
            g.matmul_batched(&a_data, &b_data, batch, m, k, n)
        } else {
            let mut c = vec![0.0f32; batch * m * n];
            for bi in 0..batch {
                let a_b = &a_data[bi * m * k..(bi + 1) * m * k];
                let b_b = &b_data[bi * k * n..(bi + 1) * k * n];
                let c_b = cpu_matmul_2d(a_b, b_b, m, k, n);
                c[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c_b);
            }
            c
        }
    } else {
        let mut c = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            let a_b = &a_data[bi * m * k..(bi + 1) * m * k];
            let b_b = &b_data[bi * k * n..(bi + 1) * k * n];
            let c_b = cpu_matmul_2d(a_b, b_b, m, k, n);
            c[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c_b);
        }
        c
    };
    if a.requires_grad() || b.requires_grad() {
        Tensor::from_op(
            data,
            &[batch, m, n],
            Arc::new(MatMulBatchedBackward {
                a: a.clone(),
                b: b.clone(),
                a_data,
                b_data,
                batch,
                m,
                k,
                n,
            }),
        )
    } else {
        Tensor::from_vec(data, &[batch, m, n])
    }
}

// ── slice_batch ───────────────────────────────────────────────────────────────

struct SliceBatchBackward {
    input: Tensor,
    idx: usize,
    item_size: usize,
    batch: usize,
}
impl GradFn for SliceBatchBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let mut d = vec![0.0f32; self.batch * self.item_size];
            d[self.idx * self.item_size..(self.idx + 1) * self.item_size].copy_from_slice(g);
            self.input.accumulate_grad(&d);
        }
    }
}

/// Extracts batch item `idx` from a tensor with shape `[B, ...]`, returning `[...]`.
///
/// Preserves gradient flow back to the original batch tensor.
///
/// # Panics
/// Panics if the tensor has fewer than 1 dimension or `idx` is out of bounds.
#[must_use]
pub fn slice_batch(x: &Tensor, idx: usize) -> Tensor {
    let s = x.shape();
    assert!(
        !s.is_empty(),
        "slice_batch: tensor must have at least 1 dimension"
    );
    let batch = s[0];
    assert!(
        idx < batch,
        "slice_batch: idx {idx} out of bounds for batch {batch}"
    );
    let item_shape: Vec<usize> = s[1..].to_vec();
    let item_size: usize = item_shape.iter().product();
    let data = x.data()[idx * item_size..(idx + 1) * item_size].to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &item_shape,
            Arc::new(SliceBatchBackward {
                input: x.clone(),
                idx,
                item_size,
                batch,
            }),
        )
    } else {
        Tensor::from_vec(data, &item_shape)
    }
}

// ── prepend_cls_batched ───────────────────────────────────────────────────────

struct PrependClsBatchedBackward {
    cls: Tensor,
    x: Tensor,
    b: usize,
    s: usize,
    d: usize,
}
impl GradFn for PrependClsBatchedBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.cls.clone(), self.x.clone()]
    }
    fn backward(&self, g: &[f32]) {
        let (b, s, d) = (self.b, self.s, self.d);
        // grad_cls: sum over B of the first D elements of each (S+1)*D chunk.
        if self.cls.requires_grad() {
            let mut d_cls = vec![0.0f32; d];
            for bi in 0..b {
                for j in 0..d {
                    d_cls[j] += g[bi * (s + 1) * d + j];
                }
            }
            self.cls.accumulate_grad(&d_cls);
        }
        // grad_x: elements after the first D of each (S+1)*D chunk.
        if self.x.requires_grad() {
            let mut d_x = vec![0.0f32; b * s * d];
            for bi in 0..b {
                let src = &g[bi * (s + 1) * d + d..(bi + 1) * (s + 1) * d];
                d_x[bi * s * d..(bi + 1) * s * d].copy_from_slice(src);
            }
            self.x.accumulate_grad(&d_x);
        }
    }
}

/// Prepends a CLS token to every sequence in a batch.
///
/// - `cls`: `[1, D]` — the learnable CLS token.
/// - `x`:   `[B, S, D]` — the batch of `S`-length sequences.
/// - Output: `[B, S+1, D]` — CLS prepended at position 0 for every item.
///
/// No per-item loop: the output is filled with a single parallel pass over batch items.
///
/// # Panics
/// Panics if `cls` is not `[1, D]` or `x` is not `[B, S, D]`, or their `D` dims differ.
#[must_use]
pub fn prepend_cls_batched(cls: &Tensor, x: &Tensor) -> Tensor {
    let sc = cls.shape();
    let sx = x.shape();
    assert_eq!(sc.len(), 2, "prepend_cls_batched: cls must be 2-D [1, D]");
    assert_eq!(sx.len(), 3, "prepend_cls_batched: x must be 3-D [B, S, D]");
    assert_eq!(sc[0], 1, "prepend_cls_batched: cls first dim must be 1");
    assert_eq!(sc[1], sx[2], "prepend_cls_batched: cls D must equal x D");
    let (b, s, d) = (sx[0], sx[1], sx[2]);
    let cls_data = cls.data();
    let x_data = x.data();
    let mut data = vec![0.0f32; b * (s + 1) * d];
    data.par_chunks_mut((s + 1) * d)
        .enumerate()
        .for_each(|(bi, chunk)| {
            chunk[..d].copy_from_slice(&cls_data);
            chunk[d..].copy_from_slice(&x_data[bi * s * d..(bi + 1) * s * d]);
        });
    if cls.requires_grad() || x.requires_grad() {
        Tensor::from_op(
            data,
            &[b, s + 1, d],
            Arc::new(PrependClsBatchedBackward {
                cls: cls.clone(),
                x: x.clone(),
                b,
                s,
                d,
            }),
        )
    } else {
        Tensor::from_vec(data, &[b, s + 1, d])
    }
}

// ── broadcast_add_batch ───────────────────────────────────────────────────────

struct BroadcastAddBatchBackward {
    x: Tensor,
    y: Tensor,
    b: usize,
    s: usize,
    d: usize,
}
impl GradFn for BroadcastAddBatchBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.x.clone(), self.y.clone()]
    }
    fn backward(&self, g: &[f32]) {
        let (b, s, d) = (self.b, self.s, self.d);
        // grad_x = grad_out (same shape).
        if self.x.requires_grad() {
            self.x.accumulate_grad(g);
        }
        // grad_y = sum over B of grad_out → shape [S, D].
        if self.y.requires_grad() {
            let mut d_y = vec![0.0f32; s * d];
            for bi in 0..b {
                for i in 0..s * d {
                    d_y[i] += g[bi * s * d + i];
                }
            }
            self.y.accumulate_grad(&d_y);
        }
    }
}

/// Adds a `[S, D]` tensor to every item in a `[B, S, D]` batch (broadcast over B).
///
/// Equivalent to `x[b] + y` for each `b`, but fully vectorised with a single
/// parallel pass — no per-item loop.
///
/// # Panics
/// Panics if `x` is not `[B, S, D]` or `y` is not `[S, D]`, or `S`/`D` differ.
#[must_use]
pub fn broadcast_add_batch(x: &Tensor, y: &Tensor) -> Tensor {
    let sx = x.shape();
    let sy = y.shape();
    assert_eq!(sx.len(), 3, "broadcast_add_batch: x must be 3-D [B, S, D]");
    assert_eq!(sy.len(), 2, "broadcast_add_batch: y must be 2-D [S, D]");
    assert_eq!(sx[1], sy[0], "broadcast_add_batch: S mismatch");
    assert_eq!(sx[2], sy[1], "broadcast_add_batch: D mismatch");
    let (b, s, d) = (sx[0], sx[1], sx[2]);
    let x_data = x.data();
    let y_data = y.data();
    let mut data = vec![0.0f32; b * s * d];
    data.par_chunks_mut(s * d)
        .enumerate()
        .for_each(|(bi, chunk)| {
            let x_item = &x_data[bi * s * d..(bi + 1) * s * d];
            for i in 0..s * d {
                chunk[i] = x_item[i] + y_data[i];
            }
        });
    if x.requires_grad() || y.requires_grad() {
        Tensor::from_op(
            data,
            &[b, s, d],
            Arc::new(BroadcastAddBatchBackward {
                x: x.clone(),
                y: y.clone(),
                b,
                s,
                d,
            }),
        )
    } else {
        Tensor::from_vec(data, &[b, s, d])
    }
}

// ── select_token ──────────────────────────────────────────────────────────────

struct SelectTokenBackward {
    input: Tensor,
    b: usize,
    s: usize,
    d: usize,
    idx: usize,
}
impl GradFn for SelectTokenBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let (b, s, d, idx) = (self.b, self.s, self.d, self.idx);
            let mut d_x = vec![0.0f32; b * s * d];
            for bi in 0..b {
                let src = &g[bi * d..(bi + 1) * d];
                let dst = &mut d_x[bi * s * d + idx * d..bi * s * d + (idx + 1) * d];
                dst.copy_from_slice(src);
            }
            self.input.accumulate_grad(&d_x);
        }
    }
}

/// Extracts one token from every sequence in a `[B, S, D]` batch.
///
/// Returns `[B, D]`: `output[b] = x[b, idx, :]`.
///
/// Fully vectorised — no per-item loop.
///
/// # Panics
/// Panics if `x` is not 3-D or `idx` is out of bounds.
#[must_use]
pub fn select_token(x: &Tensor, idx: usize) -> Tensor {
    let s = x.shape();
    assert_eq!(s.len(), 3, "select_token: x must be 3-D [B, S, D]");
    let (b, seq, d) = (s[0], s[1], s[2]);
    assert!(
        idx < seq,
        "select_token: idx {idx} out of bounds for S={seq}"
    );
    let x_data = x.data();
    let mut data = vec![0.0f32; b * d];
    data.par_chunks_mut(d).enumerate().for_each(|(bi, chunk)| {
        chunk.copy_from_slice(&x_data[bi * seq * d + idx * d..bi * seq * d + (idx + 1) * d]);
    });
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &[b, d],
            Arc::new(SelectTokenBackward {
                input: x.clone(),
                b,
                s: seq,
                d,
                idx,
            }),
        )
    } else {
        Tensor::from_vec(data, &[b, d])
    }
}

// ── mse_loss_tensor ───────────────────────────────────────────────────────────

struct MseLossTensorBackward {
    pred: Tensor,
    target_data: Vec<f32>,
    n: usize,
}
impl GradFn for MseLossTensorBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.pred.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.pred.requires_grad() {
            let scale = 2.0 * g[0] / self.n as f32;
            let pred_data = self.pred.data();
            let d: Vec<f32> = pred_data
                .iter()
                .zip(self.target_data.iter())
                .map(|(&p, &t)| scale * (p - t))
                .collect();
            self.pred.accumulate_grad(&d);
        }
    }
}

/// Mean-squared error between two tensors of identical shape.
///
/// Returns a scalar `[1]` tensor: `mean((pred - target)²)`.
///
/// # Panics
/// Panics if shapes differ.
#[must_use]
pub fn mse_loss_tensor(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        pred.shape(),
        target.shape(),
        "mse_loss_tensor: shape mismatch"
    );
    let pred_data = pred.data();
    let target_data = target.data();
    let n = pred_data.len();
    let loss: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / n as f32;
    if pred.requires_grad() {
        Tensor::from_op(
            vec![loss],
            &[1],
            Arc::new(MseLossTensorBackward {
                pred: pred.clone(),
                target_data,
                n,
            }),
        )
    } else {
        Tensor::from_vec(vec![loss], &[1])
    }
}

// ── dropout ───────────────────────────────────────────────────────────────────

struct DropoutBackward {
    input: Tensor,
    mask: Vec<f32>,
}
impl GradFn for DropoutBackward {
    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
    fn backward(&self, g: &[f32]) {
        if self.input.requires_grad() {
            let d: Vec<f32> = g
                .iter()
                .zip(self.mask.iter())
                .map(|(gv, m)| gv * m)
                .collect();
            self.input.accumulate_grad(&d);
        }
    }
}

/// Applies inverted dropout to `x` during training.
///
/// - When `training` is `false` (or `p == 0.0`), returns `x` unchanged.
/// - When training, each element is independently zeroed with probability `p`
///   and scaled by `1 / (1 - p)` to preserve expected activation magnitude
///   (inverted / "corrected" dropout).
///
/// Output shape equals input shape.
#[must_use]
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    if !training || p == 0.0 {
        return x.clone();
    }
    let scale = 1.0 / (1.0 - p);
    let mut rng = rand::thread_rng();
    let mask: Vec<f32> = x
        .data()
        .iter()
        .map(|_| if rng.gen::<f32>() < p { 0.0 } else { scale })
        .collect();
    let data: Vec<f32> = x
        .data()
        .iter()
        .zip(mask.iter())
        .map(|(v, m)| v * m)
        .collect();
    let shape = x.shape().to_vec();
    if x.requires_grad() {
        Tensor::from_op(
            data,
            &shape,
            Arc::new(DropoutBackward {
                input: x.clone(),
                mask,
            }),
        )
    } else {
        Tensor::from_vec(data, &shape)
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
