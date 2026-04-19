//! Core [`Tensor`] type and [`GradFn`] trait.
//!
//! A `Tensor` is a reference-counted handle to immutable data plus an optional
//! gradient accumulator. Operations create new tensors and attach a `GradFn`
//! that knows how to propagate gradients back through that operation.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::float_cmp)]

use std::collections::HashSet;
use std::sync::{Arc, Mutex, RwLock};

use rand::Rng;
use rand_distr::StandardNormal;

/// Trait implemented by every backward node in the computation graph.
///
/// Each op that creates a `Tensor` attaches a `GradFn` that, during
/// `backward()`, computes and accumulates gradients into the input tensors.
pub trait GradFn: Send + Sync {
    /// Returns the direct input tensors of this operation (used for
    /// topological sort during `backward()`).
    fn inputs(&self) -> Vec<Tensor>;

    /// Given `grad_output` — the gradient of the loss w.r.t. this op's
    /// output — accumulate the corresponding input gradients.
    fn backward(&self, grad_output: &[f32]);
}

pub(crate) struct TensorData {
    pub data: RwLock<Vec<f32>>,
    pub shape: Vec<usize>,
    /// Accumulated gradient (same length as `data`). Initialised to zeros.
    pub grad: Mutex<Vec<f32>>,
    pub grad_fn: Option<Arc<dyn GradFn>>,
    pub requires_grad: bool,
}

/// A multi-dimensional array of `f32` values with optional autograd support.
///
/// `Tensor` is a thin, cheaply-clonable handle backed by an [`Arc`] (reference
/// count). Cloning a `Tensor` is cheap — both the original and the clone point
/// to the *same* underlying data and gradient storage.
///
/// ## Autograd
///
/// Tensors created by calling `.with_grad()` are *leaf* tensors: they have no
/// `grad_fn` and serve as the learnable parameters of a model. When you call
/// [`Tensor::backward`] on a scalar loss, gradients flow back through the
/// computation graph and are accumulated into every leaf tensor that was
/// involved. Read the accumulated gradient with [`Tensor::grad`], and reset it
/// before the next forward pass with [`Tensor::zero_grad`] (or via the
/// optimizer's `zero_grad`).
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<TensorData>,
}

impl Tensor {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Creates a tensor from flat row-major data and a shape.
    ///
    /// # Panics
    /// Panics if `data.len()` ≠ product of `shape`.
    #[must_use]
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "from_vec: data length {len} ≠ shape product {n}",
            len = data.len()
        );
        Self {
            inner: Arc::new(TensorData {
                grad: Mutex::new(vec![0.0; n]),
                data: RwLock::new(data),
                shape: shape.to_vec(),
                grad_fn: None,
                requires_grad: false,
            }),
        }
    }

    /// Creates a zero tensor of the given shape.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_vec(vec![0.0; n], shape)
    }

    /// Creates a ones tensor of the given shape.
    #[must_use]
    pub fn ones(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self::from_vec(vec![1.0; n], shape)
    }

    /// Creates a tensor filled with the given scalar value.
    #[must_use]
    pub fn full(shape: &[usize], value: f32) -> Self {
        let n: usize = shape.iter().product();
        Self::from_vec(vec![value; n], shape)
    }

    /// Creates a tensor with values drawn from N(0, std²).
    #[must_use]
    pub fn randn(shape: &[usize], std: f32) -> Self {
        let n: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..n)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * std)
            .collect();
        Self::from_vec(data, shape)
    }

    /// Creates a scalar tensor from a single value.
    #[must_use]
    pub fn scalar(val: f32) -> Self {
        Self::from_vec(vec![val], &[1])
    }

    // ── Builder ───────────────────────────────────────────────────────────

    /// Marks this tensor as requiring gradient computation.
    ///
    /// # Panics
    /// Panics if the tensor has already been shared (Arc strong count > 1).
    #[must_use]
    pub fn with_grad(mut self) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("with_grad: tensor is already shared")
            .requires_grad = true;
        self
    }

    // ── Metadata ──────────────────────────────────────────────────────────

    /// Returns the shape of this tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.inner.shape.iter().product()
    }

    /// Returns `true` if this tensor participates in gradient computation.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    // ── Data access ───────────────────────────────────────────────────────

    /// Returns a copy of the underlying flat data.
    #[must_use]
    pub fn data(&self) -> Vec<f32> {
        self.inner
            .data
            .read()
            .expect("data RwLock poisoned")
            .clone()
    }

    /// Returns the single value for a scalar tensor.
    ///
    /// # Panics
    /// Panics if the tensor has more than one element.
    #[must_use]
    pub fn item(&self) -> f32 {
        let data = self.inner.data.read().expect("data RwLock poisoned");
        assert_eq!(
            data.len(),
            1,
            "item(): tensor has {} elements, expected 1",
            data.len()
        );
        data[0]
    }

    // ── Gradient access ───────────────────────────────────────────────────

    /// Returns a copy of the accumulated gradient.
    #[must_use]
    pub fn grad(&self) -> Vec<f32> {
        self.inner.grad.lock().expect("grad Mutex poisoned").clone()
    }

    /// Resets the gradient accumulator to zeros.
    pub fn zero_grad(&self) {
        let mut g = self.inner.grad.lock().expect("grad Mutex poisoned");
        g.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Adds `delta` element-wise to this tensor's gradient accumulator.
    pub(crate) fn accumulate_grad(&self, delta: &[f32]) {
        let mut g = self.inner.grad.lock().expect("grad Mutex poisoned");
        for (gi, di) in g.iter_mut().zip(delta) {
            *gi += di;
        }
    }

    // ── Parameter update (used by optimizer) ─────────────────────────────

    /// Subtracts `delta` element-wise from the data in-place (optimizer step).
    pub(crate) fn update_data(&self, delta: &[f32]) {
        let mut d = self.inner.data.write().expect("data RwLock poisoned");
        for (di, delta_i) in d.iter_mut().zip(delta) {
            *di -= delta_i;
        }
    }

    /// Overwrites the tensor's data in-place.
    ///
    /// # Panics
    /// Panics if `new_data.len()` does not match the tensor's element count.
    pub fn set_data(&self, new_data: &[f32]) {
        let mut d = self.inner.data.write().expect("data RwLock poisoned");
        assert_eq!(
            d.len(),
            new_data.len(),
            "set_data: length mismatch ({} vs {})",
            d.len(),
            new_data.len()
        );
        d.copy_from_slice(new_data);
    }

    // ── Transforms ────────────────────────────────────────────────────────

    /// Returns a 2-D tensor transposed: `[M, N]` → `[N, M]`.
    ///
    /// # Panics
    /// Panics if the tensor is not 2-D.
    #[must_use]
    pub fn t(&self) -> Self {
        let s = self.shape();
        assert_eq!(s.len(), 2, "t(): requires 2-D tensor, got {}D", s.len());
        let (m, n) = (s[0], s[1]);
        let src = self.data();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = src[i * n + j];
            }
        }
        if self.requires_grad() {
            Self::from_op(out, &[n, m], Arc::new(TBackward { input: self.clone(), m, n }))
        } else {
            Self::from_vec(out, &[n, m])
        }
    }

    /// Returns a view with a different shape (same total elements).
    ///
    /// # Panics
    /// Panics if the new shape has a different number of elements.
    #[must_use]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        let new_n: usize = shape.iter().product();
        assert_eq!(
            self.numel(),
            new_n,
            "reshape: {old} → {new_n} element mismatch",
            old = self.numel()
        );
        if self.requires_grad() {
            Self::from_op(self.data(), shape, Arc::new(ReshapeBackward { input: self.clone() }))
        } else {
            Self::from_vec(self.data(), shape)
        }
    }

    /// Extracts row `i` from a 2-D tensor as a 1-D tensor `[N]`.
    ///
    /// # Panics
    /// Panics if the tensor is not 2-D or `i` is out of bounds.
    #[must_use]
    pub fn row(&self, i: usize) -> Self {
        let s = self.shape();
        assert_eq!(s.len(), 2, "row(): requires 2-D tensor");
        let n = s[1];
        let data = self.data();
        Self::from_vec(data[i * n..(i + 1) * n].to_vec(), &[n])
    }

    // ── Autograd ─────────────────────────────────────────────────────────

    /// Returns the `GradFn` attached to this tensor, if any.
    #[must_use]
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFn>> {
        self.inner.grad_fn.as_ref()
    }

    /// Runs the backward pass from this scalar tensor.
    ///
    /// Seeds `self.grad = [1.0]`, then propagates gradients through the
    /// computation graph in reverse topological order.
    ///
    /// # Panics
    /// Panics if `self` is not a scalar (numel ≠ 1).
    pub fn backward(&self) {
        assert_eq!(
            self.numel(),
            1,
            "backward(): expected scalar, got {} elements",
            self.numel()
        );
        // Seed the loss gradient.
        {
            let mut g = self.inner.grad.lock().expect("grad Mutex poisoned");
            g[0] = 1.0;
        }
        // Topological sort: loss is first, leaves are last.
        let order = topo_sort(self);
        for t in &order {
            if let Some(gfn) = &t.inner.grad_fn {
                let grad = t.inner.grad.lock().expect("grad Mutex poisoned").clone();
                gfn.backward(&grad);
            }
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Creates a new tensor with the provided data, shape, and `GradFn`.
    pub(crate) fn from_op(data: Vec<f32>, shape: &[usize], grad_fn: Arc<dyn GradFn>) -> Self {
        let n = data.len();
        Self {
            inner: Arc::new(TensorData {
                data: RwLock::new(data),
                shape: shape.to_vec(),
                grad: Mutex::new(vec![0.0; n]),
                grad_fn: Some(grad_fn),
                requires_grad: true,
            }),
        }
    }
}

// ── Grad-fn nodes for shape ops ──────────────────────────────────────────────

struct ReshapeBackward {
    input: Tensor,
}

impl GradFn for ReshapeBackward {
    fn inputs(&self) -> Vec<Tensor> { vec![self.input.clone()] }
    fn backward(&self, grad_output: &[f32]) {
        self.input.accumulate_grad(grad_output);
    }
}

struct TBackward {
    input: Tensor,
    m: usize,
    n: usize,
}

impl GradFn for TBackward {
    fn inputs(&self) -> Vec<Tensor> { vec![self.input.clone()] }
    fn backward(&self, grad_output: &[f32]) {
        // grad_output is [N, M]; transpose back to [M, N]
        let (m, n) = (self.m, self.n);
        let mut g = vec![0.0f32; m * n];
        for i in 0..n {
            for j in 0..m {
                g[j * n + i] = grad_output[i * m + j];
            }
        }
        self.input.accumulate_grad(&g);
    }
}

// ── Topological sort ──────────────────────────────────────────────────────────

/// Returns tensors in reverse topological order (root/loss first, leaves last).
fn topo_sort(root: &Tensor) -> Vec<Tensor> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut order: Vec<Tensor> = Vec::new();
    dfs(root, &mut visited, &mut order);
    order.reverse();
    order
}

fn dfs(t: &Tensor, visited: &mut HashSet<usize>, order: &mut Vec<Tensor>) {
    let id = Arc::as_ptr(&t.inner) as usize;
    if !visited.insert(id) {
        return;
    }
    if let Some(gfn) = &t.inner.grad_fn {
        for inp in gfn.inputs() {
            dfs(&inp, visited, order);
        }
    }
    order.push(t.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_shape() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.numel(), 12);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn transpose_2d() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn backward_seeds_grad() {
        let t = Tensor::scalar(3.0).with_grad();
        t.backward();
        assert_eq!(t.grad()[0], 1.0);
    }
}
