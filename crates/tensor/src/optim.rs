//! Adam optimiser.
//!
//! Maintains per-parameter first and second moment estimates and applies
//! the bias-corrected update rule from Kingma & Ba (2015).

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use crate::Tensor;

/// Adam optimizer with β₁=0.9, β₂=0.999, ε=1e-8.
///
/// Maintains per-parameter first and second moment estimates that adapt the
/// effective learning rate. Suitable for sparse gradients and non-stationary objectives.
///
/// # Typical usage
///
/// ```ignore
/// let mut adam = Adam::new(model.parameters(), 1e-4);
/// // each training iteration:
/// adam.zero_grad();
/// loss.backward();
/// adam.step();
/// ```
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<Vec<f32>>, // first moment  (same shape as each param)
    v: Vec<Vec<f32>>, // second moment (same shape as each param)
    t: usize,         // global step counter
}

impl Adam {
    /// Creates an Adam optimiser for the given parameters.
    #[must_use]
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m,
            v,
            t: 0,
        }
    }

    /// Applies one Adam update step using the accumulated `.grad` on each parameter.
    pub fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, param) in self.params.iter().enumerate() {
            let grad = param.grad();
            let delta: Vec<f32> = grad
                .iter()
                .zip(self.m[i].iter_mut())
                .zip(self.v[i].iter_mut())
                .map(|((g, mi), vi)| {
                    *mi = self.beta1 * *mi + (1.0 - self.beta1) * g;
                    *vi = self.beta2 * *vi + (1.0 - self.beta2) * g * g;
                    let m_hat = *mi / bc1;
                    let v_hat = *vi / bc2;
                    self.lr * m_hat / (v_hat.sqrt() + self.eps)
                })
                .collect();
            param.update_data(&delta);
        }
    }

    /// Clips the global gradient L2 norm across all parameters.
    ///
    /// Computes the combined L2 norm of every parameter's gradient vector.
    /// If it exceeds `max_norm`, all gradients are scaled down uniformly so
    /// the global norm equals `max_norm`. Has no effect when the norm is
    /// already within bounds or is not finite (e.g. NaN/Inf).
    ///
    /// Call this after `loss.backward()` and before `step()`.
    pub fn clip_grad_norm(&self, max_norm: f32) {
        let total_sq: f32 = self
            .params
            .iter()
            .flat_map(|p| p.grad().into_iter().map(|g| g * g))
            .sum();
        let norm = total_sq.sqrt();
        if norm.is_finite() && norm > max_norm {
            let scale = max_norm / norm;
            for p in &self.params {
                let clipped: Vec<f32> = p.grad().iter().map(|g| g * scale).collect();
                p.zero_grad();
                p.accumulate_grad(&clipped);
            }
        }
    }

    /// Zeros all accumulated gradients.
    ///
    /// Must be called before each call to `backward()` to prevent gradients
    /// from accumulating across iterations.
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// Returns `(t, m, v)` — the step counter and per-parameter moment slices needed for serialisation.
    #[must_use]
    pub fn state(&self) -> (usize, &[Vec<f32>], &[Vec<f32>]) {
        (self.t, &self.m, &self.v)
    }

    /// Constructs a state-only shell (no attached parameters) from deserialised moments.
    /// Intended for use by `Persist::read_from` implementations — apply with [`Self::restore_state_from`].
    #[must_use]
    pub fn from_state(t: usize, m: Vec<Vec<f32>>, v: Vec<Vec<f32>>) -> Self {
        Self { params: vec![], lr: 0.0, beta1: 0.9, beta2: 0.999, eps: 1e-8, m, v, t }
    }

    /// Attaches live model parameters and learning rate to a state-only shell, consuming it.
    ///
    /// # Errors
    /// Returns an error if the parameter count or any per-parameter size does not match the saved moments.
    pub fn with_params(mut self, params: Vec<Tensor>, lr: f32) -> std::io::Result<Self> {
        if params.len() != self.m.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "group count mismatch: saved state has {}, model has {}",
                    self.m.len(),
                    params.len()
                ),
            ));
        }
        for (i, p) in params.iter().enumerate() {
            if p.numel() != self.m[i].len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "size mismatch for group {i}: saved has {}, model has {}",
                        self.m[i].len(),
                        p.numel()
                    ),
                ));
            }
        }
        self.params = params;
        self.lr = lr;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    #[test]
    fn adam_reduces_loss() {
        // Single parameter: p=1.0, loss = (p-0)^2, optimal p=0
        let p = Tensor::from_vec(vec![1.0f32], &[1]).with_grad();
        let mut optim = Adam::new(vec![p.clone()], 0.1);
        for _ in 0..50 {
            optim.zero_grad();
            let loss = ops::mse_loss(&p, 0.0);
            loss.backward();
            optim.step();
        }
        assert!(p.item().abs() < 0.1, "p={}", p.item());
    }
}
