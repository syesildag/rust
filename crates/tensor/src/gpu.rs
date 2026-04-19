//! wgpu GPU context — compiles to Metal on macOS, Vulkan on Linux.
//!
//! All compute pipelines are compiled once in [`GpuContext::try_new`] and
//! cached for the lifetime of the process. Call [`global_gpu`] to get the
//! process-wide singleton; it returns `None` gracefully when no GPU is found.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]

use std::sync::{Arc, OnceLock};

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

// ── global singleton ──────────────────────────────────────────────────────────

static GPU: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();

/// Returns the process-wide GPU context, initialising it on first call.
///
/// Returns `None` if no suitable adapter is available (e.g. CI, headless).
#[must_use]
pub fn global_gpu() -> Option<Arc<GpuContext>> {
    GPU.get_or_init(GpuContext::try_new).clone()
}

// ── cached pipeline ───────────────────────────────────────────────────────────

struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

// ── context ───────────────────────────────────────────────────────────────────

/// Owns the wgpu device, queue, and all pre-compiled compute pipelines.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul: CachedPipeline,
    matmul_batched: CachedPipeline,
    elementwise: CachedPipeline,
    softmax: CachedPipeline,
    layer_norm: CachedPipeline,
}

impl GpuContext {
    /// Tries to create a GPU context. Compiles all pipelines eagerly.
    ///
    /// Returns `None` if no suitable adapter is found.
    #[must_use]
    pub fn try_new() -> Option<Arc<Self>> {
        pollster::block_on(Self::try_new_async())
    }

    async fn try_new_async() -> Option<Arc<Self>> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;

        tracing::info!(backend = ?adapter.get_info().backend, "GPU adapter selected");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tensor-gpu"),
                ..Default::default()
            })
            .await
            .ok()?;

        // Compile all shaders and cache the resulting pipelines.
        let four = four_binding_layout();
        let matmul = compile_pipeline(
            &device,
            include_str!("../shaders/matmul.wgsl"),
            &four,
        );
        let matmul_batched = compile_pipeline(
            &device,
            include_str!("../shaders/matmul_batched.wgsl"),
            &four,
        );
        let elementwise = compile_pipeline(
            &device,
            include_str!("../shaders/elementwise.wgsl"),
            &four,
        );
        let softmax = compile_pipeline(
            &device,
            include_str!("../shaders/softmax.wgsl"),
            &[storage_ro_entry(0), storage_rw_entry(1), uniform_entry(2)],
        );
        let layer_norm = compile_pipeline(
            &device,
            include_str!("../shaders/layer_norm.wgsl"),
            &[
                storage_ro_entry(0),
                storage_ro_entry(1),
                storage_ro_entry(2),
                storage_rw_entry(3),
                uniform_entry(4),
            ],
        );

        tracing::info!("all GPU pipelines compiled");

        Some(Arc::new(Self {
            device,
            queue,
            matmul,
            matmul_batched,
            elementwise,
            softmax,
            layer_norm,
        }))
    }

    // ── matmul ────────────────────────────────────────────────────────────────

    /// `C = A @ B` where `A` is `[m×k]` and `B` is `[k×n]`.
    #[must_use]
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct Dims {
            m: u32,
            n: u32,
            k: u32,
            _pad: u32,
        }
        let dims = Dims { m: m as u32, n: n as u32, k: k as u32, _pad: 0 };
        let [a_buf, b_buf] = self.upload_ro_pair(a, b);
        let c_buf = self.alloc_output(m * n);
        let staging = self.alloc_staging(m * n);
        let dims_buf = self.upload_uniform(&[dims]);

        let bg = self.bind_group(&self.matmul.bgl, &[
            bind(0, a_buf.as_entire_binding()),
            bind(1, b_buf.as_entire_binding()),
            bind(2, c_buf.as_entire_binding()),
            bind(3, dims_buf.as_entire_binding()),
        ]);
        let wgx = (m as u32).div_ceil(16);
        let wgy = (n as u32).div_ceil(16);
        self.dispatch_and_readback(&self.matmul.pipeline, &bg, wgx, wgy, 1, &c_buf, &staging, m * n)
    }

    // ── matmul_batched ────────────────────────────────────────────────────────

    /// `C[b] = A[b] @ B[b]` for `b` in `0..batch`.
    ///
    /// `A`: `[batch × m × k]`, `B`: `[batch × k × n]` → `[batch × m × n]`.
    #[must_use]
    pub fn matmul_batched(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct Dims {
            batch: u32,
            m: u32,
            n: u32,
            k: u32,
        }
        let dims = Dims { batch: batch as u32, m: m as u32, n: n as u32, k: k as u32 };
        let [a_buf, b_buf] = self.upload_ro_pair(a, b);
        let c_buf = self.alloc_output(batch * m * n);
        let staging = self.alloc_staging(batch * m * n);
        let dims_buf = self.upload_uniform(&[dims]);

        let bg = self.bind_group(&self.matmul_batched.bgl, &[
            bind(0, a_buf.as_entire_binding()),
            bind(1, b_buf.as_entire_binding()),
            bind(2, c_buf.as_entire_binding()),
            bind(3, dims_buf.as_entire_binding()),
        ]);
        let wgx = (m as u32).div_ceil(16);
        let wgy = (n as u32).div_ceil(16);
        self.dispatch_and_readback(
            &self.matmul_batched.pipeline,
            &bg,
            wgx,
            wgy,
            batch as u32,
            &c_buf,
            &staging,
            batch * m * n,
        )
    }

    // ── elementwise ───────────────────────────────────────────────────────────

    /// Element-wise op: `op_code` selects relu(0), gelu(1), add(2), sub(3).
    ///
    /// For unary ops (relu, gelu) `b` is ignored and may be empty.
    #[must_use]
    pub fn elementwise(&self, a: &[f32], b: &[f32], op_code: u32) -> Vec<f32> {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct Ctrl {
            op_code: u32,
            scalar: f32,
            len: u32,
            _pad: u32,
        }
        let n = a.len();
        let ctrl = Ctrl { op_code, scalar: 0.0, len: n as u32, _pad: 0 };
        // WGPU requires a non-empty buffer; use a 1-element dummy for unused B.
        let b_eff: &[f32] = if b.is_empty() { &[0.0f32] } else { b };

        let [a_buf, b_buf] = self.upload_ro_pair(a, b_eff);
        let c_buf = self.alloc_output(n);
        let staging = self.alloc_staging(n);
        let ctrl_buf = self.upload_uniform(&[ctrl]);

        let bg = self.bind_group(&self.elementwise.bgl, &[
            bind(0, a_buf.as_entire_binding()),
            bind(1, b_buf.as_entire_binding()),
            bind(2, c_buf.as_entire_binding()),
            bind(3, ctrl_buf.as_entire_binding()),
        ]);
        let wgx = (n as u32).div_ceil(256);
        self.dispatch_and_readback(&self.elementwise.pipeline, &bg, wgx, 1, 1, &c_buf, &staging, n)
    }

    // ── softmax ───────────────────────────────────────────────────────────────

    /// Row-wise softmax: `input [rows × cols]` → `[rows × cols]`.
    #[must_use]
    pub fn softmax(&self, input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct Dims {
            rows: u32,
            cols: u32,
        }
        let dims = Dims { rows: rows as u32, cols: cols as u32 };
        let in_buf = self.upload_ro(input);
        let out_buf = self.alloc_output(rows * cols);
        let staging = self.alloc_staging(rows * cols);
        let dims_buf = self.upload_uniform(&[dims]);

        let bg = self.bind_group(&self.softmax.bgl, &[
            bind(0, in_buf.as_entire_binding()),
            bind(1, out_buf.as_entire_binding()),
            bind(2, dims_buf.as_entire_binding()),
        ]);
        // One workgroup per row.
        self.dispatch_and_readback(
            &self.softmax.pipeline,
            &bg,
            rows as u32,
            1,
            1,
            &out_buf,
            &staging,
            rows * cols,
        )
    }

    // ── layer_norm ────────────────────────────────────────────────────────────

    /// Row-wise layer normalisation: `[rows × d]` → `[rows × d]`.
    #[must_use]
    pub fn layer_norm(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
        rows: usize,
        d: usize,
    ) -> Vec<f32> {
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct Params {
            rows: u32,
            d: u32,
            eps: f32,
            _pad: u32,
        }
        let params = Params { rows: rows as u32, d: d as u32, eps, _pad: 0 };
        let in_buf = self.upload_ro(input);
        let gamma_buf = self.upload_ro(gamma);
        let beta_buf = self.upload_ro(beta);
        let out_buf = self.alloc_output(rows * d);
        let staging = self.alloc_staging(rows * d);
        let params_buf = self.upload_uniform(&[params]);

        let bg = self.bind_group(&self.layer_norm.bgl, &[
            bind(0, in_buf.as_entire_binding()),
            bind(1, gamma_buf.as_entire_binding()),
            bind(2, beta_buf.as_entire_binding()),
            bind(3, out_buf.as_entire_binding()),
            bind(4, params_buf.as_entire_binding()),
        ]);
        // One workgroup per row.
        self.dispatch_and_readback(
            &self.layer_norm.pipeline,
            &bg,
            rows as u32,
            1,
            1,
            &out_buf,
            &staging,
            rows * d,
        )
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn upload_ro(&self, data: &[f32]) -> wgpu::Buffer {
        // WGPU requires at least 4 bytes; use a zero byte if data is empty.
        let contents: &[u8] = if data.is_empty() { &[0u8; 4] } else { cast_slice(data) };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    fn upload_ro_pair(&self, a: &[f32], b: &[f32]) -> [wgpu::Buffer; 2] {
        [self.upload_ro(a), self.upload_ro(b)]
    }

    fn upload_uniform<T: bytemuck::Pod>(&self, data: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn alloc_output(&self, n: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn alloc_staging(&self, n: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn bind_group<'a>(
        &'a self,
        layout: &'a wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry<'a>],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries,
        })
    }

    fn dispatch_and_readback(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        wgx: u32,
        wgy: u32,
        wgz: u32,
        out_buf: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        n: usize,
    ) -> Vec<f32> {
        let byte_size = (n * std::mem::size_of::<f32>()) as u64;
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wgx, wgy, wgz);
        }
        encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let mapped = staging.slice(..).get_mapped_range();
        let result: Vec<f32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }
}

// ── compile helper ────────────────────────────────────────────────────────────

fn compile_pipeline(
    device: &wgpu::Device,
    shader_src: &str,
    bgl_entries: &[wgpu::BindGroupLayoutEntry],
) -> CachedPipeline {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: bgl_entries,
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    CachedPipeline { pipeline, bgl }
}

// ── BGL entry helpers ─────────────────────────────────────────────────────────

fn four_binding_layout() -> [wgpu::BindGroupLayoutEntry; 4] {
    [storage_ro_entry(0), storage_ro_entry(1), storage_rw_entry(2), uniform_entry(3)]
}

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}
