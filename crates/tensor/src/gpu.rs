//! wgpu GPU context — compiles to Metal on macOS, Vulkan on Linux.
//!
//! The context is initialised once and passed to GPU-accelerated ops.
//! Falls back gracefully if no GPU is available.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// Owns the wgpu device and command queue.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl GpuContext {
    /// Tries to create a GPU context backed by the best available adapter
    /// (Metal on macOS, Vulkan on Linux/Windows).
    ///
    /// Returns `None` if no suitable GPU adapter is found.
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tensor-gpu"),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Arc::new(Self { device, queue }))
    }

    /// Matrix multiply on GPU: `C = A @ B` where `A` is `[m×k]`, `B` is `[k×n]`.
    ///
    /// Uploads buffers, dispatches the WGSL shader, and reads back the result.
    #[must_use]
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let shader_src = include_str!("../shaders/matmul.wgsl");

        let a_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_A"),
                contents: cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_B"),
                contents: cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_C"),
            size: c_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_staging"),
            size: c_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct Dims {
            m: u32,
            n: u32,
            k: u32,
            _pad: u32,
        }
        let dims = Dims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };
        let dims_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matmul_dims"),
                contents: cast_slice(&[dims]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    storage_ro_entry(0),
                    storage_ro_entry(1),
                    storage_rw_entry(2),
                    uniform_entry(3),
                ],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    },
                )),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let wg = 16u32;
            pass.dispatch_workgroups((m as u32).div_ceil(wg), (n as u32).div_ceil(wg), 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &staging, 0, c_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back.
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let mapped = staging.slice(..).get_mapped_range();
        let result: Vec<f32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

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
