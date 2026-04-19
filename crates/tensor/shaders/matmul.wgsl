// Tiled 16×16 matrix multiply: C = A @ B
// A: [M, K]  B: [K, N]  C: [M, N]

struct Dims { m: u32, n: u32, k: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read>       A:    array<f32>;
@group(0) @binding(1) var<storage, read>       B:    array<f32>;
@group(0) @binding(2) var<storage, read_write> C:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if row >= dims.m || col >= dims.n { return; }
    var acc = 0.0f;
    for (var l = 0u; l < dims.k; l++) {
        acc += A[row * dims.k + l] * B[l * dims.n + col];
    }
    C[row * dims.n + col] = acc;
}
