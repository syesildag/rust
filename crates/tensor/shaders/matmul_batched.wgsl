// Batched matrix multiply: C[b] = A[b] @ B[b]
// A: [B*M*K]  B: [B*K*N]  C: [B*M*N]
// Dispatch: (ceil(M/16), ceil(N/16), B)

struct Dims { batch: u32, m: u32, n: u32, k: u32 }

@group(0) @binding(0) var<storage, read>       A:    array<f32>;
@group(0) @binding(1) var<storage, read>       B:    array<f32>;
@group(0) @binding(2) var<storage, read_write> C:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let b   = id.z;
    let row = id.x;
    let col = id.y;
    if b >= dims.batch || row >= dims.m || col >= dims.n { return; }
    var acc = 0.0f;
    for (var l = 0u; l < dims.k; l++) {
        acc += A[b * dims.m * dims.k + row * dims.k + l]
             * B[b * dims.k * dims.n + l * dims.n + col];
    }
    C[b * dims.m * dims.n + row * dims.n + col] = acc;
}
