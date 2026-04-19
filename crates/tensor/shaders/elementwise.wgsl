// Element-wise ops: relu, gelu, add, sub
// op_code: 0=relu  1=gelu  2=add(A+B)  3=sub(A-B)
// For unary ops (0, 1) the B buffer is ignored.

struct Ctrl { op_code: u32, scalar: f32, len: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read>       A:    array<f32>;
@group(0) @binding(1) var<storage, read>       B:    array<f32>;
@group(0) @binding(2) var<storage, read_write> C:    array<f32>;
@group(0) @binding(3) var<uniform>             ctrl: Ctrl;

fn gelu_approx(x: f32) -> f32 {
    let s     = 0.7978845608f; // sqrt(2/pi)
    let c     = 0.044715f;
    let inner = s * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= ctrl.len { return; }
    switch ctrl.op_code {
        case 0u: { C[i] = max(A[i], 0.0f); }
        case 1u: { C[i] = gelu_approx(A[i]); }
        case 2u: { C[i] = A[i] + B[i]; }
        case 3u: { C[i] = A[i] - B[i]; }
        default: { C[i] = A[i]; }
    }
}
