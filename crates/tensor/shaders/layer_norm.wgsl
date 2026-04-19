// Row-wise layer normalisation with learned gamma / beta.
// IN / OUT: [rows * D]  GAMMA / BETA: [D]  (row-major)
// Dispatch: (rows, 1, 1)  — one workgroup per row, workgroup_size = 256.

struct Params { rows: u32, d: u32, eps: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read>       IN:     array<f32>;
@group(0) @binding(1) var<storage, read>       GAMMA:  array<f32>;
@group(0) @binding(2) var<storage, read>       BETA:   array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT:    array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    let d   = params.d;
    if row >= params.rows { return; }

    // ── Phase 1: mean ─────────────────────────────────────────────────────
    var local_sum = 0.0f;
    for (var c = tid; c < d; c += 256u) {
        local_sum += IN[row * d + c];
    }
    scratch[tid] = local_sum;
    workgroupBarrier();
    if tid < 128u { scratch[tid] += scratch[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u  { scratch[tid] += scratch[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u  { scratch[tid] += scratch[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u  { scratch[tid] += scratch[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u   { scratch[tid] += scratch[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u   { scratch[tid] += scratch[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u   { scratch[tid] += scratch[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u   { scratch[tid] += scratch[tid + 1u]; }
    workgroupBarrier();
    let mean = scratch[0] / f32(d);
    workgroupBarrier();

    // ── Phase 2: variance ─────────────────────────────────────────────────
    var local_var = 0.0f;
    for (var c = tid; c < d; c += 256u) {
        let diff = IN[row * d + c] - mean;
        local_var += diff * diff;
    }
    scratch[tid] = local_var;
    workgroupBarrier();
    if tid < 128u { scratch[tid] += scratch[tid + 128u]; }
    workgroupBarrier();
    if tid < 64u  { scratch[tid] += scratch[tid + 64u]; }
    workgroupBarrier();
    if tid < 32u  { scratch[tid] += scratch[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u  { scratch[tid] += scratch[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u   { scratch[tid] += scratch[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u   { scratch[tid] += scratch[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u   { scratch[tid] += scratch[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u   { scratch[tid] += scratch[tid + 1u]; }
    workgroupBarrier();
    let inv_std = 1.0f / sqrt(scratch[0] / f32(d) + params.eps);
    workgroupBarrier();

    // ── Phase 3: normalise with gamma / beta ──────────────────────────────
    for (var c = tid; c < d; c += 256u) {
        let x_hat = (IN[row * d + c] - mean) * inv_std;
        OUT[row * d + c] = GAMMA[c] * x_hat + BETA[c];
    }
}
