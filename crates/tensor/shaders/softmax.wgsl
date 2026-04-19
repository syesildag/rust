// Row-wise numerically-stable softmax.
// IN / OUT: [rows * cols]  (row-major)
// Dispatch: (rows, 1, 1)  — one workgroup per row, workgroup_size = 256.

struct Dims { rows: u32, cols: u32 }

@group(0) @binding(0) var<storage, read>       IN:   array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT:  array<f32>;
@group(0) @binding(2) var<uniform>             dims: Dims;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let row  = wgid.x;
    let tid  = lid.x;
    let cols = dims.cols;
    if row >= dims.rows { return; }

    // ── Phase 1: row maximum (for numerical stability) ────────────────────
    var local_max = -1.0e38f;
    for (var c = tid; c < cols; c += 256u) {
        local_max = max(local_max, IN[row * cols + c]);
    }
    scratch[tid] = local_max;
    workgroupBarrier();
    if tid < 128u { scratch[tid] = max(scratch[tid], scratch[tid + 128u]); }
    workgroupBarrier();
    if tid < 64u  { scratch[tid] = max(scratch[tid], scratch[tid + 64u]); }
    workgroupBarrier();
    if tid < 32u  { scratch[tid] = max(scratch[tid], scratch[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u  { scratch[tid] = max(scratch[tid], scratch[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u   { scratch[tid] = max(scratch[tid], scratch[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u   { scratch[tid] = max(scratch[tid], scratch[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u   { scratch[tid] = max(scratch[tid], scratch[tid + 2u]); }
    workgroupBarrier();
    if tid < 1u   { scratch[tid] = max(scratch[tid], scratch[tid + 1u]); }
    workgroupBarrier();
    let row_max = scratch[0];
    workgroupBarrier();

    // ── Phase 2: exp(x − max) and partial sums ────────────────────────────
    var local_sum = 0.0f;
    for (var c = tid; c < cols; c += 256u) {
        let e = exp(IN[row * cols + c] - row_max);
        OUT[row * cols + c] = e;
        local_sum += e;
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
    let row_sum = scratch[0];
    workgroupBarrier();

    // ── Phase 3: normalise ────────────────────────────────────────────────
    for (var c = tid; c < cols; c += 256u) {
        OUT[row * cols + c] /= row_sum;
    }
}
