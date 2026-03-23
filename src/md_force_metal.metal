/*
 * Metal Compute Shader — LJ Force Kernel (MAXIMUM EFFORT)
 *
 * Every trick in the book:
 *   - fast:: namespace for all math (reduced precision, max throughput)
 *   - half precision shared memory + distance computation
 *   - [[max_total_threads_per_threadgroup]] hint for register allocation
 *   - Branchless everything
 *   - Let the Metal 3.2 compiler do the unrolling
 */

#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 128

struct MDMetalParams {
    uint   n_real;
    uint   n_padded;
    float  lbox;
    float  inv_box;
    float  rc2;
    float  v_shift;
};

kernel void lj_force_kernel(
    const device float *x       [[ buffer(0) ]],
    const device float *y       [[ buffer(1) ]],
    const device float *z       [[ buffer(2) ]],
    device float       *fx      [[ buffer(3) ]],
    device float       *fy      [[ buffer(4) ]],
    device float       *fz      [[ buffer(5) ]],
    device float       *pe_out  [[ buffer(6) ]],
    constant MDMetalParams &params [[ buffer(7) ]],
    uint tid     [[ thread_position_in_grid ]],
    uint lid     [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) [[max_total_threads_per_threadgroup(TILE_SIZE)]] {

    threadgroup half sx[TILE_SIZE];
    threadgroup half sy[TILE_SIZE];
    threadgroup half sz[TILE_SIZE];

    const uint   n_real  = params.n_real;
    const uint   n_pad   = params.n_padded;
    const float  box     = params.lbox;
    const float  inv_box = params.inv_box;
    const float  rc2     = params.rc2;
    const float  v_shift = params.v_shift;
    const half   h_box   = (half)box;
    const half   h_ibox  = (half)inv_box;

    const bool valid_i = (tid < n_real);
    const float xi_f = valid_i ? x[tid] : 0.0f;
    const float yi_f = valid_i ? y[tid] : 0.0f;
    const float zi_f = valid_i ? z[tid] : 0.0f;
    const half  xi_h = (half)xi_f;
    const half  yi_h = (half)yi_f;
    const half  zi_h = (half)zi_f;

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    for (uint tile_start = 0; tile_start < n_pad; tile_start += TILE_SIZE) {
        uint j_idx = tile_start + lid;
        if (j_idx < n_real) {
            sx[lid] = (half)x[j_idx];
            sy[lid] = (half)y[j_idx];
            sz[lid] = (half)z[j_idx];
        } else {
            sx[lid] = (half)1e4f;
            sy[lid] = (half)1e4f;
            sz[lid] = (half)1e4f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_i) {
            for (uint k = 0; k < TILE_SIZE; k++) {
                /* Distance in half — 2x ALU throughput, halved register pressure */
                half dx_h = xi_h - sx[k];
                half dy_h = yi_h - sy[k];
                half dz_h = zi_h - sz[k];

                dx_h -= h_box * rint(dx_h * h_ibox);
                dy_h -= h_box * rint(dy_h * h_ibox);
                dz_h -= h_box * rint(dz_h * h_ibox);

                /* Widen to float for LJ (half can't hold inv_r12 range) */
                float dx = (float)dx_h;
                float dy = (float)dy_h;
                float dz = (float)dz_h;

                float r2 = fma(dx, dx, fma(dy, dy, dz * dz));

                float mask = select(0.0f, 1.0f, r2 > 1e-10f && r2 < rc2);

                /*
                 * fast::rsqrt squared — the key optimization.
                 *
                 * fast::rsqrt uses the GPU's hardware RSQRT unit
                 * with reduced precision (~1 ULP). Squaring gives 1/r2.
                 * Avoids the 6-cycle RECIP instruction entirely.
                 * RSQRT is pipelined differently — may have better throughput.
                 */
                float safe_r2 = select(1.0f, r2, r2 > 1e-10f);
                float irsqrt = fast::rsqrt(safe_r2);
                float inv_r2 = irsqrt * irsqrt;

                float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                float inv_r12 = inv_r6 * inv_r6;

                float f_over_r = 24.0f * inv_r2 * fma(2.0f, inv_r12, -inv_r6) * mask;

                fxi = fma(f_over_r, dx, fxi);
                fyi = fma(f_over_r, dy, fyi);
                fzi = fma(f_over_r, dz, fzi);

                pei += (4.0f * (inv_r12 - inv_r6) - v_shift) * mask;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_i) {
        fx[tid]     = fxi;
        fy[tid]     = fyi;
        fz[tid]     = fzi;
        pe_out[tid] = pei * 0.5f;
    }
}
