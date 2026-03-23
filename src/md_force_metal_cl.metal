/*
 * Metal Cell List Kernel — O(N) GPU force evaluation.
 *
 * Particles are sorted by cell index on CPU. Each thread reads its
 * position from the sorted array, iterates 27 neighbor cells via
 * cell_start/cell_end offset arrays, and reads j-positions from
 * contiguous sorted memory. No shared memory tiling needed —
 * sorted layout gives good L2 cache behavior since threads in
 * the same SIMD group process nearby particles.
 */

#include <metal_stdlib>
using namespace metal;

struct MDMetalCLParams {
    uint   n_particles;
    uint   ncells_side;
    float  lbox;
    float  inv_box;
    float  rc2;
    float  v_shift;
    float  inv_cell;
};

kernel void lj_force_cl_kernel(
    const device float *sx         [[ buffer(0) ]],
    const device float *sy         [[ buffer(1) ]],
    const device float *sz         [[ buffer(2) ]],
    device float       *fx         [[ buffer(3) ]],
    device float       *fy         [[ buffer(4) ]],
    device float       *fz         [[ buffer(5) ]],
    device float       *pe_out     [[ buffer(6) ]],
    constant MDMetalCLParams &p    [[ buffer(7) ]],
    const device uint  *cell_start [[ buffer(8) ]],
    const device uint  *cell_end   [[ buffer(9) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= p.n_particles) return;

    const float xi = sx[tid];
    const float yi = sy[tid];
    const float zi = sz[tid];

    const uint ncs = p.ncells_side;

    uint cix = min((uint)(xi * p.inv_cell), ncs - 1);
    uint ciy = min((uint)(yi * p.inv_cell), ncs - 1);
    uint ciz = min((uint)(zi * p.inv_cell), ncs - 1);

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    for (uint di = 0; di < 3; di++) {
        uint nx = (cix + ncs + di - 1) % ncs;
        for (uint dj = 0; dj < 3; dj++) {
            uint ny = (ciy + ncs + dj - 1) % ncs;
            for (uint dk = 0; dk < 3; dk++) {
                uint nz = (ciz + ncs + dk - 1) % ncs;
                uint cidx = (nx * ncs + ny) * ncs + nz;

                uint js = cell_start[cidx];
                uint je = cell_end[cidx];

                for (uint j = js; j < je; j++) {
                    if (j == tid) continue;

                    float ddx = xi - sx[j];
                    float ddy = yi - sy[j];
                    float ddz = zi - sz[j];

                    ddx -= p.lbox * rint(ddx * p.inv_box);
                    ddy -= p.lbox * rint(ddy * p.inv_box);
                    ddz -= p.lbox * rint(ddz * p.inv_box);

                    float r2 = fma(ddx, ddx, fma(ddy, ddy, ddz * ddz));

                    if (r2 < p.rc2) {
                        float inv_r2  = 1.0f / r2;
                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;

                        float f_over_r = 24.0f * inv_r2
                                       * fma(2.0f, inv_r12, -inv_r6);

                        fxi = fma(f_over_r, ddx, fxi);
                        fyi = fma(f_over_r, ddy, fyi);
                        fzi = fma(f_over_r, ddz, fzi);

                        pei += 4.0f * (inv_r12 - inv_r6) - p.v_shift;
                    }
                }
            }
        }
    }

    fx[tid]     = fxi;
    fy[tid]     = fyi;
    fz[tid]     = fzi;
    pe_out[tid] = pei * 0.5f;
}
