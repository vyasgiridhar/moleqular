/*
 * NBNXM-style cluster pair force kernel for Metal GPU.
 *
 * One threadgroup (8×8 = 64 threads) per i-cluster.
 * Thread (tidx_i, tidx_j) computes force between i-atom tidx_i and j-atom tidx_j.
 * i-cluster preloaded to threadgroup memory, j-clusters loaded per iteration.
 * After all j-clusters: tree-reduce across tidx_j → total force on each i-atom.
 *
 * This is the GROMACS NBNXM pattern adapted for Apple Metal.
 */

#include <metal_stdlib>
using namespace metal;

#define CL_SIZE 8

struct ClusterParams {
    uint  n_clusters;
    uint  n_atoms_total;
    float lbox;
    float inv_box;
    float rc2;
    float v_shift;
};

kernel void lj_force_cluster(
    const device float  *cx         [[ buffer(0) ]],
    const device float  *cy         [[ buffer(1) ]],
    const device float  *cz         [[ buffer(2) ]],
    device float        *fx         [[ buffer(3) ]],
    device float        *fy         [[ buffer(4) ]],
    device float        *fz         [[ buffer(5) ]],
    device float        *pe_out     [[ buffer(6) ]],
    constant ClusterParams &p       [[ buffer(7) ]],
    const device uint   *pair_start [[ buffer(8) ]],
    const device uint   *pair_end   [[ buffer(9) ]],
    const device uint   *j_list     [[ buffer(10) ]],
    const device uint   *cl_count   [[ buffer(11) ]],
    uint  tgid  [[ threadgroup_position_in_grid ]],
    uint  lid   [[ thread_index_in_threadgroup ]]
) {
    uint ci = tgid;
    if (ci >= p.n_clusters) return;

    uint tidx_j = lid % CL_SIZE;   /* 0-7: which j-atom */
    uint tidx_i = lid / CL_SIZE;   /* 0-7: which i-atom */

    uint ci_base = ci * CL_SIZE;
    uint ci_real = cl_count[ci];

    /* Phase 1: load i-cluster into threadgroup memory */
    threadgroup float si_x[CL_SIZE];
    threadgroup float si_y[CL_SIZE];
    threadgroup float si_z[CL_SIZE];

    if (tidx_j == 0) {
        si_x[tidx_i] = cx[ci_base + tidx_i];
        si_y[tidx_i] = cy[ci_base + tidx_i];
        si_z[tidx_i] = cz[ci_base + tidx_i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float xi = si_x[tidx_i];
    float yi = si_y[tidx_i];
    float zi = si_z[tidx_i];
    bool valid_i = (tidx_i < ci_real);

    /* Phase 2: iterate j-clusters
     * j-atoms loaded directly from global memory (coalesced: adjacent tidx_j
     * threads read adjacent memory locations). No shared memory needed for j,
     * no barriers in the inner loop. This is the GROMACS approach. */
    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    float box = p.lbox;
    float inv_box = p.inv_box;
    float rc2 = p.rc2;
    float v_shift = p.v_shift;

    uint ps = pair_start[ci];
    uint pe = pair_end[ci];

    for (uint jp = ps; jp < pe; jp++) {
        uint cj = j_list[jp];
        uint aj = cj * CL_SIZE + tidx_j;

        /* Direct global memory read — coalesced across tidx_j threads */
        float xj = cx[aj];
        float yj = cy[aj];
        float zj = cz[aj];

        if (valid_i) {
            float dx = xi - xj;
            float dy = yi - yj;
            float dz = zi - zj;

            dx -= box * rint(dx * inv_box);
            dy -= box * rint(dy * inv_box);
            dz -= box * rint(dz * inv_box);

            float r2 = dx*dx + dy*dy + dz*dz;

            /* Self-exclusion + cutoff + padding sentinel rejection */
            if (r2 < rc2 && r2 > 1e-10f) {
                float inv_r2  = 1.0f / r2;
                float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                float inv_r12 = inv_r6 * inv_r6;

                float f_over_r = 24.0f * inv_r2 * (2.0f * inv_r12 - inv_r6);

                fxi += f_over_r * dx;
                fyi += f_over_r * dy;
                fzi += f_over_r * dz;
                pei += 4.0f * (inv_r12 - inv_r6) - v_shift;
            }
        }
    }

    /* Phase 3: reduce forces across tidx_j for each tidx_i */
    threadgroup float red[CL_SIZE * CL_SIZE * 4];  /* 64 * 4 = 256 floats */
    uint r_base = lid * 4;
    red[r_base + 0] = fxi;
    red[r_base + 1] = fyi;
    red[r_base + 2] = fzi;
    red[r_base + 3] = pei;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Tree reduction: stride 4, 2, 1 across tidx_j dimension */
    /* red layout: [tidx_i * 8 + tidx_j] * 4 */
    uint row_base = tidx_i * CL_SIZE;

    if (tidx_j < 4) {
        uint a = (row_base + tidx_j) * 4;
        uint b = (row_base + tidx_j + 4) * 4;
        red[a+0] += red[b+0]; red[a+1] += red[b+1];
        red[a+2] += red[b+2]; red[a+3] += red[b+3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tidx_j < 2) {
        uint a = (row_base + tidx_j) * 4;
        uint b = (row_base + tidx_j + 2) * 4;
        red[a+0] += red[b+0]; red[a+1] += red[b+1];
        red[a+2] += red[b+2]; red[a+3] += red[b+3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tidx_j == 0 && tidx_i < ci_real) {
        uint a0 = (row_base) * 4;
        uint a1 = (row_base + 1) * 4;
        uint atom_idx = ci_base + tidx_i;
        fx[atom_idx]     = red[a0+0] + red[a1+0];
        fy[atom_idx]     = red[a0+1] + red[a1+1];
        fz[atom_idx]     = red[a0+2] + red[a1+2];
        pe_out[atom_idx] = (red[a0+3] + red[a1+3]) * 0.5f;
    }
}
