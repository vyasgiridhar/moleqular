#include "md_force.h"
#include "md_types.h"
#include "md_celllist.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/*
 * NEON + Cell List Force Kernel — O(N) scaling.
 *
 * For each particle i, gather neighbors from 27 neighboring cells
 * into a contiguous stack buffer, then run the proven NEON inner loop
 * over the buffer. This gives us:
 *   - O(N) pair evaluations instead of O(N²)
 *   - Contiguous vld1q_f32 loads from L1-hot stack memory
 *   - Identical NEON compute to md_force_neon.c
 *
 * Crossover vs all-pairs: ~2000-3000 particles.
 * At N=10K: ~10x faster. At N=87K: ~80x faster.
 */

#define NBUF 1024   /* Max neighbors in 27-cell stencil (~400 typical) */

void md_force_neon_cl(MDSystem *sys, float *pe_out) {
    const int   n       = sys->n;
    const int   n_real  = sys->n_real;
    const float box     = sys->lbox;
    const float rc2     = MD_CUTOFF * MD_CUTOFF;
    const float v_shift = md_lj_shift();

    float *restrict x  = sys->x;
    float *restrict y  = sys->y;
    float *restrict z  = sys->z;
    float *restrict fx = sys->fx;
    float *restrict fy = sys->fy;
    float *restrict fz = sys->fz;

    /* Zero forces */
    memset(fx, 0, (size_t)n * sizeof(float));
    memset(fy, 0, (size_t)n * sizeof(float));
    memset(fz, 0, (size_t)n * sizeof(float));

    /* Lazy cell list allocation (static, like Metal host pattern) */
    static MDCellList *cl = NULL;
    if (!cl) cl = md_celllist_create(n);
    md_celllist_build(cl, sys);

    const int ncs = cl->ncells_side;

    /* NEON constants */
    float32x4_t vbox     = vdupq_n_f32(box);
    float32x4_t vinv_box = vdupq_n_f32(1.0f / box);
    float32x4_t vrc2     = vdupq_n_f32(rc2);
    float32x4_t veps24   = vdupq_n_f32(24.0f * MD_EPSILON);
    float32x4_t veps4    = vdupq_n_f32(4.0f * MD_EPSILON);
    float32x4_t vtwo     = vdupq_n_f32(2.0f);
    float32x4_t vhalf    = vdupq_n_f32(0.5f);
    float32x4_t vzero    = vdupq_n_f32(0.0f);
    float32x4_t vshift   = vdupq_n_f32(v_shift);

    float pe_accum = 0.0f;

    for (int i = 0; i < n_real; i++) {
        const float xi_s = x[i];
        const float yi_s = y[i];
        const float zi_s = z[i];

        /* Determine cell of particle i */
        int cix = (int)(xi_s * cl->inv_cell);
        int ciy = (int)(yi_s * cl->inv_cell);
        int ciz = (int)(zi_s * cl->inv_cell);
        if (cix >= ncs) cix = ncs - 1;
        if (ciy >= ncs) ciy = ncs - 1;
        if (ciz >= ncs) ciz = ncs - 1;
        if (cix < 0) cix = 0;
        if (ciy < 0) ciy = 0;
        if (ciz < 0) ciz = 0;

        /* Gather neighbors from 27 cells into contiguous buffer */
        float jx_buf[NBUF], jy_buf[NBUF], jz_buf[NBUF];
        int n_j = 0;

        for (int dx = -1; dx <= 1; dx++) {
            int nx = (cix + dx + ncs) % ncs;
            for (int dy = -1; dy <= 1; dy++) {
                int ny = (ciy + dy + ncs) % ncs;
                for (int dz = -1; dz <= 1; dz++) {
                    int nz = (ciz + dz + ncs) % ncs;
                    int cell_idx = (nx * ncs + ny) * ncs + nz;

                    int j = cl->head[cell_idx];
                    while (j != -1) {
                        if (j != i) {
                            assert(n_j < NBUF);
                            jx_buf[n_j] = x[j];
                            jy_buf[n_j] = y[j];
                            jz_buf[n_j] = z[j];
                            n_j++;
                        }
                        j = cl->next[j];
                    }
                }
            }
        }

        /* Pad to multiple of 4 (fill with self → r²=0 → masked out) */
        int n_j_pad = (n_j + 3) & ~3;
        for (int k = n_j; k < n_j_pad; k++) {
            jx_buf[k] = xi_s;
            jy_buf[k] = yi_s;
            jz_buf[k] = zi_s;
        }

        /* === NEON inner loop (identical to md_force_neon.c:74-185) === */
        float32x4_t xi = vdupq_n_f32(xi_s);
        float32x4_t yi = vdupq_n_f32(yi_s);
        float32x4_t zi = vdupq_n_f32(zi_s);

        float32x4_t fxi = vzero;
        float32x4_t fyi = vzero;
        float32x4_t fzi = vzero;
        float32x4_t pei = vzero;

        for (int j = 0; j < n_j_pad; j += 4) {
            float32x4_t xj = vld1q_f32(&jx_buf[j]);
            float32x4_t yj = vld1q_f32(&jy_buf[j]);
            float32x4_t zj = vld1q_f32(&jz_buf[j]);

            /* Minimum image distance */
            float32x4_t ddx = vsubq_f32(xi, xj);
            float32x4_t ddy = vsubq_f32(yi, yj);
            float32x4_t ddz = vsubq_f32(zi, zj);

            ddx = vsubq_f32(ddx, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(ddx, vinv_box))));
            ddy = vsubq_f32(ddy, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(ddy, vinv_box))));
            ddz = vsubq_f32(ddz, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(ddz, vinv_box))));

            /* r² */
            float32x4_t r2 = vmulq_f32(ddx, ddx);
            r2 = vfmaq_f32(r2, ddy, ddy);
            r2 = vfmaq_f32(r2, ddz, ddz);

            /* Cutoff + self-exclusion mask */
            float32x4_t eps_guard = vdupq_n_f32(1e-10f);
            uint32x4_t  mask      = vcltq_f32(eps_guard, r2);
            uint32x4_t  mask_rc   = vcltq_f32(r2, vrc2);
            mask = vandq_u32(mask, mask_rc);

            /* 1/r² via reciprocal estimate + Newton-Raphson */
            float32x4_t safe_r2 = vbslq_f32(mask, r2, vdupq_n_f32(1.0f));
            float32x4_t inv_r2  = vrecpeq_f32(safe_r2);
            inv_r2 = vmulq_f32(inv_r2, vrecpsq_f32(safe_r2, inv_r2));

            /* LJ: inv_r6, inv_r12 */
            float32x4_t inv_r6  = vmulq_f32(vmulq_f32(inv_r2, inv_r2), inv_r2);
            float32x4_t inv_r12 = vmulq_f32(inv_r6, inv_r6);

            /* F/r = 24ε * inv_r2 * (2*inv_r12 - inv_r6) */
            float32x4_t f_over_r = vmulq_f32(veps24, inv_r2);
            float32x4_t lj_term  = vsubq_f32(vmulq_f32(vtwo, inv_r12), inv_r6);
            f_over_r = vmulq_f32(f_over_r, lj_term);

            /* Mask */
            f_over_r = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(f_over_r), mask));

            /* Accumulate forces on i */
            fxi = vfmaq_f32(fxi, f_over_r, ddx);
            fyi = vfmaq_f32(fyi, f_over_r, ddy);
            fzi = vfmaq_f32(fzi, f_over_r, ddz);

            /* PE: 4ε(inv_r12 - inv_r6) - v_shift, ×0.5 for double counting */
            float32x4_t pe_pair = vmulq_f32(veps4,
                vsubq_f32(inv_r12, inv_r6));
            pe_pair = vsubq_f32(pe_pair, vshift);
            pe_pair = vmulq_f32(pe_pair, vhalf);
            pe_pair = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(pe_pair), mask));
            pei = vaddq_f32(pei, pe_pair);
        }

        /* Horizontal reduce */
        fx[i] += vaddvq_f32(fxi);
        fy[i] += vaddvq_f32(fyi);
        fz[i] += vaddvq_f32(fzi);
        pe_accum += vaddvq_f32(pei);
    }

    *pe_out = pe_accum;
}
