#include "md_force.h"
#include "md_types.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/*
 * OpenMP + NEON force kernel.
 *
 * Parallelizes the outer i-loop across cores. Each thread accumulates
 * forces for its chunk of i-particles into the shared fx/fy/fz arrays.
 * No race condition because each i writes only to fx[i] — no Newton's
 * 3rd law scatter-writes.
 *
 * On M4's asymmetric 4P+6E topology, OpenMP will spread across all 10
 * cores by default. The E-cores have half the L1D (64KB vs 128KB) and
 * 1/4 the L2 (4MB vs 16MB) — the j-particle arrays may spill on E-cores
 * but stay hot on P-cores.
 */

void md_force_omp(MDSystem *sys, float *pe_out) {
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

    memset(fx, 0, (size_t)n * sizeof(float));
    memset(fy, 0, (size_t)n * sizeof(float));
    memset(fz, 0, (size_t)n * sizeof(float));

    float pe_total = 0.0f;

    #pragma omp parallel reduction(+:pe_total)
    {
        /* Pre-broadcast constants — each thread gets its own register copy */
        float32x4_t vbox     = vdupq_n_f32(box);
        float32x4_t vinv_box = vdupq_n_f32(1.0f / box);
        float32x4_t vrc2     = vdupq_n_f32(rc2);
        float32x4_t veps24   = vdupq_n_f32(24.0f * MD_EPSILON);
        float32x4_t veps4    = vdupq_n_f32(4.0f * MD_EPSILON);
        float32x4_t vtwo     = vdupq_n_f32(2.0f);
        float32x4_t vhalf    = vdupq_n_f32(0.5f);
        float32x4_t vzero    = vdupq_n_f32(0.0f);
        float32x4_t vshift   = vdupq_n_f32(v_shift);

        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < n_real; i++) {
            float32x4_t xi = vdupq_n_f32(x[i]);
            float32x4_t yi = vdupq_n_f32(y[i]);
            float32x4_t zi = vdupq_n_f32(z[i]);

            float32x4_t fxi = vzero;
            float32x4_t fyi = vzero;
            float32x4_t fzi = vzero;
            float32x4_t pei = vzero;

            for (int j = 0; j < n; j += 4) {
                float32x4_t xj = vld1q_f32(&x[j]);
                float32x4_t yj = vld1q_f32(&y[j]);
                float32x4_t zj = vld1q_f32(&z[j]);

                float32x4_t dx = vsubq_f32(xi, xj);
                float32x4_t dy = vsubq_f32(yi, yj);
                float32x4_t dz = vsubq_f32(zi, zj);

                dx = vsubq_f32(dx, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(dx, vinv_box))));
                dy = vsubq_f32(dy, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(dy, vinv_box))));
                dz = vsubq_f32(dz, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(dz, vinv_box))));

                float32x4_t r2 = vmulq_f32(dx, dx);
                r2 = vfmaq_f32(r2, dy, dy);
                r2 = vfmaq_f32(r2, dz, dz);

                float32x4_t eps_guard = vdupq_n_f32(1e-10f);
                uint32x4_t  mask      = vcltq_f32(eps_guard, r2);
                uint32x4_t  mask_rc   = vcltq_f32(r2, vrc2);
                mask = vandq_u32(mask, mask_rc);

                float32x4_t safe_r2 = vbslq_f32(mask, r2, vdupq_n_f32(1.0f));
                float32x4_t inv_r2  = vrecpeq_f32(safe_r2);
                inv_r2 = vmulq_f32(inv_r2, vrecpsq_f32(safe_r2, inv_r2));

                float32x4_t inv_r6  = vmulq_f32(vmulq_f32(inv_r2, inv_r2), inv_r2);
                float32x4_t inv_r12 = vmulq_f32(inv_r6, inv_r6);

                float32x4_t f_over_r = vmulq_f32(veps24, inv_r2);
                float32x4_t lj_term  = vsubq_f32(vmulq_f32(vtwo, inv_r12), inv_r6);
                f_over_r = vmulq_f32(f_over_r, lj_term);

                f_over_r = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(f_over_r), mask));

                fxi = vfmaq_f32(fxi, f_over_r, dx);
                fyi = vfmaq_f32(fyi, f_over_r, dy);
                fzi = vfmaq_f32(fzi, f_over_r, dz);

                float32x4_t pe_pair = vmulq_f32(veps4,
                    vsubq_f32(inv_r12, inv_r6));
                pe_pair = vsubq_f32(pe_pair, vshift);
                pe_pair = vmulq_f32(pe_pair, vhalf);
                pe_pair = vreinterpretq_f32_u32(
                    vandq_u32(vreinterpretq_u32_f32(pe_pair), mask));
                pei = vaddq_f32(pei, pe_pair);
            }

            fx[i] = vaddvq_f32(fxi);
            fy[i] = vaddvq_f32(fyi);
            fz[i] = vaddvq_f32(fzi);
            pe_total += vaddvq_f32(pei);
        }
    }

    *pe_out = pe_total;
}
