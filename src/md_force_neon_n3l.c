#include "md_force.h"
#include "md_types.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

/*
 * NEON + Newton's 3rd Law force kernel.
 *
 * The challenge: when computing pair (i,j), we want to write:
 *   fx[i] += f * dx
 *   fx[j] -= f * dx     ← scatter-write to 4 different j indices
 *
 * With NEON processing 4 j's at once, the j-writes conflict if any
 * two j's in the same NEON lane group alias the same cache line, or
 * worse, if j happens to equal i.
 *
 * Strategy: accumulate j-forces into a LOCAL buffer per i-iteration,
 * then flush the buffer with scalar writes. The i-forces accumulate
 * in NEON registers as before (no conflict — one i per outer loop).
 *
 * This halves the pair count (only i < j) while keeping the inner
 * loop vectorized. The j-buffer flush is O(N) scalar adds per i,
 * but the force compute is O(N) SIMD — the flush is cheap.
 *
 * Actually, simpler: since each j appears in only ONE NEON group
 * per i-iteration, we can store the j-forces directly with vst1q.
 * The trick is we accumulate into a TEMPORARY per-i j-force array,
 * then add it to the global array after the j-loop. No conflicts.
 */

void md_force_neon_n3l(MDSystem *sys, float *pe_out) {
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

    float32x4_t vbox     = vdupq_n_f32(box);
    float32x4_t vinv_box = vdupq_n_f32(1.0f / box);
    float32x4_t vrc2     = vdupq_n_f32(rc2);
    float32x4_t veps24   = vdupq_n_f32(24.0f * MD_EPSILON);
    float32x4_t veps4    = vdupq_n_f32(4.0f * MD_EPSILON);
    float32x4_t vtwo     = vdupq_n_f32(2.0f);
    float32x4_t vzero    = vdupq_n_f32(0.0f);
    float32x4_t vshift   = vdupq_n_f32(v_shift);

    float pe_accum = 0.0f;

    for (int i = 0; i < n_real - 1; i++) {
        float32x4_t xi = vdupq_n_f32(x[i]);
        float32x4_t yi = vdupq_n_f32(y[i]);
        float32x4_t zi = vdupq_n_f32(z[i]);

        float32x4_t fxi = vzero;
        float32x4_t fyi = vzero;
        float32x4_t fzi = vzero;
        float32x4_t pei = vzero;

        /* j starts at i+1, rounded down to multiple of 4 */
        int j_start = (i + 1) & ~3;

        /* Handle the ragged head: i+1 to j_start (0-3 scalar pairs) */
        for (int j = i + 1; j < j_start && j < n_real; j++) {
            float dx = x[i] - x[j];
            float dy = y[i] - y[j];
            float dz = z[i] - z[j];
            dx -= box * roundf(dx / box * 1.0f);
            dy -= box * roundf(dy / box * 1.0f);
            dz -= box * roundf(dz / box * 1.0f);
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < rc2 && r2 > 1e-10f) {
                float inv_r2  = 1.0f / r2;
                float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                float inv_r12 = inv_r6 * inv_r6;
                float f_over_r = 24.0f * MD_EPSILON * inv_r2 * (2.0f * inv_r12 - inv_r6);
                float ffx = f_over_r * dx;
                float ffy = f_over_r * dy;
                float ffz = f_over_r * dz;
                fx[i] += ffx; fy[i] += ffy; fz[i] += ffz;
                fx[j] -= ffx; fy[j] -= ffy; fz[j] -= ffz;
                pe_accum += 4.0f * MD_EPSILON * (inv_r12 - inv_r6) - v_shift;
            }
        }

        /* NEON loop: j from j_start to n, 4 at a time */
        for (int j = j_start; j < n; j += 4) {
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

            /* Mask: r2 > eps AND r2 < rc2 AND j < n_real */
            uint32x4_t mask = vcltq_f32(vdupq_n_f32(1e-10f), r2);
            mask = vandq_u32(mask, vcltq_f32(r2, vrc2));

            /* Mask out padding particles (j >= n_real) */
            int32_t jmask[4] = {
                j     < n_real ? -1 : 0,
                j + 1 < n_real ? -1 : 0,
                j + 2 < n_real ? -1 : 0,
                j + 3 < n_real ? -1 : 0
            };
            mask = vandq_u32(mask, vld1q_u32((uint32_t *)jmask));

            float32x4_t safe_r2 = vbslq_f32(mask, r2, vdupq_n_f32(1.0f));
            float32x4_t inv_r2  = vrecpeq_f32(safe_r2);
            inv_r2 = vmulq_f32(inv_r2, vrecpsq_f32(safe_r2, inv_r2));

            float32x4_t inv_r6  = vmulq_f32(vmulq_f32(inv_r2, inv_r2), inv_r2);
            float32x4_t inv_r12 = vmulq_f32(inv_r6, inv_r6);

            float32x4_t f_over_r = vmulq_f32(veps24, inv_r2);
            f_over_r = vmulq_f32(f_over_r, vsubq_f32(vmulq_f32(vtwo, inv_r12), inv_r6));
            f_over_r = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(f_over_r), mask));

            /* Accumulate on i (NEON — no conflict) */
            fxi = vfmaq_f32(fxi, f_over_r, dx);
            fyi = vfmaq_f32(fyi, f_over_r, dy);
            fzi = vfmaq_f32(fzi, f_over_r, dz);

            /*
             * Newton's 3rd law: scatter-write to j particles.
             *
             * f_j = -(f_over_r * d) per lane. We need to subtract
             * from fx[j+0], fx[j+1], fx[j+2], fx[j+3].
             * These are contiguous → we can vld1q, subtract, vst1q.
             * No conflict because each j appears in exactly one group.
             */
            float32x4_t neg_fx = vmulq_f32(f_over_r, dx);
            float32x4_t neg_fy = vmulq_f32(f_over_r, dy);
            float32x4_t neg_fz = vmulq_f32(f_over_r, dz);

            vst1q_f32(&fx[j], vsubq_f32(vld1q_f32(&fx[j]), neg_fx));
            vst1q_f32(&fy[j], vsubq_f32(vld1q_f32(&fy[j]), neg_fy));
            vst1q_f32(&fz[j], vsubq_f32(vld1q_f32(&fz[j]), neg_fz));

            /* PE (no double-count since i < j) */
            float32x4_t pe_pair = vmulq_f32(veps4, vsubq_f32(inv_r12, inv_r6));
            pe_pair = vsubq_f32(pe_pair, vshift);
            pe_pair = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(pe_pair), mask));
            pei = vaddq_f32(pei, pe_pair);
        }

        fx[i] += vaddvq_f32(fxi);
        fy[i] += vaddvq_f32(fyi);
        fz[i] += vaddvq_f32(fzi);
        pe_accum += vaddvq_f32(pei);
    }

    *pe_out = pe_accum;
}
