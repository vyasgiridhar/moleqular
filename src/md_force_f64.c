#include "md_force.h"
#include "md_types.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

/*
 * Double-precision NEON force kernel.
 *
 * NEON float64x2_t: 2 doubles per 128-bit register (vs 4 floats).
 * Half the throughput per instruction, but full 64-bit precision.
 *
 * The question: does single precision break MD? Over 10^6 steps,
 * energy drift from float32 rounding accumulates. This kernel lets
 * you measure the drift difference directly.
 *
 * On M4: same 4 NEON pipes, same registers, just 2 lanes instead of 4.
 * Expected throughput: ~half of float32 NEON.
 */

void md_force_f64(MDSystem *sys, float *pe_out) {
    const int   n_real  = sys->n_real;
    const int   n       = sys->n;
    const double box     = (double)sys->lbox;
    const double inv_box = 1.0 / box;
    const double rc2     = (double)(MD_CUTOFF * MD_CUTOFF);
    const double eps24   = 24.0 * (double)MD_EPSILON;
    const double eps4    = 4.0 * (double)MD_EPSILON;

    /* Shifted potential at cutoff */
    double rc2_inv = 1.0 / rc2;
    double rc6_inv = rc2_inv * rc2_inv * rc2_inv;
    double rc12_inv = rc6_inv * rc6_inv;
    double v_shift = 4.0 * (double)MD_EPSILON * (rc12_inv - rc6_inv);

    float *restrict x  = sys->x;
    float *restrict y  = sys->y;
    float *restrict z  = sys->z;
    float *restrict fx = sys->fx;
    float *restrict fy = sys->fy;
    float *restrict fz = sys->fz;

    memset(fx, 0, (size_t)n * sizeof(float));
    memset(fy, 0, (size_t)n * sizeof(float));
    memset(fz, 0, (size_t)n * sizeof(float));

    /* Broadcast constants to f64 NEON vectors (2 lanes each) */
    float64x2_t vbox     = vdupq_n_f64(box);
    float64x2_t vinv_box = vdupq_n_f64(inv_box);
    float64x2_t vrc2     = vdupq_n_f64(rc2);
    float64x2_t veps24   = vdupq_n_f64(eps24);
    float64x2_t veps4    = vdupq_n_f64(eps4);
    float64x2_t vtwo     = vdupq_n_f64(2.0);
    float64x2_t vhalf    = vdupq_n_f64(0.5);
    float64x2_t vzero    = vdupq_n_f64(0.0);
    float64x2_t vshift   = vdupq_n_f64(v_shift);
    float64x2_t veps     = vdupq_n_f64(1e-20);

    double pe_accum = 0.0;

    /* Pad n_real up to multiple of 2 for f64 NEON */
    int n_pad2 = (n_real + 1) & ~1;

    for (int i = 0; i < n_real; i++) {
        /* Widen particle i from float to double, broadcast to 2 lanes */
        float64x2_t xi = vdupq_n_f64((double)x[i]);
        float64x2_t yi = vdupq_n_f64((double)y[i]);
        float64x2_t zi = vdupq_n_f64((double)z[i]);

        float64x2_t fxi = vzero;
        float64x2_t fyi = vzero;
        float64x2_t fzi = vzero;
        float64x2_t pei = vzero;

        for (int j = 0; j < n_pad2; j += 2) {
            /*
             * Load 2 floats, widen to doubles.
             * No vld1q_f64-from-float — must load f32 then convert.
             */
            float32x2_t xj_f32 = vld1_f32(&x[j]);  /* 2 floats */
            float32x2_t yj_f32 = vld1_f32(&y[j]);
            float32x2_t zj_f32 = vld1_f32(&z[j]);

            float64x2_t xj = vcvt_f64_f32(xj_f32);
            float64x2_t yj = vcvt_f64_f32(yj_f32);
            float64x2_t zj = vcvt_f64_f32(zj_f32);

            float64x2_t dx = vsubq_f64(xi, xj);
            float64x2_t dy = vsubq_f64(yi, yj);
            float64x2_t dz = vsubq_f64(zi, zj);

            /* Minimum image */
            dx = vsubq_f64(dx, vmulq_f64(vbox, vrndnq_f64(vmulq_f64(dx, vinv_box))));
            dy = vsubq_f64(dy, vmulq_f64(vbox, vrndnq_f64(vmulq_f64(dy, vinv_box))));
            dz = vsubq_f64(dz, vmulq_f64(vbox, vrndnq_f64(vmulq_f64(dz, vinv_box))));

            /* r² */
            float64x2_t r2 = vmulq_f64(dx, dx);
            r2 = vfmaq_f64(r2, dy, dy);
            r2 = vfmaq_f64(r2, dz, dz);

            /* Masks */
            uint64x2_t mask = vcgtq_f64(r2, veps);
            mask = vandq_u64(mask, vcltq_f64(r2, vrc2));

            /* Mask out j >= n_real */
            uint64_t jmask[2] = {
                j     < n_real ? ~0ULL : 0ULL,
                j + 1 < n_real ? ~0ULL : 0ULL
            };
            mask = vandq_u64(mask, vld1q_u64(jmask));

            /* Reciprocal — no vrecpeq for f64, use division */
            float64x2_t safe_r2 = vbslq_f64(mask, r2, vdupq_n_f64(1.0));
            float64x2_t inv_r2 = vdivq_f64(vdupq_n_f64(1.0), safe_r2);

            /* LJ */
            float64x2_t inv_r6  = vmulq_f64(vmulq_f64(inv_r2, inv_r2), inv_r2);
            float64x2_t inv_r12 = vmulq_f64(inv_r6, inv_r6);

            float64x2_t f_over_r = vmulq_f64(veps24, inv_r2);
            f_over_r = vmulq_f64(f_over_r, vsubq_f64(vmulq_f64(vtwo, inv_r12), inv_r6));
            f_over_r = vreinterpretq_f64_u64(
                vandq_u64(vreinterpretq_u64_f64(f_over_r), mask));

            fxi = vfmaq_f64(fxi, f_over_r, dx);
            fyi = vfmaq_f64(fyi, f_over_r, dy);
            fzi = vfmaq_f64(fzi, f_over_r, dz);

            float64x2_t pe_pair = vmulq_f64(veps4, vsubq_f64(inv_r12, inv_r6));
            pe_pair = vsubq_f64(pe_pair, vshift);
            pe_pair = vmulq_f64(pe_pair, vhalf);
            pe_pair = vreinterpretq_f64_u64(
                vandq_u64(vreinterpretq_u64_f64(pe_pair), mask));
            pei = vaddq_f64(pei, pe_pair);
        }

        /* Horizontal reduction — sum 2 f64 lanes */
        fx[i] += (float)vaddvq_f64(fxi);
        fy[i] += (float)vaddvq_f64(fyi);
        fz[i] += (float)vaddvq_f64(fzi);
        pe_accum += vaddvq_f64(pei);
    }

    *pe_out = (float)pe_accum;
}
