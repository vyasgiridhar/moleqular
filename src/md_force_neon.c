#include "md_force.h"
#include "md_types.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

/*
 * NEON LJ Force Kernel — the hot loop.
 *
 * NEON register cheat sheet (ARM 128-bit SIMD):
 *   float32x4_t  = 4 floats packed in one 128-bit V register
 *   vld1q_f32    = load 4 contiguous floats into V register
 *   vst1q_f32    = store V register to memory
 *   vdupq_n_f32  = broadcast one float to all 4 lanes
 *   vfmaq_f32    = fused multiply-accumulate: a + b*c (single instruction!)
 *   vrecpeq_f32  = reciprocal estimate (~12 bits accuracy)
 *   vrecpsq_f32  = Newton-Raphson refinement step for reciprocal
 *   vcltq_f32    = compare less-than, returns bitmask (all 1s or all 0s per lane)
 *   vbslq_f32    = bitwise select using mask (branchless conditional)
 *   vaddvq_f32   = horizontal sum of all 4 lanes → scalar
 *   vrndnq_f32   = round to nearest (for minimum image convention)
 *
 * The inner loop processes 4 j-particles per iteration.
 * For particle i, we broadcast xi/yi/zi to all 4 lanes, then sweep j.
 */

void md_force_neon(MDSystem *sys, float *pe_out) {
    const int   n       = sys->n;       /* padded to multiple of 4 */
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

    /* Broadcast constants to NEON registers */
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
        /*
         * Broadcast particle i's position to all 4 NEON lanes.
         * vdupq_n_f32(scalar) → [scalar, scalar, scalar, scalar]
         */
        float32x4_t xi = vdupq_n_f32(x[i]);
        float32x4_t yi = vdupq_n_f32(y[i]);
        float32x4_t zi = vdupq_n_f32(z[i]);

        /* Force accumulators for particle i (4 lanes, reduced at end) */
        float32x4_t fxi = vzero;
        float32x4_t fyi = vzero;
        float32x4_t fzi = vzero;
        float32x4_t pei = vzero;

        for (int j = 0; j < n; j += 4) {
            /*
             * Load 4 j-particle positions in one shot.
             * vld1q_f32(&x[j]) reads x[j], x[j+1], x[j+2], x[j+3]
             * into a single 128-bit register. This is why SoA layout matters:
             * these 4 floats are contiguous in memory → one cache line access.
             */
            float32x4_t xj = vld1q_f32(&x[j]);
            float32x4_t yj = vld1q_f32(&y[j]);
            float32x4_t zj = vld1q_f32(&z[j]);

            /* --- Distance with minimum image convention ---
             *
             * dx = xi - xj
             * dx -= box * round(dx / box)
             *
             * vrndnq_f32 = round to nearest even (IEEE 754).
             * This implements minimum image: if dx > box/2, wrap it.
             */
            float32x4_t dx = vsubq_f32(xi, xj);
            float32x4_t dy = vsubq_f32(yi, yj);
            float32x4_t dz = vsubq_f32(zi, zj);

            dx = vsubq_f32(dx, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(dx, vinv_box))));
            dy = vsubq_f32(dy, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(dy, vinv_box))));
            dz = vsubq_f32(dz, vmulq_f32(vbox, vrndnq_f32(vmulq_f32(dz, vinv_box))));

            /* --- r² = dx² + dy² + dz² ---
             *
             * vfmaq_f32(a, b, c) = a + b*c  (fused multiply-accumulate)
             * This is the M4's bread and butter — single-cycle FMA.
             * We chain two FMAs to compute the full dot product.
             */
            float32x4_t r2 = vmulq_f32(dx, dx);
            r2 = vfmaq_f32(r2, dy, dy);        /* r2 += dy*dy */
            r2 = vfmaq_f32(r2, dz, dz);        /* r2 += dz*dz */

            /* --- Cutoff mask ---
             *
             * vcltq_f32(a, b) → per-lane: a < b ? 0xFFFFFFFF : 0x00000000
             * No branch, no divergence. Lanes outside cutoff get zeroed.
             * Also mask out self-interaction (r2 == 0 → r2 < tiny is false).
             */
            float32x4_t eps_guard = vdupq_n_f32(1e-10f);
            uint32x4_t  mask      = vcltq_f32(eps_guard, r2);  /* r2 > eps (not self) */
            uint32x4_t  mask_rc   = vcltq_f32(r2, vrc2);       /* r2 < rc² */
            mask = vandq_u32(mask, mask_rc);                     /* both conditions */

            /* --- Reciprocal: 1/r² ---
             *
             * vrecpeq_f32 gives ~12 bits of 1/x.
             * One Newton-Raphson step (vrecpsq_f32) refines to ~24 bits.
             * This avoids expensive division entirely.
             *
             * Newton-Raphson for reciprocal:
             *   x_{n+1} = x_n * (2 - a * x_n)
             * vrecpsq_f32(a, x) computes (2 - a*x), so:
             *   refined = estimate * vrecpsq_f32(r2, estimate)
             */
            float32x4_t safe_r2 = vbslq_f32(mask, r2, vdupq_n_f32(1.0f));
            float32x4_t inv_r2  = vrecpeq_f32(safe_r2);             /* ~12 bit estimate */
            inv_r2 = vmulq_f32(inv_r2, vrecpsq_f32(safe_r2, inv_r2)); /* ~24 bit refined */

            /* --- LJ terms ---
             *
             * inv_r6  = inv_r2 * inv_r2 * inv_r2  = (1/r²)³ = 1/r⁶
             * inv_r12 = inv_r6 * inv_r6             = 1/r¹²
             *
             * No pow() calls — just multiplies. This is how MD codes
             * avoid transcendentals in the hot loop.
             */
            float32x4_t inv_r6  = vmulq_f32(vmulq_f32(inv_r2, inv_r2), inv_r2);
            float32x4_t inv_r12 = vmulq_f32(inv_r6, inv_r6);

            /* --- Force magnitude ---
             *
             * F/r = 24ε * (2σ¹²/r¹⁴ - σ⁶/r⁸)
             *      = 24ε * (2*inv_r12 - inv_r6) * inv_r2
             *
             * vfmsq_f32(a, b, c) = a - b*c would be nice but we use
             * the equivalent: 24ε * inv_r2 * (2*inv_r12 - inv_r6)
             */
            float32x4_t f_over_r = vmulq_f32(veps24, inv_r2);
            float32x4_t lj_term  = vsubq_f32(vmulq_f32(vtwo, inv_r12), inv_r6);
            f_over_r = vmulq_f32(f_over_r, lj_term);

            /* Apply cutoff mask — zero out forces for pairs beyond rc or self */
            f_over_r = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(f_over_r), mask));

            /* --- Accumulate forces on particle i ---
             *
             * fi += (F/r) * dr
             * Using FMA: fxi = fxi + f_over_r * dx
             */
            fxi = vfmaq_f32(fxi, f_over_r, dx);
            fyi = vfmaq_f32(fyi, f_over_r, dy);
            fzi = vfmaq_f32(fzi, f_over_r, dz);

            /* --- Potential energy ---
             *
             * V = 4ε(1/r¹² - 1/r⁶) - V_shift
             * Multiply by 0.5 to avoid double-counting (we sum all i,j pairs)
             */
            float32x4_t pe_pair = vmulq_f32(veps4,
                vsubq_f32(inv_r12, inv_r6));
            pe_pair = vsubq_f32(pe_pair, vshift);
            pe_pair = vmulq_f32(pe_pair, vhalf);
            pe_pair = vreinterpretq_f32_u32(
                vandq_u32(vreinterpretq_u32_f32(pe_pair), mask));
            pei = vaddq_f32(pei, pe_pair);
        }

        /*
         * Horizontal reduction — sum all 4 lanes into a scalar.
         * vaddvq_f32 is ARMv8.2+: adds all 4 lanes in one instruction.
         * On older ARM you'd need vpadd pairs. M4 has this natively.
         */
        fx[i] += vaddvq_f32(fxi);
        fy[i] += vaddvq_f32(fyi);
        fz[i] += vaddvq_f32(fzi);
        pe_accum += vaddvq_f32(pei);
    }

    *pe_out = pe_accum;
}
