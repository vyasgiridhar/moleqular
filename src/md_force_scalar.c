#include "md_force.h"
#include "md_types.h"
#include <math.h>
#include <string.h>

/*
 * Scalar (plain C) LJ Force Kernel — the baseline.
 *
 * Identical physics to md_force_neon.c, but no intrinsics.
 * Compare wall time to see exactly what NEON buys you.
 */

void md_force_scalar(MDSystem *sys, float *pe_out) {
    const int   n_real  = sys->n_real;
    const float box     = sys->lbox;
    const float inv_box = 1.0f / box;
    const float rc2     = MD_CUTOFF * MD_CUTOFF;
    const float v_shift = md_lj_shift();

    float *restrict x  = sys->x;
    float *restrict y  = sys->y;
    float *restrict z  = sys->z;
    float *restrict fx = sys->fx;
    float *restrict fy = sys->fy;
    float *restrict fz = sys->fz;

    /* Zero forces */
    memset(fx, 0, (size_t)n_real * sizeof(float));
    memset(fy, 0, (size_t)n_real * sizeof(float));
    memset(fz, 0, (size_t)n_real * sizeof(float));

    float pe = 0.0f;

    /*
     * Newton's 3rd law optimization: only compute i < j pairs,
     * apply force to both particles. Halves the work.
     * (The NEON version does all-pairs for simplicity — this is
     * the scalar advantage: easier to exploit N3L.)
     */
    for (int i = 0; i < n_real; i++) {
        float xi = x[i], yi = y[i], zi = z[i];
        float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;

        for (int j = i + 1; j < n_real; j++) {
            /* Distance */
            float dx = xi - x[j];
            float dy = yi - y[j];
            float dz = zi - z[j];

            /* Minimum image convention */
            dx -= box * roundf(dx * inv_box);
            dy -= box * roundf(dy * inv_box);
            dz -= box * roundf(dz * inv_box);

            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < rc2) {
                float inv_r2  = 1.0f / r2;
                float inv_r6  = inv_r2 * inv_r2 * inv_r2;
                float inv_r12 = inv_r6 * inv_r6;

                /* Force: 24ε(2/r¹⁴ - 1/r⁸) * dr */
                float f_over_r = 24.0f * MD_EPSILON * inv_r2
                               * (2.0f * inv_r12 - inv_r6);

                float ffx = f_over_r * dx;
                float ffy = f_over_r * dy;
                float ffz = f_over_r * dz;

                fxi   += ffx;
                fyi   += ffy;
                fzi   += ffz;
                fx[j] -= ffx;    /* Newton's 3rd law */
                fy[j] -= ffy;
                fz[j] -= ffz;

                /* Potential energy (no double-count since i < j) */
                pe += 4.0f * MD_EPSILON * (inv_r12 - inv_r6) - v_shift;
            }
        }

        fx[i] += fxi;
        fy[i] += fyi;
        fz[i] += fzi;
    }

    *pe_out = pe;
}
