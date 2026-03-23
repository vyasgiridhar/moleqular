#include "md_force.h"
#include "md_types.h"
#include <arm_sme.h>
#include <math.h>
#include <string.h>

/*
 * SME2 Force Kernel — distance matrix via outer products.
 *
 * SME2 crash course:
 *
 *   STREAMING MODE: entered via __arm_locally_streaming attribute.
 *     Regular NEON is UNAVAILABLE. Only SVE (streaming subset) and
 *     ZA tile operations work. NEON intrinsics = SIGILL.
 *
 *   ZA TILES: 16x16 float32 matrix registers (4KB total on M4).
 *     The killer instruction is FMOPA (floating-point outer product
 *     accumulate): ZA += col * row^T. One instruction computes a
 *     full 16x16 outer product. 2 TFLOPS peak on M4 P-cores.
 *
 *   SVL: Scalable Vector Length = 512 bits on M4 = 16 floats.
 *     svfloat32_t holds 16 floats (not 4 like NEON's float32x4_t).
 *
 *   DATA FLOW: CPU ←→ L2 cache ←→ SME coprocessor.
 *     No direct register path — all transfers go through L2.
 *
 * Strategy:
 *   Process particles in 16x16 tiles (matching ZA dimensions).
 *   For each tile of (i-particles × j-particles):
 *     1. Compute dx, dy, dz vectors (SVE ops in streaming mode)
 *     2. Use FMOPA to accumulate r² = dx·dx^T + dy·dy^T + dz·dz^T
 *        Wait — this gives the CROSS terms, not what we want.
 *
 *   Actually, the distance matrix approach is:
 *     r²_ij = (xi-xj)² + (yi-yj)² + (zi-zj)²
 *
 *   We can't directly use outer products for pairwise distances.
 *   But we CAN use the Euclidean trick:
 *     r²_ij = ||pi||² + ||pj||² - 2(pi · pj)
 *     where pi·pj = xi*xj + yi*yj + zi*zj
 *
 *   The Gram matrix (pi · pj) IS an outer product over coordinates:
 *     G = X_i * X_j^T + Y_i * Y_j^T + Z_i * Z_j^T
 *   Each of those is a FMOPA. Three FMOPAs per 16x16 tile.
 *
 *   Then: r²_ij = norms_i + norms_j - 2*G_ij
 *   Then apply LJ using the r² tile, accumulate forces.
 *
 *   But we also need dx, dy, dz for force DIRECTION.
 *   So we compute those separately with SVE subtract.
 *
 * Note: minimum image convention is applied to dx/dy/dz before
 * computing r². The Gram matrix trick assumes unwrapped coordinates,
 * so we can't use it directly with PBC. Instead we compute the
 * difference vectors first, apply minimum image, then use FMOPA
 * to build r² from the wrapped differences.
 *
 *   r²_ij = dx_ij² + dy_ij² + dz_ij²
 *
 *   Using outer products on DIFFERENCE vectors:
 *   For a fixed i, dx_ij for j=0..15 is a vector.
 *   r²_i,j=0..15 = dx_i,* .* dx_i,* + dy_i,* .* dy_i,* + dz_i,* .* dz_i,*
 *   That's element-wise, not outer product.
 *
 *   So the real SME2 win here is using SVE's wider vectors (16 floats
 *   vs NEON's 4) in streaming mode, plus FMOPA for any matrix-shaped
 *   subproblems we can find.
 *
 * What we actually do:
 *   Use streaming SVE (16-wide) for the force inner loop.
 *   That alone is 4x wider than NEON per instruction.
 *   Use ZA tiles to accumulate per-tile force/PE sums where beneficial.
 */

/*
 * Streaming SVE force kernel.
 * __arm_locally_streaming: compiler wraps with smstart/smstop.
 * __arm_new("za"): fresh ZA state, zeroed on entry.
 *
 * SVE is vector-length agnostic. svcntw() returns the number of
 * 32-bit elements per vector (16 on M4). The same code would work
 * on a chip with SVL=256 (8 floats) or SVL=2048 (64 floats).
 */
__arm_locally_streaming
void md_force_sme2(MDSystem *sys, float *pe_out) {
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

    memset(fx, 0, (size_t)sys->n * sizeof(float));
    memset(fy, 0, (size_t)sys->n * sizeof(float));
    memset(fz, 0, (size_t)sys->n * sizeof(float));

    /*
     * svcntw() — number of 32-bit lanes in an SVE vector.
     * On M4: returns 16. On other hardware: could be 4, 8, 32, 64.
     * This is the entire point of SVE: one binary, any vector width.
     */
    const uint64_t vl = svcntw();

    /* Broadcast constants to SVE vectors */
    svbool_t    ptrue    = svptrue_b32();
    svfloat32_t sv_box   = svdup_f32(box);
    svfloat32_t sv_ibox  = svdup_f32(inv_box);
    svfloat32_t sv_rc2   = svdup_f32(rc2);
    svfloat32_t sv_eps24 = svdup_f32(24.0f * MD_EPSILON);
    svfloat32_t sv_eps4  = svdup_f32(4.0f * MD_EPSILON);
    svfloat32_t sv_two   = svdup_f32(2.0f);
    svfloat32_t sv_half  = svdup_f32(0.5f);
    svfloat32_t sv_shift = svdup_f32(v_shift);
    svfloat32_t sv_eps   = svdup_f32(1e-10f);
    svfloat32_t sv_one   = svdup_f32(1.0f);
    svfloat32_t sv_zero  = svdup_f32(0.0f);

    float pe_accum = 0.0f;

    for (int i = 0; i < n_real; i++) {
        svfloat32_t xi = svdup_f32(x[i]);
        svfloat32_t yi = svdup_f32(y[i]);
        svfloat32_t zi = svdup_f32(z[i]);

        svfloat32_t fxi = sv_zero;
        svfloat32_t fyi = sv_zero;
        svfloat32_t fzi = sv_zero;
        svfloat32_t pei = sv_zero;

        for (int j = 0; j < sys->n; j += (int)vl) {
            /*
             * svld1_f32(predicate, ptr) — predicated load.
             * ptrue = all lanes active. For the tail (n not multiple of vl),
             * use svwhilelt to mask off invalid lanes.
             *
             * On M4 this loads 16 floats = 64 bytes per instruction.
             * NEON loads 4 floats = 16 bytes. 4x more data per load.
             */
            svbool_t pg = svwhilelt_b32(j, sys->n);

            svfloat32_t xj = svld1_f32(pg, &x[j]);
            svfloat32_t yj = svld1_f32(pg, &y[j]);
            svfloat32_t zj = svld1_f32(pg, &z[j]);

            /* Distance with minimum image */
            svfloat32_t dx = svsub_f32_x(ptrue, xi, xj);
            svfloat32_t dy = svsub_f32_x(ptrue, yi, yj);
            svfloat32_t dz = svsub_f32_x(ptrue, zi, zj);

            /*
             * Minimum image: dx -= box * round(dx / box)
             *
             * svrintn_f32_x = round to nearest (like NEON's vrndnq_f32)
             * svmul/svsub = element-wise multiply/subtract
             * svmls_f32_x(a, b, c) = a - b*c (fused multiply-subtract)
             */
            dx = svmls_f32_x(ptrue, dx, sv_box,
                    svrintn_f32_x(ptrue, svmul_f32_x(ptrue, dx, sv_ibox)));
            dy = svmls_f32_x(ptrue, dy, sv_box,
                    svrintn_f32_x(ptrue, svmul_f32_x(ptrue, dy, sv_ibox)));
            dz = svmls_f32_x(ptrue, dz, sv_box,
                    svrintn_f32_x(ptrue, svmul_f32_x(ptrue, dz, sv_ibox)));

            /*
             * r² = dx² + dy² + dz²
             *
             * svmla_f32_x(a, b, c) = a + b*c  (FMA — same as NEON vfmaq)
             * On M4 SVE: processes 16 lanes per FMA. NEON: 4.
             */
            svfloat32_t r2 = svmul_f32_x(ptrue, dx, dx);
            r2 = svmla_f32_x(ptrue, r2, dy, dy);
            r2 = svmla_f32_x(ptrue, r2, dz, dz);

            /*
             * Masks: self-exclusion (r2 > eps) AND cutoff (r2 < rc²)
             *
             * SVE predicates are first-class — not reinterpret hacks
             * like NEON's uint32x4_t bitmask dance. Much cleaner.
             */
            svbool_t mask_self = svcmpgt_f32(ptrue, r2, sv_eps);
            svbool_t mask_rc   = svcmplt_f32(ptrue, r2, sv_rc2);
            svbool_t mask      = svand_b_z(ptrue, mask_self, mask_rc);
            mask = svand_b_z(ptrue, mask, pg);  /* also mask tail */

            /*
             * Reciprocal: 1/r²
             *
             * svrecpe_f32 + svrecps_f32 — same Newton-Raphson as NEON
             * but operating on 16 lanes instead of 4.
             */
            svfloat32_t safe_r2 = svsel_f32(mask, r2, sv_one);
            svfloat32_t inv_r2  = svrecpe_f32(safe_r2);
            inv_r2 = svmul_f32_x(ptrue, inv_r2,
                        svrecps_f32(safe_r2, inv_r2));

            /* LJ: inv_r6, inv_r12 */
            svfloat32_t inv_r6  = svmul_f32_x(ptrue,
                                    svmul_f32_x(ptrue, inv_r2, inv_r2), inv_r2);
            svfloat32_t inv_r12 = svmul_f32_x(ptrue, inv_r6, inv_r6);

            /* Force magnitude: 24ε * (2/r¹⁴ - 1/r⁸) */
            svfloat32_t lj_term = svmls_f32_x(ptrue,
                svmul_f32_x(ptrue, sv_two, inv_r12), sv_one, inv_r6);
            /* lj_term = 2*inv_r12 - inv_r6 */

            svfloat32_t f_over_r = svmul_f32_x(ptrue, sv_eps24, inv_r2);
            f_over_r = svmul_f32_x(ptrue, f_over_r, lj_term);

            /* Zero inactive lanes */
            f_over_r = svsel_f32(mask, f_over_r, sv_zero);

            /* Accumulate forces: fi += (F/r) * dr */
            fxi = svmla_f32_x(ptrue, fxi, f_over_r, dx);
            fyi = svmla_f32_x(ptrue, fyi, f_over_r, dy);
            fzi = svmla_f32_x(ptrue, fzi, f_over_r, dz);

            /* PE: 4ε(1/r¹² - 1/r⁶) - shift, half for double-count */
            svfloat32_t pe_pair = svmul_f32_x(ptrue, sv_eps4,
                svsub_f32_x(ptrue, inv_r12, inv_r6));
            pe_pair = svsub_f32_x(ptrue, pe_pair, sv_shift);
            pe_pair = svmul_f32_x(ptrue, pe_pair, sv_half);
            pe_pair = svsel_f32(mask, pe_pair, sv_zero);
            pei = svadd_f32_x(ptrue, pei, pe_pair);
        }

        /*
         * Horizontal reduction: sum all 16 lanes → scalar.
         * svaddv_f32 reduces the full SVE vector to a scalar float.
         */
        fx[i] += svaddv_f32(ptrue, fxi);
        fy[i] += svaddv_f32(ptrue, fyi);
        fz[i] += svaddv_f32(ptrue, fzi);
        pe_accum += svaddv_f32(ptrue, pei);
    }

    *pe_out = pe_accum;
}
