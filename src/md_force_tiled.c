#include "md_force.h"
#include "md_types.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>

/*
 * Tiled OpenMP + NEON force kernel with CPU pinning.
 *
 * Two optimizations over the naive OMP version:
 *
 * 1. J-LOOP TILING:
 *    M4 cache line = 128 bytes = 32 floats.
 *    Tile the j-loop in blocks of 32 so each coordinate array (x,y,z)
 *    loads exactly one cache line per tile. Prefetch the next tile while
 *    computing the current one — hides memory latency.
 *
 *    For extra reuse, we also tile in i: process I_TILE particles against
 *    each j-tile before moving on. The j-data stays hot in L1 while
 *    multiple i-particles read it.
 *
 * 2. CPU PINNING:
 *    macOS doesn't have sched_setaffinity. We use two mechanisms:
 *    - QOS_CLASS_USER_INTERACTIVE: biases the scheduler toward P-cores
 *    - THREAD_AFFINITY_POLICY: hints the kernel to keep threads on the
 *      same cache-sharing cluster
 *
 *    Set OMP_NUM_THREADS=4 to limit to P-cores.
 */

#define J_TILE  32   /* 32 floats = 128 bytes = 1 M4 cache line */
#define I_TILE  4    /* 4 i-particles reuse each j-tile from L1 */

/* Pin calling thread to P-cores via macOS QoS + affinity tag */
static void pin_to_performance(int thread_id) {
    /* QoS hint: USER_INTERACTIVE biases toward P-cores */
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    /* Affinity tag: threads with same tag get co-scheduled on same L2 domain */
    thread_affinity_policy_data_t policy = { .affinity_tag = 1 };
    thread_policy_set(
        mach_thread_self(),
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT
    );
    (void)thread_id;
}

/*
 * Process one i-particle against a j-tile [j0, j0+J_TILE).
 * Returns force contributions in fxi/fyi/fzi/pei accumulators.
 * All arguments are NEON vectors — caller holds accumulators across tiles.
 */
static inline void force_i_vs_jtile(
    float32x4_t xi, float32x4_t yi, float32x4_t zi,
    float32x4_t *fxi, float32x4_t *fyi, float32x4_t *fzi, float32x4_t *pei,
    const float *restrict x, const float *restrict y, const float *restrict z,
    int j0, int j_end,
    float32x4_t vbox, float32x4_t vinv_box, float32x4_t vrc2,
    float32x4_t veps24, float32x4_t veps4, float32x4_t vtwo,
    float32x4_t vhalf, float32x4_t vshift)
{
    for (int j = j0; j < j_end; j += 4) {
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

        uint32x4_t mask    = vcltq_f32(vdupq_n_f32(1e-10f), r2);
        uint32x4_t mask_rc = vcltq_f32(r2, vrc2);
        mask = vandq_u32(mask, mask_rc);

        float32x4_t safe_r2 = vbslq_f32(mask, r2, vdupq_n_f32(1.0f));
        float32x4_t inv_r2  = vrecpeq_f32(safe_r2);
        inv_r2 = vmulq_f32(inv_r2, vrecpsq_f32(safe_r2, inv_r2));

        float32x4_t inv_r6  = vmulq_f32(vmulq_f32(inv_r2, inv_r2), inv_r2);
        float32x4_t inv_r12 = vmulq_f32(inv_r6, inv_r6);

        float32x4_t f_over_r = vmulq_f32(veps24, inv_r2);
        f_over_r = vmulq_f32(f_over_r, vsubq_f32(vmulq_f32(vtwo, inv_r12), inv_r6));
        f_over_r = vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(f_over_r), mask));

        *fxi = vfmaq_f32(*fxi, f_over_r, dx);
        *fyi = vfmaq_f32(*fyi, f_over_r, dy);
        *fzi = vfmaq_f32(*fzi, f_over_r, dz);

        float32x4_t pe_pair = vmulq_f32(veps4, vsubq_f32(inv_r12, inv_r6));
        pe_pair = vsubq_f32(pe_pair, vshift);
        pe_pair = vmulq_f32(pe_pair, vhalf);
        pe_pair = vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(pe_pair), mask));
        *pei = vaddq_f32(*pei, pe_pair);
    }
}

void md_force_tiled(MDSystem *sys, float *pe_out) {
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
        pin_to_performance(omp_get_thread_num());

        float32x4_t vbox     = vdupq_n_f32(box);
        float32x4_t vinv_box = vdupq_n_f32(1.0f / box);
        float32x4_t vrc2     = vdupq_n_f32(rc2);
        float32x4_t veps24   = vdupq_n_f32(24.0f * MD_EPSILON);
        float32x4_t veps4    = vdupq_n_f32(4.0f * MD_EPSILON);
        float32x4_t vtwo     = vdupq_n_f32(2.0f);
        float32x4_t vhalf    = vdupq_n_f32(0.5f);
        float32x4_t vzero    = vdupq_n_f32(0.0f);
        float32x4_t vshift   = vdupq_n_f32(v_shift);

        /*
         * Double tiling: i-blocks of I_TILE, j-blocks of J_TILE.
         *
         * The j-tile data (32 x/y/z floats = 3 cache lines = 384 bytes)
         * stays in L1 while I_TILE particles read it. Without tiling,
         * each i sweeps the entire j-array before the next i touches it —
         * the head of j has been evicted by the time i+1 needs it.
         *
         * With tiling:
         *   for each j_tile:
         *     prefetch next j_tile
         *     for each i in i_block:
         *       compute forces(i, j_tile)   ← j_tile hot in L1
         */
        #pragma omp for schedule(static)
        for (int i0 = 0; i0 < n_real; i0 += I_TILE) {
            int i_end = i0 + I_TILE < n_real ? i0 + I_TILE : n_real;
            int i_count = i_end - i0;

            /* Per-i accumulators (persist across j-tiles) */
            float32x4_t fxa[I_TILE], fya[I_TILE], fza[I_TILE], pea[I_TILE];
            float32x4_t xia[I_TILE], yia[I_TILE], zia[I_TILE];

            for (int ii = 0; ii < i_count; ii++) {
                fxa[ii] = vzero; fya[ii] = vzero; fza[ii] = vzero; pea[ii] = vzero;
                xia[ii] = vdupq_n_f32(x[i0 + ii]);
                yia[ii] = vdupq_n_f32(y[i0 + ii]);
                zia[ii] = vdupq_n_f32(z[i0 + ii]);
            }

            /* Sweep j in tiles */
            for (int j0 = 0; j0 < n; j0 += J_TILE) {
                int j_end = j0 + J_TILE < n ? j0 + J_TILE : n;

                /* Prefetch next j-tile into L1 (locality hint 3 = keep in all caches) */
                if (j0 + J_TILE < n) {
                    __builtin_prefetch(&x[j0 + J_TILE], 0, 3);
                    __builtin_prefetch(&y[j0 + J_TILE], 0, 3);
                    __builtin_prefetch(&z[j0 + J_TILE], 0, 3);
                }

                /* All i's in this block hit the same j-tile — reuse from L1 */
                for (int ii = 0; ii < i_count; ii++) {
                    force_i_vs_jtile(
                        xia[ii], yia[ii], zia[ii],
                        &fxa[ii], &fya[ii], &fza[ii], &pea[ii],
                        x, y, z, j0, j_end,
                        vbox, vinv_box, vrc2, veps24, veps4, vtwo, vhalf, vshift);
                }
            }

            /* Reduce accumulators → scalar force arrays */
            for (int ii = 0; ii < i_count; ii++) {
                fx[i0 + ii] = vaddvq_f32(fxa[ii]);
                fy[i0 + ii] = vaddvq_f32(fya[ii]);
                fz[i0 + ii] = vaddvq_f32(fza[ii]);
                pe_total += vaddvq_f32(pea[ii]);
            }
        }
    }

    *pe_out = pe_total;
}
