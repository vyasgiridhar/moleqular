/*
 * Direct ANE LJ force — hybrid CPU↔ANE pipeline.
 *
 * ANE: LJ force chain on [1,N,1,N] tensor (r² → fm, pm). FP16 throughout.
 *      Bypasses CoreML via maderix's reverse-engineered _ANEClient APIs.
 *
 * CPU (NEON): distance matrix (r², dx, dy, dz), force accumulation (fm*dr),
 *             PE summation. All vectorized with ARM NEON float32x4.
 *
 * DISCOVERY: ANE compiler corrupts multi-input complex chains.
 * Single-input chains work perfectly. Cross-input ops must stay on CPU.
 */

#import <Foundation/Foundation.h>
#include "md_types.h"
#include "md_force.h"
#include "ane_runtime.h"
#include <string.h>
#include <math.h>
#include <arm_neon.h>

/*
 * MIL graph: single-input LJ chain from r².
 *
 * Input:  r2 [1,N,1,N] (FP32 — cast to FP16 on ANE)
 * Output: fm [1,N,1,N] (masked f/r, FP16), pm [1,N,1,N] (masked PE per pair, FP16)
 */
static NSString *generate_lj_ane_mil(int N) {
    float rc2 = MD_CUTOFF * MD_CUTOFF;
    float ri2 = 1.0f / rc2;
    float ri6 = ri2 * ri2 * ri2;
    float v_shift = 4.0f * MD_EPSILON * (ri6 * ri6 - ri6);

    NSMutableString *m = [NSMutableString stringWithCapacity:4096];

    [m appendString:@"program(1.3)\n"];
    [m appendString:@"[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                     "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                     "{\"coremltools-version\", \"9.0\"}})]\n{\n"];

    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> r2_in) {\n", N, N];

    [m appendString:@"        string t16 = const()[name=string(\"t16\"), val=string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> r2 = cast(dtype=t16, x=r2_in)[name=string(\"r2\")];\n", N, N];

    /* Cutoff mask: sigmoid(500 * (rc² - r²)) */
    [m appendFormat:@"        fp16 rc2v = const()[name=string(\"rc2v\"), val=fp16(%.8f)];\n", rc2];
    [m appendString:@"        fp16 msc = const()[name=string(\"msc\"), val=fp16(500.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> df = sub(x=rc2v, y=r2)[name=string(\"df\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> sc = mul(x=df, y=msc)[name=string(\"sc\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> mask = sigmoid(x=sc)[name=string(\"mask\")];\n", N, N];

    /* LJ chain: ir2 → ir6 → ir12 → f_over_r */
    [m appendString:@"        fp16 one_h = const()[name=string(\"one_h\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> ir2 = real_div(x=one_h, y=r2)[name=string(\"ir2\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> ir4 = mul(x=ir2, y=ir2)[name=string(\"ir4\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> ir6 = mul(x=ir4, y=ir2)[name=string(\"ir6\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> ir12 = mul(x=ir6, y=ir6)[name=string(\"ir12\")];\n", N, N];

    [m appendString:@"        fp16 two = const()[name=string(\"two\"), val=fp16(2.0)];\n"];
    [m appendString:@"        fp16 c24 = const()[name=string(\"c24\"), val=fp16(24.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> t1 = mul(x=two, y=ir12)[name=string(\"t1\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> t2 = sub(x=t1, y=ir6)[name=string(\"t2\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> t3 = mul(x=ir2, y=t2)[name=string(\"t3\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> fr = mul(x=c24, y=t3)[name=string(\"fr\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> fm = mul(x=fr, y=mask)[name=string(\"fm\")];\n", N, N];

    /* PE per pair: 4*(ir12 - ir6) - v_shift, masked */
    [m appendString:@"        fp16 four = const()[name=string(\"four\"), val=fp16(4.0)];\n"];
    [m appendFormat:@"        fp16 vs = const()[name=string(\"vs\"), val=fp16(%.8f)];\n", v_shift];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> p1 = sub(x=ir12, y=ir6)[name=string(\"p1\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> p2 = mul(x=four, y=p1)[name=string(\"p2\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> p3 = sub(x=p2, y=vs)[name=string(\"p3\")];\n", N, N];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> pm = mul(x=p3, y=mask)[name=string(\"pm\")];\n", N, N];

    [m appendString:@"    } -> (fm, pm);\n}\n"];
    return m;
}

/* --- Static state --- */
static ANEKernel *g_kernel = NULL;
static int g_N = 0;

static void ane_init_kernel(int N) {
    if (g_kernel && g_N == N) return;
    if (g_kernel) { ane_free(g_kernel); g_kernel = NULL; }

    NSString *mil_str = generate_lj_ane_mil(N);
    NSData *mil = [mil_str dataUsingEncoding:NSUTF8StringEncoding];

    uint8_t *wbuf = (uint8_t *)calloc(128, 1);
    wbuf[0] = 0x01; wbuf[4] = 0x02;
    wbuf[64] = 0xEF; wbuf[65] = 0xBE; wbuf[66] = 0xAD; wbuf[67] = 0xDE; wbuf[68] = 0x01;
    NSData *wb = [NSData dataWithBytesNoCopy:wbuf length:128 freeWhenDone:YES];

    size_t mat_f32 = (size_t)N * N * sizeof(float);
    size_t in_sizes[] = { mat_f32 };
    size_t mat_f16 = (size_t)N * N * sizeof(uint16_t);
    size_t out_sizes[] = { mat_f16, mat_f16 };

    printf("[ANE Direct] Compiling single-input LJ kernel for N=%d...\n", N);
    g_kernel = ane_compile(mil, wb, 1, in_sizes, 2, out_sizes);
    if (g_kernel) {
        printf("[ANE Direct] Compiled. N=%d, FP16 LJ on ANE + NEON accumulation.\n", N);
        g_N = N;
    } else {
        fprintf(stderr, "[ANE Direct] Compilation FAILED.\n");
    }
}

/* --- NEON distance matrix: r², dx, dy, dz with PBC minimum image --- */
static void compute_distance_matrix_neon(const MDSystem *sys,
                                          float *restrict r2,
                                          float *restrict dx,
                                          float *restrict dy,
                                          float *restrict dz) {
    const int n = sys->n_real;
    const float lbox = sys->lbox;
    const float inv_box = 1.0f / lbox;
    const float32x4_t lbox_v = vdupq_n_f32(lbox);
    const float32x4_t inv_box_v = vdupq_n_f32(inv_box);
    const float *restrict px = sys->x;
    const float *restrict py = sys->y;
    const float *restrict pz = sys->z;

    /* n is padded to multiple of 4 by MDSystem */
    const int n4 = n & ~3;

    for (int i = 0; i < n; i++) {
        const float32x4_t xi = vdupq_n_f32(px[i]);
        const float32x4_t yi = vdupq_n_f32(py[i]);
        const float32x4_t zi = vdupq_n_f32(pz[i]);
        float *restrict r2_row = r2 + (size_t)i * n;
        float *restrict dx_row = dx + (size_t)i * n;
        float *restrict dy_row = dy + (size_t)i * n;
        float *restrict dz_row = dz + (size_t)i * n;

        int j = 0;
        for (; j < n4; j += 4) {
            /* Load 4 j-particle positions */
            float32x4_t xj = vld1q_f32(px + j);
            float32x4_t yj = vld1q_f32(py + j);
            float32x4_t zj = vld1q_f32(pz + j);

            /* Displacement */
            float32x4_t ddx = vsubq_f32(xi, xj);
            float32x4_t ddy = vsubq_f32(yi, yj);
            float32x4_t ddz = vsubq_f32(zi, zj);

            /* PBC minimum image: d -= L * round(d / L) */
            ddx = vsubq_f32(ddx, vmulq_f32(lbox_v, vrndnq_f32(vmulq_f32(ddx, inv_box_v))));
            ddy = vsubq_f32(ddy, vmulq_f32(lbox_v, vrndnq_f32(vmulq_f32(ddy, inv_box_v))));
            ddz = vsubq_f32(ddz, vmulq_f32(lbox_v, vrndnq_f32(vmulq_f32(ddz, inv_box_v))));

            /* r² = dx² + dy² + dz² */
            float32x4_t rr = vmulq_f32(ddx, ddx);
            rr = vfmaq_f32(rr, ddy, ddy);
            rr = vfmaq_f32(rr, ddz, ddz);

            vst1q_f32(dx_row + j, ddx);
            vst1q_f32(dy_row + j, ddy);
            vst1q_f32(dz_row + j, ddz);
            vst1q_f32(r2_row + j, rr);
        }
        /* Scalar remainder (n not multiple of 4) */
        for (; j < n; j++) {
            float ddx = px[i] - px[j];
            float ddy = py[i] - py[j];
            float ddz = pz[i] - pz[j];
            ddx -= lbox * rintf(ddx * inv_box);
            ddy -= lbox * rintf(ddy * inv_box);
            ddz -= lbox * rintf(ddz * inv_box);
            dx_row[j] = ddx;
            dy_row[j] = ddy;
            dz_row[j] = ddz;
            r2_row[j] = ddx*ddx + ddy*ddy + ddz*ddz;
        }
        /* Self-exclusion: push r²[i,i] beyond cutoff */
        r2_row[i] = 100.0f;
    }
}

/* --- NEON accumulation: fm(FP16) × dx/dy/dz(FP32) → forces, pm(FP16) → PE --- */
static void accumulate_forces_neon(const uint16_t *restrict fm_h,
                                    const uint16_t *restrict pm_h,
                                    const float *restrict dx_mat,
                                    const float *restrict dy_mat,
                                    const float *restrict dz_mat,
                                    MDSystem *sys, float *pe_out, int n) {
    const int n4 = n & ~3;
    double pe_accum = 0.0;

    for (int i = 0; i < n; i++) {
        float32x4_t fx_v = vdupq_n_f32(0.0f);
        float32x4_t fy_v = vdupq_n_f32(0.0f);
        float32x4_t fz_v = vdupq_n_f32(0.0f);
        float32x4_t pe_v = vdupq_n_f32(0.0f);

        const size_t row = (size_t)i * n;
        const uint16_t *fm_row = fm_h + row;
        const uint16_t *pm_row = pm_h + row;
        const float *dx_row = dx_mat + row;
        const float *dy_row = dy_mat + row;
        const float *dz_row = dz_mat + row;

        int j = 0;
        for (; j < n4; j += 4) {
            /* Convert 4 FP16 → FP32 in one instruction */
            float32x4_t fm4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(fm_row + j)));
            float32x4_t pm4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(pm_row + j)));

            /* Load 4 FP32 displacement components */
            float32x4_t dx4 = vld1q_f32(dx_row + j);
            float32x4_t dy4 = vld1q_f32(dy_row + j);
            float32x4_t dz4 = vld1q_f32(dz_row + j);

            /* FMA: force += fm * dr */
            fx_v = vfmaq_f32(fx_v, fm4, dx4);
            fy_v = vfmaq_f32(fy_v, fm4, dy4);
            fz_v = vfmaq_f32(fz_v, fm4, dz4);
            pe_v = vaddq_f32(pe_v, pm4);
        }

        /* Horizontal sum of 4 lanes */
        float fx_sum = vaddvq_f32(fx_v);
        float fy_sum = vaddvq_f32(fy_v);
        float fz_sum = vaddvq_f32(fz_v);
        float pe_sum = vaddvq_f32(pe_v);

        /* Scalar remainder */
        for (; j < n; j++) {
            /* Scalar FP16→F32 fallback for tail elements */
            uint16_t hfm = fm_row[j];
            uint32_t s = (hfm >> 15) & 1, e = (hfm >> 10) & 0x1F, mt = hfm & 0x3FF;
            float fmv;
            if (e == 0) fmv = (s ? -1.f : 1.f) * (mt / 1024.f) / 16384.f;
            else if (e == 31) fmv = mt ? NAN : (s ? -INFINITY : INFINITY);
            else fmv = (s ? -1.f : 1.f) * (1.f + mt / 1024.f) * powf(2.f, (float)e - 15.f);

            uint16_t hpm = pm_row[j];
            s = (hpm >> 15) & 1; e = (hpm >> 10) & 0x1F; mt = hpm & 0x3FF;
            float pmv;
            if (e == 0) pmv = (s ? -1.f : 1.f) * (mt / 1024.f) / 16384.f;
            else if (e == 31) pmv = mt ? NAN : (s ? -INFINITY : INFINITY);
            else pmv = (s ? -1.f : 1.f) * (1.f + mt / 1024.f) * powf(2.f, (float)e - 15.f);

            fx_sum += fmv * dx_row[j];
            fy_sum += fmv * dy_row[j];
            fz_sum += fmv * dz_row[j];
            pe_sum += pmv;
        }

        sys->fx[i] = fx_sum;
        sys->fy[i] = fy_sum;
        sys->fz[i] = fz_sum;
        pe_accum += (double)pe_sum;
    }
    *pe_out = (float)(pe_accum * 0.5);
}

/* --- ForceFunc interface --- */
void md_force_ane_direct(MDSystem *sys, float *pe_out) {
    int n = sys->n_real;

    @autoreleasepool {
        ane_init_kernel(n);
        if (!g_kernel) {
            memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fz, 0, (size_t)sys->n * sizeof(float));
            *pe_out = 0.0f;
            return;
        }

        size_t mat_f32 = (size_t)n * n * sizeof(float);

        /* Zero-copy: write distance matrix directly into IOSurface (unified memory) */
        IOSurfaceLock(g_kernel->ioInputs[0], 0, NULL);
        float *r2_mat = (float *)IOSurfaceGetBaseAddress(g_kernel->ioInputs[0]);

        /* dx/dy/dz still need separate buffers for CPU accumulation */
        float *dx_mat = (float *)malloc(mat_f32);
        float *dy_mat = (float *)malloc(mat_f32);
        float *dz_mat = (float *)malloc(mat_f32);

        /* CPU NEON: write r² directly into ANE IOSurface, dx/dy/dz to malloc'd buffers */
        compute_distance_matrix_neon(sys, r2_mat, dx_mat, dy_mat, dz_mat);
        IOSurfaceUnlock(g_kernel->ioInputs[0], 0, NULL);

        /* ANE eval — r² already in IOSurface, zero memcpy */
        bool ok = ane_eval(g_kernel);
        if (!ok) {
            free(dx_mat); free(dy_mat); free(dz_mat);
            *pe_out = 0.0f;
            return;
        }

        /* Zero-copy read: accumulate directly from output IOSurface */
        IOSurfaceLock(g_kernel->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(g_kernel->ioOutputs[1], kIOSurfaceLockReadOnly, NULL);
        const uint16_t *fm_h = (const uint16_t *)IOSurfaceGetBaseAddress(g_kernel->ioOutputs[0]);
        const uint16_t *pm_h = (const uint16_t *)IOSurfaceGetBaseAddress(g_kernel->ioOutputs[1]);

        accumulate_forces_neon(fm_h, pm_h, dx_mat, dy_mat, dz_mat, sys, pe_out, n);

        IOSurfaceUnlock(g_kernel->ioOutputs[0], kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(g_kernel->ioOutputs[1], kIOSurfaceLockReadOnly, NULL);

        free(dx_mat); free(dy_mat); free(dz_mat);

        /* Zero padding */
        for (int i = n; i < sys->n; i++)
            sys->fx[i] = sys->fy[i] = sys->fz[i] = 0.0f;
    }
}
