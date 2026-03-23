/*
 * Apple Neural Engine Force Kernel — LJ via CoreML inference.
 *
 * The 38 TOPS Neural Engine evaluates a small MLP (3×64 SiLU)
 * that approximates F/r as a function of r². Pairwise distances
 * computed on CPU via cell lists, batched into MLMultiArray,
 * predicted on ANE, forces scattered back.
 *
 * This is an experiment, not an optimization. The ANE is designed
 * for neural network inference patterns (conv, matmul, attention),
 * not arbitrary function evaluation. But it's fun to try.
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "md_types.h"
#include "md_force.h"
#include "md_celllist.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Normalization constants (must match gen_ane_model.py) */
#define R2_NORM_MIN 0.64f
#define R2_NORM_MAX 6.25f
/* f_min/f_max loaded from model metadata at runtime */

static MLModel        *g_ane_model    = nil;
static MDCellList     *g_ane_celllist = NULL;
static float           g_ane_f_min    = 0.0f;
static float           g_ane_f_max    = 0.0f;
static int             g_ane_ready    = 0;

/* Pair storage for batching */
#define ANE_MAX_PAIRS (1024 * 1024)
static float *g_ane_r2_buf  = NULL;   /* normalized r² for each pair */
static float *g_ane_dx_buf  = NULL;   /* displacement vectors */
static float *g_ane_dy_buf  = NULL;
static float *g_ane_dz_buf  = NULL;
static int   *g_ane_i_buf   = NULL;   /* particle index i for each pair */

static int ane_init(void) {
    if (g_ane_ready) return 0;

    @autoreleasepool {
        /* Find model: try next to executable, then cwd */
        NSString *paths[] = {
            @"ane_lj.mlmodelc",
            @"ane_lj.mlpackage",
        };

        NSURL *modelURL = nil;
        for (int p = 0; p < 2; p++) {
            /* Try cwd */
            NSURL *url = [NSURL fileURLWithPath:paths[p]];
            if ([[NSFileManager defaultManager] fileExistsAtPath:paths[p]]) {
                modelURL = url;
                break;
            }
            /* Try next to executable */
            NSString *execDir = [[[NSProcessInfo processInfo] arguments][0]
                                  stringByDeletingLastPathComponent];
            NSString *full = [execDir stringByAppendingPathComponent:paths[p]];
            if ([[NSFileManager defaultManager] fileExistsAtPath:full]) {
                modelURL = [NSURL fileURLWithPath:full];
                break;
            }
        }

        if (!modelURL) {
            fprintf(stderr, "[ANE] Model not found. Run: uv run tools/gen_ane_model.py\n");
            return -1;
        }

        /* Compile .mlpackage to .mlmodelc if needed */
        NSError *error = nil;
        NSURL *compiledURL = nil;

        if ([[modelURL pathExtension] isEqualToString:@"mlpackage"]) {
            fprintf(stderr, "[ANE] Compiling model...\n");
            compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
            if (!compiledURL) {
                fprintf(stderr, "[ANE] Model compilation failed: %s\n",
                        [[error localizedDescription] UTF8String]);
                return -1;
            }
        } else {
            compiledURL = modelURL;
        }

        /* Configure for Neural Engine */
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        g_ane_model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:config
                                                error:&error];
        if (!g_ane_model) {
            fprintf(stderr, "[ANE] Failed to load model: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        /* Read normalization constants from metadata */
        NSDictionary *meta = g_ane_model.modelDescription.metadata;
        NSDictionary *userMeta = meta[MLModelCreatorDefinedKey];
        if (userMeta[@"f_min"] && userMeta[@"f_max"]) {
            g_ane_f_min = [userMeta[@"f_min"] floatValue];
            g_ane_f_max = [userMeta[@"f_max"] floatValue];
        } else {
            /* Fallback: compute from analytical LJ at boundaries */
            float inv2 = 1.0f / R2_NORM_MIN;
            float inv6 = inv2 * inv2 * inv2;
            float inv12 = inv6 * inv6;
            g_ane_f_max = 24.0f * inv2 * (2.0f * inv12 - inv6);
            g_ane_f_min = 0.0f;  /* F/r → 0 at cutoff */
        }

        fprintf(stderr, "[ANE] Loaded model from %s\n", [[modelURL path] UTF8String]);
        fprintf(stderr, "[ANE] F/r normalization: [%.2f, %.2f]\n", g_ane_f_min, g_ane_f_max);

        /* Allocate pair buffers */
        g_ane_r2_buf = malloc(ANE_MAX_PAIRS * sizeof(float));
        g_ane_dx_buf = malloc(ANE_MAX_PAIRS * sizeof(float));
        g_ane_dy_buf = malloc(ANE_MAX_PAIRS * sizeof(float));
        g_ane_dz_buf = malloc(ANE_MAX_PAIRS * sizeof(float));
        g_ane_i_buf  = malloc(ANE_MAX_PAIRS * sizeof(int));

        g_ane_ready = 1;
    }
    return 0;
}

void md_force_ane(MDSystem *sys, float *pe_out) {
    @autoreleasepool {
        if (ane_init() != 0) {
            memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fz, 0, (size_t)sys->n * sizeof(float));
            *pe_out = 0.0f;
            return;
        }

        const int   n_real  = sys->n_real;
        const float box     = sys->lbox;
        const float inv_box = 1.0f / box;
        const float rc2     = MD_CUTOFF * MD_CUTOFF;
        const float v_shift = md_lj_shift();
        const float r2_range = R2_NORM_MAX - R2_NORM_MIN;

        /* Zero forces */
        memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
        memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
        memset(sys->fz, 0, (size_t)sys->n * sizeof(float));

        /* Build cell list */
        if (!g_ane_celllist) g_ane_celllist = md_celllist_create(sys->n);
        md_celllist_build(g_ane_celllist, sys);

        const int ncs = g_ane_celllist->ncells_side;

        /* Phase 1: Gather pairs using cell list (CPU) */
        int n_pairs = 0;

        for (int i = 0; i < n_real && n_pairs < ANE_MAX_PAIRS - 512; i++) {
            float xi = sys->x[i], yi = sys->y[i], zi = sys->z[i];

            int cix = (int)(xi * g_ane_celllist->inv_cell);
            int ciy = (int)(yi * g_ane_celllist->inv_cell);
            int ciz = (int)(zi * g_ane_celllist->inv_cell);
            if (cix >= ncs) cix = ncs - 1;
            if (ciy >= ncs) ciy = ncs - 1;
            if (ciz >= ncs) ciz = ncs - 1;
            if (cix < 0) cix = 0;
            if (ciy < 0) ciy = 0;
            if (ciz < 0) ciz = 0;

            for (int dx = -1; dx <= 1; dx++) {
                int nx = (cix + dx + ncs) % ncs;
                for (int dy = -1; dy <= 1; dy++) {
                    int ny = (ciy + dy + ncs) % ncs;
                    for (int dz = -1; dz <= 1; dz++) {
                        int nz = (ciz + dz + ncs) % ncs;
                        int cell_idx = (nx * ncs + ny) * ncs + nz;

                        int j = g_ane_celllist->head[cell_idx];
                        while (j != -1) {
                            if (j != i) {
                                float ddx = xi - sys->x[j];
                                float ddy = yi - sys->y[j];
                                float ddz = zi - sys->z[j];
                                ddx -= box * roundf(ddx * inv_box);
                                ddy -= box * roundf(ddy * inv_box);
                                ddz -= box * roundf(ddz * inv_box);
                                float r2 = ddx*ddx + ddy*ddy + ddz*ddz;

                                if (r2 > R2_NORM_MIN && r2 < rc2) {
                                    g_ane_r2_buf[n_pairs] = (r2 - R2_NORM_MIN) / r2_range;
                                    g_ane_dx_buf[n_pairs] = ddx;
                                    g_ane_dy_buf[n_pairs] = ddy;
                                    g_ane_dz_buf[n_pairs] = ddz;
                                    g_ane_i_buf[n_pairs]  = i;
                                    n_pairs++;
                                }
                            }
                            j = g_ane_celllist->next[j];
                        }
                    }
                }
            }
        }

        if (n_pairs == 0) {
            *pe_out = 0.0f;
            return;
        }

        /* Phase 2: Batch predict F/r on Neural Engine */

        /* Round up to supported batch size */
        int batch_sizes[] = {256, 1024, 4096, 16384, 65536};
        int padded_n = n_pairs;
        for (int b = 0; b < 5; b++) {
            if (batch_sizes[b] >= n_pairs) {
                padded_n = batch_sizes[b];
                break;
            }
        }
        if (padded_n < n_pairs) padded_n = 65536;

        /* Create input MLMultiArray */
        NSArray *shape = @[@(padded_n), @1];
        NSArray *strides = @[@1, @1];
        NSError *error = nil;

        MLMultiArray *input = [[MLMultiArray alloc]
            initWithShape:shape
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&error];

        if (!input) {
            fprintf(stderr, "[ANE] Failed to create input array: %s\n",
                    [[error localizedDescription] UTF8String]);
            *pe_out = 0.0f;
            return;
        }

        /* Copy normalized r² values */
        float *input_ptr = (float *)input.dataPointer;
        memcpy(input_ptr, g_ane_r2_buf, (size_t)n_pairs * sizeof(float));
        /* Zero padding */
        for (int i = n_pairs; i < padded_n; i++) {
            input_ptr[i] = 0.5f;  /* mid-range, harmless */
        }

        /* Create feature provider */
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"r2_norm": input}
                         error:&error];

        /* Predict */
        id<MLFeatureProvider> result = [g_ane_model predictionFromFeatures:provider
                                                                    error:&error];
        if (!result) {
            fprintf(stderr, "[ANE] Prediction failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            *pe_out = 0.0f;
            return;
        }

        /* Extract output */
        MLMultiArray *output = (MLMultiArray *)[result featureValueForName:@"f_over_r_norm"].multiArrayValue;
        float *out_ptr = (float *)output.dataPointer;

        /* Phase 3: Scatter forces using F/r predictions */
        float pe_total = 0.0f;
        float f_range = g_ane_f_max - g_ane_f_min + 1e-8f;

        for (int p = 0; p < n_pairs; p++) {
            /* Denormalize F/r */
            float f_over_r = out_ptr[p] * f_range + g_ane_f_min;

            float ddx = g_ane_dx_buf[p];
            float ddy = g_ane_dy_buf[p];
            float ddz = g_ane_dz_buf[p];
            int i = g_ane_i_buf[p];

            sys->fx[i] += f_over_r * ddx;
            sys->fy[i] += f_over_r * ddy;
            sys->fz[i] += f_over_r * ddz;

            /* PE analytical (for energy tracking) */
            float r2 = g_ane_r2_buf[p] * r2_range + R2_NORM_MIN;
            float inv_r2 = 1.0f / r2;
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;
            pe_total += 4.0f * (inv_r12 - inv_r6) - v_shift;
        }

        *pe_out = pe_total * 0.5f;
    }
}
