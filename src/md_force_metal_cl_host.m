/*
 * Metal Cell List Host — GPU O(N) force evaluation.
 *
 * CPU-side: build cell list, sort particles by cell, create
 * cell_start/cell_end offset arrays, upload to GPU.
 * GPU-side: each thread iterates 27 neighbor cells via offsets,
 * reads j-positions from contiguous sorted memory.
 * CPU-side: unsort forces back to original particle order.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "md_types.h"
#include "md_force.h"
#include "md_celllist.h"
#include <string.h>
#include <stdio.h>

typedef struct {
    uint32_t n_particles;
    uint32_t ncells_side;
    float    lbox;
    float    inv_box;
    float    rc2;
    float    v_shift;
    float    inv_cell;
} MDMetalCLParams;

/* Embedded shader source for runtime compilation */
static NSString *const kMetalCLShaderSource =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct MDMetalCLParams {\n"
    "    uint   n_particles;\n"
    "    uint   ncells_side;\n"
    "    float  lbox;\n"
    "    float  inv_box;\n"
    "    float  rc2;\n"
    "    float  v_shift;\n"
    "    float  inv_cell;\n"
    "};\n"
    "\n"
    "kernel void lj_force_cl_kernel(\n"
    "    const device float *sx         [[ buffer(0) ]],\n"
    "    const device float *sy         [[ buffer(1) ]],\n"
    "    const device float *sz         [[ buffer(2) ]],\n"
    "    device float       *fx         [[ buffer(3) ]],\n"
    "    device float       *fy         [[ buffer(4) ]],\n"
    "    device float       *fz         [[ buffer(5) ]],\n"
    "    device float       *pe_out     [[ buffer(6) ]],\n"
    "    constant MDMetalCLParams &p    [[ buffer(7) ]],\n"
    "    const device uint  *cell_start [[ buffer(8) ]],\n"
    "    const device uint  *cell_end   [[ buffer(9) ]],\n"
    "    uint tid [[ thread_position_in_grid ]]\n"
    ") {\n"
    "    if (tid >= p.n_particles) return;\n"
    "\n"
    "    const float xi = sx[tid];\n"
    "    const float yi = sy[tid];\n"
    "    const float zi = sz[tid];\n"
    "\n"
    "    const uint ncs = p.ncells_side;\n"
    "\n"
    "    uint cix = min((uint)(xi * p.inv_cell), ncs - 1);\n"
    "    uint ciy = min((uint)(yi * p.inv_cell), ncs - 1);\n"
    "    uint ciz = min((uint)(zi * p.inv_cell), ncs - 1);\n"
    "\n"
    "    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;\n"
    "    float pei = 0.0f;\n"
    "\n"
    "    for (uint di = 0; di < 3; di++) {\n"
    "        uint nx = (cix + ncs + di - 1) % ncs;\n"
    "        for (uint dj = 0; dj < 3; dj++) {\n"
    "            uint ny = (ciy + ncs + dj - 1) % ncs;\n"
    "            for (uint dk = 0; dk < 3; dk++) {\n"
    "                uint nz = (ciz + ncs + dk - 1) % ncs;\n"
    "                uint cidx = (nx * ncs + ny) * ncs + nz;\n"
    "\n"
    "                uint js = cell_start[cidx];\n"
    "                uint je = cell_end[cidx];\n"
    "\n"
    "                for (uint j = js; j < je; j++) {\n"
    "                    if (j == tid) continue;\n"
    "\n"
    "                    float ddx = xi - sx[j];\n"
    "                    float ddy = yi - sy[j];\n"
    "                    float ddz = zi - sz[j];\n"
    "\n"
    "                    ddx -= p.lbox * rint(ddx * p.inv_box);\n"
    "                    ddy -= p.lbox * rint(ddy * p.inv_box);\n"
    "                    ddz -= p.lbox * rint(ddz * p.inv_box);\n"
    "\n"
    "                    float r2 = fma(ddx, ddx, fma(ddy, ddy, ddz * ddz));\n"
    "\n"
    "                    if (r2 < p.rc2) {\n"
    "                        float inv_r2  = 1.0f / r2;\n"
    "                        float inv_r6  = inv_r2 * inv_r2 * inv_r2;\n"
    "                        float inv_r12 = inv_r6 * inv_r6;\n"
    "\n"
    "                        float f_over_r = 24.0f * inv_r2\n"
    "                                       * fma(2.0f, inv_r12, -inv_r6);\n"
    "\n"
    "                        fxi = fma(f_over_r, ddx, fxi);\n"
    "                        fyi = fma(f_over_r, ddy, fyi);\n"
    "                        fzi = fma(f_over_r, ddz, fzi);\n"
    "\n"
    "                        pei += 4.0f * (inv_r12 - inv_r6) - p.v_shift;\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    fx[tid]     = fxi;\n"
    "    fy[tid]     = fyi;\n"
    "    fz[tid]     = fzi;\n"
    "    pe_out[tid] = pei * 0.5f;\n"
    "}\n";

/* Persistent Metal state */
static id<MTLDevice>               g_cl_device   = nil;
static id<MTLCommandQueue>         g_cl_queue    = nil;
static id<MTLComputePipelineState> g_cl_pipeline = nil;

/* GPU buffers */
static id<MTLBuffer> g_cl_buf_sx     = nil;  /* sorted positions */
static id<MTLBuffer> g_cl_buf_sy     = nil;
static id<MTLBuffer> g_cl_buf_sz     = nil;
static id<MTLBuffer> g_cl_buf_fx     = nil;  /* sorted forces out */
static id<MTLBuffer> g_cl_buf_fy     = nil;
static id<MTLBuffer> g_cl_buf_fz     = nil;
static id<MTLBuffer> g_cl_buf_pe     = nil;
static id<MTLBuffer> g_cl_buf_params = nil;
static id<MTLBuffer> g_cl_buf_cstart = nil;  /* cell offsets */
static id<MTLBuffer> g_cl_buf_cend   = nil;
static int           g_cl_buf_n      = 0;
static int           g_cl_buf_ncells = 0;

/* CPU scratch for sorting */
static int  *g_cl_orig_idx  = NULL;
static int   g_cl_scratch_n = 0;

/* Cell list */
static MDCellList *g_cl_celllist = NULL;

static int metal_cl_init(void) {
    if (g_cl_device) return 0;

    @autoreleasepool {
        g_cl_device = MTLCreateSystemDefaultDevice();
        if (!g_cl_device) {
            fprintf(stderr, "[Metal-CL] No GPU device found.\n");
            return -1;
        }

        g_cl_queue = [g_cl_device newCommandQueue];

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeFast;
            opts.languageVersion = MTLLanguageVersion3_2;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }

        NSError *error = nil;
        id<MTLLibrary> lib = [g_cl_device newLibraryWithSource:kMetalCLShaderSource
                                                       options:opts
                                                         error:&error];
        if (!lib) {
            fprintf(stderr, "[Metal-CL] Shader compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"lj_force_cl_kernel"];
        if (!fn) {
            fprintf(stderr, "[Metal-CL] Kernel function not found.\n");
            return -1;
        }

        g_cl_pipeline = [g_cl_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_cl_pipeline) {
            fprintf(stderr, "[Metal-CL] Pipeline creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        fprintf(stderr, "[Metal-CL] Compiled cell-list GPU kernel (runtime JIT)\n");
    }
    return 0;
}

static int metal_cl_alloc_buffers(int n, int ncells_total) {
    @autoreleasepool {
        MTLResourceOptions mopts = MTLResourceStorageModeShared;

        if (n > g_cl_buf_n) {
            NSUInteger sz = (NSUInteger)n * sizeof(float);
            g_cl_buf_sx = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_sy = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_sz = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_fx = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_fy = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_fz = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_pe = [g_cl_device newBufferWithLength:sz options:mopts];
            g_cl_buf_params = [g_cl_device newBufferWithLength:sizeof(MDMetalCLParams)
                                                       options:mopts];
            g_cl_buf_n = n;
        }

        if (ncells_total > g_cl_buf_ncells) {
            NSUInteger csz = (NSUInteger)ncells_total * sizeof(uint32_t);
            g_cl_buf_cstart = [g_cl_device newBufferWithLength:csz options:mopts];
            g_cl_buf_cend   = [g_cl_device newBufferWithLength:csz options:mopts];
            g_cl_buf_ncells = ncells_total;
        }
    }

    if (n > g_cl_scratch_n) {
        free(g_cl_orig_idx);
        g_cl_orig_idx  = malloc((size_t)n * sizeof(int));
        g_cl_scratch_n = n;
    }

    return 0;
}

void md_force_metal_cl(MDSystem *sys, float *pe_out) {
    @autoreleasepool {
        if (metal_cl_init() != 0) {
            memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fz, 0, (size_t)sys->n * sizeof(float));
            *pe_out = 0.0f;
            return;
        }

        const int n_real = sys->n_real;

        /* Build cell list on CPU */
        if (!g_cl_celllist) g_cl_celllist = md_celllist_create(n_real);
        md_celllist_build(g_cl_celllist, sys);

        const int ncells_total = g_cl_celllist->ncells_total;
        const int ncs = g_cl_celllist->ncells_side;

        metal_cl_alloc_buffers(n_real, ncells_total);

        /* Sort particles by cell into Metal buffers */
        float    *sorted_x = (float *)[g_cl_buf_sx contents];
        float    *sorted_y = (float *)[g_cl_buf_sy contents];
        float    *sorted_z = (float *)[g_cl_buf_sz contents];
        uint32_t *cs_arr   = (uint32_t *)[g_cl_buf_cstart contents];
        uint32_t *ce_arr   = (uint32_t *)[g_cl_buf_cend contents];

        int pos = 0;
        for (int c = 0; c < ncells_total; c++) {
            cs_arr[c] = (uint32_t)pos;
            int j = g_cl_celllist->head[c];
            while (j != -1) {
                sorted_x[pos] = sys->x[j];
                sorted_y[pos] = sys->y[j];
                sorted_z[pos] = sys->z[j];
                g_cl_orig_idx[pos] = j;
                pos++;
                j = g_cl_celllist->next[j];
            }
            ce_arr[c] = (uint32_t)pos;
        }

        /* Set params */
        MDMetalCLParams *params = (MDMetalCLParams *)[g_cl_buf_params contents];
        params->n_particles = (uint32_t)n_real;
        params->ncells_side = (uint32_t)ncs;
        params->lbox        = sys->lbox;
        params->inv_box     = 1.0f / sys->lbox;
        params->rc2         = MD_CUTOFF * MD_CUTOFF;
        params->v_shift     = md_lj_shift();
        params->inv_cell    = g_cl_celllist->inv_cell;

        /* Encode + dispatch */
        id<MTLCommandBuffer> cmdBuf = [g_cl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_cl_pipeline];
        [enc setBuffer:g_cl_buf_sx     offset:0 atIndex:0];
        [enc setBuffer:g_cl_buf_sy     offset:0 atIndex:1];
        [enc setBuffer:g_cl_buf_sz     offset:0 atIndex:2];
        [enc setBuffer:g_cl_buf_fx     offset:0 atIndex:3];
        [enc setBuffer:g_cl_buf_fy     offset:0 atIndex:4];
        [enc setBuffer:g_cl_buf_fz     offset:0 atIndex:5];
        [enc setBuffer:g_cl_buf_pe     offset:0 atIndex:6];
        [enc setBuffer:g_cl_buf_params offset:0 atIndex:7];
        [enc setBuffer:g_cl_buf_cstart offset:0 atIndex:8];
        [enc setBuffer:g_cl_buf_cend   offset:0 atIndex:9];

        MTLSize grid = MTLSizeMake((NSUInteger)n_real, 1, 1);
        MTLSize tg   = MTLSizeMake(256, 1, 1);
        NSUInteger max_tg = [g_cl_pipeline maxTotalThreadsPerThreadgroup];
        if (tg.width > max_tg) tg.width = max_tg;

        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* Unsort forces back to original particle order */
        float *gpu_fx = (float *)[g_cl_buf_fx contents];
        float *gpu_fy = (float *)[g_cl_buf_fy contents];
        float *gpu_fz = (float *)[g_cl_buf_fz contents];

        memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
        memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
        memset(sys->fz, 0, (size_t)sys->n * sizeof(float));

        for (int i = 0; i < n_real; i++) {
            int orig = g_cl_orig_idx[i];
            sys->fx[orig] = gpu_fx[i];
            sys->fy[orig] = gpu_fy[i];
            sys->fz[orig] = gpu_fz[i];
        }

        /* Reduce PE */
        float *pe_buf = (float *)[g_cl_buf_pe contents];
        float pe_total = 0.0f;
        for (int i = 0; i < n_real; i++) {
            pe_total += pe_buf[i];
        }
        *pe_out = pe_total;
    }
}
