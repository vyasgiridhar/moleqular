/*
 * Metal host bridge for BVH two-pass force kernel.
 *
 * Pass 1: BVH traversal → neighbor list (GPU)
 * Pass 2: Force computation from neighbor list (GPU)
 *
 * BVH built on CPU each timestep (cheap at <100K particles).
 * Sorted positions + tree uploaded via shared memory (zero-copy on Apple Silicon).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "md_types.h"
#include "md_force.h"
#include "md_bvh.h"
#include <string.h>
#include <math.h>

/* Matches BVHParams in the Metal shader */
typedef struct {
    uint32_t n_particles;
    uint32_t n_internal;
    float    lbox;
    float    inv_box;
    float    rc2;
    float    v_shift;
    float    inv_quant;
} BVHParamsCPU;

/* Embedded shader source */
static NSString *shader_source(void) {
    NSError *err = nil;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"md_force_metal_bvh"
                                                     ofType:@"metal"];
    if (path) {
        return [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    }
    /* Fallback: load from working directory */
    return [NSString stringWithContentsOfFile:@"src/md_force_metal_bvh.metal"
                                     encoding:NSUTF8StringEncoding
                                        error:&err];
}

/* --- Static Metal state --- */
static id<MTLDevice>              g_device   = nil;
static id<MTLCommandQueue>        g_queue    = nil;
static id<MTLComputePipelineState> g_pipe_traverse = nil;
static id<MTLComputePipelineState> g_pipe_forces   = nil;

/* Buffers */
static id<MTLBuffer> g_buf_sx  = nil, g_buf_sy = nil, g_buf_sz = nil;
static id<MTLBuffer> g_buf_nodes    = nil;
static id<MTLBuffer> g_buf_nbrs     = nil;
static id<MTLBuffer> g_buf_n_nbrs   = nil;
static id<MTLBuffer> g_buf_params   = nil;
static id<MTLBuffer> g_buf_fx  = nil, g_buf_fy = nil, g_buf_fz = nil;
static id<MTLBuffer> g_buf_pe  = nil;

static int  g_buf_n      = 0;   /* capacity for particles */
static int  g_buf_nodes_n = 0;  /* capacity for BVH nodes */
static int *g_sorted_idx = NULL;
static int  g_sorted_n   = 0;

#define MAX_NBRS 128

static void ensure_buffers(int n, int n_nodes) {
    MTLResourceOptions opts = MTLResourceStorageModeShared;

    if (n > g_buf_n) {
        NSUInteger sz_f = (NSUInteger)n * sizeof(float);
        NSUInteger sz_nbrs = (NSUInteger)n * MAX_NBRS * sizeof(uint32_t);
        NSUInteger sz_cnt  = (NSUInteger)n * sizeof(uint32_t);

        g_buf_sx     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_sy     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_sz     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_nbrs   = [g_device newBufferWithLength:sz_nbrs options:opts];
        g_buf_n_nbrs = [g_device newBufferWithLength:sz_cnt  options:opts];
        g_buf_fx     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_fy     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_fz     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_pe     = [g_device newBufferWithLength:sz_f    options:opts];
        g_buf_n = n;
    }

    if (n_nodes > g_buf_nodes_n) {
        NSUInteger sz_nodes = (NSUInteger)n_nodes * sizeof(uint32_t) * 4; /* BVHNode = 16 bytes */
        g_buf_nodes = [g_device newBufferWithLength:sz_nodes options:opts];
        g_buf_nodes_n = n_nodes;
    }
}

static void metal_init(void) {
    if (g_device) return;

    g_device = MTLCreateSystemDefaultDevice();
    g_queue  = [g_device newCommandQueue];

    NSString *src = shader_source();
    NSError *error = nil;
    MTLCompileOptions *copts = [[MTLCompileOptions alloc] init];
    copts.fastMathEnabled = YES;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:src options:copts error:&error];
    if (!lib) {
        fprintf(stderr, "[BVH Metal] Shader compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return;
    }

    g_pipe_traverse = [g_device newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@"bvh_traverse"] error:&error];
    g_pipe_forces   = [g_device newComputePipelineStateWithFunction:
                       [lib newFunctionWithName:@"bvh_forces"] error:&error];

    g_buf_params = [g_device newBufferWithLength:sizeof(BVHParamsCPU)
                                         options:MTLResourceStorageModeShared];
}

static float lj_shift(void) {
    float rc2 = MD_CUTOFF * MD_CUTOFF;
    float ri2 = 1.0f / rc2;
    float ri6 = ri2 * ri2 * ri2;
    return 4.0f * MD_EPSILON * (ri6 * ri6 - ri6);
}

void md_force_metal_bvh(MDSystem *sys, float *pe_out) {
    metal_init();

    int n = sys->n_real;
    float lbox = sys->lbox;

    /* Build BVH on CPU */
    MDBVH *bvh = md_bvh_build(sys->x, sys->y, sys->z, n, lbox);
    int n_nodes = bvh->n_nodes;

    /* Ensure GPU buffers */
    ensure_buffers(n, n_nodes);

    /* Upload sorted positions */
    memcpy([g_buf_sx contents], bvh->sorted_x, (size_t)n * sizeof(float));
    memcpy([g_buf_sy contents], bvh->sorted_y, (size_t)n * sizeof(float));
    memcpy([g_buf_sz contents], bvh->sorted_z, (size_t)n * sizeof(float));

    /* Upload BVH nodes (only internal: N-1 nodes) */
    int n_internal = n - 1;
    if (n_internal > 0) {
        memcpy([g_buf_nodes contents], bvh->nodes,
               (size_t)n_internal * sizeof(BVHNode));
    }

    /* Save sorted_idx for unsort */
    if (n > g_sorted_n) {
        free(g_sorted_idx);
        g_sorted_idx = (int *)malloc((size_t)n * sizeof(int));
        g_sorted_n = n;
    }
    memcpy(g_sorted_idx, bvh->sorted_idx, (size_t)n * sizeof(int));

    /* Set params */
    BVHParamsCPU *params = (BVHParamsCPU *)[g_buf_params contents];
    params->n_particles = (uint32_t)n;
    params->n_internal  = (uint32_t)n_internal;
    params->lbox        = lbox;
    params->inv_box     = 1.0f / lbox;
    params->rc2         = MD_CUTOFF * MD_CUTOFF;
    params->v_shift     = lj_shift();
    params->inv_quant   = bvh->inv_quant;

    md_bvh_destroy(bvh);

    /* Dispatch */
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);

    id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];

    /* Pass 1: BVH traversal → neighbor list */
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_traverse];
        [enc setBuffer:g_buf_sx     offset:0 atIndex:0];
        [enc setBuffer:g_buf_sy     offset:0 atIndex:1];
        [enc setBuffer:g_buf_sz     offset:0 atIndex:2];
        [enc setBuffer:g_buf_nodes  offset:0 atIndex:3];
        [enc setBuffer:g_buf_nbrs   offset:0 atIndex:4];
        [enc setBuffer:g_buf_n_nbrs offset:0 atIndex:5];
        [enc setBuffer:g_buf_params offset:0 atIndex:6];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* Pass 2: Force computation from neighbor list */
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:g_pipe_forces];
        [enc setBuffer:g_buf_sx     offset:0 atIndex:0];
        [enc setBuffer:g_buf_sy     offset:0 atIndex:1];
        [enc setBuffer:g_buf_sz     offset:0 atIndex:2];
        [enc setBuffer:g_buf_nbrs   offset:0 atIndex:4];
        [enc setBuffer:g_buf_n_nbrs offset:0 atIndex:5];
        [enc setBuffer:g_buf_params offset:0 atIndex:6];
        [enc setBuffer:g_buf_fx     offset:0 atIndex:7];
        [enc setBuffer:g_buf_fy     offset:0 atIndex:8];
        [enc setBuffer:g_buf_fz     offset:0 atIndex:9];
        [enc setBuffer:g_buf_pe     offset:0 atIndex:10];
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    /* Unsort forces back to original particle order */
    float *gpu_fx = (float *)[g_buf_fx contents];
    float *gpu_fy = (float *)[g_buf_fy contents];
    float *gpu_fz = (float *)[g_buf_fz contents];
    float *gpu_pe = (float *)[g_buf_pe contents];

    float pe_total = 0.0f;
    for (int i = 0; i < n; i++) {
        int orig = g_sorted_idx[i];
        sys->fx[orig] = gpu_fx[i];
        sys->fy[orig] = gpu_fy[i];
        sys->fz[orig] = gpu_fz[i];
        pe_total += gpu_pe[i];
    }

    /* Zero ghost forces */
    for (int i = n; i < sys->n; i++) {
        sys->fx[i] = 0.0f;
        sys->fy[i] = 0.0f;
        sys->fz[i] = 0.0f;
    }

    *pe_out = pe_total;
}
