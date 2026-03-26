/*
 * Metal host bridge for NBNXM cluster pair force kernel.
 *
 * CPU: build cluster pair list each timestep (cheap at <100K particles)
 * GPU: dispatch 8×8 threadgroup per i-cluster, force reduction in threadgroup memory
 * CPU: unsort forces back to original particle order
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "md_types.h"
#include "md_force.h"
#include "md_cluster.h"
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t n_clusters;
    uint32_t n_atoms_total;
    float    lbox;
    float    inv_box;
    float    rc2;
    float    v_shift;
} ClusterParamsCPU;

static NSString *load_shader(void) {
    NSError *err = nil;
    NSString *path = @"src/md_force_metal_nbnxm.metal";
    return [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
}

/* Static Metal state */
static id<MTLDevice>               g_device = nil;
static id<MTLCommandQueue>         g_queue  = nil;
static id<MTLComputePipelineState> g_pipe   = nil;

static id<MTLBuffer> g_buf_cx = nil, g_buf_cy = nil, g_buf_cz = nil;
static id<MTLBuffer> g_buf_fx = nil, g_buf_fy = nil, g_buf_fz = nil;
static id<MTLBuffer> g_buf_pe = nil;
static id<MTLBuffer> g_buf_params = nil;
static id<MTLBuffer> g_buf_pair_start = nil, g_buf_pair_end = nil;
static id<MTLBuffer> g_buf_j_list = nil;
static id<MTLBuffer> g_buf_cl_count = nil;

static int g_buf_n_atoms = 0;      /* capacity: n_clusters * 8 */
static int g_buf_n_clusters = 0;
static int g_buf_n_pairs = 0;

static void metal_init(void) {
    if (g_device) return;

    g_device = MTLCreateSystemDefaultDevice();
    g_queue  = [g_device newCommandQueue];

    NSString *src = load_shader();
    NSError *error = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:src options:opts error:&error];
    if (!lib) {
        fprintf(stderr, "[NBNXM] Shader compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return;
    }

    g_pipe = [g_device newComputePipelineStateWithFunction:
              [lib newFunctionWithName:@"lj_force_cluster"] error:&error];

    g_buf_params = [g_device newBufferWithLength:sizeof(ClusterParamsCPU)
                                         options:MTLResourceStorageModeShared];
}

static void ensure_buffers(int n_atoms, int n_clusters, int n_pairs) {
    MTLResourceOptions opts = MTLResourceStorageModeShared;

    if (n_atoms > g_buf_n_atoms) {
        NSUInteger sz = (NSUInteger)n_atoms * sizeof(float);
        g_buf_cx = [g_device newBufferWithLength:sz options:opts];
        g_buf_cy = [g_device newBufferWithLength:sz options:opts];
        g_buf_cz = [g_device newBufferWithLength:sz options:opts];
        g_buf_fx = [g_device newBufferWithLength:sz options:opts];
        g_buf_fy = [g_device newBufferWithLength:sz options:opts];
        g_buf_fz = [g_device newBufferWithLength:sz options:opts];
        g_buf_pe = [g_device newBufferWithLength:sz options:opts];
        g_buf_n_atoms = n_atoms;
    }

    if (n_clusters > g_buf_n_clusters) {
        NSUInteger sz = (NSUInteger)n_clusters * sizeof(uint32_t);
        g_buf_pair_start = [g_device newBufferWithLength:sz options:opts];
        g_buf_pair_end   = [g_device newBufferWithLength:sz options:opts];
        g_buf_cl_count   = [g_device newBufferWithLength:sz options:opts];
        g_buf_n_clusters = n_clusters;
    }

    if (n_pairs > g_buf_n_pairs) {
        NSUInteger sz = (NSUInteger)n_pairs * sizeof(uint32_t);
        g_buf_j_list = [g_device newBufferWithLength:sz options:opts];
        g_buf_n_pairs = n_pairs;
    }
}

static float lj_shift(void) {
    float rc2 = MD_CUTOFF * MD_CUTOFF;
    float ri2 = 1.0f / rc2;
    float ri6 = ri2 * ri2 * ri2;
    return 4.0f * MD_EPSILON * (ri6 * ri6 - ri6);
}

void md_force_metal_nbnxm(MDSystem *sys, float *pe_out) {
    metal_init();
    if (!g_pipe) {
        fprintf(stderr, "[NBNXM] Pipeline not ready\n");
        *pe_out = 0.0f;
        return;
    }

    /* Build cluster pair list on CPU */
    MDClusterPairList *cpl = md_cluster_build(sys);
    int n_clusters = cpl->n_clusters;
    int n_atoms = n_clusters * CLUSTER_SIZE;
    int n_pairs = cpl->n_pairs_total;

    ensure_buffers(n_atoms, n_clusters, n_pairs);

    /* Upload cluster positions */
    memcpy([g_buf_cx contents], cpl->cluster_x, (size_t)n_atoms * sizeof(float));
    memcpy([g_buf_cy contents], cpl->cluster_y, (size_t)n_atoms * sizeof(float));
    memcpy([g_buf_cz contents], cpl->cluster_z, (size_t)n_atoms * sizeof(float));

    /* Upload pair list */
    uint32_t *ps_buf = (uint32_t *)[g_buf_pair_start contents];
    uint32_t *pe_buf = (uint32_t *)[g_buf_pair_end contents];
    uint32_t *jl_buf = (uint32_t *)[g_buf_j_list contents];
    uint32_t *cc_buf = (uint32_t *)[g_buf_cl_count contents];

    for (int i = 0; i < n_clusters; i++) {
        ps_buf[i] = (uint32_t)cpl->pair_start[i];
        pe_buf[i] = (uint32_t)cpl->pair_end[i];
        cc_buf[i] = (uint32_t)cpl->cluster_count[i];
    }
    for (int i = 0; i < n_pairs; i++) {
        jl_buf[i] = (uint32_t)cpl->j_list[i];
    }

    /* Set params */
    ClusterParamsCPU *params = (ClusterParamsCPU *)[g_buf_params contents];
    params->n_clusters   = (uint32_t)n_clusters;
    params->n_atoms_total = (uint32_t)n_atoms;
    params->lbox         = sys->lbox;
    params->inv_box      = 1.0f / sys->lbox;
    params->rc2          = MD_CUTOFF * MD_CUTOFF;
    params->v_shift      = lj_shift();

    /* Dispatch: one threadgroup per i-cluster, 64 threads (8 i-atoms × 8 j-atoms) */
    MTLSize grid = MTLSizeMake((NSUInteger)n_clusters, 1, 1);
    MTLSize tg   = MTLSizeMake(64, 1, 1);

    id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:g_pipe];
    [enc setBuffer:g_buf_cx         offset:0 atIndex:0];
    [enc setBuffer:g_buf_cy         offset:0 atIndex:1];
    [enc setBuffer:g_buf_cz         offset:0 atIndex:2];
    [enc setBuffer:g_buf_fx         offset:0 atIndex:3];
    [enc setBuffer:g_buf_fy         offset:0 atIndex:4];
    [enc setBuffer:g_buf_fz         offset:0 atIndex:5];
    [enc setBuffer:g_buf_pe         offset:0 atIndex:6];
    [enc setBuffer:g_buf_params     offset:0 atIndex:7];
    [enc setBuffer:g_buf_pair_start offset:0 atIndex:8];
    [enc setBuffer:g_buf_pair_end   offset:0 atIndex:9];
    [enc setBuffer:g_buf_j_list     offset:0 atIndex:10];
    [enc setBuffer:g_buf_cl_count   offset:0 atIndex:11];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    /* Unsort forces back to original particle order */
    float *gpu_fx = (float *)[g_buf_fx contents];
    float *gpu_fy = (float *)[g_buf_fy contents];
    float *gpu_fz = (float *)[g_buf_fz contents];
    float *gpu_pe = (float *)[g_buf_pe contents];

    /* Zero all forces first */
    memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
    memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
    memset(sys->fz, 0, (size_t)sys->n * sizeof(float));

    float pe_total = 0.0f;
    int n_atom_slots = n_clusters * CLUSTER_SIZE;
    for (int i = 0; i < n_atom_slots; i++) {
        int orig = cpl->cluster_orig_idx[i];
        if (orig >= 0) {
            sys->fx[orig] = gpu_fx[i];
            sys->fy[orig] = gpu_fy[i];
            sys->fz[orig] = gpu_fz[i];
            pe_total += gpu_pe[i];
        }
    }

    *pe_out = pe_total;
    md_cluster_destroy(cpl);
}
