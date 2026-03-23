/*
 * Metal Host Bridge — Objective-C glue between C and the GPU.
 *
 * The dame walked in and said she needed her forces computed.
 * "864 particles," she whispered, "Lennard-Jones, periodic box."
 * I reached for the GPU — unified memory, zero-copy buffers.
 * No memcpy, no staging. Just one address space, shared
 * between the CPU and the M4's 10 cores like a secret
 * between old partners on Lamington Road.
 *
 * Architecture:
 *   1. On first call, lazily initialize Metal device, command queue,
 *      compiled pipeline, and GPU buffers (all persistent).
 *   2. On each call, copy positions into GPU buffers,
 *      encode the compute command, dispatch, wait, read back forces.
 *   3. Buffers use MTLResourceStorageModeShared — CPU and GPU see
 *      the same physical memory on Apple Silicon. No DMA, no fuss.
 *
 * Shader loading strategy (tried in order):
 *   1. Precompiled md_force.metallib next to executable (fastest, ~0.7ms)
 *   2. Precompiled md_force.metallib in current directory
 *   3. Runtime compilation from embedded MSL source string (~20-100ms first call)
 *
 * The runtime compilation fallback means this works even without Xcode
 * installed (only Command Line Tools needed). The Metal runtime compiler
 * is always available on macOS.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "md_types.h"
#include "md_force.h"
#include <string.h>
#include <stdio.h>

/* Must match the struct in md_force_metal.metal */
typedef struct {
    uint32_t n_real;
    uint32_t n_padded;
    float    lbox;
    float    inv_box;
    float    rc2;
    float    v_shift;
} MDMetalParams;

#define TILE_SIZE 128

/* ---------------------------------------------------------------------------
 * Embedded Metal shader source — compiled at runtime if no .metallib found.
 *
 * This is the same code as md_force_metal.metal, embedded as a C string.
 * Advantage: zero external file dependencies. The binary is self-contained.
 * Cost: ~20-100ms on first call for runtime compilation (amortized to zero).
 *
 * NOTE: We use adjacent string literal concatenation and escape the MSL
 * preprocessor directives with backslash-newline tricks to prevent the
 * C preprocessor from expanding them. The #include and #define lines
 * are written as plain strings — the Metal runtime compiler handles them.
 * --------------------------------------------------------------------------- */
static NSString *const kMetalShaderSource =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "#define TILE_SIZE 128\n"
    "#define MD_EPSILON 1.0f\n"
    "\n"
    "struct MDMetalParams {\n"
    "    uint   n_real;\n"
    "    uint   n_padded;\n"
    "    float  lbox;\n"
    "    float  inv_box;\n"
    "    float  rc2;\n"
    "    float  v_shift;\n"
    "};\n"
    "\n"
    "kernel void lj_force_kernel(\n"
    "    const device float *x       [[ buffer(0) ]],\n"
    "    const device float *y       [[ buffer(1) ]],\n"
    "    const device float *z       [[ buffer(2) ]],\n"
    "    device float       *fx      [[ buffer(3) ]],\n"
    "    device float       *fy      [[ buffer(4) ]],\n"
    "    device float       *fz      [[ buffer(5) ]],\n"
    "    device float       *pe_out  [[ buffer(6) ]],\n"
    "    constant MDMetalParams &params [[ buffer(7) ]],\n"
    "    uint tid     [[ thread_position_in_grid ]],\n"
    "    uint lid     [[ thread_position_in_threadgroup ]],\n"
    "    uint tg_size [[ threads_per_threadgroup ]]\n"
    ") {\n"
    "    threadgroup float sx[TILE_SIZE];\n"
    "    threadgroup float sy[TILE_SIZE];\n"
    "    threadgroup float sz[TILE_SIZE];\n"
    "\n"
    "    const uint   n_real  = params.n_real;\n"
    "    const uint   n_pad   = params.n_padded;\n"
    "    const float  box     = params.lbox;\n"
    "    const float  inv_box = params.inv_box;\n"
    "    const float  rc2     = params.rc2;\n"
    "    const float  v_shift = params.v_shift;\n"
    "\n"
    "    const bool valid_i = (tid < n_real);\n"
    "    const float xi = valid_i ? x[tid] : 0.0f;\n"
    "    const float yi = valid_i ? y[tid] : 0.0f;\n"
    "    const float zi = valid_i ? z[tid] : 0.0f;\n"
    "\n"
    "    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;\n"
    "    float pei = 0.0f;\n"
    "\n"
    "    for (uint tile_start = 0; tile_start < n_pad; tile_start += TILE_SIZE) {\n"
    "        uint j_idx = tile_start + lid;\n"
    "        sx[lid] = (j_idx < n_real) ? x[j_idx] : 0.0f;\n"
    "        sy[lid] = (j_idx < n_real) ? y[j_idx] : 0.0f;\n"
    "        sz[lid] = (j_idx < n_real) ? z[j_idx] : 0.0f;\n"
    "\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        if (valid_i) {\n"
    "            for (uint k = 0; k < TILE_SIZE; k++) {\n"
    "                uint j_global = tile_start + k;\n"
    "                if (j_global == tid || j_global >= n_real) continue;\n"
    "\n"
    "                float dx = xi - sx[k];\n"
    "                float dy = yi - sy[k];\n"
    "                float dz = zi - sz[k];\n"
    "\n"
    "                dx -= box * rint(dx * inv_box);\n"
    "                dy -= box * rint(dy * inv_box);\n"
    "                dz -= box * rint(dz * inv_box);\n"
    "\n"
    "                float r2 = fma(dx, dx, fma(dy, dy, dz * dz));\n"
    "\n"
    "                if (r2 < rc2) {\n"
    "                    float inv_r2  = 1.0f / r2;\n"
    "                    float inv_r6  = inv_r2 * inv_r2 * inv_r2;\n"
    "                    float inv_r12 = inv_r6 * inv_r6;\n"
    "\n"
    "                    float f_over_r = 24.0f * MD_EPSILON * inv_r2\n"
    "                                   * fma(2.0f, inv_r12, -inv_r6);\n"
    "\n"
    "                    fxi = fma(f_over_r, dx, fxi);\n"
    "                    fyi = fma(f_over_r, dy, fyi);\n"
    "                    fzi = fma(f_over_r, dz, fzi);\n"
    "\n"
    "                    pei += 4.0f * MD_EPSILON * (inv_r12 - inv_r6) - v_shift;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    if (valid_i) {\n"
    "        fx[tid]     = fxi;\n"
    "        fy[tid]     = fyi;\n"
    "        fz[tid]     = fzi;\n"
    "        pe_out[tid] = pei * 0.5f;\n"
    "    }\n"
    "}\n";

/* ---------------------------------------------------------------------------
 * Persistent Metal state — initialized once, reused every force call.
 * The dame doesn't like waiting in the rain while we set up the gear.
 * --------------------------------------------------------------------------- */
static id<MTLDevice>               g_device       = nil;
static id<MTLCommandQueue>         g_queue        = nil;
static id<MTLComputePipelineState> g_pipeline     = nil;
static id<MTLLibrary>              g_library      = nil;

/* GPU buffers — allocated once, resized if particle count changes */
static id<MTLBuffer> g_buf_x       = nil;
static id<MTLBuffer> g_buf_y       = nil;
static id<MTLBuffer> g_buf_z       = nil;
static id<MTLBuffer> g_buf_fx      = nil;
static id<MTLBuffer> g_buf_fy      = nil;
static id<MTLBuffer> g_buf_fz      = nil;
static id<MTLBuffer> g_buf_pe      = nil;
static id<MTLBuffer> g_buf_params  = nil;
static int           g_buf_n       = 0;   /* allocated size (n_padded) */

/* Cached pipeline properties */
static NSUInteger g_max_threads_per_tg = 0;
static NSUInteger g_thread_exec_width  = 0;

/* ---------------------------------------------------------------------------
 * Try to load a precompiled .metallib from disk.
 * Returns nil if not found — caller falls back to runtime compilation.
 * --------------------------------------------------------------------------- */
static id<MTLLibrary> metal_load_precompiled(id<MTLDevice> device) {
    NSError *error = nil;
    id<MTLLibrary> library = nil;

    /* Try next to executable */
    NSString *execPath = [[NSProcessInfo processInfo] arguments][0];
    NSString *execDir  = [execPath stringByDeletingLastPathComponent];
    NSString *libPath  = [execDir stringByAppendingPathComponent:@"md_force.metallib"];

    NSURL *libURL = [NSURL fileURLWithPath:libPath];
    library = [device newLibraryWithURL:libURL error:&error];
    if (library) {
        fprintf(stderr, "[Metal] Loaded precompiled metallib: %s\n",
                [libPath UTF8String]);
        return library;
    }

    /* Try current working directory */
    libURL = [NSURL fileURLWithPath:@"md_force.metallib"];
    library = [device newLibraryWithURL:libURL error:&error];
    if (library) {
        fprintf(stderr, "[Metal] Loaded precompiled metallib from cwd\n");
        return library;
    }

    return nil;
}

/* ---------------------------------------------------------------------------
 * Compile Metal shader from embedded source string at runtime.
 * This is the fallback when no .metallib exists (e.g., no Xcode installed).
 * Takes ~20-100ms on first call. One-time cost, amortized over 1000+ steps.
 * --------------------------------------------------------------------------- */
static id<MTLLibrary> metal_compile_from_source(id<MTLDevice> device) {
    NSError *error = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    /* Use mathMode on macOS 15+ (replaces deprecated fastMathEnabled) */
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_2;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
    }

    id<MTLLibrary> library = [device newLibraryWithSource:kMetalShaderSource
                                                  options:opts
                                                    error:&error];
    if (!library) {
        fprintf(stderr, "[Metal] Runtime shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    fprintf(stderr, "[Metal] Compiled shader from embedded source (runtime JIT)\n");
    return library;
}

/* ---------------------------------------------------------------------------
 * Initialize Metal device, queue, pipeline.
 * Returns 0 on success, -1 on failure.
 * --------------------------------------------------------------------------- */
static int metal_init(void) {
    if (g_device != nil) return 0;  /* already initialized */

    @autoreleasepool {
        /* --- Device --- */
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "[Metal] No GPU device found.\n");
            return -1;
        }

        /* --- Command Queue --- */
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "[Metal] Failed to create command queue.\n");
            return -1;
        }

        /* --- Load shader library ---
         * Try precompiled first (fast), fall back to runtime compilation. */
        g_library = metal_load_precompiled(g_device);
        if (!g_library) {
            g_library = metal_compile_from_source(g_device);
        }
        if (!g_library) {
            fprintf(stderr, "[Metal] All shader loading methods failed.\n");
            return -1;
        }

        /* --- Kernel function --- */
        id<MTLFunction> kernel = [g_library newFunctionWithName:@"lj_force_kernel"];
        if (!kernel) {
            fprintf(stderr, "[Metal] Kernel 'lj_force_kernel' not found in library.\n");
            return -1;
        }

        /* --- Compute pipeline --- */
        NSError *error = nil;
        g_pipeline = [g_device newComputePipelineStateWithFunction:kernel error:&error];
        if (!g_pipeline) {
            fprintf(stderr, "[Metal] Failed to create pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            return -1;
        }

        /* --- Query device capabilities ---
         *
         * On Apple M4 (apple9 GPU family):
         *   threadExecutionWidth = 32  (SIMD width, like a CUDA warp)
         *   maxTotalThreadsPerThreadgroup = 1024
         *
         * Our TILE_SIZE of 256 = 8 SIMD groups per threadgroup.
         * This leaves room for the GPU to schedule multiple threadgroups
         * per compute unit for latency hiding. */
        g_max_threads_per_tg = [g_pipeline maxTotalThreadsPerThreadgroup];
        g_thread_exec_width  = [g_pipeline threadExecutionWidth];

        fprintf(stderr, "[Metal] Device: %s\n", [[g_device name] UTF8String]);
        fprintf(stderr, "[Metal] Thread execution width: %lu\n",
                (unsigned long)g_thread_exec_width);
        fprintf(stderr, "[Metal] Max threads/threadgroup: %lu\n",
                (unsigned long)g_max_threads_per_tg);
        fprintf(stderr, "[Metal] Using threadgroup size: %d (TILE_SIZE)\n", TILE_SIZE);
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Allocate (or reallocate) GPU buffers for n_padded particles.
 * Uses MTLResourceStorageModeShared — zero-copy on Apple Silicon unified memory.
 * --------------------------------------------------------------------------- */
static int metal_alloc_buffers(int n_padded) {
    if (g_buf_n >= n_padded) return 0;  /* already big enough */

    @autoreleasepool {
        NSUInteger sz = (NSUInteger)n_padded * sizeof(float);

        /* StorageModeShared: CPU and GPU share the same physical memory.
         * No DMA transfer needed. buffer.contents returns a pointer
         * the CPU can read/write directly. This is the killer feature
         * of Apple Silicon unified memory for compute workloads.
         *
         * On discrete GPU systems (AMD/Intel Macs), you'd want
         * MTLResourceStorageModeManaged and explicit synchronize calls.
         * But Apple Silicon is unified — Shared is optimal. */
        MTLResourceOptions opts = MTLResourceStorageModeShared;

        g_buf_x      = [g_device newBufferWithLength:sz options:opts];
        g_buf_y      = [g_device newBufferWithLength:sz options:opts];
        g_buf_z      = [g_device newBufferWithLength:sz options:opts];
        g_buf_fx     = [g_device newBufferWithLength:sz options:opts];
        g_buf_fy     = [g_device newBufferWithLength:sz options:opts];
        g_buf_fz     = [g_device newBufferWithLength:sz options:opts];
        g_buf_pe     = [g_device newBufferWithLength:sz options:opts];
        g_buf_params = [g_device newBufferWithLength:sizeof(MDMetalParams) options:opts];

        if (!g_buf_x || !g_buf_y || !g_buf_z ||
            !g_buf_fx || !g_buf_fy || !g_buf_fz ||
            !g_buf_pe || !g_buf_params) {
            fprintf(stderr, "[Metal] Buffer allocation failed.\n");
            return -1;
        }

        g_buf_n = n_padded;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * md_force_metal — the public C function, matches ForceFunc signature.
 *
 * Called from main.c via the function pointer. The outside world
 * never knows it's talking to Objective-C and Metal underneath.
 * Like a good informant — clean interface, messy internals.
 * --------------------------------------------------------------------------- */
void md_force_metal(MDSystem *sys, float *pe_out) {
    @autoreleasepool {
        /* Lazy init */
        if (metal_init() != 0) {
            fprintf(stderr, "[Metal] Initialization failed, falling back to zeros.\n");
            memset(sys->fx, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fy, 0, (size_t)sys->n * sizeof(float));
            memset(sys->fz, 0, (size_t)sys->n * sizeof(float));
            *pe_out = 0.0f;
            return;
        }

        const int n_real = sys->n_real;

        /* Pad particle count to multiple of TILE_SIZE for the tiled kernel.
         * Padded slots have positions = 0 and are skipped by the kernel
         * (j_global >= n_real check). */
        int n_padded = (n_real + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

        /* Allocate GPU buffers if needed */
        if (metal_alloc_buffers(n_padded) != 0) {
            *pe_out = 0.0f;
            return;
        }

        /* --- Copy positions into GPU buffers ---
         * With StorageModeShared, buffer.contents IS the memory.
         * memcpy into it is our only "transfer" — and it's just a CPU memcpy
         * into unified memory that the GPU will read. On M4 with 120 GB/s
         * bandwidth, 864 particles * 3 * 4 bytes = 10 KB. Negligible. */
        memcpy([g_buf_x contents], sys->x, (size_t)n_real * sizeof(float));
        memcpy([g_buf_y contents], sys->y, (size_t)n_real * sizeof(float));
        memcpy([g_buf_z contents], sys->z, (size_t)n_real * sizeof(float));

        /* Zero-fill padding region */
        if (n_padded > n_real) {
            size_t pad_bytes = (size_t)(n_padded - n_real) * sizeof(float);
            size_t offset    = (size_t)n_real * sizeof(float);
            memset((char *)[g_buf_x contents] + offset, 0, pad_bytes);
            memset((char *)[g_buf_y contents] + offset, 0, pad_bytes);
            memset((char *)[g_buf_z contents] + offset, 0, pad_bytes);
        }

        /* --- Set kernel parameters --- */
        MDMetalParams *params = (MDMetalParams *)[g_buf_params contents];
        params->n_real   = (uint32_t)n_real;
        params->n_padded = (uint32_t)n_padded;
        params->lbox     = sys->lbox;
        params->inv_box  = 1.0f / sys->lbox;
        params->rc2      = MD_CUTOFF * MD_CUTOFF;
        params->v_shift  = md_lj_shift();

        /* --- Encode compute command --- */
        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_pipeline];
        [enc setBuffer:g_buf_x      offset:0 atIndex:0];
        [enc setBuffer:g_buf_y      offset:0 atIndex:1];
        [enc setBuffer:g_buf_z      offset:0 atIndex:2];
        [enc setBuffer:g_buf_fx     offset:0 atIndex:3];
        [enc setBuffer:g_buf_fy     offset:0 atIndex:4];
        [enc setBuffer:g_buf_fz     offset:0 atIndex:5];
        [enc setBuffer:g_buf_pe     offset:0 atIndex:6];
        [enc setBuffer:g_buf_params offset:0 atIndex:7];

        /* --- Dispatch ---
         * Grid: n_padded threads total (one per particle slot)
         * Threadgroup: TILE_SIZE threads (matches shared memory tile)
         *
         * n_padded is guaranteed to be a multiple of TILE_SIZE,
         * so the grid divides evenly into threadgroups.
         *
         * We use dispatchThreads (non-uniform threadgroups, available
         * on apple3+ GPU family, which includes all Apple Silicon).
         * This handles the case where n_padded isn't a perfect multiple
         * of the threadgroup size — but we've already padded, so it's moot.
         * Using it anyway because it's the modern API. */
        MTLSize grid_size = MTLSizeMake((NSUInteger)n_padded, 1, 1);
        MTLSize tg_size   = MTLSizeMake(TILE_SIZE, 1, 1);

        /* Safety clamp (TILE_SIZE=256 is well under the 1024 limit) */
        if (tg_size.width > g_max_threads_per_tg) {
            tg_size.width = g_max_threads_per_tg;
        }

        [enc dispatchThreads:grid_size threadsPerThreadgroup:tg_size];
        [enc endEncoding];

        /* --- Submit and wait ---
         * commit() sends the command buffer to the GPU.
         * waitUntilCompleted() blocks the CPU until the GPU finishes.
         * In a game engine you'd use callbacks, but for MD we need
         * the forces before the next integration step. Synchronous is correct. */
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        /* --- Read results ---
         * Forces and PE are already in shared memory. Just memcpy out.
         * On Apple Silicon this is CPU-to-CPU copy (same physical RAM). */
        memcpy(sys->fx, [g_buf_fx contents], (size_t)n_real * sizeof(float));
        memcpy(sys->fy, [g_buf_fy contents], (size_t)n_real * sizeof(float));
        memcpy(sys->fz, [g_buf_fz contents], (size_t)n_real * sizeof(float));

        /* Zero remaining padded forces (match other kernels' behavior) */
        if (sys->n > n_real) {
            size_t pad_bytes = (size_t)(sys->n - n_real) * sizeof(float);
            memset(sys->fx + n_real, 0, pad_bytes);
            memset(sys->fy + n_real, 0, pad_bytes);
            memset(sys->fz + n_real, 0, pad_bytes);
        }

        /* --- Reduce per-particle PE to scalar ---
         * The GPU wrote pe_out[i] for each particle. Sum them on CPU.
         * For 864 particles this is trivial. For millions, you'd want
         * a GPU reduction kernel (simd_sum + threadgroup reduce). */
        float *pe_buf = (float *)[g_buf_pe contents];
        float pe_total = 0.0f;
        for (int i = 0; i < n_real; i++) {
            pe_total += pe_buf[i];
        }
        *pe_out = pe_total;
    }
}
