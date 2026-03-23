/*
 * CUDA LJ Force Kernels — tiled all-pairs + cell list.
 *
 * Direct port of the Metal compute shaders to CUDA.
 * Tiled all-pairs: shared memory tiles, same algorithm as Metal.
 * Cell list: sorted particles, cell_start/cell_end offsets.
 *
 * Build: nvcc -O2 -arch=sm_89 (Ada/L4) or sm_80 (A100)
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

/* Must match md_types.h */
#define MD_CUTOFF   2.5f
#define MD_EPSILON  1.0f
#define MD_DENSITY  0.8f

typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *fx, *fy, *fz;
    int n;
    int n_real;
    float lbox;
} MDSystem;

static inline float md_lj_shift(void) {
    float rc2 = MD_CUTOFF * MD_CUTOFF;
    float ri2 = 1.0f / rc2;
    float ri6 = ri2 * ri2 * ri2;
    float ri12 = ri6 * ri6;
    return 4.0f * MD_EPSILON * (ri12 - ri6);
}

/* --- Device globals --- */
static float *d_x = NULL, *d_y = NULL, *d_z = NULL;
static float *d_fx = NULL, *d_fy = NULL, *d_fz = NULL;
static float *d_pe = NULL;
static int    d_n = 0;

#define TILE 256

/* --- All-pairs tiled kernel --- */
__global__ void lj_force_tiled_kernel(
    const float *x, const float *y, const float *z,
    float *fx, float *fy, float *fz, float *pe_out,
    int n_real, int n_padded, float box, float inv_box, float rc2, float v_shift
) {
    __shared__ float sx[TILE], sy[TILE], sz[TILE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    bool valid = (tid < n_real);
    float xi = valid ? x[tid] : 0.0f;
    float yi = valid ? y[tid] : 0.0f;
    float zi = valid ? z[tid] : 0.0f;

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    for (int tile_start = 0; tile_start < n_padded; tile_start += TILE) {
        int j_idx = tile_start + lid;
        sx[lid] = (j_idx < n_real) ? x[j_idx] : 1e10f;
        sy[lid] = (j_idx < n_real) ? y[j_idx] : 1e10f;
        sz[lid] = (j_idx < n_real) ? z[j_idx] : 1e10f;

        __syncthreads();

        if (valid) {
            #pragma unroll 8
            for (int k = 0; k < TILE; k++) {
                float dx = xi - sx[k];
                float dy = yi - sy[k];
                float dz = zi - sz[k];

                dx -= box * rintf(dx * inv_box);
                dy -= box * rintf(dy * inv_box);
                dz -= box * rintf(dz * inv_box);

                float r2 = dx*dx + dy*dy + dz*dz;

                if (r2 > 1e-10f && r2 < rc2) {
                    float inv_r2 = 1.0f / r2;
                    float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    float inv_r12 = inv_r6 * inv_r6;

                    float f_over_r = 24.0f * inv_r2 * (2.0f * inv_r12 - inv_r6);

                    fxi += f_over_r * dx;
                    fyi += f_over_r * dy;
                    fzi += f_over_r * dz;

                    pei += 4.0f * (inv_r12 - inv_r6) - v_shift;
                }
            }
        }

        __syncthreads();
    }

    if (valid) {
        fx[tid] = fxi;
        fy[tid] = fyi;
        fz[tid] = fzi;
        pe_out[tid] = pei * 0.5f;
    }
}

/* --- Cell list kernel --- */
__global__ void lj_force_cl_kernel(
    const float *sx, const float *sy, const float *sz,
    float *fx, float *fy, float *fz, float *pe_out,
    const int *cell_start, const int *cell_end,
    int n_real, int ncells_side, float box, float inv_box, float rc2, float v_shift, float inv_cell
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_real) return;

    float xi = sx[tid];
    float yi = sy[tid];
    float zi = sz[tid];

    int ncs = ncells_side;
    int cix = min((int)(xi * inv_cell), ncs - 1);
    int ciy = min((int)(yi * inv_cell), ncs - 1);
    int ciz = min((int)(zi * inv_cell), ncs - 1);

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    for (int di = -1; di <= 1; di++) {
        int nx = (cix + di + ncs) % ncs;
        for (int dj = -1; dj <= 1; dj++) {
            int ny = (ciy + dj + ncs) % ncs;
            for (int dk = -1; dk <= 1; dk++) {
                int nz = (ciz + dk + ncs) % ncs;
                int cidx = (nx * ncs + ny) * ncs + nz;

                int js = cell_start[cidx];
                int je = cell_end[cidx];

                for (int j = js; j < je; j++) {
                    if (j == tid) continue;

                    float dx = xi - sx[j];
                    float dy = yi - sy[j];
                    float dz = zi - sz[j];

                    dx -= box * rintf(dx * inv_box);
                    dy -= box * rintf(dy * inv_box);
                    dz -= box * rintf(dz * inv_box);

                    float r2 = dx*dx + dy*dy + dz*dz;

                    if (r2 < rc2) {
                        float inv_r2 = 1.0f / r2;
                        float inv_r6 = inv_r2 * inv_r2 * inv_r2;
                        float inv_r12 = inv_r6 * inv_r6;

                        float f_over_r = 24.0f * inv_r2 * (2.0f * inv_r12 - inv_r6);

                        fxi += f_over_r * dx;
                        fyi += f_over_r * dy;
                        fzi += f_over_r * dz;

                        pei += 4.0f * (inv_r12 - inv_r6) - v_shift;
                    }
                }
            }
        }
    }

    fx[tid] = fxi;
    fy[tid] = fyi;
    fz[tid] = fzi;
    pe_out[tid] = pei * 0.5f;
}

/* --- Host interface --- */

static void cuda_alloc(int n) {
    if (n <= d_n) return;
    if (d_x) { cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
               cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_pe); }
    size_t sz = (size_t)n * sizeof(float);
    cudaMalloc(&d_x, sz); cudaMalloc(&d_y, sz); cudaMalloc(&d_z, sz);
    cudaMalloc(&d_fx, sz); cudaMalloc(&d_fy, sz); cudaMalloc(&d_fz, sz);
    cudaMalloc(&d_pe, sz);
    d_n = n;
}

extern "C" void md_force_cuda(MDSystem *sys, float *pe_out) {
    int n_real = sys->n_real;
    int n_padded = ((n_real + TILE - 1) / TILE) * TILE;
    float box = sys->lbox;
    float rc2 = MD_CUTOFF * MD_CUTOFF;
    float v_shift = md_lj_shift();

    cuda_alloc(n_padded);

    cudaMemcpy(d_x, sys->x, (size_t)n_real * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, sys->y, (size_t)n_real * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, sys->z, (size_t)n_real * sizeof(float), cudaMemcpyHostToDevice);

    if (n_padded > n_real) {
        cudaMemset(d_x + n_real, 0, (size_t)(n_padded - n_real) * sizeof(float));
        cudaMemset(d_y + n_real, 0, (size_t)(n_padded - n_real) * sizeof(float));
        cudaMemset(d_z + n_real, 0, (size_t)(n_padded - n_real) * sizeof(float));
    }

    int blocks = (n_padded + TILE - 1) / TILE;
    lj_force_tiled_kernel<<<blocks, TILE>>>(
        d_x, d_y, d_z, d_fx, d_fy, d_fz, d_pe,
        n_real, n_padded, box, 1.0f / box, rc2, v_shift
    );
    cudaDeviceSynchronize();

    cudaMemcpy(sys->fx, d_fx, (size_t)n_real * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sys->fy, d_fy, (size_t)n_real * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sys->fz, d_fz, (size_t)n_real * sizeof(float), cudaMemcpyDeviceToHost);

    memset(sys->fx + n_real, 0, (size_t)(sys->n - n_real) * sizeof(float));
    memset(sys->fy + n_real, 0, (size_t)(sys->n - n_real) * sizeof(float));
    memset(sys->fz + n_real, 0, (size_t)(sys->n - n_real) * sizeof(float));

    float *pe_buf = (float *)malloc((size_t)n_real * sizeof(float));
    cudaMemcpy(pe_buf, d_pe, (size_t)n_real * sizeof(float), cudaMemcpyDeviceToHost);
    float pe_total = 0.0f;
    for (int i = 0; i < n_real; i++) pe_total += pe_buf[i];
    *pe_out = pe_total;
    free(pe_buf);
}

/* --- Simple standalone benchmark (no main.c dependency) --- */
#include <stdlib.h>
#include <time.h>

static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Simple FCC lattice init */
static MDSystem *create_system(int ncells) {
    int n_real = 4 * ncells * ncells * ncells;
    int n_pad = (n_real + 3) & ~3;

    MDSystem *sys = (MDSystem *)calloc(1, sizeof(MDSystem));
    sys->n_real = n_real;
    sys->n = n_pad;
    sys->lbox = cbrtf((float)n_real / MD_DENSITY);

    size_t sz = (size_t)n_pad * sizeof(float);
    sys->x  = (float *)calloc(n_pad, sizeof(float));
    sys->y  = (float *)calloc(n_pad, sizeof(float));
    sys->z  = (float *)calloc(n_pad, sizeof(float));
    sys->vx = (float *)calloc(n_pad, sizeof(float));
    sys->vy = (float *)calloc(n_pad, sizeof(float));
    sys->vz = (float *)calloc(n_pad, sizeof(float));
    sys->fx = (float *)calloc(n_pad, sizeof(float));
    sys->fy = (float *)calloc(n_pad, sizeof(float));
    sys->fz = (float *)calloc(n_pad, sizeof(float));

    float a = sys->lbox / (float)ncells;
    float basis[4][3] = {{0,0,0},{0.5f*a,0.5f*a,0},{0.5f*a,0,0.5f*a},{0,0.5f*a,0.5f*a}};
    int idx = 0;
    for (int ix = 0; ix < ncells; ix++)
        for (int iy = 0; iy < ncells; iy++)
            for (int iz = 0; iz < ncells; iz++)
                for (int b = 0; b < 4; b++) {
                    sys->x[idx] = ix * a + basis[b][0];
                    sys->y[idx] = iy * a + basis[b][1];
                    sys->z[idx] = iz * a + basis[b][2];
                    idx++;
                }
    return sys;
}

int main(int argc, char **argv) {
    int ncells = 6;
    int nsteps = 100;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--ncells=", 9) == 0) ncells = atoi(argv[i] + 9);
        if (strncmp(argv[i], "--steps=", 8) == 0) nsteps = atoi(argv[i] + 8);
    }

    MDSystem *sys = create_system(ncells);
    printf("CUDA LJ Benchmark — NVIDIA L4 (Ada Lovelace)\n");
    printf("Particles: %d, Box: %.4f\n", sys->n_real, sys->lbox);

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, %d SMs, %.0f MHz, %.1f GB\n",
           prop.name, prop.multiProcessorCount,
           prop.clockRate / 1000.0f, prop.totalGlobalMem / 1e9);

    /* Warmup */
    float pe;
    md_force_cuda(sys, &pe);
    printf("Initial PE: %.4f\n\n", pe);

    /* Benchmark */
    double t0 = wtime();
    for (int s = 0; s < nsteps; s++) {
        md_force_cuda(sys, &pe);
    }
    double elapsed = wtime() - t0;

    double n = (double)sys->n_real;
    double pairs = n * n;
    double gflops = pairs * 20.0 * nsteps / elapsed / 1e9;

    printf("%d steps in %.3f s (%.1f steps/s)\n", nsteps, elapsed, nsteps / elapsed);
    printf("%.3f ms per force evaluation\n", elapsed / nsteps * 1000.0);
    printf("%.2f GFLOPS (%.0f pairs/eval, 20 FLOPs/pair)\n", gflops, pairs);

    free(sys->x); free(sys->y); free(sys->z);
    free(sys->vx); free(sys->vy); free(sys->vz);
    free(sys->fx); free(sys->fy); free(sys->fz);
    free(sys);
    return 0;
}
