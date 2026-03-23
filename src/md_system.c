#include "md_system.h"
#include "md_types.h"
#include <math.h>
#include <string.h>
#include <time.h>

/* Simple xorshift64 PRNG — fast, deterministic if seeded */
static uint64_t rng_state = 0;

static void rng_seed(uint64_t s) {
    rng_state = s ? s : 0xDEADBEEFCAFEBABEULL;
}

static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* Uniform [0,1) */
static float rng_uniform(void) {
    return (float)(rng_next() >> 11) * 0x1.0p-53f;
}

/* Box-Muller: two standard normals from two uniforms */
static float rng_gaussian(void) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    /* guard against log(0) */
    if (u1 < 1e-30f) u1 = 1e-30f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static float *alloc_aligned(int n) {
    void *p = NULL;
    posix_memalign(&p, 16, (size_t)n * sizeof(float));
    memset(p, 0, (size_t)n * sizeof(float));
    return (float *)p;
}

MDSystem *md_system_create(int ncells, float density, float temp) {
    int n_real = 4 * ncells * ncells * ncells;   /* FCC: 4 atoms per unit cell */
    int n_pad  = (n_real + 3) & ~3;               /* round up to multiple of 4 */

    MDSystem *sys = calloc(1, sizeof(MDSystem));
    sys->n      = n_pad;
    sys->n_real = n_real;
    sys->lbox   = cbrtf((float)n_real / density);

    sys->x  = alloc_aligned(n_pad);
    sys->y  = alloc_aligned(n_pad);
    sys->z  = alloc_aligned(n_pad);
    sys->vx = alloc_aligned(n_pad);
    sys->vy = alloc_aligned(n_pad);
    sys->vz = alloc_aligned(n_pad);
    sys->fx = alloc_aligned(n_pad);
    sys->fy = alloc_aligned(n_pad);
    sys->fz = alloc_aligned(n_pad);

    /* --- Place particles on FCC lattice --- */
    float a = sys->lbox / (float)ncells;   /* lattice constant */

    /* FCC basis vectors (in units of a) */
    static const float basis[4][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.0f},
        {0.5f, 0.0f, 0.5f},
        {0.0f, 0.5f, 0.5f},
    };

    int idx = 0;
    for (int ix = 0; ix < ncells; ix++) {
        for (int iy = 0; iy < ncells; iy++) {
            for (int iz = 0; iz < ncells; iz++) {
                for (int b = 0; b < 4; b++) {
                    sys->x[idx] = ((float)ix + basis[b][0]) * a;
                    sys->y[idx] = ((float)iy + basis[b][1]) * a;
                    sys->z[idx] = ((float)iz + basis[b][2]) * a;
                    idx++;
                }
            }
        }
    }

    /* --- Maxwell-Boltzmann velocities --- */
    rng_seed((uint64_t)time(NULL));
    float sigma_v = sqrtf(temp / MD_MASS);   /* sqrt(kT/m), kB=1 */

    for (int i = 0; i < n_real; i++) {
        sys->vx[i] = sigma_v * rng_gaussian();
        sys->vy[i] = sigma_v * rng_gaussian();
        sys->vz[i] = sigma_v * rng_gaussian();
    }

    /* Remove net momentum (so system doesn't drift) */
    float svx = 0, svy = 0, svz = 0;
    for (int i = 0; i < n_real; i++) {
        svx += sys->vx[i];
        svy += sys->vy[i];
        svz += sys->vz[i];
    }
    svx /= (float)n_real;
    svy /= (float)n_real;
    svz /= (float)n_real;
    for (int i = 0; i < n_real; i++) {
        sys->vx[i] -= svx;
        sys->vy[i] -= svy;
        sys->vz[i] -= svz;
    }

    /* Rescale to exact target temperature */
    float t_actual = 0.0f;
    for (int i = 0; i < n_real; i++) {
        t_actual += sys->vx[i]*sys->vx[i]
                  + sys->vy[i]*sys->vy[i]
                  + sys->vz[i]*sys->vz[i];
    }
    t_actual /= (3.0f * (float)n_real);  /* T = 2KE/(3NkB), KE=0.5*m*v^2, m=1 */
    float scale = sqrtf(temp / t_actual);
    for (int i = 0; i < n_real; i++) {
        sys->vx[i] *= scale;
        sys->vy[i] *= scale;
        sys->vz[i] *= scale;
    }

    return sys;
}

void md_system_destroy(MDSystem *sys) {
    if (!sys) return;
    free(sys->x);  free(sys->y);  free(sys->z);
    free(sys->vx); free(sys->vy); free(sys->vz);
    free(sys->fx); free(sys->fy); free(sys->fz);
    free(sys);
}

float md_kinetic_energy(const MDSystem *sys) {
    float ke = 0.0f;
    for (int i = 0; i < sys->n_real; i++) {
        ke += sys->vx[i]*sys->vx[i]
            + sys->vy[i]*sys->vy[i]
            + sys->vz[i]*sys->vz[i];
    }
    return 0.5f * MD_MASS * ke;
}

float md_temperature(const MDSystem *sys) {
    float ke = md_kinetic_energy(sys);
    return 2.0f * ke / (3.0f * (float)sys->n_real);
}
