#ifndef MD_TYPES_H
#define MD_TYPES_H

#include <stdlib.h>

/* --- LJ reduced units: sigma=1, epsilon=1, mass=1 --- */
#define MD_SIGMA    1.0f
#define MD_EPSILON  1.0f
#define MD_MASS     1.0f
#define MD_CUTOFF   2.5f          /* rc in units of sigma */
#define MD_DT       0.005f        /* timestep */
#define MD_DENSITY  0.8f          /* reduced number density */
#define MD_TEMP0    1.0f          /* initial temperature */
#define MD_FCC_N    6             /* unit cells per side → 864 particles */

/* Shifted potential constant: V(rc) so V is continuous at cutoff */
static inline float md_lj_shift(void) {
    float rc2  = MD_CUTOFF * MD_CUTOFF;
    float ri2  = 1.0f / rc2;
    float ri6  = ri2 * ri2 * ri2;
    float ri12 = ri6 * ri6;
    return 4.0f * MD_EPSILON * (ri12 - ri6);
}

/* SoA particle system */
typedef struct {
    float *x,  *y,  *z;          /* positions  */
    float *vx, *vy, *vz;         /* velocities */
    float *fx, *fy, *fz;         /* forces     */
    int    n;                     /* particle count (padded to multiple of 4) */
    int    n_real;                /* actual particle count before padding     */
    float  lbox;                  /* cubic box side length */
} MDSystem;

/* Force kernel function pointer — swap scalar/neon at runtime */
typedef void (*ForceFunc)(MDSystem *sys, float *pe_out);

#endif /* MD_TYPES_H */
