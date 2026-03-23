#include "md_integrate.h"
#include "md_types.h"
#include <math.h>

/*
 * Velocity Verlet — the MD standard.
 *
 * Split into two calls per timestep:
 *   1. md_integrate_positions:  x += v*dt + 0.5*a*dt^2;  v += 0.5*a*dt
 *   2. [compute forces → new a]
 *   3. md_integrate_velocities: v += 0.5*a_new*dt
 *
 * This two-phase split lets us slot the (expensive) force computation
 * between the half-kicks without storing old forces separately.
 */

void md_integrate_positions(MDSystem *sys, float dt) {
    const float half_dt  = 0.5f * dt;
    const float half_dt2 = 0.5f * dt * dt;
    const float inv_mass = 1.0f / MD_MASS;
    const int   n        = sys->n_real;
    const float box      = sys->lbox;

    for (int i = 0; i < n; i++) {
        /* half-kick velocity */
        sys->vx[i] += half_dt * sys->fx[i] * inv_mass;
        sys->vy[i] += half_dt * sys->fy[i] * inv_mass;
        sys->vz[i] += half_dt * sys->fz[i] * inv_mass;

        /* full position update */
        sys->x[i] += sys->vx[i] * dt;
        sys->y[i] += sys->vy[i] * dt;
        sys->z[i] += sys->vz[i] * dt;

        /* wrap into periodic box */
        sys->x[i] -= box * floorf(sys->x[i] / box);
        sys->y[i] -= box * floorf(sys->y[i] / box);
        sys->z[i] -= box * floorf(sys->z[i] / box);
    }

    (void)half_dt2; /* verlet position uses v*dt; half_dt2 folded into half-kick */
}

void md_integrate_velocities(MDSystem *sys, float dt) {
    const float half_dt  = 0.5f * dt;
    const float inv_mass = 1.0f / MD_MASS;
    const int   n        = sys->n_real;

    for (int i = 0; i < n; i++) {
        sys->vx[i] += half_dt * sys->fx[i] * inv_mass;
        sys->vy[i] += half_dt * sys->fy[i] * inv_mass;
        sys->vz[i] += half_dt * sys->fz[i] * inv_mass;
    }
}
