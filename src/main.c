#include "md_types.h"
#include "md_system.h"
#include "md_force.h"
#include "md_integrate.h"
#include "md_io.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#define NSTEPS      1000
#define THERMO_FREQ 100
#define XYZ_FREQ    100

static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    /* Pick force kernel: --scalar or default NEON */
    ForceFunc compute_forces = md_force_neon;
    const char *mode = "NEON";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scalar") == 0) {
            compute_forces = md_force_scalar;
            mode = "scalar";
        }
    }

    printf("moleqular — LJ Molecular Dynamics on Apple M4\n");
    printf("Force kernel: %s\n", mode);

    MDSystem *sys = md_system_create(MD_FCC_N, MD_DENSITY, MD_TEMP0);
    printf("Particles: %d (padded: %d), Box: %.4f\n\n",
           sys->n_real, sys->n, sys->lbox);

    /* Open trajectory file */
    FILE *traj = fopen("trajectory.xyz", "w");

    /* Initial force computation */
    float pe = 0.0f;
    compute_forces(sys, &pe);

    double t_start = wtime();

    for (int step = 0; step <= NSTEPS; step++) {
        if (step > 0) {
            /* Velocity Verlet: position half-step */
            md_integrate_positions(sys, MD_DT);

            /* Recompute forces */
            compute_forces(sys, &pe);

            /* Velocity Verlet: velocity half-step */
            md_integrate_velocities(sys, MD_DT);
        }

        /* Thermodynamics output */
        if (step % THERMO_FREQ == 0) {
            float ke   = md_kinetic_energy(sys);
            float temp = md_temperature(sys);
            md_print_thermo(step, ke, pe, temp);
        }

        /* Trajectory output */
        if (traj && step % XYZ_FREQ == 0) {
            md_write_xyz(traj, sys, step);
        }
    }

    double t_end = wtime();
    double elapsed = t_end - t_start;

    printf("\n%d steps in %.3f s (%.1f steps/s)\n",
           NSTEPS, elapsed, (double)NSTEPS / elapsed);
    printf("%.3f ms per force evaluation\n",
           elapsed / (double)NSTEPS * 1000.0);

    if (traj) fclose(traj);
    md_system_destroy(sys);
    return 0;
}
