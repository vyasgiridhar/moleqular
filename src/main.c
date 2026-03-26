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

    int ncells = MD_FCC_N;
    int nsteps = NSTEPS;
    int thermo_freq = 0;  /* 0 = auto */
    int no_traj = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scalar") == 0) {
            compute_forces = md_force_scalar;
            mode = "scalar";
        } else if (strcmp(argv[i], "--omp") == 0) {
            compute_forces = md_force_omp;
            mode = "OpenMP+NEON";
        } else if (strcmp(argv[i], "--tiled") == 0) {
            compute_forces = md_force_tiled;
            mode = "Tiled+OMP+NEON";
        } else if (strcmp(argv[i], "--sme2") == 0) {
            compute_forces = md_force_sme2;
            mode = "SME2";
        } else if (strcmp(argv[i], "--n3l") == 0) {
            compute_forces = md_force_neon_n3l;
            mode = "NEON+N3L";
        } else if (strcmp(argv[i], "--f64") == 0) {
            compute_forces = md_force_f64;
            mode = "NEON-f64";
        } else if (strcmp(argv[i], "--metal") == 0) {
            compute_forces = md_force_metal;
            mode = "Metal GPU";
        } else if (strcmp(argv[i], "--cl") == 0) {
            compute_forces = md_force_neon_cl;
            mode = "NEON+CellList";
        } else if (strcmp(argv[i], "--omp-cl") == 0) {
            compute_forces = md_force_omp_cl;
            mode = "OMP+NEON+CellList";
        } else if (strcmp(argv[i], "--metal-cl") == 0) {
            compute_forces = md_force_metal_cl;
            mode = "Metal+CellList";
        } else if (strcmp(argv[i], "--ane") == 0) {
            compute_forces = md_force_ane;
            mode = "ANE (Neural Engine)";
        } else if (strcmp(argv[i], "--bvh") == 0) {
            compute_forces = md_force_metal_bvh;
            mode = "Metal+BVH";
        } else if (strcmp(argv[i], "--nbnxm") == 0) {
            compute_forces = md_force_metal_nbnxm;
            mode = "Metal+NBNXM";
        } else if (strcmp(argv[i], "--ane-direct") == 0) {
            compute_forces = md_force_ane_direct;
            mode = "ANE Direct";
        } else if (strncmp(argv[i], "--ncells=", 9) == 0) {
            ncells = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--steps=", 8) == 0) {
            nsteps = atoi(argv[i] + 8);
        } else if (strncmp(argv[i], "--thermo=", 9) == 0) {
            thermo_freq = atoi(argv[i] + 9);
        } else if (strcmp(argv[i], "--no-traj") == 0) {
            no_traj = 1;
        }
    }

    printf("moleqular — LJ Molecular Dynamics on Apple M4\n");
    printf("Force kernel: %s\n", mode);

    MDSystem *sys = md_system_create(ncells, MD_DENSITY, MD_TEMP0);
    printf("Particles: %d (padded: %d), Box: %.4f\n\n",
           sys->n_real, sys->n, sys->lbox);

    /* Open trajectory file (skip for long runs) */
    FILE *traj = no_traj ? NULL : fopen("trajectory.xyz", "w");

    /* Auto thermo frequency */
    if (thermo_freq == 0)
        thermo_freq = nsteps >= 100000 ? nsteps / 10 : THERMO_FREQ;

    /* Initial force computation */
    float pe = 0.0f;
    compute_forces(sys, &pe);

    double t_start = wtime();

    for (int step = 0; step <= nsteps; step++) {
        if (step > 0) {
            /* Velocity Verlet: position half-step */
            md_integrate_positions(sys, MD_DT);

            /* Recompute forces */
            compute_forces(sys, &pe);

            /* Velocity Verlet: velocity half-step */
            md_integrate_velocities(sys, MD_DT);
        }

        /* Thermodynamics output */
        if (step % thermo_freq == 0) {
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

    /*
     * FLOP count per force evaluation:
     * Each pair (i,j) costs ~20 FLOPs:
     *   3 sub (dx,dy,dz) + 3 mul+2 FMA (r²) + 1 recip est + 1 NR step
     *   + 3 mul (inv_r6, inv_r12) + 2 mul+1 sub (LJ term)
     *   + 1 mul (f_over_r) + 3 FMA (force accum) + ~2 (PE)
     *
     * Scalar kernel: N*(N-1)/2 pairs (Newton's 3rd law)
     * NEON/SME2/OMP: N*N pairs (all-pairs, no N3L)
     */
    int uses_n3l = (compute_forces == md_force_scalar || compute_forces == md_force_neon_n3l);
    int uses_cl  = (compute_forces == md_force_neon_cl || compute_forces == md_force_omp_cl
                    || compute_forces == md_force_metal_cl
                    || compute_forces == md_force_metal_bvh
                    || compute_forces == md_force_metal_nbnxm);
    double n = (double)sys->n_real;
    double pairs_per_eval;
    if (uses_cl) {
        /* Cell list: ~27 cells × particles_per_cell neighbors per particle */
        double cs = (double)(int)(sys->lbox / MD_CUTOFF);
        if (cs < 3.0) cs = 3.0;
        double cvol = sys->lbox / cs;
        cvol = cvol * cvol * cvol;
        double per_cell = (double)MD_DENSITY * cvol;
        pairs_per_eval = n * 27.0 * per_cell;
    } else if (uses_n3l) {
        pairs_per_eval = n*(n-1.0)/2.0;
    } else {
        pairs_per_eval = n*n;
    }
    double flops_per_eval = pairs_per_eval * 20.0;
    double flops_total    = flops_per_eval * (double)nsteps;
    double gflops         = flops_total / elapsed / 1e9;

    printf("\n%d steps in %.3f s (%.1f steps/s)\n",
           nsteps, elapsed, (double)nsteps / elapsed);
    printf("%.3f ms per force evaluation\n",
           elapsed / (double)nsteps * 1000.0);
    printf("%.2f GFLOPS (%.0f pairs/eval, %d FLOPs/pair)\n",
           gflops, pairs_per_eval, 20);

    if (traj) fclose(traj);
    md_system_destroy(sys);
    return 0;
}
