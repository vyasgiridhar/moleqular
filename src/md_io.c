#include "md_io.h"

void md_write_xyz(FILE *fp, const MDSystem *sys, int step) {
    fprintf(fp, "%d\n", sys->n_real);
    fprintf(fp, "step=%d lbox=%.6f\n", step, sys->lbox);
    for (int i = 0; i < sys->n_real; i++) {
        fprintf(fp, "Ar  %.6f  %.6f  %.6f\n",
                sys->x[i], sys->y[i], sys->z[i]);
    }
}

void md_print_thermo(int step, float ke, float pe, float temp) {
    printf("step=%6d  KE=%10.4f  PE=%10.4f  E=%10.4f  T=%8.4f\n",
           step, ke, pe, ke + pe, temp);
}
