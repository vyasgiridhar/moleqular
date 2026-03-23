#include "md_celllist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

MDCellList *md_celllist_create(int n_max) {
    MDCellList *cl = malloc(sizeof(MDCellList));

    cl->n_alloc = n_max;
    cl->next    = malloc((size_t)n_max * sizeof(int));

    /* Conservative initial allocation — resized in build if needed */
    int max_side = 64;
    cl->head_alloc = max_side * max_side * max_side;
    cl->head = malloc((size_t)cl->head_alloc * sizeof(int));

    cl->ncells_side  = 0;
    cl->ncells_total = 0;
    cl->cell_size    = 0.0f;
    cl->inv_cell     = 0.0f;

    return cl;
}

void md_celllist_destroy(MDCellList *cl) {
    if (!cl) return;
    free(cl->head);
    free(cl->next);
    free(cl);
}

void md_celllist_build(MDCellList *cl, const MDSystem *sys) {
    const int   n_real = sys->n_real;
    const float lbox   = sys->lbox;

    /* Cells per side: floor(lbox / rc), minimum 3 to avoid PBC stencil aliasing */
    int ncs = (int)(lbox / MD_CUTOFF);
    if (ncs < 3) ncs = 3;

    cl->ncells_side  = ncs;
    cl->ncells_total = ncs * ncs * ncs;
    cl->cell_size    = lbox / (float)ncs;
    cl->inv_cell     = (float)ncs / lbox;

    /* Grow arrays if needed */
    if (n_real > cl->n_alloc) {
        cl->n_alloc = n_real;
        free(cl->next);
        cl->next = malloc((size_t)n_real * sizeof(int));
    }
    if (cl->ncells_total > cl->head_alloc) {
        cl->head_alloc = cl->ncells_total;
        free(cl->head);
        cl->head = malloc((size_t)cl->ncells_total * sizeof(int));
    }

    /* Clear heads */
    memset(cl->head, -1, (size_t)cl->ncells_total * sizeof(int));

    /* Insert particles into cells */
    const float inv_cell = cl->inv_cell;
    const int   ncs_max  = ncs - 1;

    for (int i = 0; i < n_real; i++) {
        int cx = (int)(sys->x[i] * inv_cell);
        int cy = (int)(sys->y[i] * inv_cell);
        int cz = (int)(sys->z[i] * inv_cell);

        /* Clamp for boundary safety (x[i] == lbox exactly) */
        if (cx > ncs_max) cx = ncs_max;
        if (cy > ncs_max) cy = ncs_max;
        if (cz > ncs_max) cz = ncs_max;
        if (cx < 0) cx = 0;
        if (cy < 0) cy = 0;
        if (cz < 0) cz = 0;

        int cell_idx = (cx * ncs + cy) * ncs + cz;
        cl->next[i]       = cl->head[cell_idx];
        cl->head[cell_idx] = i;
    }
}
