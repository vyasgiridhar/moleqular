#ifndef MD_CELLLIST_H
#define MD_CELLLIST_H

#include "md_types.h"

/*
 * Cell List — O(N) neighbor acceleration for pairwise force kernels.
 *
 * Divides the simulation box into cubic cells of side >= rc.
 * Each particle belongs to exactly one cell. Force kernels only
 * check the 27 neighboring cells (3×3×3 stencil) instead of all N.
 *
 * Linked-list storage: head[cell] → first particle, next[i] → next in cell.
 * O(N) build, zero sorting, ~13 particles/cell at LJ density 0.8.
 */

typedef struct {
    int   *head;          /* head[cell] = first particle in cell, -1 if empty */
    int   *next;          /* next[i] = next particle in same cell, -1 if last */
    int    ncells_side;   /* cells per box side */
    int    ncells_total;  /* ncells_side^3 */
    float  cell_size;     /* lbox / ncells_side */
    float  inv_cell;      /* 1.0 / cell_size */
    int    n_alloc;       /* allocated size of next[] */
    int    head_alloc;    /* allocated size of head[] */
} MDCellList;

MDCellList *md_celllist_create(int n_max);
void        md_celllist_destroy(MDCellList *cl);
void        md_celllist_build(MDCellList *cl, const MDSystem *sys);

#endif /* MD_CELLLIST_H */
