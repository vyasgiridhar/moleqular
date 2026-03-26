#ifndef MD_CLUSTER_H
#define MD_CLUSTER_H

#include "md_types.h"

#define CLUSTER_SIZE 8

typedef struct {
    float *cluster_x, *cluster_y, *cluster_z;  /* SoA, [n_clusters * CLUSTER_SIZE] */
    int   *cluster_orig_idx;    /* original particle index per slot */
    int   *cluster_count;       /* real atoms per cluster (1-8) */
    int   *pair_start;          /* [n_clusters] start into j_list */
    int   *pair_end;            /* [n_clusters] end into j_list */
    int   *j_list;              /* flat array of j-cluster indices */
    int   *cell_cluster_start;  /* [ncells_total] first cluster in cell */
    int   *cell_cluster_end;    /* [ncells_total] one-past-last cluster in cell */
    int    n_clusters;
    int    n_pairs_total;       /* length of j_list */
    int    ncells_side;
    int    ncells_total;
    float  cell_size;
    float  inv_cell;
} MDClusterPairList;

MDClusterPairList *md_cluster_build(const MDSystem *sys);
void md_cluster_destroy(MDClusterPairList *cpl);

#endif /* MD_CLUSTER_H */
