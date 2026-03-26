/*
 * NBNXM-style cluster pair list builder (CPU side).
 *
 * 1. Build cell list (reuse md_celllist)
 * 2. Walk cells, pack particles into clusters of 8
 * 3. Build pair list: for each i-cluster, find j-clusters in 27 neighbor cells
 */

#include "md_cluster.h"
#include "md_celllist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

MDClusterPairList *md_cluster_build(const MDSystem *sys) {
    int n = sys->n_real;
    float lbox = sys->lbox;

    /* Step 1: spatial grid parameters (same as cell list) */
    int ncs = (int)(lbox / MD_CUTOFF);
    if (ncs < 3) ncs = 3;
    int ncells_total = ncs * ncs * ncs;
    float cell_size = lbox / (float)ncs;
    float inv_cell = (float)ncs / lbox;

    /* Step 2: bin particles into cells using a simple count+offset approach */
    int *cell_count = (int *)calloc((size_t)ncells_total, sizeof(int));
    int *particle_cell = (int *)malloc((size_t)n * sizeof(int));

    for (int i = 0; i < n; i++) {
        int cx = (int)(sys->x[i] * inv_cell);
        int cy = (int)(sys->y[i] * inv_cell);
        int cz = (int)(sys->z[i] * inv_cell);
        if (cx >= ncs) cx = ncs - 1;
        if (cy >= ncs) cy = ncs - 1;
        if (cz >= ncs) cz = ncs - 1;
        int cell_idx = (cx * ncs + cy) * ncs + cz;
        particle_cell[i] = cell_idx;
        cell_count[cell_idx]++;
    }

    /* Compute cluster counts per cell and total clusters */
    int n_clusters = 0;
    int *cell_n_clusters = (int *)malloc((size_t)ncells_total * sizeof(int));
    int *cell_cl_start = (int *)malloc((size_t)ncells_total * sizeof(int));

    for (int c = 0; c < ncells_total; c++) {
        int nc = (cell_count[c] + CLUSTER_SIZE - 1) / CLUSTER_SIZE;
        if (nc == 0) nc = 0;  /* empty cell → 0 clusters */
        cell_n_clusters[c] = nc;
        cell_cl_start[c] = n_clusters;
        n_clusters += nc;
    }

    /* Allocate cluster pair list */
    MDClusterPairList *cpl = (MDClusterPairList *)calloc(1, sizeof(MDClusterPairList));
    cpl->n_clusters = n_clusters;
    cpl->ncells_side = ncs;
    cpl->ncells_total = ncells_total;
    cpl->cell_size = cell_size;
    cpl->inv_cell = inv_cell;

    int n_slots = n_clusters * CLUSTER_SIZE;
    cpl->cluster_x = (float *)malloc((size_t)n_slots * sizeof(float));
    cpl->cluster_y = (float *)malloc((size_t)n_slots * sizeof(float));
    cpl->cluster_z = (float *)malloc((size_t)n_slots * sizeof(float));
    cpl->cluster_orig_idx = (int *)malloc((size_t)n_slots * sizeof(int));
    cpl->cluster_count = (int *)calloc((size_t)n_clusters, sizeof(int));
    cpl->cell_cluster_start = (int *)malloc((size_t)ncells_total * sizeof(int));
    cpl->cell_cluster_end = (int *)malloc((size_t)ncells_total * sizeof(int));

    /* Initialize all slots to sentinel position (far beyond cutoff) */
    for (int i = 0; i < n_slots; i++) {
        cpl->cluster_x[i] = 1e10f;
        cpl->cluster_y[i] = 1e10f;
        cpl->cluster_z[i] = 1e10f;
        cpl->cluster_orig_idx[i] = -1;
    }

    /* Step 3: fill clusters from cells */
    /* We need to walk each cell's particles and pack them into clusters */
    /* First, build per-cell particle lists */
    int *cell_offset = (int *)calloc((size_t)ncells_total, sizeof(int));

    for (int i = 0; i < n; i++) {
        int cell_idx = particle_cell[i];
        int ci = cell_cl_start[cell_idx];       /* first cluster for this cell */
        int off = cell_offset[cell_idx];         /* how many atoms already placed */
        int local_cl = off / CLUSTER_SIZE;       /* which cluster within cell */
        int local_at = off % CLUSTER_SIZE;       /* which slot within cluster */

        int slot = (ci + local_cl) * CLUSTER_SIZE + local_at;
        cpl->cluster_x[slot] = sys->x[i];
        cpl->cluster_y[slot] = sys->y[i];
        cpl->cluster_z[slot] = sys->z[i];
        cpl->cluster_orig_idx[slot] = i;

        cell_offset[cell_idx]++;
    }

    /* Set cluster counts and cell→cluster mapping */
    for (int c = 0; c < ncells_total; c++) {
        int ci_start = cell_cl_start[c];
        int nc = cell_n_clusters[c];
        cpl->cell_cluster_start[c] = ci_start;
        cpl->cell_cluster_end[c] = ci_start + nc;

        int atoms_left = cell_count[c];
        for (int k = 0; k < nc; k++) {
            int count = (atoms_left >= CLUSTER_SIZE) ? CLUSTER_SIZE : atoms_left;
            cpl->cluster_count[ci_start + k] = count;
            atoms_left -= count;
        }
    }

    /* Step 4: build pair list */
    /* First pass: count j-clusters per i-cluster to allocate j_list */
    cpl->pair_start = (int *)malloc((size_t)n_clusters * sizeof(int));
    cpl->pair_end = (int *)malloc((size_t)n_clusters * sizeof(int));

    int *pair_count = (int *)calloc((size_t)n_clusters, sizeof(int));

    for (int c = 0; c < ncells_total; c++) {
        int cx = c / (ncs * ncs);
        int cy = (c / ncs) % ncs;
        int cz = c % ncs;

        for (int ci = cpl->cell_cluster_start[c]; ci < cpl->cell_cluster_end[c]; ci++) {
            /* 27 neighbor cells */
            for (int di = -1; di <= 1; di++) {
                int nx = (cx + di + ncs) % ncs;
                for (int dj = -1; dj <= 1; dj++) {
                    int ny = (cy + dj + ncs) % ncs;
                    for (int dk = -1; dk <= 1; dk++) {
                        int nz = (cz + dk + ncs) % ncs;
                        int nc_idx = (nx * ncs + ny) * ncs + nz;

                        int n_j_cl = cpl->cell_cluster_end[nc_idx] -
                                     cpl->cell_cluster_start[nc_idx];
                        pair_count[ci] += n_j_cl;
                    }
                }
            }
        }
    }

    /* Compute offsets */
    int total_pairs = 0;
    for (int ci = 0; ci < n_clusters; ci++) {
        cpl->pair_start[ci] = total_pairs;
        total_pairs += pair_count[ci];
        cpl->pair_end[ci] = total_pairs;
    }
    cpl->n_pairs_total = total_pairs;

    /* Allocate and fill j_list */
    cpl->j_list = (int *)malloc((size_t)total_pairs * sizeof(int));
    memset(pair_count, 0, (size_t)n_clusters * sizeof(int));

    for (int c = 0; c < ncells_total; c++) {
        int cx = c / (ncs * ncs);
        int cy = (c / ncs) % ncs;
        int cz = c % ncs;

        for (int ci = cpl->cell_cluster_start[c]; ci < cpl->cell_cluster_end[c]; ci++) {
            int base = cpl->pair_start[ci];

            for (int di = -1; di <= 1; di++) {
                int nx = (cx + di + ncs) % ncs;
                for (int dj = -1; dj <= 1; dj++) {
                    int ny = (cy + dj + ncs) % ncs;
                    for (int dk = -1; dk <= 1; dk++) {
                        int nz = (cz + dk + ncs) % ncs;
                        int nc_idx = (nx * ncs + ny) * ncs + nz;

                        for (int cj = cpl->cell_cluster_start[nc_idx];
                             cj < cpl->cell_cluster_end[nc_idx]; cj++) {
                            cpl->j_list[base + pair_count[ci]] = cj;
                            pair_count[ci]++;
                        }
                    }
                }
            }
        }
    }

    free(cell_count);
    free(particle_cell);
    free(cell_n_clusters);
    free(cell_cl_start);
    free(cell_offset);
    free(pair_count);
    return cpl;
}

void md_cluster_destroy(MDClusterPairList *cpl) {
    if (!cpl) return;
    free(cpl->cluster_x);
    free(cpl->cluster_y);
    free(cpl->cluster_z);
    free(cpl->cluster_orig_idx);
    free(cpl->cluster_count);
    free(cpl->cell_cluster_start);
    free(cpl->cell_cluster_end);
    free(cpl->pair_start);
    free(cpl->pair_end);
    free(cpl->j_list);
    free(cpl);
}
