/*
 * BVH construction — CPU side.
 *
 * Morton encode → sort → Karras radix tree → bottom-up AABB → quantize.
 * Based on Karras 2012 "Maximizing Parallelism in the Construction of BVHs,
 * Octrees, and k-d Trees" — the radix tree algorithm.
 */

#include "md_bvh.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Morton encoding --- */

uint32_t md_morton_expand(uint32_t v) {
    /* Expand 10-bit integer into 30 bits with 2 zeros between each bit */
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

uint32_t md_morton_encode(uint32_t x, uint32_t y, uint32_t z) {
    return (md_morton_expand(x) << 2) |
           (md_morton_expand(y) << 1) |
            md_morton_expand(z);
}

/* --- Sort helpers --- */

typedef struct {
    uint32_t code;
    int      idx;
} MortonPair;

static int morton_cmp(const void *a, const void *b) {
    uint32_t ca = ((const MortonPair *)a)->code;
    uint32_t cb = ((const MortonPair *)b)->code;
    return (ca > cb) - (ca < cb);
}

/* --- Karras radix tree --- */

/* Count leading zeros of a 32-bit integer */
static int clz32(uint32_t x) {
    if (x == 0) return 32;
    return __builtin_clz(x);
}

/* Longest common prefix between Morton codes at indices i and j.
 * Returns -1 if j is out of range. */
static int delta(const uint32_t *codes, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j]) {
        /* Tie-break on index to ensure unique ordering */
        return 32 + clz32((uint32_t)(i ^ j));
    }
    return clz32(codes[i] ^ codes[j]);
}

/* Build the radix tree structure for internal nodes.
 * Internal node i covers a contiguous range of leaves.
 * Sets left/right child indices for each internal node. */
static void build_radix_tree(BVHNode *nodes, const uint32_t *codes,
                             int n_leaves) {
    int n_internal = n_leaves - 1;

    for (int i = 0; i < n_internal; i++) {
        /* Determine direction of the range */
        int d_left  = delta(codes, n_leaves, i, i - 1);
        int d_right = delta(codes, n_leaves, i, i + 1);
        int d = (d_right > d_left) ? 1 : -1;

        /* Compute upper bound for the length of the range */
        int d_min = delta(codes, n_leaves, i, i - d);
        int lmax = 2;
        while (delta(codes, n_leaves, i, i + lmax * d) > d_min)
            lmax *= 2;

        /* Binary search for the other end */
        int l = 0;
        for (int t = lmax / 2; t >= 1; t /= 2) {
            if (delta(codes, n_leaves, i, i + (l + t) * d) > d_min)
                l += t;
        }
        int j = i + l * d;

        /* Find the split position */
        int d_node = delta(codes, n_leaves, i, j);
        int s = 0;
        int t = (l + 1) / 2;  /* Start with half the range */
        /* Round up division for the binary search */
        if (t == 0) t = 1;

        for (;;) {
            int new_s = s + t;
            if (new_s <= l) {
                int split_idx = i + new_s * d;
                if (delta(codes, n_leaves, i, split_idx) > d_node)
                    s = new_s;
            }
            if (t <= 1) break;
            t = (t + 1) / 2;
        }

        int split = i + s * d + (d > 0 ? 0 : -1);

        /* Left child */
        int left_range_start = (d > 0) ? i : j;
        int left_range_end   = split;
        if (left_range_start == left_range_end) {
            /* Leaf */
            nodes[i].left = ~left_range_start;  /* bitwise NOT encodes leaf */
        } else {
            /* Internal node — find its index.
             * In Karras construction, the internal node covering [a..b]
             * has index = split position when d > 0, or split+1 when d < 0.
             * Simpler: left child internal node index = split */
            nodes[i].left = split;
        }

        /* Right child */
        int right_range_start = split + 1;
        int right_range_end   = (d > 0) ? j : i;
        if (right_range_start == right_range_end) {
            nodes[i].right = ~right_range_start;
        } else {
            nodes[i].right = split + 1;
        }
    }
}

/* --- AABB computation (bottom-up) --- */

/* Pack 3 × 10-bit values into uint32 */
static uint32_t pack_10bit(uint32_t x, uint32_t y, uint32_t z) {
    return (x & 0x3FFu) | ((y & 0x3FFu) << 10) | ((z & 0x3FFu) << 20);
}

static void unpack_10bit(uint32_t packed, uint32_t *x, uint32_t *y, uint32_t *z) {
    *x = packed & 0x3FFu;
    *y = (packed >> 10) & 0x3FFu;
    *z = (packed >> 20) & 0x3FFu;
}

/* Quantize a position to [0, 1023] */
static uint32_t quantize(float v, float inv_lbox) {
    int q = (int)(v * inv_lbox * 1024.0f);
    if (q < 0) q = 0;
    if (q > 1023) q = 1023;
    return (uint32_t)q;
}

/* Recursive AABB computation.
 * For leaves, AABB = quantized particle position (point).
 * For internal, AABB = union of children. */
static void compute_aabb(BVHNode *nodes, const float *sx, const float *sy,
                         const float *sz, int n_leaves, float inv_lbox,
                         int node_idx) {
    BVHNode *node = &nodes[node_idx];

    uint32_t min_x, min_y, min_z, max_x, max_y, max_z;

    /* Process left child */
    uint32_t lmin_x, lmin_y, lmin_z, lmax_x, lmax_y, lmax_z;
    if (node->left < 0) {
        int leaf = ~node->left;
        uint32_t qx = quantize(sx[leaf], inv_lbox);
        uint32_t qy = quantize(sy[leaf], inv_lbox);
        uint32_t qz = quantize(sz[leaf], inv_lbox);
        lmin_x = lmax_x = qx;
        lmin_y = lmax_y = qy;
        lmin_z = lmax_z = qz;
    } else {
        compute_aabb(nodes, sx, sy, sz, n_leaves, inv_lbox, node->left);
        unpack_10bit(nodes[node->left].aabb_min, &lmin_x, &lmin_y, &lmin_z);
        unpack_10bit(nodes[node->left].aabb_max, &lmax_x, &lmax_y, &lmax_z);
    }

    /* Process right child */
    uint32_t rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z;
    if (node->right < 0) {
        int leaf = ~node->right;
        uint32_t qx = quantize(sx[leaf], inv_lbox);
        uint32_t qy = quantize(sy[leaf], inv_lbox);
        uint32_t qz = quantize(sz[leaf], inv_lbox);
        rmin_x = rmax_x = qx;
        rmin_y = rmax_y = qy;
        rmin_z = rmax_z = qz;
    } else {
        compute_aabb(nodes, sx, sy, sz, n_leaves, inv_lbox, node->right);
        unpack_10bit(nodes[node->right].aabb_min, &rmin_x, &rmin_y, &rmin_z);
        unpack_10bit(nodes[node->right].aabb_max, &rmax_x, &rmax_y, &rmax_z);
    }

    /* Union */
    min_x = (lmin_x < rmin_x) ? lmin_x : rmin_x;
    min_y = (lmin_y < rmin_y) ? lmin_y : rmin_y;
    min_z = (lmin_z < rmin_z) ? lmin_z : rmin_z;
    max_x = (lmax_x > rmax_x) ? lmax_x : rmax_x;
    max_y = (lmax_y > rmax_y) ? lmax_y : rmax_y;
    max_z = (lmax_z > rmax_z) ? lmax_z : rmax_z;

    node->aabb_min = pack_10bit(min_x, min_y, min_z);
    node->aabb_max = pack_10bit(max_x, max_y, max_z);
}

/* --- Public API --- */

MDBVH *md_bvh_build(const float *x, const float *y, const float *z,
                     int n, float lbox) {
    float inv_lbox = 1.0f / lbox;

    /* Step 1: Morton encode */
    MortonPair *pairs = (MortonPair *)malloc((size_t)n * sizeof(MortonPair));
    for (int i = 0; i < n; i++) {
        uint32_t qx = quantize(x[i], inv_lbox);
        uint32_t qy = quantize(y[i], inv_lbox);
        uint32_t qz = quantize(z[i], inv_lbox);
        pairs[i].code = md_morton_encode(qx, qy, qz);
        pairs[i].idx  = i;
    }

    /* Step 2: Sort by Morton code */
    qsort(pairs, (size_t)n, sizeof(MortonPair), morton_cmp);

    /* Step 3: Reorder positions and extract sorted codes */
    MDBVH *bvh = (MDBVH *)calloc(1, sizeof(MDBVH));
    bvh->n = n;
    bvh->n_nodes = 2 * n - 1;
    bvh->lbox = lbox;
    bvh->inv_quant = lbox / 1024.0f;

    bvh->sorted_idx = (int *)malloc((size_t)n * sizeof(int));
    bvh->sorted_x   = (float *)malloc((size_t)n * sizeof(float));
    bvh->sorted_y   = (float *)malloc((size_t)n * sizeof(float));
    bvh->sorted_z   = (float *)malloc((size_t)n * sizeof(float));

    uint32_t *codes = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));

    for (int i = 0; i < n; i++) {
        int orig = pairs[i].idx;
        bvh->sorted_idx[i] = orig;
        bvh->sorted_x[i]   = x[orig];
        bvh->sorted_y[i]   = y[orig];
        bvh->sorted_z[i]   = z[orig];
        codes[i]            = pairs[i].code;
    }
    free(pairs);

    /* Step 4: Build Karras radix tree (N-1 internal nodes) */
    bvh->nodes = (BVHNode *)calloc((size_t)(2 * n - 1), sizeof(BVHNode));
    if (n > 1) {
        build_radix_tree(bvh->nodes, codes, n);
    } else {
        /* Degenerate: single particle */
        bvh->nodes[0].left  = ~0;
        bvh->nodes[0].right = -1;
    }

    /* Step 5: Compute AABBs bottom-up (for internal nodes) */
    /* First set leaf AABBs (not stored in nodes array — computed inline during traversal) */
    /* Internal node AABBs computed recursively from root */
    if (n > 1) {
        compute_aabb(bvh->nodes, bvh->sorted_x, bvh->sorted_y, bvh->sorted_z,
                     n, inv_lbox, 0);
    }

    free(codes);
    return bvh;
}

void md_bvh_destroy(MDBVH *bvh) {
    if (!bvh) return;
    free(bvh->nodes);
    free(bvh->sorted_idx);
    free(bvh->sorted_x);
    free(bvh->sorted_y);
    free(bvh->sorted_z);
    free(bvh);
}
