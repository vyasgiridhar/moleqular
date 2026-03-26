#ifndef MD_BVH_H
#define MD_BVH_H

#include <stdint.h>

/* BVH node — 16 bytes, cache-line friendly.
 * Internal nodes: left/right >= 0 (index into nodes array)
 * Leaf nodes:     left = ~particle_index (bitwise NOT), right = -1
 */
typedef struct {
    uint32_t aabb_min;   /* 10-bit x,y,z packed: bits 0-9, 10-19, 20-29 */
    uint32_t aabb_max;
    int32_t  left;
    int32_t  right;
} BVHNode;

/* CPU-side BVH handle */
typedef struct {
    BVHNode  *nodes;       /* 2N-1 nodes: [0..N-2] internal, [N-1..2N-2] leaves */
    int      *sorted_idx;  /* sorted_idx[i] = original particle index for leaf i */
    float    *sorted_x, *sorted_y, *sorted_z;
    int       n;           /* particle count (leaves) */
    int       n_nodes;     /* 2N-1 */
    float     lbox;
    float     inv_quant;   /* lbox / 1024.0 for AABB dequantization */
} MDBVH;

/* Build BVH from particle positions. Caller must destroy. */
MDBVH *md_bvh_build(const float *x, const float *y, const float *z,
                     int n, float lbox);

void md_bvh_destroy(MDBVH *bvh);

/* Morton encoding helpers (exposed for testing) */
uint32_t md_morton_expand(uint32_t v);
uint32_t md_morton_encode(uint32_t x, uint32_t y, uint32_t z);

#endif /* MD_BVH_H */
