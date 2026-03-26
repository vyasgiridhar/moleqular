/*
 * BVH two-pass Metal kernels.
 *
 * Kernel 1: bvh_traverse — stack-based BVH descent → neighbor list
 * Kernel 2: bvh_forces   — flat loop over neighbor list → LJ forces
 *
 * The key insight: divergent tree traversal lives in the cheap kernel,
 * uniform force computation lives in the expensive kernel.
 */

#include <metal_stdlib>
using namespace metal;

#define MAX_NBRS 128
#define BVH_STACK_DEPTH 64

struct BVHNode {
    uint  aabb_min;   /* 10-bit x,y,z packed */
    uint  aabb_max;
    int   left;       /* >= 0: internal, < 0: ~leaf_index */
    int   right;
};

struct BVHParams {
    uint  n_particles;
    uint  n_internal;   /* N-1 */
    float lbox;
    float inv_box;
    float rc2;
    float v_shift;
    float inv_quant;    /* lbox / 1024.0 */
};

/* Unpack 10-bit quantized coordinates to float */
static inline float3 unpack_min(uint packed, float inv_quant) {
    float x = float(packed & 0x3FFu) * inv_quant;
    float y = float((packed >> 10) & 0x3FFu) * inv_quant;
    float z = float((packed >> 20) & 0x3FFu) * inv_quant;
    return float3(x, y, z);
}

static inline float3 unpack_max(uint packed, float inv_quant) {
    float x = float((packed & 0x3FFu) + 1) * inv_quant;
    float y = float(((packed >> 10) & 0x3FFu) + 1) * inv_quant;
    float z = float(((packed >> 20) & 0x3FFu) + 1) * inv_quant;
    return float3(x, y, z);
}

/*
 * AABB-sphere intersection with periodic boundary conditions.
 *
 * For each axis: compute minimum image distance from particle to the
 * AABB center, then subtract the half-extent. This correctly handles
 * particles near box boundaries finding neighbors that wrap around.
 */
static inline bool aabb_intersects_cutoff(
    float3 pos, float3 bmin, float3 bmax,
    float rc2, float lbox, float inv_box
) {
    float dist2 = 0.0f;

    for (int axis = 0; axis < 3; axis++) {
        float center = (bmin[axis] + bmax[axis]) * 0.5f;
        float half_ext = (bmax[axis] - bmin[axis]) * 0.5f;

        /* Minimum image distance from particle to AABB center */
        float dp = pos[axis] - center;
        dp -= lbox * rint(dp * inv_box);
        dp = abs(dp);

        /* Distance to interval = max(0, |dp| - half_extent) */
        float d = max(0.0f, dp - half_ext);
        dist2 += d * d;
    }

    return dist2 < rc2;
}

/* --- Kernel 1: BVH Traversal → Neighbor List --- */

kernel void bvh_traverse(
    const device float   *sx         [[ buffer(0) ]],
    const device float   *sy         [[ buffer(1) ]],
    const device float   *sz         [[ buffer(2) ]],
    const device BVHNode *nodes      [[ buffer(3) ]],
    device uint          *neighbors  [[ buffer(4) ]],
    device uint          *n_nbrs     [[ buffer(5) ]],
    constant BVHParams   &p          [[ buffer(6) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= p.n_particles) return;

    float3 pos = float3(sx[tid], sy[tid], sz[tid]);
    float rc2 = p.rc2;
    float lbox = p.lbox;
    float inv_box = p.inv_box;
    float inv_quant = p.inv_quant;
    uint count = 0;
    uint base = tid * MAX_NBRS;

    /* Stack-based traversal */
    int stack[BVH_STACK_DEPTH];
    int sp = 0;
    stack[sp++] = 0;  /* root = internal node 0 */

    while (sp > 0) {
        int node_idx = stack[--sp];
        BVHNode node = nodes[node_idx];

        /* Test AABB against cutoff sphere */
        float3 bmin = unpack_min(node.aabb_min, inv_quant);
        float3 bmax = unpack_max(node.aabb_max, inv_quant);

        if (!aabb_intersects_cutoff(pos, bmin, bmax, rc2, lbox, inv_box))
            continue;

        /* Process left child */
        if (node.left < 0) {
            /* Leaf */
            uint j = uint(~node.left);
            if (j != tid) {
                float dx = pos.x - sx[j];
                float dy = pos.y - sy[j];
                float dz = pos.z - sz[j];
                dx -= lbox * rint(dx * inv_box);
                dy -= lbox * rint(dy * inv_box);
                dz -= lbox * rint(dz * inv_box);
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < rc2 && count < MAX_NBRS)
                    neighbors[base + count++] = j;
            }
        } else {
            if (sp < BVH_STACK_DEPTH)
                stack[sp++] = node.left;
        }

        /* Process right child */
        if (node.right < 0) {
            uint j = uint(~node.right);
            if (j != tid) {
                float dx = pos.x - sx[j];
                float dy = pos.y - sy[j];
                float dz = pos.z - sz[j];
                dx -= lbox * rint(dx * inv_box);
                dy -= lbox * rint(dy * inv_box);
                dz -= lbox * rint(dz * inv_box);
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < rc2 && count < MAX_NBRS)
                    neighbors[base + count++] = j;
            }
        } else {
            if (sp < BVH_STACK_DEPTH)
                stack[sp++] = node.right;
        }
    }

    n_nbrs[tid] = count;
}

/* --- Kernel 2: Force Computation from Neighbor List --- */

kernel void bvh_forces(
    const device float  *sx         [[ buffer(0) ]],
    const device float  *sy         [[ buffer(1) ]],
    const device float  *sz         [[ buffer(2) ]],
    const device uint   *neighbors  [[ buffer(4) ]],
    const device uint   *n_nbrs     [[ buffer(5) ]],
    constant BVHParams  &p          [[ buffer(6) ]],
    device float        *fx         [[ buffer(7) ]],
    device float        *fy         [[ buffer(8) ]],
    device float        *fz         [[ buffer(9) ]],
    device float        *pe_out     [[ buffer(10) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= p.n_particles) return;

    float xi = sx[tid], yi = sy[tid], zi = sz[tid];
    float lbox = p.lbox;
    float inv_box = p.inv_box;
    float rc2 = p.rc2;
    float v_shift = p.v_shift;

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float pei = 0.0f;

    uint count = n_nbrs[tid];
    uint base = tid * MAX_NBRS;

    for (uint k = 0; k < count; k++) {
        uint j = neighbors[base + k];

        float dx = xi - sx[j];
        float dy = yi - sy[j];
        float dz = zi - sz[j];

        dx -= lbox * rint(dx * inv_box);
        dy -= lbox * rint(dy * inv_box);
        dz -= lbox * rint(dz * inv_box);

        float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 > 1e-10f && r2 < rc2) {
            float inv_r2  = 1.0f / r2;
            float inv_r6  = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;

            float f_over_r = 24.0f * inv_r2 * (2.0f * inv_r12 - inv_r6);

            fxi += f_over_r * dx;
            fyi += f_over_r * dy;
            fzi += f_over_r * dz;

            pei += 4.0f * (inv_r12 - inv_r6) - v_shift;
        }
    }

    fx[tid] = fxi;
    fy[tid] = fyi;
    fz[tid] = fzi;
    pe_out[tid] = pei * 0.5f;
}
