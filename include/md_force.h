#ifndef MD_FORCE_H
#define MD_FORCE_H

#include "md_types.h"

/* Scalar (plain C) LJ force computation — baseline */
void md_force_scalar(MDSystem *sys, float *pe_out);

/* NEON intrinsics LJ force computation — optimized */
void md_force_neon(MDSystem *sys, float *pe_out);

/* OpenMP + NEON — parallel across cores */
void md_force_omp(MDSystem *sys, float *pe_out);

/* Tiled OpenMP + NEON + CPU pinning — cache-line aware */
void md_force_tiled(MDSystem *sys, float *pe_out);

/* SME2 streaming SVE — 16-wide vectors */
void md_force_sme2(MDSystem *sys, float *pe_out);

/* NEON + Newton's 3rd law — half the pairs, SIMD scatter-write */
void md_force_neon_n3l(MDSystem *sys, float *pe_out);

/* Double-precision NEON (float64x2) — precision vs performance */
void md_force_f64(MDSystem *sys, float *pe_out);

/* Metal GPU compute — tiled all-pairs on M4 GPU cores */
void md_force_metal(MDSystem *sys, float *pe_out);

/* NEON + cell list neighbor list — O(N) scaling */
void md_force_neon_cl(MDSystem *sys, float *pe_out);

/* OpenMP + NEON + cell list — O(N) multi-core */
void md_force_omp_cl(MDSystem *sys, float *pe_out);

/* Metal GPU + cell list — O(N) on GPU */
void md_force_metal_cl(MDSystem *sys, float *pe_out);

/* Apple Neural Engine — LJ force via CoreML MLP inference */
void md_force_ane(MDSystem *sys, float *pe_out);

/* Metal GPU + BVH two-pass — neighbor list via tree traversal, then force compute */
void md_force_metal_bvh(MDSystem *sys, float *pe_out);

/* Metal GPU + NBNXM cluster pairs — GROMACS-style 8×8 cluster pair lists */
void md_force_metal_nbnxm(MDSystem *sys, float *pe_out);

/* Direct ANE — exact LJ force via private _ANEClient APIs, FP16, no CoreML */
void md_force_ane_direct(MDSystem *sys, float *pe_out);

#endif /* MD_FORCE_H */
