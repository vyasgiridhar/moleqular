#ifndef MD_FORCE_H
#define MD_FORCE_H

#include "md_types.h"

/* Scalar (plain C) LJ force computation — baseline */
void md_force_scalar(MDSystem *sys, float *pe_out);

/* NEON intrinsics LJ force computation — optimized */
void md_force_neon(MDSystem *sys, float *pe_out);

#endif /* MD_FORCE_H */
