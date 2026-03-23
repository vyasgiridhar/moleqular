#ifndef MD_INTEGRATE_H
#define MD_INTEGRATE_H

#include "md_types.h"

/* Velocity Verlet step 1: update positions + half-kick velocities */
void md_integrate_positions(MDSystem *sys, float dt);

/* Velocity Verlet step 2: finish velocity update with new forces */
void md_integrate_velocities(MDSystem *sys, float dt);

#endif /* MD_INTEGRATE_H */
