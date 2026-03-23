#ifndef MD_SYSTEM_H
#define MD_SYSTEM_H

#include "md_types.h"

/* Allocate arrays (16-byte aligned), init FCC lattice + Maxwell-Boltzmann */
MDSystem *md_system_create(int ncells, float density, float temp);

/* Free everything */
void md_system_destroy(MDSystem *sys);

/* Compute kinetic energy */
float md_kinetic_energy(const MDSystem *sys);

/* Compute instantaneous temperature: T = 2*KE / (3*N*kB), kB=1 in reduced */
float md_temperature(const MDSystem *sys);

#endif /* MD_SYSTEM_H */
