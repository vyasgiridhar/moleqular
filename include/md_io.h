#ifndef MD_IO_H
#define MD_IO_H

#include "md_types.h"
#include <stdio.h>

/* Write one frame in XYZ format (VMD/OVITO compatible) */
void md_write_xyz(FILE *fp, const MDSystem *sys, int step);

/* Print energy/temperature line to stdout */
void md_print_thermo(int step, float ke, float pe, float temp);

#endif /* MD_IO_H */
