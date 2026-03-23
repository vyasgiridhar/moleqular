# moleqular — CLAUDE.md

## Project

Molecular dynamics simulation targeting Apple M4 hardware. Multiple force kernel implementations exploring NEON, SME2, OpenMP, Metal GPU. The goal is pushing each compute path to its hardware limits, not production MD.

## Build

```bash
brew install libomp   # once
make                  # build
make bench            # run all kernels
```

SME2 kernel compiles with `-march=armv8-a+sme2` (not `-mcpu=native`).
Metal host is Objective-C (`-fobjc-arc`), compiled separately from C sources.

## Architecture

- `include/md_types.h` — MDSystem struct, LJ constants, ForceFunc typedef
- `include/md_force.h` — all force kernel declarations
- `include/md_celllist.h` — cell list acceleration structure
- `src/md_force_*.c` — one file per kernel implementation
- `src/md_celllist.c` — cell list build/create/destroy
- `src/md_force_metal.metal` — Metal compute shader (MSL), all-pairs tiled
- `src/md_force_metal_cl.metal` — Metal cell list shader
- `src/md_force_metal_host.m` — Objective-C Metal bridge (all-pairs)
- `src/md_force_metal_cl_host.m` — Objective-C Metal bridge (cell list)
- `src/main.c` — simulation loop, CLI parsing, timing, GFLOPS reporting

## Adding a new kernel

1. Create `src/md_force_<name>.c` implementing `void md_force_<name>(MDSystem *sys, float *pe_out)`
2. Declare in `include/md_force.h`
3. Add CLI flag in `src/main.c`
4. Add source to `SRC` in `Makefile` (or special compile rule if needed)
5. Update N3L detection in main.c GFLOPS counter if applicable

## Key constraints

- SoA (Structure of Arrays) data layout — all kernels expect contiguous x[], y[], z[] arrays
- Particle count padded to multiple of 4 (NEON width)
- Metal kernel pads to TILE_SIZE (128)
- All arrays 16-byte aligned (posix_memalign)
- No full `-ffast-math` — use safe sub-flags only to avoid energy drift

## Current performance ceiling

- NEON: ~29 GFLOPS (single core, compute-bound)
- OpenMP: ~112 GFLOPS (4 P-cores)
- Metal all-pairs: ~810 GFLOPS (10 GPU cores, ALU-bound on LJ chain)
- Metal+CellList: ~128 GFLOPS stencil, 4.3ms/step at 70K (memory-bound, scattered access)
- OMP+CellList: ~34 GFLOPS, 15.7ms/step at 87K
- SME2: ~5.5 GFLOPS (wrong problem shape — designed for FMOPA)

## Next steps

- Cluster-pair neighbor lists (GROMACS NBNXM style) for GPU SIMD coherence
- GCP cross-architecture benchmarking (Axion SVE2, Sapphire Rapids AVX-512, H100 CUDA)
