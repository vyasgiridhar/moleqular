# Molecular Dynamics NEON Kernel — Design Document

## Goal

Build a classic N-body molecular dynamics kernel in C, optimized for Apple M4 using ARM NEON intrinsics. The project is educational: understand NEON SIMD at the register level while producing a working, benchmarkable MD simulation.

## Physics

- **Potential:** Lennard-Jones with cutoff and shifted potential
  - `V(r) = 4e [(s/r)^12 - (s/r)^6] - V(rc)` for `r < rc`
  - `F(r) = 24e/r [2(s/r)^12 - (s/r)^6] * r_hat`
- **Integrator:** Velocity Verlet (symplectic, time-reversible)
  - `x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2`
  - `v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt`
- **Boundary conditions:** Periodic, cubic box, minimum image convention
- **Scale:** ~1,000 particles, O(N^2) all-pairs (fits in L1 cache)
- **Initialization:** FCC lattice + Maxwell-Boltzmann velocity distribution

## Target Hardware

- Apple M4 (MacBook Air, Mac16,12)
- 4 Performance cores (128KB L1D, 16MB shared L2) + 6 Efficiency cores
- **128-byte cache lines** (critical for data layout)
- NEON: 32 x 128-bit registers, FMA, reciprocal estimate
- Available features: FP16, BF16, DotProd, SME2, LSE2 (NEON only for v1)

## Data Layout — Structure of Arrays

```c
typedef struct {
    float *x, *y, *z;       // positions
    float *vx, *vy, *vz;    // velocities
    float *fx, *fy, *fz;    // forces
    int n;                   // particle count (padded to multiple of 4)
    float lbox;              // box side length
} MDSystem;
```

All arrays 16-byte aligned via `posix_memalign`. SoA ensures contiguous access
for NEON loads: `vld1q_f32(&x[j])` fetches 4 particle positions in one shot.

## NEON Force Kernel — Inner Loop

For each particle `i`, broadcast its position to all 4 NEON lanes, then sweep
`j` in steps of 4:

1. **Load:** `vld1q_f32` for xj, yj, zj (4 particles per load)
2. **Distance:** dx, dy, dz via subtract; apply minimum image with `vrndnq_f32`
3. **r^2:** `vfmaq_f32` chain: `dx*dx + dy*dy + dz*dz`
4. **Reciprocal:** `vrecpeq_f32` + Newton-Raphson via `vrecpsq_f32` (~24-bit precision)
5. **LJ terms:** `inv_r6 = inv_r2^3`, `inv_r12 = inv_r6^2`
6. **Force magnitude:** `24e * (2*inv_r12 - inv_r6) * inv_r2`
7. **Cutoff mask:** `vcltq_f32(r2, rc2)` — branchless, zeroes out beyond cutoff
8. **Accumulate:** `fxi += f_mag * dx` (horizontal reduction after j-loop)

Key NEON intrinsics: `vld1q_f32`, `vst1q_f32`, `vfmaq_f32`, `vrecpeq_f32`,
`vrecpsq_f32`, `vcltq_f32`, `vdupq_n_f32`, `vaddvq_f32`.

## Two Implementations

- `md_force_scalar.c` — plain C, no intrinsics. Baseline for benchmarking.
- `md_force_neon.c` — hand-written NEON intrinsics. Compare wall time.

Side-by-side comparison demonstrates exactly what NEON buys.

## File Structure

```
moleqular/
  include/
    md_types.h          # MDSystem struct, physical constants
    md_neon.h           # NEON force kernel declaration
    md_integrate.h      # Velocity Verlet declaration
    md_io.h             # XYZ output declaration
  src/
    main.c              # simulation loop, timing, CLI
    md_system.c         # alloc, FCC init, Maxwell-Boltzmann velocities
    md_force_scalar.c   # plain C force loop (baseline)
    md_force_neon.c     # NEON-intrinsics force kernel
    md_integrate.c      # Velocity Verlet integrator
    md_io.c             # .xyz file output (VMD/OVITO compatible)
  Makefile
```

## Build Configuration

```makefile
CC = clang
CFLAGS  = -O2 -mcpu=native
CFLAGS += -ffp-contract=fast
CFLAGS += -fno-math-errno -fno-signed-zeros -freciprocal-math -fno-trapping-math
CFLAGS += -funroll-loops -flto=thin -fno-stack-check
LDFLAGS = -flto=thin -Wl,-dead_strip
```

Rationale:
- `-O2` not `-O3`: identical assembly for NEON intrinsic code, safer
- No full `-ffast-math`: `-fassociative-math` causes energy drift over 10^6 steps
- Safe sub-flags only: `freciprocal-math`, `fno-signed-zeros`, `fno-math-errno`
- ThinLTO: parallel, often outperforms full LTO
- Diagnostics available: `-Rpass=loop-vectorize`, `-fsave-optimization-record`

## Output

`.xyz` format — simple text, one frame per file or concatenated. Viewable in
VMD or OVITO. Also print per-step: kinetic energy, potential energy, total
energy (conservation check), temperature.

## Future Extensions (not in v1)

- Coulomb potential (electrostatics)
- Cell lists / Verlet neighbor lists for O(N) scaling
- Multi-threaded with GCD or pthreads (exploit P+E cores)
- SME2 matrix operations for distance matrices
- Double-precision mode
