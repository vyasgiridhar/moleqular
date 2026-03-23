# moleqular

Molecular dynamics on Apple M4 — pushing every compute path to its limits.

LJ (Lennard-Jones) N-body simulation with **11 force kernels** targeting different hardware units on Apple Silicon. Same physics, same particles, wildly different performance characteristics.

## Kernels

| Kernel | Flag | Hardware | Description |
|--------|------|----------|-------------|
| Scalar | `--scalar` | CPU (1 core) | Plain C, Newton's 3rd law, real division. Baseline. |
| NEON | *(default)* | CPU SIMD | Hand-written ARM NEON intrinsics, float32x4, 4 particles/op |
| NEON+N3L | `--n3l` | CPU SIMD | NEON with Newton's 3rd law — half the pairs, SIMD scatter-write |
| NEON f64 | `--f64` | CPU SIMD | Double-precision NEON (float64x2), 2 particles/op |
| OpenMP | `--omp` | CPU multi-core | NEON kernel parallelized across P-cores |
| SME2 | `--sme2` | SME coprocessor | ARM Scalable Matrix Extension 2, streaming SVE (16-wide) |
| Tiled | `--tiled` | CPU multi-core | Cache-tiled OpenMP+NEON with CPU pinning |
| **Metal** | `--metal` | **M4 GPU** | Metal compute shader, tiled all-pairs, Horner polynomial |
| NEON+CL | `--cl` | CPU SIMD | NEON with cell list neighbor lists — O(N) scaling |
| OMP+CL | `--omp-cl` | CPU multi-core | OpenMP+NEON with cell lists |
| **Metal+CL** | `--metal-cl` | **M4 GPU** | Metal compute with cell lists — O(N) GPU |

## Benchmark Results (Apple M4, MacBook Air, 16GB)

### Scaling by particle count (GFLOPS)

```
N        Scalar   NEON     NEON+N3L  OMP(4P)  Metal GPU
864      13.6     28.9     22.6      86.3     23.5
2,048    16.3     28.8     22.4      86.2     71.3
4,000    17.8     28.5     22.5      95.1     189.2
6,912    18.5     28.8     22.6      93.6     436.2
10,976   18.9     28.7     22.4      98.1     421.5
32,000   —        —        —         —        780.2
87,808   —        —        —         —        808.2
```

Metal GPU all-pairs: **810 GFLOPS peak** (~19% of 4.26 TFLOPS theoretical).

### Cell list scaling (ms/step, wall clock)

```
N        NEON     NEON+CL  OMP+CL(4P)  Metal AP   Metal+CL
864      0.58     0.73     —           1.5        1.2
4,000    10.9     4.3      —           1.33       0.64
10,976   83.1     9.6      2.65        3.70       0.64
32,000   698      26.1     7.56        26.5       2.29
70,304   —        56.3     15.7        ~190       4.26
```

Cell lists turn O(N²) into O(N). At 70K particles, Metal+CL is **45× faster** than Metal all-pairs.
OMP+CL on 4 P-cores beats all-pairs Metal GPU above 32K particles.

### Why different kernels exist

Each kernel exercises different M4 hardware:

- **NEON** (29 GFLOPS) — single-core SIMD ceiling. 4 FMA pipes × 4 floats.
- **NEON+N3L** — lower GFLOPS but faster wall time (half the pairs).
- **Scalar** beats NEON at large N because Newton's 3rd law halves work.
- **f64** — 2.2x slower than f32. Proves float32 is sufficient for LJ MD (0.02% energy drift over 10^6 steps vs 0.009% for f64 — timestep error dominates).
- **SME2** (5.5 GFLOPS) — streaming SVE through L2 data path. Wrong tool for element-wise work; designed for FMOPA matrix outer products.
- **OpenMP** (~112 GFLOPS) — 4 P-cores saturated. E-cores hurt performance at small N due to smaller caches.
- **Metal** (~810 GFLOPS) — 10 GPU cores on unified memory. Zero-copy buffers. Crosses over OMP at ~3K particles.
- **Cell list kernels** — linked-list cell lists, gather neighbors into contiguous buffer, same NEON/GPU inner loop. Crossover vs all-pairs at ~2K particles.

## Build

Requires macOS on Apple Silicon (M1+). Needs Homebrew `libomp` for OpenMP kernels.

```bash
brew install libomp
make            # builds everything
make bench      # runs all kernels
make metallib   # (optional) precompile Metal shader (needs Xcode)
```

### Compiler flags

```
-O2 -mcpu=native -ffp-contract=fast
-fno-math-errno -fno-signed-zeros -freciprocal-math -fno-trapping-math
-funroll-loops -flto=thin
```

No full `-ffast-math` — it enables `-fassociative-math` which causes energy drift over long simulations.

## Usage

```bash
./moleqular                          # default: NEON, 864 particles, 1000 steps
./moleqular --metal --ncells=14      # Metal GPU, 10976 particles
./moleqular --omp --steps=5000       # OpenMP, 5000 steps
OMP_NUM_THREADS=4 ./moleqular --omp  # Pin to 4 P-cores
./moleqular --f64 --steps=1000000 --no-traj  # precision drift test
```

| Flag | Description |
|------|-------------|
| `--ncells=N` | FCC unit cells per side (particles = 4*N^3) |
| `--steps=N` | Number of timesteps |
| `--no-traj` | Skip trajectory file output |
| `--thermo=N` | Thermodynamics output frequency |

## Physics

- Lennard-Jones potential: V(r) = 4e[(s/r)^12 - (s/r)^6]
- Velocity Verlet integrator (symplectic, time-reversible)
- Periodic boundary conditions, minimum image convention
- Cutoff at 2.5 sigma with shifted potential
- FCC lattice initialization + Maxwell-Boltzmann velocities
- Reduced units: sigma=1, epsilon=1, mass=1, kB=1

## What I learned

- M4 NEON: 4 parallel FMA pipes = 16 floats/cycle. Wider than it looks at 128-bit.
- SME2 streaming SVE: 16-wide vectors but through L2 (not in-core). 5x slower than NEON for element-wise work. Only wins with FMOPA outer products.
- AVX-512 frequency throttling is real — ARM avoids it with multiple narrow pipes.
- Metal unified memory: zero-copy CPU↔GPU. `MTLResourceStorageModeShared` + `newBufferWithBytesNoCopy` wraps existing allocations.
- GPU all-pairs bottleneck is the serial LJ dependency chain (10 cycles per pair), not parallelism. ~19% of theoretical peak is the expected efficiency for this workload.
- Cell lists on GPU: O(N) scaling works great (45× at 70K) but per-pair GPU utilization drops from 19% to 3% due to scattered memory access (each thread reads different neighbor cells).
- float32 precision is fine for MD — energy drift is dominated by integrator timestep, not FP rounding.

## Experiment log — what I tried to optimize and what happened

### Hits

| Optimization | Result | Why it worked |
|---|---|---|
| **NEON intrinsics** (all-pairs) | 29 GFLOPS (2× scalar) | 4-wide FMA, reciprocal estimate + NR avoids division |
| **Newton's 3rd law** (scalar) | Fastest wall time at small N | Half the pairs, even though scalar |
| **OpenMP on 4 P-cores only** | 112 GFLOPS (not 10 cores) | E-cores have 1/2 L1, 1/4 L2 — j-arrays spill |
| **Metal tiled all-pairs** | 810 GFLOPS | Shared memory tiles amortize j-particle loads across 128 threads |
| **Cell lists** (CPU) | 340× speedup at 87K (NEON→OMP+CL) | O(N) vs O(N²). Only ~52 neighbors within cutoff at density 0.8 |
| **Cell lists** (GPU) | 45× speedup at 70K | Same O(N) win; sorted particles give decent L2 behavior |

### Misses

| Optimization | Result | Why it failed |
|---|---|---|
| **Cache tiling** (CPU, I/J blocks) | No improvement | Not a stencil kernel. Hardware prefetcher handles sequential SoA access perfectly. Manual tiling adds overhead for no cache benefit. |
| **SME2 streaming SVE** | 5.5 GFLOPS (5× slower than NEON) | Data flows Core→L2→SME coprocessor→L2→Core. Designed for FMOPA matrix outer products, not element-wise pairwise forces. |
| **Piecewise Horner polynomial** (Metal) | 812 GFLOPS (no change from 810) | Replaces 8-cycle analytical LJ with 4-FMA Horner. But total per-pair cycle count (index + Horner + mask + accum) ≈ 8 anyway. The dependency chain is the bottleneck, not the specific ops. |
| **GPU shared memory tiled cell list** (32 threads/threadgroup) | 2-3× slower | Apple GPU needs 128-256 threads per threadgroup for latency hiding. 32 threads = 1 SIMD group = zero overlap when stalling on memory. Occupancy starvation killed it. |
| **Morton Z-order sort** (GPU cell list) | No improvement | Adjacent threads already in nearby cells from row-major sort. Morton improves 3D locality, but the L2 cache is big enough that row-major works fine at these sizes. |
| **Padded uniform cells** (CELL_CAP=32) | Slower at small N | 3× more array slots than real particles (32 slots/cell × ~13 actual). Dispatch overhead dominates at small-medium N. |
| **Horner polynomial in cell list kernel** | Worse energy conservation, no speed gain | Force (Horner) and PE (analytical) computed from different functions → non-conservative forces → systematic energy drift. Speed unchanged because cell list kernel is memory-bound, not compute-bound. |

### Key insight

The all-pairs tiled Metal kernel gets 810 GFLOPS because all threads in a threadgroup read the **same** j-particles from shared memory — perfect SIMD coherence. Cell lists break this: each thread reads **different** j-particles from different cells. This is a fundamental tension between algorithmic efficiency (O(N)) and GPU execution efficiency (coherent SIMD). Production MD codes (GROMACS, LAMMPS) resolve this with cluster-pair neighbor lists — a research-grade data structure, not a quick optimization.

## License

MIT
