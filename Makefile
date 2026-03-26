CC      = clang
CFLAGS  = -O2 -mcpu=native -std=c17
CFLAGS += -Iinclude
CFLAGS += -ffp-contract=fast
CFLAGS += -fno-math-errno -fno-signed-zeros -freciprocal-math -fno-trapping-math
CFLAGS += -funroll-loops -flto=thin -fno-stack-check
CFLAGS += -Wall -Wextra -Wpedantic

# OpenMP via Homebrew libomp
OMP_PREFIX = $(shell brew --prefix libomp 2>/dev/null)
CFLAGS  += -Xclang -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS  = -flto=thin -Wl,-dead_strip -lm -L$(OMP_PREFIX)/lib -lomp

# Metal frameworks (needed for GPU force kernel)
LDFLAGS += -framework Metal -framework Foundation

# Diagnostic flags (uncomment to see vectorization remarks)
# CFLAGS += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize

# Objective-C flags for Metal host code (no -Wpedantic: ObjC extensions are fine)
OBJCFLAGS = -O2 -std=c17 -fobjc-arc -Iinclude -Wall -Wextra

# Detect if Metal offline compiler is available (requires full Xcode)
METAL_COMPILER := $(shell xcrun --find metal 2>/dev/null)

SRC     = src/main.c src/md_system.c src/md_force_scalar.c src/md_force_neon.c \
          src/md_force_omp.c src/md_force_tiled.c src/md_force_sme2.c \
          src/md_force_neon_n3l.c src/md_force_f64.c \
          src/md_celllist.c src/md_force_neon_cl.c src/md_force_omp_cl.c \
          src/md_bvh.c src/md_cluster.c \
          src/md_integrate.c src/md_io.c
OBJ     = $(SRC:.c=.o)

# Metal sources (Objective-C host + optional compiled shader library)
METAL_HOST_SRC    = src/md_force_metal_host.m
METAL_HOST_OBJ    = src/md_force_metal_host.o
METAL_CL_HOST_SRC = src/md_force_metal_cl_host.m
METAL_CL_HOST_OBJ = src/md_force_metal_cl_host.o
ANE_HOST_SRC      = src/md_force_ane_host.m
ANE_HOST_OBJ      = src/md_force_ane_host.o
BVH_HOST_SRC      = src/md_force_metal_bvh_host.m
BVH_HOST_OBJ      = src/md_force_metal_bvh_host.o
NBNXM_HOST_SRC    = src/md_force_metal_nbnxm_host.m
NBNXM_HOST_OBJ    = src/md_force_metal_nbnxm_host.o
ANE_DIRECT_SRC    = src/md_force_ane_direct.m
ANE_DIRECT_OBJ    = src/md_force_ane_direct.o
METAL_SHADER      = src/md_force_metal.metal
METAL_AIR         = src/md_force_metal.air
METAL_LIB         = md_force.metallib

# Visualization sources
VIZ_SRC = src/md_viz.m
VIZ_OBJ = src/md_viz.o
VIZ_BIN = moleqular-viz

# Shared objects (everything except main.o for the viz binary)
SHARED_SRC = src/md_system.c src/md_force_scalar.c src/md_force_neon.c \
             src/md_force_omp.c src/md_force_tiled.c src/md_force_sme2.c \
             src/md_force_neon_n3l.c src/md_force_f64.c \
             src/md_celllist.c src/md_force_neon_cl.c src/md_force_omp_cl.c \
             src/md_bvh.c src/md_cluster.c \
             src/md_integrate.c src/md_io.c
SHARED_OBJ = $(SHARED_SRC:.c=.o)

BIN     = moleqular

.PHONY: all clean run bench metallib viz

all: $(BIN)

viz: $(VIZ_BIN)

# Link: C objects + Objective-C host objects
$(BIN): $(OBJ) $(METAL_HOST_OBJ) $(METAL_CL_HOST_OBJ) $(ANE_HOST_OBJ) $(BVH_HOST_OBJ) $(NBNXM_HOST_OBJ) $(ANE_DIRECT_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -framework CoreML -framework IOSurface -o $@ $(OBJ) $(METAL_HOST_OBJ) $(METAL_CL_HOST_OBJ) $(ANE_HOST_OBJ) $(BVH_HOST_OBJ) $(NBNXM_HOST_OBJ) $(ANE_DIRECT_OBJ)

# Visualization binary (separate executable with its own main)
$(VIZ_BIN): $(SHARED_OBJ) $(METAL_HOST_OBJ) $(METAL_CL_HOST_OBJ) $(ANE_HOST_OBJ) $(BVH_HOST_OBJ) $(NBNXM_HOST_OBJ) $(VIZ_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -framework CoreML -framework MetalKit -framework AppKit \
		-o $@ $(SHARED_OBJ) $(METAL_HOST_OBJ) $(METAL_CL_HOST_OBJ) $(ANE_HOST_OBJ) $(BVH_HOST_OBJ) $(NBNXM_HOST_OBJ) $(VIZ_OBJ)

# --- Metal shader precompilation (optional, requires full Xcode) ---
# If you have Xcode installed, `make metallib` creates a precompiled .metallib
# for faster startup (~0.7ms vs ~20-100ms runtime compilation).
# Without it, the shader compiles from embedded source on first --metal run.
metallib: $(METAL_LIB)

# Step 1: .metal -> .air (intermediate representation)
$(METAL_AIR): $(METAL_SHADER)
ifdef METAL_COMPILER
	xcrun -sdk macosx metal -std=metal3.2 -ffast-math -c $< -o $@
else
	@echo "[Metal] Offline compiler not found (need full Xcode, not just CLT)."
	@echo "[Metal] Shader will be compiled at runtime from embedded source."
	@false
endif

# Step 2: .air -> .metallib (GPU binary library)
$(METAL_LIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $< -o $@

# --- Metal host (Objective-C) compilation ---
# Compiled separately: needs -fobjc-arc. No OpenMP, no NEON intrinsics.
$(METAL_HOST_OBJ): $(METAL_HOST_SRC) include/md_types.h include/md_force.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(METAL_CL_HOST_OBJ): $(METAL_CL_HOST_SRC) include/md_types.h include/md_force.h include/md_celllist.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(ANE_HOST_OBJ): $(ANE_HOST_SRC) include/md_types.h include/md_force.h include/md_celllist.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(BVH_HOST_OBJ): $(BVH_HOST_SRC) include/md_types.h include/md_force.h include/md_bvh.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(NBNXM_HOST_OBJ): $(NBNXM_HOST_SRC) include/md_types.h include/md_force.h include/md_cluster.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(ANE_DIRECT_OBJ): $(ANE_DIRECT_SRC) include/md_types.h include/md_force.h src/ane_runtime.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

$(VIZ_OBJ): $(VIZ_SRC) include/md_types.h include/md_force.h
	$(CC) $(OBJCFLAGS) -c -o $@ $<

# SME2 file needs -march instead of -mcpu (streaming SVE mode)
src/md_force_sme2.o: src/md_force_sme2.c
	$(CC) -O2 -march=armv8-a+sme2 -std=c17 -Iinclude -ffp-contract=fast -fno-math-errno -fno-signed-zeros -freciprocal-math -fno-trapping-math -funroll-loops -flto=thin -fno-stack-check -Wall -Wextra -c -o $@ $<

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(BIN)
	./$(BIN)

bench: $(BIN)
	@echo "=== Scalar ===" && ./$(BIN) --scalar
	@echo ""
	@echo "=== NEON ===" && ./$(BIN)
	@echo ""
	@echo "=== OpenMP+NEON (all cores) ===" && OMP_NUM_THREADS=10 ./$(BIN) --omp
	@echo ""
	@echo "=== OpenMP+NEON (P-cores only) ===" && OMP_NUM_THREADS=4 ./$(BIN) --omp
	@echo ""
	@echo "=== Tiled+OMP+NEON (4 P-cores) ===" && OMP_NUM_THREADS=4 ./$(BIN) --tiled
	@echo ""
	@echo "=== NEON+CellList ===" && ./$(BIN) --cl
	@echo ""
	@echo "=== OMP+NEON+CellList (4 P-cores) ===" && OMP_NUM_THREADS=4 ./$(BIN) --omp-cl
	@echo ""
	@echo "=== Metal GPU ===" && ./$(BIN) --metal
	@echo ""
	@echo "=== Metal+CellList ===" && ./$(BIN) --metal-cl

clean:
	rm -f $(OBJ) $(METAL_HOST_OBJ) $(METAL_CL_HOST_OBJ) $(ANE_HOST_OBJ) $(BVH_HOST_OBJ) $(NBNXM_HOST_OBJ) $(ANE_DIRECT_OBJ) $(VIZ_OBJ) \
		$(METAL_AIR) $(METAL_LIB) $(BIN) $(VIZ_BIN) trajectory.xyz
