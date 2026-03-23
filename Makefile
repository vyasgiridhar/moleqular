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
          src/md_integrate.c src/md_io.c
OBJ     = $(SRC:.c=.o)

# Metal sources (Objective-C host + optional compiled shader library)
METAL_HOST_SRC = src/md_force_metal_host.m
METAL_HOST_OBJ = src/md_force_metal_host.o
METAL_SHADER   = src/md_force_metal.metal
METAL_AIR      = src/md_force_metal.air
METAL_LIB      = md_force.metallib

BIN     = moleqular

.PHONY: all clean run bench metallib

all: $(BIN)

# Link: C objects + Objective-C Metal host object
$(BIN): $(OBJ) $(METAL_HOST_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJ) $(METAL_HOST_OBJ)

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
	@echo "=== Metal GPU ===" && ./$(BIN) --metal

clean:
	rm -f $(OBJ) $(METAL_HOST_OBJ) $(METAL_AIR) $(METAL_LIB) $(BIN) trajectory.xyz
