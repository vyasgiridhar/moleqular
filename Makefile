CC      = clang
CFLAGS  = -O2 -mcpu=native -std=c17
CFLAGS += -Iinclude
CFLAGS += -ffp-contract=fast
CFLAGS += -fno-math-errno -fno-signed-zeros -freciprocal-math -fno-trapping-math
CFLAGS += -funroll-loops -flto=thin -fno-stack-check
CFLAGS += -Wall -Wextra -Wpedantic

LDFLAGS = -flto=thin -Wl,-dead_strip -lm

# Diagnostic flags (uncomment to see vectorization remarks)
# CFLAGS += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize

SRC     = src/main.c src/md_system.c src/md_force_scalar.c src/md_force_neon.c \
          src/md_integrate.c src/md_io.c
OBJ     = $(SRC:.c=.o)
BIN     = moleqular

.PHONY: all clean run

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(OBJ) $(BIN) trajectory.xyz
