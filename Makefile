CC ?= gcc
CFLAGS ?= -std=gnu17 -O2
SOLVERS = c/radau5.c c/dc_decsol.c c/decsol.c c/dop853.c c/dopri5.c c/libf2c_stubs.c c/lsoda.c c/mxs_lapack.c
# Every .c file in c/sundials is needed: scripts/import-sundials.sh copies only CVODE's.
SUNDIALS = $(wildcard c/sundials/src/*/*.c c/sundials/src/*/*/*.c)
SUNDIALS_INC = -Ic/sundials/include -Ic/sundials/src/sundials
INC = -Ic $(SUNDIALS_INC)

.PHONY: test lint clean
test: build/test_solvers
	./build/test_solvers

# The f2c output and SUNDIALS are left as upstream wrote them, so only our own C is held
# to -Wall.
lint:
	$(CC) $(CFLAGS) -Wall -Wextra -Werror $(INC) -fsyntax-only c/mxs_lapack.c tests/test_solvers.c

build/test_solvers: tests/test_solvers.c $(SOLVERS) $(SUNDIALS) $(wildcard c/*.h) $(shell find c/sundials -name '*.h')
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_solvers.c $(SOLVERS) $(SUNDIALS) -lm

clean:
	rm -rf build
