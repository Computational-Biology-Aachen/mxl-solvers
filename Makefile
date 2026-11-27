CC ?= gcc
CFLAGS ?= -std=gnu17 -O2
SOLVERS = c/radau5.c c/dc_decsol.c c/decsol.c c/dop853.c c/dopri5.c c/libf2c_stubs.c c/lsoda.c c/mxs_lapack.c
# Every .c file in c/sundials is needed: scripts/import-sundials.sh copies only CVODE's.
SUNDIALS = $(wildcard c/sundials/src/*/*.c c/sundials/src/*/*/*.c)
SUNDIALS_INC = -Ic/sundials/include -Ic/sundials/src/sundials
# Fit drivers: Levenberg-Marquardt (cminpack lmdif / lmder with forward sensitivities) and Adam over
# the adjoint, all over the experiment layout of fit_experiments.h.
FIT = c/radau5_wrapper.c c/solver_common.c c/fit_experiments.c c/fit_wrapper.c c/jacobian_wrapper.c \
	c/adjoint_wrapper.c $(wildcard c/minpack/*.c)
INC = -Ic $(SUNDIALS_INC)

.PHONY: test lint asan clean
test: build/test_solvers build/test_problems build/test_wrappers build/test_linalg build/test_minpack build/test_fit build/test_discrete_adjoint
	./build/test_solvers
	./build/test_problems
	./build/test_wrappers
	./build/test_linalg
	./build/test_minpack
	./build/test_fit
	./build/test_discrete_adjoint

# The f2c output and SUNDIALS are left as upstream wrote them, so only our own C is held
# to -Wall.
lint:
	$(CC) $(CFLAGS) -Wall -Wextra -Werror $(INC) -Ic/minpack -Itests -fsyntax-only c/mxs_lapack.c c/radau5_wrapper.c c/solver_common.c c/discrete_adjoint.c tests/test_solvers.c tests/test_problems.c tests/test_wrappers.c tests/test_linalg.c tests/test_minpack.c tests/test_fit.c \
		tests/test_discrete_adjoint.c

build/test_solvers: tests/test_solvers.c $(SOLVERS) $(SUNDIALS) $(wildcard c/*.h) $(shell find c/sundials -name '*.h')
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_solvers.c $(SOLVERS) $(SUNDIALS) -lm

build/test_problems: tests/test_problems.c $(SOLVERS) $(SUNDIALS) $(wildcard c/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_problems.c $(SOLVERS) $(SUNDIALS) -lm

# Single-segment drivers (radau5_wrapper.c) and the output grid.
WRAPPERS = c/radau5_wrapper.c c/solver_common.c

build/test_wrappers: tests/test_wrappers.c $(SOLVERS) $(WRAPPERS) $(wildcard c/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_wrappers.c $(SOLVERS) $(WRAPPERS) -lm

# Dense and banded LU: decsol.c (RADAU5) and mxs_lapack.c (LSODA).
build/test_linalg: tests/test_linalg.c c/decsol.c c/mxs_lapack.c c/libf2c_stubs.c $(wildcard c/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_linalg.c c/decsol.c c/mxs_lapack.c c/libf2c_stubs.c -lm

# cminpack's own tests (tests/cminpack, test-only copies of its examples) against c/minpack.
MINPACK = $(wildcard c/minpack/*.c)
MINPACK_TESTS = tests/cminpack/lmder1.c tests/cminpack/lmdif1.c tests/cminpack/ssqfcn.c tests/cminpack/ssqjac.c \
	tests/cminpack/ssqipt.c

build/test_minpack: tests/test_minpack.c $(MINPACK) $(MINPACK_TESTS) $(wildcard c/minpack/*.h tests/cminpack/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w -Ic/minpack -Itests -o $@ tests/test_minpack.c $(MINPACK_TESTS) $(MINPACK) -lm

build/test_fit: tests/test_fit.c $(SOLVERS) $(FIT) $(wildcard c/*.h c/minpack/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_fit.c $(SOLVERS) $(FIT) -lm

# Discrete adjoint (discretise-then-optimise) over taped RADAU5 steps and LSODA / CVODE operation logs.
DISCRETE = c/discrete_adjoint.c

build/test_discrete_adjoint: tests/test_discrete_adjoint.c $(SOLVERS) $(SUNDIALS) $(DISCRETE) $(wildcard c/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_discrete_adjoint.c $(SOLVERS) $(SUNDIALS) $(DISCRETE) -lm

# Every test again under AddressSanitizer + UndefinedBehaviorSanitizer (leaks, out-of-bounds
# reads and writes, overflow), from separate objects so the plain build stays untouched.
ASAN_FLAGS = -std=gnu17 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
asan:
	mkdir -p build/asan
	$(CC) $(ASAN_FLAGS) -w $(INC) -o build/asan/test_solvers tests/test_solvers.c $(SOLVERS) $(SUNDIALS) -lm
	$(CC) $(ASAN_FLAGS) -w $(INC) -o build/asan/test_problems tests/test_problems.c $(SOLVERS) $(SUNDIALS) -lm
	$(CC) $(ASAN_FLAGS) -w $(INC) -o build/asan/test_wrappers tests/test_wrappers.c $(SOLVERS) $(WRAPPERS) -lm
	$(CC) $(ASAN_FLAGS) -w $(INC) -o build/asan/test_linalg tests/test_linalg.c c/decsol.c c/mxs_lapack.c c/libf2c_stubs.c -lm
	$(CC) $(ASAN_FLAGS) -w -Ic/minpack -Itests -o build/asan/test_minpack tests/test_minpack.c $(MINPACK_TESTS) $(MINPACK) -lm
	$(CC) $(ASAN_FLAGS) -w $(INC) -o build/asan/test_fit tests/test_fit.c $(SOLVERS) $(FIT) -lm
	$(CC) $(ASAN_FLAGS) -w $(INC) -o build/asan/test_discrete_adjoint tests/test_discrete_adjoint.c $(SOLVERS) \
		$(SUNDIALS) $(DISCRETE) -lm
	./build/asan/test_solvers > /dev/null
	./build/asan/test_problems > /dev/null
	./build/asan/test_wrappers > /dev/null
	./build/asan/test_linalg > /dev/null
	./build/asan/test_minpack > /dev/null
	./build/asan/test_fit > /dev/null
	./build/asan/test_discrete_adjoint > /dev/null

clean:
	rm -rf build
