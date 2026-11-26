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

.PHONY: test lint clean
test: build/test_solvers build/test_fit build/test_discrete_adjoint
	./build/test_solvers
	./build/test_fit
	./build/test_discrete_adjoint

# The f2c output and SUNDIALS are left as upstream wrote them, so only our own C is held
# to -Wall.
lint:
	$(CC) $(CFLAGS) -Wall -Wextra -Werror $(INC) -fsyntax-only c/mxs_lapack.c c/discrete_adjoint.c tests/test_solvers.c tests/test_fit.c \
		tests/test_discrete_adjoint.c

build/test_solvers: tests/test_solvers.c $(SOLVERS) $(SUNDIALS) $(wildcard c/*.h) $(shell find c/sundials -name '*.h')
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_solvers.c $(SOLVERS) $(SUNDIALS) -lm

build/test_fit: tests/test_fit.c $(SOLVERS) $(FIT) $(wildcard c/*.h c/minpack/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_fit.c $(SOLVERS) $(FIT) -lm

# Discrete adjoint (discretise-then-optimise) over taped RADAU5 steps and LSODA / CVODE operation logs.
DISCRETE = c/discrete_adjoint.c

build/test_discrete_adjoint: tests/test_discrete_adjoint.c $(SOLVERS) $(SUNDIALS) $(DISCRETE) $(wildcard c/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w $(INC) -o $@ tests/test_discrete_adjoint.c $(SOLVERS) $(SUNDIALS) $(DISCRETE) -lm

clean:
	rm -rf build
