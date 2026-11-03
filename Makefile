CC ?= gcc
CFLAGS ?= -std=gnu17 -O2
SOLVERS = c/radau5.c c/dc_decsol.c c/decsol.c c/dop853.c c/dopri5.c c/libf2c_stubs.c c/lsoda.c c/mxs_lapack.c

.PHONY: test lint clean
test: build/test_solvers
	./build/test_solvers

# The f2c output is left as generated, so only our own C is held to -Wall.
lint:
	$(CC) $(CFLAGS) -Wall -Wextra -Werror -Ic -fsyntax-only c/mxs_lapack.c tests/test_solvers.c

build/test_solvers: tests/test_solvers.c $(SOLVERS) $(wildcard c/*.h)
	mkdir -p build
	$(CC) $(CFLAGS) -w -Ic -o $@ tests/test_solvers.c $(SOLVERS) -lm

clean:
	rm -rf build
