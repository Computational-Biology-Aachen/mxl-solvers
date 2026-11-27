# cminpack's own tests

Test-only files from [cminpack](https://github.com/devernay/cminpack) at commit `07eea43dd3`, unchanged:

- `ssqfcn.c`, `ssqjac.c`, `ssqipt.c`, `ssq.h`: the 18 Moré–Garbow–Hillstrom least-squares problems, with their Jacobians and starting points.
- `ssq.data`: the problems and dimensions that cminpack's drivers run.
- `lmddrvc.ref`, `lmfdrvc.ref`: the output of those drivers for `lmder1` and `lmdif1`.
- `lmder1.c`, `lmdif1.c`: the simple drivers they call. `c/minpack` doesn't need them.

`tests/test_minpack.c` reruns the drivers' loop against our `c/minpack` and compares the result with the summary tables in the `.ref` files. It compares the final residual norms, using the same tolerance as cminpack's own `cmpfiles` (1% relative). Evaluation counts and exit codes aren't compared: on the rounding-sensitive problems (zero residuals, Powell singular, Kowalik–Osborne) they shift with FMA contraction and the libm, for example on macOS arm64.

The licence is in `c/minpack/COPYRIGHT.txt`.
