# mxl-solvers

The C ODE solvers and fit drivers shared by [MxlPy](https://github.com/Computational-Biology-Aachen/MxlPy)'s `Native` integrator (via [mxlpy-native](https://github.com/Computational-Biology-Aachen/mxlpy-native)) and [mxlweb-core](https://github.com/Computational-Biology-Aachen/mxlweb-core)'s WASM backend. Both repos used to keep their own copies, which drifted apart. This repo is now the only place these files change.

| File | Solver | Origin | Licence |
|---|---|---|---|
| `c/radau5.c` | RADAU5, implicit Runge-Kutta for stiff problems | Hairer & Wanner, `fortran/radau5.f` via f2c, plus local fixes (below) | `licences/HAIRER-LICENCE.txt` |
| `c/dopri5.c`, `c/dop853.c` | Explicit Runge-Kutta 5(4) and 8(5,3) | Hairer & Wanner via f2c | `licences/HAIRER-LICENCE.txt` |
| `c/decsol.c`, `c/dc_decsol.c` | Linear algebra for RADAU5 | Hairer & Wanner via f2c | `licences/HAIRER-LICENCE.txt` |
| `c/f2c.h`, `c/libf2c_stubs.c` | The parts of libf2c the f2c output needs | f2c | |
| `c/lsoda.c`, `c/lsoda.h` | LSODA, switching automatically between Adams (non-stiff) and BDF (stiff) | SciPy `scipy/integrate/src` at main d1d4a8aa11 (v1.18.1 plus the fix that honours the minimum step `RWORK(7)`, scipy/scipy#26092), plus optional discrete-adjoint hooks (`nordsieck_tape.h`, unset = unchanged behaviour) and a corrector-convergence factor (`conit_scale`). Otherwise the code `scipy.integrate` runs. | `licences/SCIPY-LICENCE.txt` |
| `c/sundials/` | CVODE: variable-order BDF (stiff) and Adams-Moulton (non-stiff), with dense/band matrices, the dense direct linear solver and Newton/fixed-point iteration | [SUNDIALS](https://github.com/LLNL/sundials) v7.9.0 (tag `v7.9.0`), upstream layout, unchanged except the two config headers (below) and `scripts/patches/sundials-nordsieck-tape.patch` (discrete-adjoint hooks on the Nordsieck array, re-applied by `scripts/import-sundials.sh`) | `licences/SUNDIALS-LICENCE.txt` (BSD-3-Clause), `licences/SUNDIALS-NOTICE.txt` |
| `c/radau5_wrapper.c`, `c/solver_common.{c,h}` | Single-segment drivers for RADAU5, DOP853, DOPRI5 and LSODA, the registered model function and the output buffer / exact output grid | Written for mxlweb-core | `LICENSE` |
| `c/fit_experiments.{c,h}` | Experiment layout of the fit drivers: time courses with piecewise-constant parameter protocols and steady states, residuals | Written for mxlweb-core | `LICENSE` |
| `c/fit_wrapper.c`, `c/jacobian_wrapper.c`, `c/adjoint_wrapper.c` | Fit drivers: Levenberg-Marquardt with finite differences (`lmdif`) or forward sensitivities (`lmder`, an augmented system of states and sensitivities), and Adam over the continuous adjoint; `adjoint_vjp`, one adjoint gradient (vector-Jacobian product) for MxlPy's fits | Written for mxlweb-core and MxlPy | `LICENSE` |
| `c/discrete_adjoint.{c,h}`, `c/nordsieck_tape.h` | Discrete (discretise-then-optimise) adjoint gradients: a forward solve with RADAU5, LSODA or CVODE (BDF / Adams) that records every step (RADAU5's stages, the Nordsieck solvers' operations on their history array), then any number of backward passes over it, exact for the computed solution with frozen step sizes; `da_replay` re-solves the recorded steps for gradient checks | Written for MxlPy | `LICENSE` |
| `c/finite_jacobian.h` | `finite_jacobian`: a compiled Jacobian whose non-finite columns (e.g. a symbolic derivative at an `exp` overflow) are replaced by central differences of the model; the discrete adjoint and mxlpy-native's solvers call Jacobians through it | Written for MxlPy | `LICENSE` |
| `c/minpack/` | `lmdif`, `lmder` and their helpers | [cminpack](https://github.com/devernay/cminpack) | `c/minpack/COPYRIGHT.txt` |
| `c/blaslapack_declarations.h`, `c/mxs_lapack.c` | The four LAPACK routines `lsoda.c` calls (`dgetrf`/`dgetrs`/`dgbtrf`/`dgbtrs`), unblocked | Written here; replaces SciPy's header of the same name so `lsoda.c` builds without BLAS | `LICENSE` |

`fortran/` keeps the Hairer originals and their example drivers (`dr_*.f`).

## Local changes to `radau5.c`

`radau5.c` isn't the plain f2c output any more:
- Non-finite steps are rejected.
- `radau5_jac_growth` limits how far the step size can grow while a Jacobian is reused.

Don't regenerate it from `fortran/radau5.f` without carrying these changes over.

## SUNDIALS

`c/sundials/` holds the part of [SUNDIALS](https://github.com/LLNL/sundials) that CVODE needs,
copied by `scripts/import-sundials.sh <sundials-source-dir>` from the v7.9.0 release. It keeps
upstream's `include/` and `src/` layout, so the files build unchanged. Upstream's copyright and
SPDX headers stay in every file. Don't edit them here; to upgrade, re-run the script on a newer
release.

The only files that aren't upstream's are `include/sundials/sundials_config.h` and
`sundials_export.h`, which SUNDIALS' CMake would generate. They were generated once with CMake
(CVODE only, 32-bit indices, logging level 0, error checks and profiling off), then trimmed so
the same files build with gcc, clang, MSVC and emcc. Their headers say what changed. The
profiler (`sundials_profiler.c`) is left out: it's off, every call to it is behind
`SUNDIALS_ENABLE_PROFILING`, and its timers don't build in wasm.

Build every `.c` file below `c/sundials/src`, with `-Ic/sundials/include -Ic/sundials/src/sundials`.

SUNDIALS is © Lawrence Livermore National Security, Southern Methodist University, University
of Maryland Baltimore County and the SUNDIALS contributors, under the BSD 3-Clause licence. Work
that uses CVODE through these solvers should cite SUNDIALS as its authors ask
([CITATIONS.md](https://github.com/LLNL/sundials/blob/v7.9.0/CITATIONS.md)):

- S. B. Roberts, M. Aggul, D. R. Reynolds, C. J. Balos, D. J. Gardner and C. S. Woodward, "New
  Time Integrators and Capabilities in SUNDIALS Versions 6.2.0-7.4.0", *ACM Transactions on
  Mathematical Software* 52(1), 8:1–8:14, 2026. doi:10.1145/3797888
- D. J. Gardner, D. R. Reynolds, C. S. Woodward and C. J. Balos, "Enabling new flexibility in the
  SUNDIALS suite of nonlinear and differential/algebraic equation solvers", *ACM Transactions on
  Mathematical Software* 48(3), 1–24, 2022. doi:10.1145/3539801
- A. C. Hindmarsh, P. N. Brown, K. E. Grant, S. L. Lee, R. Serban, D. E. Shumaker and
  C. S. Woodward, "SUNDIALS: Suite of Nonlinear and Differential/Algebraic Equation Solvers",
  *ACM Transactions on Mathematical Software* 31(3), 363–396, 2005. doi:10.1145/1089014.1089020
- A. C. Hindmarsh, R. Serban, C. J. Balos, D. J. Gardner, D. R. Reynolds and C. S. Woodward,
  "User Documentation for CVODE", v7.9.0, 2026.

## Tests

```bash
make test   # reference problems for every solver, as relative errors
make lint   # -Wall -Wextra -Werror on the code written here
```

`tests/test_fit.c` fits problems with known optima: exponential decay with both Levenberg-Marquardt drivers (RADAU5 and LSODA, linear and log parameters) to 1e-8 relative precision, and the parameter ratio of a reversible reaction at steady state. It checks `adjoint_vjp` against analytic gradients (exponential decay with a state and a derived target on RADAU5, LSODA and DOPRI5, and a reversible reaction's steady state with its conserved total) to 1e-6 relative precision, and that Adam over the adjoint fits the decay rate.

`tests/test_solvers.c` checks Robertson against Hairer & Wanner's reference values (RADAU5, LSODA with full and banded Jacobians, and CVODE BDF), and that LSODA stops on Robertson when given a minimum step too large for its initial transient. It checks that Lotka-Volterra's first integral stays constant (LSODA, DOPRI5, DOP853, RADAU5, CVODE Adams and BDF), and that Van der Pol agrees with a tight DOP853 solution.

## Using these files in another repo

Copy the files with `scripts/vendor.sh` instead of by hand:

```bash
scripts/vendor.sh <dest-dir> c/radau5.c c/decsol.c ... c/sundials licences/HAIRER-LICENCE.txt
```

Files land at their path below `c/` or `licences/` (so `c/sundials/...` becomes `<dest-dir>/sundials/...`), and a directory stands for all its files. The script needs a clean, pushed commit. It writes `<dest-dir>/VENDORED` with that commit's SHA and a sha256 for every file, and each consuming repo has a test that checks these hashes. `VENDORED` is rewritten on every run, so always pass the full list. To fix a solver, change it here, push, then re-vendor into the consumers.

Current consumers:
- [mxlpy-native](https://github.com/Computational-Biology-Aachen/mxlpy-native): `c/vendor/`, built into its wheels and into MxlPy's `solvers.wasm`.
- mxlweb-core: `src-c/`, built into `static/wasm/radau5.wasm` (and SUNDIALS into the separate, lazily loaded `static/wasm/cvode.wasm`). Only its CVODE wrapper (`cvode_wrapper.c`) stays in mxlweb-core.

The fit drivers keep their state in globals, so only one fit can run at a time per process (or WASM instance). Functions are registered as `intptr_t`: a WASM table index from JavaScript, a function pointer from native code.
