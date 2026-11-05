# mxl-solvers

The C ODE solvers shared by [MxlPy](https://github.com/Computational-Biology-Aachen/MxlPy)'s `Native` integrator (via [mxlpy-native](https://github.com/Computational-Biology-Aachen/mxlpy-native)) and [mxlweb-core](https://github.com/Computational-Biology-Aachen/mxlweb-core)'s WASM backend. Both repos used to keep their own copies, which drifted apart. This repo is now the only place these files change.

| File | Solver | Origin | Licence |
|---|---|---|---|
| `c/radau5.c` | RADAU5, implicit Runge-Kutta for stiff problems | Hairer & Wanner, `fortran/radau5.f` via f2c, plus local fixes (below) | `licences/HAIRER-LICENCE.txt` |
| `c/dopri5.c`, `c/dop853.c` | Explicit Runge-Kutta 5(4) and 8(5,3) | Hairer & Wanner via f2c | `licences/HAIRER-LICENCE.txt` |
| `c/decsol.c`, `c/dc_decsol.c` | Linear algebra for RADAU5 | Hairer & Wanner via f2c | `licences/HAIRER-LICENCE.txt` |
| `c/f2c.h`, `c/libf2c_stubs.c` | The parts of libf2c the f2c output needs | f2c | |
| `c/lsoda.c`, `c/lsoda.h` | LSODA, switching automatically between Adams (non-stiff) and BDF (stiff) | SciPy v1.18.1 `scipy/integrate/src`, unchanged. This is the code `scipy.integrate` runs. | `licences/SCIPY-LICENCE.txt` |
| `c/blaslapack_declarations.h`, `c/mxs_lapack.c` | The four LAPACK routines `lsoda.c` calls (`dgetrf`/`dgetrs`/`dgbtrf`/`dgbtrs`), unblocked | Written here; replaces SciPy's header of the same name so `lsoda.c` builds without BLAS | `LICENSE` |

`fortran/` keeps the Hairer originals and their example drivers (`dr_*.f`).

## Local changes to `radau5.c`

`radau5.c` isn't the plain f2c output any more:
- Non-finite steps are rejected.
- `radau5_jac_growth` limits how far the step size can grow while a Jacobian is reused.

Don't regenerate it from `fortran/radau5.f` without carrying these changes over.

## Tests

```bash
make test   # reference problems for every solver, as relative errors
make lint   # -Wall -Wextra -Werror on the code written here
```

`tests/test_solvers.c` checks Robertson against Hairer & Wanner's reference values (RADAU5, and LSODA with full and banded Jacobians). It checks that Lotka-Volterra's first integral stays constant (LSODA, DOPRI5, DOP853, RADAU5), and that Van der Pol agrees with a tight DOP853 solution.

## Using these files in another repo

Copy the files with `scripts/vendor.sh` instead of by hand:

```bash
scripts/vendor.sh <dest-dir> c/radau5.c c/decsol.c ... licences/HAIRER-LICENCE.txt
```

The script needs a clean, pushed commit. It writes `<dest-dir>/VENDORED` with that commit's SHA and a sha256 for every file, and each consuming repo has a test that checks these hashes. To fix a solver, change it here, push, then re-vendor into the consumers.

Current consumers:
- [mxlpy-native](https://github.com/Computational-Biology-Aachen/mxlpy-native): `c/vendor/`, built into its wheels and into MxlPy's `solvers.wasm`.
- mxlweb-core: `src-c/`, built into `static/wasm/radau5.wasm`. Its wrappers and minpack stay in mxlweb-core.
