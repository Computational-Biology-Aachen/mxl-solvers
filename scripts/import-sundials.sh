#!/usr/bin/env bash
# Copy the parts of SUNDIALS that CVODE needs into c/sundials/, keeping upstream's layout.
#
#   scripts/import-sundials.sh <sundials-source-dir>
#   e.g. scripts/import-sundials.sh ~/src/sundials-7.9.0
#
# The source is a release of https://github.com/LLNL/sundials (tarball or checkout of the
# tag). Files are copied unchanged. The only files in c/sundials/ that aren't upstream's are
# include/sundials/sundials_config.h and sundials_export.h, which CMake would generate; they
# are kept by hand (see their headers) and this script leaves them alone.
#
# The list below is CVODE with dense and band matrices, the dense direct linear solver and
# the Newton and fixed-point nonlinear solvers, plus every header those include. The
# profiler is left out: it's off (see sundials_config.h), every call to it is behind
# SUNDIALS_ENABLE_PROFILING, and its timers don't build in wasm. To upgrade,
# change VERSION, re-run, rebuild (`make test`) and add any header the compiler asks for.
set -euo pipefail

VERSION=7.9.0
src=${1:?usage: import-sundials.sh <sundials-source-dir>}
root=$(cd "$(dirname "$0")/.." && pwd)
dest="$root/c/sundials"

grep -q "set(PACKAGE_STRING \"SUNDIALS $VERSION\")" "$src/CMakeLists.txt" || {
    echo "$src is not SUNDIALS $VERSION" >&2
    exit 1
}

files=(
    include/cvode/cvode.h
    include/cvode/cvode_diag.h
    include/cvode/cvode_ls.h
    include/cvode/cvode_proj.h
    include/nvector/nvector_serial.h
    include/sundials/priv/sundials_context_impl.h
    include/sundials/priv/sundials_errors_impl.h
    include/sundials/priv/sundials_logger_macros.h
    include/sundials/sundials_adaptcontroller.h
    include/sundials/sundials_adjointcheckpointscheme.h
    include/sundials/sundials_adjointstepper.h
    include/sundials/sundials_band.h
    include/sundials/sundials_context.h
    include/sundials/sundials_core.h
    include/sundials/sundials_dense.h
    include/sundials/sundials_direct.h
    include/sundials/sundials_domeigestimator.h
    include/sundials/sundials_errors.h
    include/sundials/sundials_futils.h
    include/sundials/sundials_iterative.h
    include/sundials/sundials_linearsolver.h
    include/sundials/sundials_logger.h
    include/sundials/sundials_math.h
    include/sundials/sundials_matrix.h
    include/sundials/sundials_memory.h
    include/sundials/sundials_nonlinearsolver.h
    include/sundials/sundials_nvector.h
    include/sundials/sundials_nvector_senswrapper.h
    include/sundials/sundials_profiler.h
    include/sundials/sundials_stepper.h
    include/sundials/sundials_types.h
    include/sundials/sundials_version.h
    include/sunlinsol/sunlinsol_dense.h
    include/sunmatrix/sunmatrix_band.h
    include/sunmatrix/sunmatrix_dense.h
    include/sunmatrix/sunmatrix_sparse.h
    include/sunnonlinsol/sunnonlinsol_fixedpoint.h
    include/sunnonlinsol/sunnonlinsol_newton.h
    src/cvode/cvode.c
    src/cvode/cvode_diag_impl.h
    src/cvode/cvode_fused_stubs.c
    src/cvode/cvode_impl.h
    src/cvode/cvode_io.c
    src/cvode/cvode_ls.c
    src/cvode/cvode_ls_impl.h
    src/cvode/cvode_nls.c
    src/cvode/cvode_proj.c
    src/cvode/cvode_proj_impl.h
    src/nvector/serial/nvector_serial.c
    src/sundials/stl/sunstl_vector.h
    src/sundials/sundials_adiak_metadata.h
    src/sundials/sundials_band.c
    src/sundials/sundials_context.c
    src/sundials/sundials_debug.h
    src/sundials/sundials_dense.c
    src/sundials/sundials_errors.c
    src/sundials/sundials_futils.c
    src/sundials/sundials_hashmap_impl.h
    src/sundials/sundials_linearsolver.c
    src/sundials/sundials_logger.c
    src/sundials/sundials_logger_impl.h
    src/sundials/sundials_macros.h
    src/sundials/sundials_math.c
    src/sundials/sundials_matrix.c
    src/sundials/sundials_memory.c
    src/sundials/sundials_nonlinearsolver.c
    src/sundials/sundials_nvector.c
    src/sundials/sundials_nvector_senswrapper.c
    src/sundials/sundials_utils.h
    src/sundials/sundials_version.c
    src/sunlinsol/dense/sunlinsol_dense.c
    src/sunmatrix/band/sunmatrix_band.c
    src/sunmatrix/dense/sunmatrix_dense.c
    src/sunnonlinsol/fixedpoint/sunnonlinsol_fixedpoint.c
    src/sunnonlinsol/newton/sunnonlinsol_newton.c
)

# Start from a clean tree so files dropped from the list don't linger, but keep the two
# hand-maintained config headers.
keep=(include/sundials/sundials_config.h include/sundials/sundials_export.h)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in "${keep[@]}"; do
    [ -f "$dest/$f" ] && install -D -m 644 "$dest/$f" "$tmp/$f"
done
rm -rf "$dest"
for f in "${keep[@]}"; do
    [ -f "$tmp/$f" ] && install -D -m 644 "$tmp/$f" "$dest/$f"
done

for f in "${files[@]}"; do
    install -D -m 644 "$src/$f" "$dest/$f"
done
install -m 644 "$src/LICENSE" "$root/licences/SUNDIALS-LICENCE.txt"
install -m 644 "$src/NOTICE" "$root/licences/SUNDIALS-NOTICE.txt"
echo "imported ${#files[@]} files from SUNDIALS $VERSION into $dest"
