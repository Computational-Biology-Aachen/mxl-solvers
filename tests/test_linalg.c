/* The dense LU routines under the solvers: Hairer's DEC / SOL (decsol.c, inside RADAU5) and the
 * LAPACK subset for LSODA (mxs_lapack.c: dgetrf/dgetrs, dgbtrf/dgbtrs). Cases from
 * lineax (test_well_posed, test_singular, test_misc::test_zero_matrix), LinearSolve.jl
 * (retcodes, lu_refactorization) and SUNDIALS (test_sunlinsol_dense: an anti-identity that
 * forces row pivoting). Errors are relative, max-norm. Build and run with `make test`. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blaslapack_declarations.h"
#include "f2c.h"

extern int dec_(integer *n, integer *ndim, doublereal *a, integer *ip, integer *ier);
extern int sol_(integer *n, integer *ndim, doublereal *a, doublereal *b, integer *ip);

static int failures;
static void check(const char *name, int ok, const char *fmt, double v) {
    printf("%-4s %-62s ", ok ? "ok" : "FAIL", name);
    printf(fmt, v);
    printf("\n");
    if (!ok) failures++;
}

/* Deterministic uniform numbers in [0, 1). */
static unsigned long long rng = 88172645463325252ULL;
static double uniform(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (double)(rng >> 11) / 9007199254740992.0;
}

/* b = A x for column-major A (lda = n). */
static void matvec(int n, const double *A, const double *x, double *b) {
    for (int i = 0; i < n; i++) {
        b[i] = 0;
        for (int j = 0; j < n; j++) b[i] += A[i + j * n] * x[j];
    }
}

static double rel_err(const double *got, const double *want, int n) {
    double worst = 0, scale = 0;
    for (int i = 0; i < n; i++) scale = fmax(scale, fabs(want[i]));
    for (int i = 0; i < n; i++) {
        double e = fabs(got[i] - want[i]) / scale;
        if (!(e <= worst)) worst = e;
    }
    return worst;
}

/* Solves A x = b with DEC/SOL (which == 0) or dgetrf/dgetrs (which == 1). Returns the
 * factorisation's status (ier / info): 0, or the stage at which A turned out singular. */
static int lu_solve(int which, int n, const double *A, double *b) {
    double *a = malloc((size_t)(n > 0 ? n * n : 1) * sizeof(double));
    int *ip = malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    memcpy(a, A, (size_t)n * (size_t)n * sizeof(double));
    int status;
    if (which == 0) {
        integer N = n, ier = 0;
        integer *ipi = malloc((size_t)(n > 0 ? n : 1) * sizeof(integer));
        dec_(&N, &N, a, ipi, &ier);
        if (ier == 0 && n > 0) sol_(&N, &N, a, b, ipi);
        status = (int)ier;
        free(ipi);
    } else {
        int info = 0, one = 1, lda = n > 0 ? n : 1;
        char t = 'N';
        mxs_dgetrf(&n, &n, a, &lda, ip, &info);
        if (info == 0 && n > 0) mxs_dgetrs(&t, &n, &one, a, &lda, ip, b, &lda, &info);
        status = info;
    }
    free(a);
    free(ip);
    return status;
}

static const char *NAMES[2] = {"DEC/SOL", "dgetrf/dgetrs"};

/* lineax test_well_posed: random well-conditioned systems (diagonally dominant). */
static void well_posed(int which) {
    const int sizes[4] = {1, 3, 10, 50};
    double worst = 0;
    int ok = 1;
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        double *A = malloc((size_t)n * n * sizeof(double)), *x = malloc(n * sizeof(double)),
               *b = malloc(n * sizeof(double));
        for (int i = 0; i < n * n; i++) A[i] = uniform() - 0.5;
        for (int i = 0; i < n; i++) {
            A[i + i * n] += n;
            x[i] = uniform() - 0.5;
        }
        matvec(n, A, x, b);
        ok = ok && lu_solve(which, n, A, b) == 0;
        worst = fmax(worst, rel_err(b, x, n));
        free(A);
        free(x);
        free(b);
    }
    char name[128];
    snprintf(name, sizeof name, "%s: well-conditioned n = 1, 3, 10, 50", NAMES[which]);
    check(name, ok && worst < 1e-12, "rel err %.1e", worst);
}

/* SUNDIALS test_sunlinsol_dense: small random entries plus an anti-identity, so every column
 * needs a row swap. */
static void pivoting(int which) {
    const int n = 100;
    double *A = calloc((size_t)n * n, sizeof(double)), *x = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double));
    for (int i = 0; i < n * n; i++) A[i] = uniform() / n;
    for (int i = 0; i < n; i++) {
        A[i + (n - 1 - i) * n] += 1.0;
        x[i] = uniform();
    }
    matvec(n, A, x, b);
    int status = lu_solve(which, n, A, b);
    char name[128];
    snprintf(name, sizeof name, "%s: anti-identity n = 100 (row pivoting)", NAMES[which]);
    check(name, status == 0 && rel_err(b, x, n) < 1e-10, "rel err %.1e", rel_err(b, x, n));
    free(A);
    free(x);
    free(b);
}

/* LinearSolve retcodes, lineax test_singular / test_zero_matrix: singular matrices are reported,
 * without a crash, and an empty system is a no-op. */
static void singular(int which) {
    const double ones[4] = {1, 1, 1, 1}, zero[4] = {0, 0, 0, 0}, zero_row[9] = {1, 0, 2, 3, 0, 4, 5, 0, 6};
    double b[3] = {1, 2, 3};
    char name[128];
    snprintf(name, sizeof name, "%s: [1 1; 1 1] is singular", NAMES[which]);
    check(name, lu_solve(which, 2, ones, b) != 0, "status %g", 0);
    snprintf(name, sizeof name, "%s: zero matrix is singular", NAMES[which]);
    check(name, lu_solve(which, 2, zero, b) != 0, "status %g", 0);
    snprintf(name, sizeof name, "%s: a zero row is singular", NAMES[which]);
    check(name, lu_solve(which, 3, zero_row, b) != 0, "status %g", 0);
    snprintf(name, sizeof name, "%s: n = 0 is a no-op", NAMES[which]);
    check(name, lu_solve(which, 0, zero, b) == 0, "status %g", 0);
}

/* LinearSolve lu_refactorization: factor a good matrix, then a singular one (column 1 zeroed),
 * then the good one again; the last solve must be right (RADAU5 refactors in place after a
 * singular matrix, with a smaller step). */
static void refactorization(int which) {
    const int n = 4;
    double A[16], S[16], x[4] = {1, -2, 3, -4}, b[4];
    for (int i = 0; i < 16; i++) A[i] = uniform() - 0.5;
    for (int i = 0; i < n; i++) A[i + i * n] += n;
    memcpy(S, A, sizeof A);
    for (int i = 0; i < n; i++) S[i] = 0.0;
    matvec(n, A, x, b);
    int s1 = lu_solve(which, n, A, b);
    double c[4] = {1, 1, 1, 1};
    int s2 = lu_solve(which, n, S, c);
    matvec(n, A, x, b);
    int s3 = lu_solve(which, n, A, b);
    char name[128];
    snprintf(name, sizeof name, "%s: good, singular, good again", NAMES[which]);
    check(name, s1 == 0 && s2 != 0 && s3 == 0 && rel_err(b, x, n) < 1e-12, "rel err %.1e", rel_err(b, x, n));
}

/* dgbtrf/dgbtrs (LSODA's banded Jacobians) agree with the dense factorisation on a band matrix
 * with kl = 2, ku = 1, in LAPACK band storage (ldab = 2 kl + ku + 1). */
static void banded(void) {
    const int n = 8, kl = 2, ku = 1, ldab = 2 * kl + ku + 1;
    double A[64] = {0}, AB[6 * 8] = {0}, x[8], b[8], bb[8];
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (i - j <= kl && j - i <= ku) A[i + j * n] = uniform() - 0.5 + (i == j ? 0.1 : 0.0);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            if (i - j <= kl && j - i <= ku) AB[(kl + ku + i - j) + j * ldab] = A[i + j * n];
    for (int i = 0; i < n; i++) x[i] = uniform() - 0.5;
    matvec(n, A, x, b);
    memcpy(bb, b, sizeof b);
    int ipiv[8], info = 0, nn = n, KL = kl, KU = ku, L = ldab, one = 1;
    char t = 'N';
    mxs_dgbtrf(&nn, &nn, &KL, &KU, AB, &L, ipiv, &info);
    int info2 = 0;
    if (info == 0) mxs_dgbtrs(&t, &nn, &KL, &KU, &one, AB, &L, ipiv, bb, &nn, &info2);
    check("dgbtrf/dgbtrs: band matrix kl=2, ku=1 solved", info == 0 && info2 == 0 && rel_err(bb, x, n) < 1e-10,
          "rel err %.1e", rel_err(bb, x, n));
    memcpy(bb, b, sizeof b);
    lu_solve(1, n, A, bb);
    check("dgbtrf/dgbtrs agrees with dgetrf/dgetrs", rel_err(bb, x, n) < 1e-10, "rel err %.1e", rel_err(bb, x, n));
}

int main(void) {
    for (int which = 0; which < 2; which++) {
        well_posed(which);
        pivoting(which);
        singular(which);
        refactorization(which);
    }
    banded();
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
