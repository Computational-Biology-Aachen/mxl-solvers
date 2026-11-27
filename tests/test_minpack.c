/* cminpack's own tests against our c/minpack (lmdif, lmder), plus regression tests for bugs fixed
 * upstream:
 *   - The Moré–Garbow–Hillstrom suite: cminpack's lmddrv/lmfdrv loop over tests/cminpack/ssq.data,
 *     compared with the summary tables of its lmddrvc.ref / lmfdrvc.ref like cminpack's cmpfiles
 *     (1% relative), with convergence codes 1-4 treated as one class (see tests/cminpack/README.md).
 *   - tlmdifc / tlmderc: the 15-point rational fit from cminpack's examples.
 *   - tissue76: lmpar must not divide 0/0 when the step shrinks to zero (cminpack issue #76).
 *   - fdjac2 must restore x when the function aborts during the Jacobian (cminpack #89).
 * Run from the repository root (`make test`), which the data paths are relative to. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cminpack.h"
#include "cminpack/ssq.h"


static int failures;
static void check(const char *name, int ok, const char *fmt, double v) {
    printf("%-4s %-66s ", ok ? "ok" : "FAIL", name);
    printf(fmt, v);
    printf("\n");
    if (!ok) failures++;
}

/* ------------------------------------------------------------------ */
/* MGH suite                                                           */
/* ------------------------------------------------------------------ */
typedef struct {
    int nprob, nfev, njev;
} Count;

static int fcn_der(void *p, int m, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag) {
    Count *c = (Count *)p;
    if (iflag == 1) {
        ssqfcn(m, n, x, fvec, c->nprob);
        c->nfev++;
    }
    if (iflag == 2) {
        ssqjac(m, n, x, fjac, ldfjac, c->nprob);
        c->njev++;
    }
    return 0;
}
static int fcn_dif(void *p, int m, int n, const double *x, double *fvec, int iflag) {
    Count *c = (Count *)p;
    ssqfcn(m, n, x, fvec, c->nprob);
    if (iflag == 1) c->nfev++;
    if (iflag == 2) c->njev++;
    return 0;
}

typedef struct {
    int nprob, n, m, nfev, njev, info;
    double fnorm;
} Row;

/* The summary table at the end of a cminpack driver's output. */
static int read_summary(const char *path, Row *rows, int cap) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    int in_table = 0, k = 0;
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, "nprob")) {
            in_table = 1;
            continue;
        }
        Row r;
        if (in_table && k < cap &&
            sscanf(line, "%d %d %d %d %d %d %lf", &r.nprob, &r.n, &r.m, &r.nfev, &r.njev, &r.info, &r.fnorm) == 7)
            rows[k++] = r;
    }
    fclose(f);
    return k;
}

static int info_class(int info) { return info >= 1 && info <= 4 ? 1 : info; }
static int close_rel(double a, double b, double reltol, double abstol) {
    double d = fabs(a - b);
    return d <= abstol || d <= reltol * fmax(fabs(a), fabs(b));
}

static void mgh(int use_lmder) {
    Row ref[64];
    const char *ref_path = use_lmder ? "tests/cminpack/lmddrvc.ref" : "tests/cminpack/lmfdrvc.ref";
    int n_ref = read_summary(ref_path, ref, 64);
    FILE *data = fopen("tests/cminpack/ssq.data", "r");
    if (n_ref <= 0 || !data) {
        check("MGH: reference and data files readable (run from the repo root)", 0, "%g", 0);
        if (data) fclose(data);
        return;
    }
    const double tol = sqrt(dpmpar(1));
    int nprob, n, m, ntries, ic = 0, bad = 0;
    double x[40], fvec[65], fjac[65 * 40], wa[65 * 40 + 5 * 40 + 65]; /* lmdif1 needs m n + 5 n + m */
    int ipvt[40];
    while (fscanf(data, "%d %d %d %d", &nprob, &n, &m, &ntries) == 4 && nprob > 0) {
        double factor = 1.0;
        for (int k = 0; k < ntries; k++, ic++, factor *= 10) {
            lmdipt(n, x, nprob, factor);
            Count c = {nprob, 0, 0};
            int info = use_lmder ? lmder1(fcn_der, &c, m, n, x, fvec, fjac, m, tol, ipvt, wa, (int)(sizeof wa / sizeof wa[0]))
                                 : lmdif1(fcn_dif, &c, m, n, x, fvec, tol, ipvt, wa, (int)(sizeof wa / sizeof wa[0]));
            ssqfcn(m, n, x, fvec, nprob);
            double fnorm = enorm(m, fvec);
            const Row *r = ic < n_ref ? &ref[ic] : NULL;
            int ok = r && r->nprob == nprob && r->n == n && r->m == m && info_class(info) == info_class(r->info) &&
                     close_rel(fnorm, r->fnorm, 1e-2, 1e-10) && close_rel(c.nfev, r->nfev, 0.05, 1) &&
                     isfinite(fnorm);
            if (!ok) {
                bad++;
                printf("     problem %2d n=%2d m=%2d try %d: info %d nfev %d fnorm %.7e; reference info %d nfev %d "
                       "fnorm %.7e\n",
                       nprob, n, m, k, info, c.nfev, fnorm, r ? r->info : -1, r ? r->nfev : -1, r ? r->fnorm : NAN);
            }
        }
    }
    fclose(data);
    char name[128];
    snprintf(name, sizeof name, "MGH suite, %s: %d runs match cminpack's reference", use_lmder ? "lmder" : "lmdif",
             n_ref);
    check(name, bad == 0 && ic == n_ref, "mismatches %g", bad);
}

/* ------------------------------------------------------------------ */
/* tlmdifc / tlmderc                                                   */
/* ------------------------------------------------------------------ */
static const double Y15[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1,
                               3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34,   2.1,    4.39};
static int rational_f(void *p, int m, int n, const double *x, double *fvec, int iflag) {
    (void)p; (void)m; (void)n; (void)iflag;
    for (int i = 0; i < 15; i++) {
        double t1 = i + 1, t2 = 15 - i, t3 = t1 > 8 ? t2 : t1;
        fvec[i] = Y15[i] - (x[0] + t1 / (x[1] * t2 + x[2] * t3));
    }
    return 0;
}
static int rational_fj(void *p, int m, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag) {
    if (iflag != 2) return rational_f(p, m, n, x, fvec, iflag);
    for (int i = 0; i < 15; i++) {
        double t1 = i + 1, t2 = 15 - i, t3 = t1 > 8 ? t2 : t1, t4 = (x[1] * t2 + x[2] * t3);
        t4 = t4 * t4;
        fjac[i] = -1;
        fjac[i + ldfjac] = t1 * t2 / t4;
        fjac[i + 2 * ldfjac] = t1 * t3 / t4;
    }
    return 0;
}

static void rational_fit(void) {
    const double xref[3] = {0.08241058, 1.133037, 2.343695};
    double x[3] = {1, 1, 1}, fvec[15], diag[3], fjac[45], qtf[3], wa1[3], wa2[3], wa3[3], wa4[15];
    int ipvt[3], nfev = 0, njev = 0;
    const double tol = sqrt(dpmpar(1));
    int info = lmdif(rational_f, NULL, 15, 3, x, fvec, tol, tol, 0., 800, 0., diag, 1, 100., 0, &nfev, fjac, 15,
                     ipvt, qtf, wa1, wa2, wa3, wa4);
    double fn = enorm(15, fvec), dx = 0;
    for (int j = 0; j < 3; j++) dx = fmax(dx, fabs(x[j] - xref[j]) / xref[j]);
    check("tlmdifc: info 1, |f| = 0.09063596, 21 evaluations, x as in tlmdifc.ref",
          info == 1 && fabs(fn - 0.09063596) < 1e-8 && nfev == 21 && dx < 1e-6, "|f| %.8f", fn);
    x[0] = x[1] = x[2] = 1;
    info = lmder(rational_fj, NULL, 15, 3, x, fvec, fjac, 15, tol, tol, 0., 100, diag, 1, 100., 0, &nfev, &njev,
                 ipvt, qtf, wa1, wa2, wa3, wa4);
    fn = enorm(15, fvec);
    dx = 0;
    for (int j = 0; j < 3; j++) dx = fmax(dx, fabs(x[j] - xref[j]) / xref[j]);
    check("tlmderc: info 1, |f| = 0.09063596, 6 function + 5 Jacobian evaluations",
          info == 1 && fabs(fn - 0.09063596) < 1e-8 && nfev == 6 && njev == 5 && dx < 1e-6, "|f| %.8f", fn);
}

/* ------------------------------------------------------------------ */
/* tissue76: F(x) = (-x0, M x0 - x1, M x1 - x2, M x2 - x3), solution 0  */
/* ------------------------------------------------------------------ */
#define I76_N 4
#define I76_M (36. / 73.)
static int i76_f(void *p, int m, int n, const double *x, double *fvec, int iflag) {
    (void)p; (void)m; (void)iflag;
    fvec[0] = -x[0];
    for (int i = 0; i < n - 1; i++) fvec[i + 1] = I76_M * x[i] - x[i + 1];
    return 0;
}
static int i76_fj(void *p, int m, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag) {
    if (iflag != 2) return i76_f(p, m, n, x, fvec, iflag);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) fjac[i + j * ldfjac] = i == j ? -1. : i == j + 1 ? I76_M : 0.;
    return 0;
}

static void issue76(void) {
    for (int der = 0; der < 2; der++) {
        double x[I76_N] = {1, 0, 0, 0}, fvec[I76_N], fjac[I76_N * I76_N], diag[I76_N], qtf[I76_N], wa1[I76_N],
               wa2[I76_N], wa3[I76_N], wa4[I76_N];
        int ipvt[I76_N], nfev = 0, njev = 0, info;
        const double tol = sqrt(dpmpar(1));
        if (der) {
            info = lmder(i76_fj, NULL, I76_N, I76_N, x, fvec, fjac, I76_N, tol, tol, 0., 10000, diag, 1, 100., 0,
                         &nfev, &njev, ipvt, qtf, wa1, wa2, wa3, wa4);
        } else {
            info = lmdif(i76_f, NULL, I76_N, I76_N, x, fvec, tol, tol, 0., 10000, 0., diag, 1, 100., 0, &nfev, fjac,
                         I76_N, ipvt, qtf, wa1, wa2, wa3, wa4);
        }
        int finite = 1;
        for (int i = 0; i < I76_N; i++) finite = finite && isfinite(x[i]) && isfinite(fvec[i]);
        check(der ? "tissue76 (lmpar 0/0): lmder converges to 0 quickly"
                  : "tissue76 (lmpar 0/0): lmdif converges to 0 quickly",
              info >= 1 && info <= 4 && finite && nfev < 5000, "nfev %g", nfev);
    }
}

/* cminpack #89: when the function aborts (iflag < 0) during fdjac2's finite differences, lmdif
 * must return with x as it was, not with one component still perturbed. */
static int abort_calls;
static int abort_in_jacobian(void *p, int m, int n, const double *x, double *fvec, int iflag) {
    (void)p;
    for (int i = 0; i < m; i++) fvec[i] = x[i % n] - 1.0 - i;
    if (iflag == 2 && ++abort_calls == 2) return -7;
    return 0;
}

static void fdjac2_restores_x(void) {
    double x[2] = {3, 4}, fvec[3], fjac[6], diag[2], qtf[2], wa1[2], wa2[2], wa3[2], wa4[3];
    int ipvt[2], nfev = 0;
    abort_calls = 0;
    int info = lmdif(abort_in_jacobian, NULL, 3, 2, x, fvec, 1e-10, 1e-10, 0., 200, 0., diag, 1, 100., 0, &nfev,
                     fjac, 3, ipvt, qtf, wa1, wa2, wa3, wa4);
    check("fdjac2 abort: info is the abort code and x is restored (cminpack #89)",
          info == -7 && x[0] == 3 && x[1] == 4, "x[1] %.17g", x[1]);
}

int main(void) {
    mgh(1);
    mgh(0);
    rational_fit();
    issue76();
    fdjac2_restores_x();
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
