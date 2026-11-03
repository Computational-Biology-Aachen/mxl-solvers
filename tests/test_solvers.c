/* Reference problems for every solver in c/. Errors are relative, max-norm unless
 * stated. Build and run with `make test`. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "f2c.h"
#include "lsoda.h"

extern int radau5_(integer *n, U_fp fcn, doublereal *x, doublereal *y, doublereal *xend, doublereal *h,
                   doublereal *rtol, doublereal *atol, integer *itol, U_fp jac, integer *ijac, integer *mljac,
                   integer *mujac, U_fp mas, integer *imas, integer *mlmas, integer *mumas, U_fp solout,
                   integer *iout, doublereal *work, integer *lwork, integer *iwork, integer *liwork,
                   doublereal *rpar, integer *ipar, integer *idid);
extern int dopri5_(integer *n, U_fp fcn, doublereal *x, doublereal *y, doublereal *xend, doublereal *rtol,
                   doublereal *atol, integer *itol, U_fp solout, integer *iout, doublereal *work, integer *lwork,
                   integer *iwork, integer *liwork, doublereal *rpar, integer *ipar, integer *idid);
extern int dop853_(integer *n, U_fp fcn, doublereal *x, doublereal *y, doublereal *xend, doublereal *rtol,
                   doublereal *atol, integer *itol, U_fp solout, integer *iout, doublereal *work, integer *lwork,
                   integer *iwork, integer *liwork, doublereal *rpar, integer *ipar, integer *idid);

typedef void (*Rhs)(int n, double t, const double *y, double *dy);
static Rhs g_rhs;
static int failures;

static void robertson(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
    dy[2] = 3e7 * y[1] * y[1];
    dy[1] = -dy[0] - dy[2];
}
static void lotka_volterra(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = 1.5 * y[0] - y[0] * y[1];
    dy[1] = -3.0 * y[1] + y[0] * y[1];
}
static void van_der_pol(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = y[1];
    dy[1] = (1.0 - y[0] * y[0]) * y[1] - y[0];
}

/* Calling-convention adapters. */
static int f2c_fcn(integer *n, doublereal *x, doublereal *y, doublereal *f, doublereal *rpar, integer *ipar) {
    (void)rpar; (void)ipar;
    g_rhs((int)*n, *x, y, f);
    return 0;
}
static void lsoda_fcn(int *n, double *t, double *y, double *dy) { g_rhs(*n, *t, y, dy); }
static int no_op(void) { return 0; }

static int solve_radau5(Rhs f, int n, double t0, double t1, double *y, double rtol, double atol) {
    integer N = n, itol = 0, ijac = 0, ml = n, mu = n, imas = 0, iout = 0, idid = 0, ipar = 0;
    integer lwork = 4 * N * N + 12 * N + 20, liwork = 3 * N + 20;
    doublereal *work = calloc((size_t)lwork, sizeof(doublereal));
    integer *iwork = calloc((size_t)liwork, sizeof(integer));
    doublereal x = t0, xend = t1, h = 0.0, rpar = 0.0;
    g_rhs = f;
    radau5_(&N, (U_fp)f2c_fcn, &x, y, &xend, &h, &rtol, &atol, &itol, (U_fp)no_op, &ijac, &ml, &mu, (U_fp)no_op,
            &imas, &ml, &mu, (U_fp)no_op, &iout, work, &lwork, iwork, &liwork, &rpar, &ipar, &idid);
    free(work);
    free(iwork);
    return (int)idid;
}

static int solve_dopri(int which, Rhs f, int n, double t0, double t1, double *y, double rtol, double atol) {
    integer N = n, itol = 0, iout = 0, idid = 0, ipar = 0, lwork = 11 * N + 21, liwork = 21;
    doublereal *work = calloc((size_t)lwork, sizeof(doublereal));
    integer *iwork = calloc((size_t)liwork, sizeof(integer));
    doublereal x = t0, xend = t1, rpar = 0.0;
    g_rhs = f;
    (which == 5 ? dopri5_ : dop853_)(&N, (U_fp)f2c_fcn, &x, y, &xend, &rtol, &atol, &itol, (U_fp)no_op, &iout,
                                     work, &lwork, iwork, &liwork, &rpar, &ipar, &idid);
    free(work);
    free(iwork);
    return (int)idid;
}

/* jt = 2: internal full Jacobian; jt = 5: internal banded Jacobian with ml = mu = n - 1. */
static int solve_lsoda(Rhs f, int n, double t0, double t1, double *y, double rtol, double atol, int jt) {
    int ml = n - 1, mu = n - 1;
    int lmat = jt == 2 ? n * n + 2 : (2 * ml + mu + 1) * n + 2;
    int lrn = 20 + n * 13 + 3 * n, lrs = 20 + n * 6 + 3 * n + lmat;
    int lrw = lrn > lrs ? lrn : lrs, liw = 20 + n;
    double *rwork = calloc((size_t)lrw, sizeof(double));
    int *iwork = calloc((size_t)liw, sizeof(int));
    int itask = 1, istate = 1, iopt = 1;
    iwork[0] = ml;
    iwork[1] = mu;
    iwork[5] = 100000; /* mxstep */
    double t = t0, tout = t1;
    lsoda_common_struct_t S = {0};
    g_rhs = f;
    lsoda(lsoda_fcn, n, y, &t, &tout, 1, &rtol, &atol, &itask, &istate, &iopt, rwork, lrw, iwork, liw, NULL, jt, &S);
    free(rwork);
    free(iwork);
    return istate;
}

static double rel_err(const double *got, const double *want, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        double e = fabs(got[i] - want[i]) / fabs(want[i]);
        if (e > worst) worst = e;
    }
    return worst;
}

static void check(const char *name, int status_ok, double err, double limit) {
    int ok = status_ok && err <= limit;
    printf("%-4s %-34s relative error %.2e (limit %.0e)\n", ok ? "ok" : "FAIL", name, err, limit);
    if (!ok) failures++;
}

static double lv_invariant(const double *y) { return y[0] - 3.0 * log(y[0]) + y[1] - 1.5 * log(y[1]); }

int main(void) {
    /* Robertson to t = 40, against the reference values of Hairer and Wanner. */
    const double rob_ref[3] = {0.7158270687193941, 9.185534764557338e-6, 0.2841637457458413};
    double y[3];
    y[0] = 1; y[1] = 0; y[2] = 0;
    int s = solve_radau5(robertson, 3, 0, 40, y, 1e-10, 1e-14);
    check("robertson radau5", s > 0, rel_err(y, rob_ref, 3), 1e-6);
    for (int jt = 2; jt <= 5; jt += 3) {
        y[0] = 1; y[1] = 0; y[2] = 0;
        s = solve_lsoda(robertson, 3, 0, 40, y, 1e-10, 1e-14, jt);
        check(jt == 2 ? "robertson lsoda (full jacobian)" : "robertson lsoda (banded jacobian)", s == 2,
              rel_err(y, rob_ref, 3), 1e-6);
    }

    /* Lotka-Volterra over many periods: the first integral must stay constant. */
    const double lv0[2] = {1.0, 1.0};
    const double inv0 = lv_invariant(lv0);
    const char *lv_names[] = {"lotka-volterra lsoda", "lotka-volterra dopri5", "lotka-volterra dop853",
                              "lotka-volterra radau5"};
    for (int k = 0; k < 4; k++) {
        memcpy(y, lv0, sizeof lv0);
        if (k == 0) s = solve_lsoda(lotka_volterra, 2, 0, 50, y, 1e-10, 1e-12, 2) == 2;
        else if (k == 1) s = solve_dopri(5, lotka_volterra, 2, 0, 50, y, 1e-10, 1e-12) > 0;
        else if (k == 2) s = solve_dopri(853, lotka_volterra, 2, 0, 50, y, 1e-10, 1e-12) > 0;
        else s = solve_radau5(lotka_volterra, 2, 0, 50, y, 1e-10, 1e-12) > 0;
        check(lv_names[k], s, fabs(lv_invariant(y) - inv0) / fabs(inv0), 1e-7);
    }

    /* Van der Pol (mu = 1): every solver agrees with DOP853 at a tight tolerance. */
    double ref[2] = {2.0, 0.0};
    solve_dopri(853, van_der_pol, 2, 0, 20, ref, 1e-13, 1e-13);
    const char *vdp_names[] = {"van der pol lsoda", "van der pol dopri5", "van der pol radau5"};
    for (int k = 0; k < 3; k++) {
        y[0] = 2.0; y[1] = 0.0;
        if (k == 0) s = solve_lsoda(van_der_pol, 2, 0, 20, y, 1e-10, 1e-10, 2) == 2;
        else if (k == 1) s = solve_dopri(5, van_der_pol, 2, 0, 20, y, 1e-10, 1e-10) > 0;
        else s = solve_radau5(van_der_pol, 2, 0, 20, y, 1e-10, 1e-10) > 0;
        check(vdp_names[k], s, rel_err(y, ref, 2), 1e-6);
    }

    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
