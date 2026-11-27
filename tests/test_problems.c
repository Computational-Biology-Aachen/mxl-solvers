/* Reference problems and solver properties, taken from the test suites of the codes' own
 * authors and of other ODE packages:
 *   - Hairer's example drivers (fortran/dr_dopri5.f, dr_dop853.f), built with gfortran: our
 *     f2c translations must reproduce their results and statistics exactly.
 *   - The Bari/CWI test set for IVP solvers (Mazzia & Magherini): HIRES, ROBER, OREGO, E5 and
 *     VDPOL with their published reference solutions.
 *   - SUNDIALS v7.9.0 test/unit_tests/cvode/CXX_serial/cv_test_kpr (golden statistics).
 *   - scipy.integrate (Jacobian orientation gh-24933, banded LSODA, stiffness switching),
 *     ODEInterface.jl (banded and time-dependent Jacobians), OrdinaryDiffEq.jl (FIRK
 *     Jacobian reuse, polynomial exactness, convergence order), GlobalDiffEq.jl (pure
 *     relative tolerance).
 * Errors are relative, max-norm over components, unless stated. "scd" is the number of
 * significant correct digits, -log10(relative error). Build and run with `make test`. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "f2c.h"
#include "lsoda.h"

#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

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
/* Dense Jacobian, column-major: J[i + j*n] = d f_i / d y_j. */
typedef void (*Jac)(int n, double t, const double *y, double *J);

static Rhs g_rhs;
static Jac g_jac;
static long g_jac_calls;
static int failures;

static void check(const char *name, int ok, const char *fmt, double v) {
    printf("%-4s %-60s ", ok ? "ok" : "FAIL", name);
    printf(fmt, v);
    printf("\n");
    if (!ok) failures++;
}

static double rel_err(const double *got, const double *want, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        double e = want[i] == 0 ? fabs(got[i]) : fabs(got[i] - want[i]) / fabs(want[i]);
        if (!(e <= worst)) worst = e;
    }
    return worst;
}
/* |got - want| / (|want| + floor): relative, except for components decayed below floor. */
static double mixed_err(const double *got, const double *want, int n, double floor) {
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        double e = fabs(got[i] - want[i]) / (fabs(want[i]) + floor);
        if (!(e <= worst)) worst = e;
    }
    return worst;
}
/* Golden statistics compare like SUNDIALS' testRunner: within 10%. Exact on x86-64 Linux, but
 * FMA contraction and a different libm (e.g. clang on arm64 macOS) change the rounding, and
 * with it the step sequence. */
static int stat_close(long got, long want) { return labs(got - want) <= want / 10; }

static double scd(const double *got, const double *want, int n) { return -log10(rel_err(got, want, n)); }

/* ------------------------------------------------------------------ */
/* Solvers (raw interfaces), with statistics                           */
/* ------------------------------------------------------------------ */
typedef struct {
    double rtol, atol, h0, hmax; /* h0 = 0: the solver's guess; hmax = 0: no limit */
    int user_jac;                /* g_jac instead of finite differences */
    int nmax;
} Opts;

typedef struct {
    int status;
    long nfev, njac, nstep, naccept, nreject;
} Stats;

static int f2c_fcn(integer *n, doublereal *x, doublereal *y, doublereal *f, doublereal *rpar, integer *ipar) {
    (void)rpar; (void)ipar;
    g_rhs((int)*n, *x, y, f);
    return 0;
}
static int f2c_jac(integer *n, doublereal *x, doublereal *y, doublereal *dfy, integer *ldfy, doublereal *rpar,
                   integer *ipar) {
    (void)ldfy; (void)rpar; (void)ipar;
    g_jac_calls++;
    g_jac((int)*n, *x, y, dfy);
    return 0;
}
static int no_op(void) { return 0; }
static int solout_continue(integer *nr, doublereal *xold, doublereal *x, doublereal *y, integer *n,
                           doublereal *con, integer *icomp, integer *nd, doublereal *rpar, integer *ipar,
                           integer *irtrn) {
    (void)nr; (void)xold; (void)x; (void)y; (void)n; (void)con; (void)icomp; (void)nd; (void)rpar; (void)ipar;
    *irtrn = 1;
    return 0;
}

/* DOP853 with IOUT = 3 computes dense output only when a step passes XOUT, which SOLOUT
 * advances; like dr_dop853.f, ask for it every 0.1. */
static int solout_dop853(integer *nr, doublereal *xold, doublereal *x, doublereal *y, integer *n,
                         doublereal *con, integer *icomp, integer *nd, doublereal *rpar, integer *ipar,
                         integer *irtrn, doublereal *xout) {
    (void)xold; (void)y; (void)n; (void)con; (void)icomp; (void)nd; (void)rpar; (void)ipar;
    if (*nr == 1) {
        *xout = 0.1;
    } else {
        while (*x >= *xout) *xout += 0.1;
    }
    *irtrn = 1;
    return 0;
}

static Stats radau5(Rhs f, int n, double t0, double t1, double *y, Opts o) {
    integer N = n, itol = 0, ijac = o.user_jac, ml = n, mu = n, imas = 0, iout = 0, idid = 0, ipar = 0;
    integer lwork = 4 * N * N + 12 * N + 20, liwork = 3 * N + 20;
    doublereal *work = calloc((size_t)lwork, sizeof(doublereal));
    integer *iwork = calloc((size_t)liwork, sizeof(integer));
    doublereal x = t0, xend = t1, h = o.h0, rpar = 0.0;
    iwork[1] = o.nmax;
    work[6] = o.hmax;
    g_rhs = f;
    radau5_(&N, (U_fp)f2c_fcn, &x, y, &xend, &h, &o.rtol, &o.atol, &itol, (U_fp)f2c_jac, &ijac, &ml, &mu,
            (U_fp)no_op, &imas, &ml, &mu, (U_fp)no_op, &iout, work, &lwork, iwork, &liwork, &rpar, &ipar, &idid);
    Stats s = {(int)idid, iwork[13], iwork[14], iwork[15], iwork[16], iwork[17]};
    free(work);
    free(iwork);
    return s;
}

/* order 5: DOPRI5, 8: DOP853. iout > 0 asks for dense output of every component, which costs
 * DOP853 extra RHS calls. iwork4: Hairer's stiffness test IWORK(4) (0: every 1000 steps, -1:
 * off). */
static Stats dopri(int order, Rhs f, int n, double t0, double t1, double *y, Opts o, int iout, int iwork4) {
    integer N = n, itol = 0, io = iout, idid = 0, ipar = 0, nrd = iout ? n : 0;
    integer lwork = order == 8 ? 11 * N + 8 * nrd + 21 : 8 * N + 5 * nrd + 21, liwork = nrd + 21;
    doublereal *work = calloc((size_t)lwork, sizeof(doublereal));
    integer *iwork = calloc((size_t)liwork, sizeof(integer));
    doublereal x = t0, xend = t1, rpar = 0.0;
    iwork[0] = o.nmax;
    iwork[3] = iwork4;
    iwork[4] = nrd;
    work[5] = o.hmax;
    work[6] = o.h0;
    g_rhs = f;
    (order == 5 ? dopri5_ : dop853_)(&N, (U_fp)f2c_fcn, &x, y, &xend, &o.rtol, &o.atol, &itol,
                                     order == 8 ? (U_fp)solout_dop853 : (U_fp)solout_continue, &io, work, &lwork, iwork, &liwork, &rpar, &ipar,
                                     &idid);
    Stats s = {(int)idid, iwork[16], 0, iwork[17], iwork[18], iwork[19]};
    free(work);
    free(iwork);
    return s;
}

static void lsoda_fcn(int *n, double *t, double *y, double *dy) { g_rhs(*n, *t, y, dy); }
/* LSODA's Jacobian: full (column-major, nrowpd = n) or banded, PD(i-j+mu, j) 0-based. */
static int g_ml, g_mu;
static void lsoda_jac(int *n, double *t, double *y, int *ml, int *mu, double *pd, int *nrowpd) {
    (void)ml; (void)mu;
    g_jac_calls++;
    double *J = calloc((size_t)(*n) * (size_t)(*n), sizeof(double));
    g_jac(*n, *t, y, J);
    if (*nrowpd == *n && g_ml < 0) {
        memcpy(pd, J, (size_t)(*n) * (size_t)(*n) * sizeof(double));
    } else {
        for (int j = 0; j < *n; j++)
            for (int i = 0; i < *n; i++)
                if (i - j <= g_ml && j - i <= g_mu) pd[(i - j + g_mu) + j * *nrowpd] = J[i + j * *n];
    }
    free(J);
}

typedef struct {
    Stats s;
    int first_method, last_method; /* IWORK(19) MUSED after the first and the last step: 1 Adams, 2 BDF */
    int method_switches;
} LsodaRun;

/* jt: 1 user full, 2 internal full, 4 user banded, 5 internal banded (bands ml, mu). Steps
 * one at a time (ITASK = 5, TCRIT = t1) to track the method used. */
static LsodaRun lsoda_run(Rhs f, int n, double t0, double t1, double *y, Opts o, int jt, int ml, int mu) {
    int banded = jt == 4 || jt == 5;
    int lmat = banded ? (2 * ml + mu + 1) * n + 2 : n * n + 2;
    int lrn = 20 + n * 13 + 3 * n, lrs = 20 + n * 6 + 3 * n + lmat;
    int lrw = lrn > lrs ? lrn : lrs, liw = 20 + n;
    double *rwork = calloc((size_t)lrw, sizeof(double));
    int *iwork = calloc((size_t)liw, sizeof(int));
    int itask = 5, istate = 1, iopt = 1;
    iwork[0] = ml;
    iwork[1] = mu;
    iwork[5] = o.nmax ? o.nmax : 100000;
    rwork[0] = t1;
    rwork[4] = o.h0;
    rwork[5] = o.hmax;
    g_ml = banded ? ml : -1;
    g_mu = mu;
    double t = t0, tout = t1;
    lsoda_common_struct_t S = {0};
    g_rhs = f;
    LsodaRun r = {{0}, 0, 0, 0};
    int prev = 0;
    while ((t1 - t) * (t1 >= t0 ? 1 : -1) > 0 && istate > 0) {
        lsoda(lsoda_fcn, n, y, &t, &tout, 1, &o.rtol, &o.atol, &itask, &istate, &iopt, rwork, lrw, iwork, liw,
              (jt == 1 || jt == 4) ? lsoda_jac : NULL, jt, &S);
        if (istate < 0) break;
        int m = iwork[18];
        if (!r.first_method) r.first_method = m;
        if (prev && m != prev) r.method_switches++;
        prev = m;
        r.last_method = m;
    }
    r.s = (Stats){istate, iwork[11], iwork[12], iwork[10], iwork[10], 0};
    free(rwork);
    free(iwork);
    return r;
}

static int cvode_rhs(sunrealtype t, N_Vector y, N_Vector dy, void *user_data) {
    (void)user_data;
    g_rhs((int)N_VGetLength(y), t, N_VGetArrayPointer(y), N_VGetArrayPointer(dy));
    return 0;
}
static int cvode_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector t1,
                     N_Vector t2, N_Vector t3) {
    (void)fy; (void)user_data; (void)t1; (void)t2; (void)t3;
    g_jac_calls++;
    g_jac((int)N_VGetLength(y), t, N_VGetArrayPointer(y), SUNDenseMatrix_Data(J));
    return 0;
}

/* BDF with Newton (dense, FD or user Jacobian), or Adams with fixed-point iteration. Integrates
 * to each of the n_out output times in turn (CV_NORMAL), like SUNDIALS' own examples. */
static Stats cvode_run(Rhs f, int n, double t0, const double *t_out, int n_out, double *y, Opts o, int adams) {
    SUNContext ctx;
    Stats s = {-99, 0, 0, 0, 0, 0};
    if (SUNContext_Create(SUN_COMM_NULL, &ctx) != 0) return s;
    N_Vector v = N_VMake_Serial(n, y, ctx);
    void *mem = CVodeCreate(adams ? CV_ADAMS : CV_BDF, ctx);
    SUNMatrix A = NULL;
    SUNLinearSolver ls = NULL;
    SUNNonlinearSolver nls = NULL;
    g_rhs = f;
    int flag = CVodeInit(mem, cvode_rhs, t0, v);
    if (flag == CV_SUCCESS) flag = CVodeSStolerances(mem, o.rtol, o.atol);
    if (flag == CV_SUCCESS) flag = CVodeSetMaxNumSteps(mem, o.nmax ? o.nmax : 100000);
    if (flag == CV_SUCCESS && o.h0 != 0) flag = CVodeSetInitStep(mem, o.h0);
    if (flag == CV_SUCCESS && adams) {
        nls = SUNNonlinSol_FixedPoint(v, 0, ctx);
        flag = CVodeSetNonlinearSolver(mem, nls);
    } else if (flag == CV_SUCCESS) {
        A = SUNDenseMatrix(n, n, ctx);
        ls = SUNLinSol_Dense(v, A, ctx);
        flag = CVodeSetLinearSolver(mem, ls, A);
        if (flag == CV_SUCCESS && o.user_jac) flag = CVodeSetJacFn(mem, cvode_jac);
    }
    double t = t0;
    for (int k = 0; flag == CV_SUCCESS && k < n_out; k++) flag = CVode(mem, t_out[k], v, &t, CV_NORMAL);
    long nst = 0, nfe = 0, nje = 0, netf = 0;
    CVodeGetNumSteps(mem, &nst);
    CVodeGetNumRhsEvals(mem, &nfe);
    CVodeGetNumErrTestFails(mem, &netf);
    if (!adams) CVodeGetNumJacEvals(mem, &nje);
    s = (Stats){flag, nfe, nje, nst, nst, netf};
    CVodeFree(&mem);
    SUNNonlinSolFree(nls);
    SUNLinSolFree(ls);
    SUNMatDestroy(A);
    N_VDestroy(v);
    SUNContext_Free(&ctx);
    return s;
}
static Stats cvode(Rhs f, int n, double t0, double t1, double *y, Opts o, int adams) {
    return cvode_run(f, n, t0, &t1, 1, y, o, adams);
}

/* ------------------------------------------------------------------ */
/* Problems                                                            */
/* ------------------------------------------------------------------ */
static double g_mu_vdp = 1.0; /* Van der Pol: y2' = mu ((1 - y1^2) y2 - y1) */
static void vdpol(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = y[1];
    dy[1] = g_mu_vdp * ((1 - y[0] * y[0]) * y[1] - y[0]);
}
/* Hairer's dr_dop853.f form: y2' = ((1 - y1^2) y2 - y1) / eps */
static double g_eps_vdp = 1e-3;
static void vdpol_eps(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = y[1];
    dy[1] = ((1 - y[0] * y[0]) * y[1] - y[0]) / g_eps_vdp;
}
static void arenstorf(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    const double amu = 0.012277471, amup = 1 - amu;
    dy[0] = y[2];
    dy[1] = y[3];
    double r1 = (y[0] + amu) * (y[0] + amu) + y[1] * y[1];
    r1 = r1 * sqrt(r1);
    double r2 = (y[0] - amup) * (y[0] - amup) + y[1] * y[1];
    r2 = r2 * sqrt(r2);
    dy[2] = y[0] + 2 * y[3] - amup * (y[0] + amu) / r1 - amu * (y[0] - amup) / r2;
    dy[3] = y[1] - 2 * y[2] - amup * y[1] / r1 - amu * y[1] / r2;
}
static void robertson(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
    dy[2] = 3e7 * y[1] * y[1];
    dy[1] = -dy[0] - dy[2];
}
static void robertson_jac(int n, double t, const double *y, double *J) {
    (void)n; (void)t;
    /* column-major */
    J[0] = -0.04;  J[3] = 1e4 * y[2];                       J[6] = 1e4 * y[1];
    J[1] = 0.04;   J[4] = -1e4 * y[2] - 6e7 * y[1];         J[7] = -1e4 * y[1];
    J[2] = 0.0;    J[5] = 6e7 * y[1];                       J[8] = 0.0;
}
static void robertson_jac_transposed(int n, double t, const double *y, double *J) {
    double T[9];
    robertson_jac(n, t, y, T);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) J[i + 3 * j] = T[j + 3 * i];
}
static void hires(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
    dy[1] = 1.71 * y[0] - 8.75 * y[1];
    dy[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
    dy[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
    dy[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
    dy[5] = -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
    dy[6] = 280.0 * y[5] * y[7] - 1.81 * y[6];
    dy[7] = -dy[6];
}
static void orego(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    dy[0] = 77.27 * (y[1] + y[0] * (1 - 8.375e-6 * y[0] - y[1]));
    dy[1] = (y[2] - (1 + y[0]) * y[1]) / 77.27;
    dy[2] = 0.161 * (y[0] - y[2]);
}
static void e5(int n, double t, const double *y, double *dy) {
    (void)n; (void)t;
    const double A = 7.89e-10, B = 1.1e7, C = 1.13e3, M = 1e6;
    dy[0] = -A * y[0] - B * y[0] * y[2];
    dy[1] = A * y[0] - M * C * y[1] * y[2];
    dy[3] = B * y[0] * y[2] - C * y[3];
    dy[2] = dy[1] - dy[3];
}
/* SUNDIALS KPR (problems/kpr.hpp): a, b, c, d = -2, 0.5, 0.5, -1. */
static void kpr(int n, double t, const double *y, double *dy) {
    (void)n;
    const double a = -2, b = 0.5, c = 0.5, d = -1;
    const double rho = 0.5 * cos(t), rho_p = -0.5 * sin(t), sig = cos(20 * t), sig_p = -20 * sin(20 * t);
    const double u = y[0], v = y[1];
    const double tmp1 = (-1 + u * u - rho) / (2 * u), tmp2 = (-2 + v * v - sig) / (2 * v);
    dy[0] = a * tmp1 + b * tmp2 + rho_p / (2 * u);
    dy[1] = c * tmp1 + d * tmp2 + sig_p / (2 * v);
}
static void kpr_jac(int n, double t, const double *y, double *J) {
    (void)n;
    const double a = -2, b = 0.5, c = 0.5, d = -1;
    const double rho = 0.5 * cos(t), rho_p = -0.5 * sin(t), sig = cos(20 * t), sig_p = -20 * sin(20 * t);
    const double u = y[0], v = y[1];
    J[0] = a / 2 + (a * (1 + rho) - rho_p) / (2 * u * u);
    J[1] = c / 2 + c * (1 + rho) / (2 * u * u);
    J[2] = b / 2 + b * (2 + sig) / (2 * v * v);
    J[3] = d / 2 + (d * (2 + sig) - sig_p) / (2 * v * v);
}
/* ODEInterface.jl test_jacode2: x1' = x1, xk' = x(k-1) + xk; lower bidiagonal, non-symmetric.
 * Exact from x(0) = (1, 0, 0, 0): xk = e^t t^(k-1) / (k-1)!. */
static void chain(int n, double t, const double *y, double *dy) {
    (void)t;
    dy[0] = y[0];
    for (int k = 1; k < n; k++) dy[k] = y[k - 1] + y[k];
}
static void chain_jac(int n, double t, const double *y, double *J) {
    (void)t; (void)y;
    memset(J, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) {
        J[k + k * n] = 1.0;
        if (k > 0) J[k + (k - 1) * n] = 1.0;
    }
}
/* A stiff version of the chain, so the implicit methods use its Jacobian:
 * x1' = -1000 x1, xk' = x(k-1) - lambda_k xk with lambda = 1000, 100, 10, 1. */
static const double LAMBDA[4] = {1000, 100, 10, 1};
static void stiff_chain(int n, double t, const double *y, double *dy) {
    (void)t;
    dy[0] = -LAMBDA[0] * y[0];
    for (int k = 1; k < n; k++) dy[k] = y[k - 1] - LAMBDA[k] * y[k];
}
static void stiff_chain_jac(int n, double t, const double *y, double *J) {
    (void)t; (void)y;
    memset(J, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) {
        J[k + k * n] = -LAMBDA[k];
        if (k > 0) J[k + (k - 1) * n] = 1.0;
    }
}
/* ODEInterface.jl test_rhstimederiv1: x' = (t x2, 4 t x1), exact x1 = cosh t^2 + sinh(t^2)/2,
 * x2 = cosh t^2 + 2 sinh t^2. The Jacobian depends on t. */
static void tdep(int n, double t, const double *y, double *dy) {
    (void)n;
    dy[0] = t * y[1];
    dy[1] = 4 * t * y[0];
}
static void tdep_jac(int n, double t, const double *y, double *J) {
    (void)n; (void)y;
    J[0] = 0;     J[2] = t;
    J[1] = 4 * t; J[3] = 0;
}
static void rotation(int n, double t, const double *y, double *dy) { /* y' = [[-0.1, -1], [1, -0.1]] y */
    (void)n; (void)t;
    dy[0] = -0.1 * y[0] - y[1];
    dy[1] = y[0] - 0.1 * y[1];
}
static void minus_t(int n, double t, const double *y, double *dy) { /* u' = -t: polynomial solution */
    (void)n; (void)y;
    dy[0] = -t;
}
static void linear_growth(int n, double t, const double *y, double *dy) { /* u' = 1.01 u */
    (void)t;
    for (int i = 0; i < n; i++) dy[i] = 1.01 * y[i];
}
/* GlobalDiffEq.jl: forced pendulum, theta crosses zero repeatedly. */
static void pendulum(int n, double t, const double *y, double *dy) {
    (void)n;
    dy[0] = y[1];
    dy[1] = -1.5 * 9.81 * sin(y[0]) + 3 * 0.1 * sin(t);
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

/* Golden values from Hairer's drivers, compiled from fortran/ with gfortran (-O0
 * -ffp-contract=off): the C translations take the same steps. Identical on x86-64 Linux; the
 * limits leave room for FMA contraction and other libms. */
static void hairer_drivers(void) {
    double y[4] = {0.994, 0.0, 0.0, -2.00158510637908252240537862224};
    Opts o = {1e-7, 1e-7, 0, 0, 0, 0};
    Stats s = dopri(5, arenstorf, 4, 0.0, 17.0652165601579625588917206249, y, o, 2, 0);
    const double ref5[2] = {0.9940021016, 0.8911185472e-05};
    check("dr_dopri5 (Arenstorf): y(XEND) as printed by the Fortran", s.status == 1 && rel_err(y, ref5, 2) < 1e-6,
          "rel err %.1e", rel_err(y, ref5, 2));
    check("dr_dopri5: fcn=1442 step=240 accpt=216 rejct=22 (+-10%)",
          stat_close(s.nfev, 1442) && stat_close(s.nstep, 240) && stat_close(s.naccept, 216) &&
              labs(s.nreject - 22) <= 5,
          "fcn %g", s.nfev);
    const double y0[2] = {0.994, 0.0};
    check("dr_dopri5: the orbit closes after one period", rel_err(y, y0, 1) < 1e-5, "rel err %.1e",
          rel_err(y, y0, 1));

    double z[2] = {2.0, 0.0};
    g_eps_vdp = 1e-3;
    Opts o8 = {1e-9, 1e-9, 0, 0, 0, 0};
    s = dopri(8, vdpol_eps, 2, 0.0, 2.0, z, o8, 3, 1);
    const double ref8[2] = {0.1763234540e+01, -0.8356886819e+00};
    check("dr_dop853 (VDPOL eps=1e-3): y(2) as printed by the Fortran", s.status == 1 && rel_err(z, ref8, 2) < 1e-6,
          "rel err %.1e", rel_err(z, ref8, 2));
    check("dr_dop853: fcn=17552 step=1468 accpt=1345 rejct=122 (+-10%)",
          stat_close(s.nfev, 17552) && stat_close(s.nstep, 1468) && stat_close(s.naccept, 1345) &&
              stat_close(s.nreject, 122),
          "fcn %g", s.nfev);
}

/* The Bari test set's reference solutions. Each stiff solver must reach the given number of
 * significant correct digits at a tolerance fit for it. */
typedef struct {
    const char *name;
    Rhs f;
    int n;
    double t1;
    double y0[8];
    double ref[8];
    double rtol, atol, min_scd;
} TestSetProblem;

static void test_set(void) {
    const TestSetProblem P[] = {
        {"HIRES", hires, 8, 321.8122, {1, 0, 0, 0, 0, 0, 0, 0.0057},
         {7.371312573325668e-4, 1.442485726316185e-4, 5.888729740967575e-5, 1.175651343283149e-3,
          2.386356198831331e-3, 6.238968252742796e-3, 2.849998395185769e-3, 2.850001604814231e-3},
         1e-10, 1e-12, 6.0},
        {"ROBER to 1e11", robertson, 3, 1e11, {1, 0, 0},
         {2.083340149701255e-8, 8.333360770334713e-14, 0.9999999791665050}, 1e-10, 1e-20, 5.0},
        {"OREGO", orego, 3, 360.0, {1, 2, 3}, {1.000814870318523, 1228.178521549917, 132.0554942846706}, 1e-10,
         1e-10, 5.0},
        {"VDPOL eps=1e-6", vdpol_eps, 2, 2.0, {2, -0.66}, {1.706167732170483, -0.8928097010247975}, 1e-10, 1e-10,
         6.0},
    };
    for (size_t k = 0; k < sizeof P / sizeof P[0]; k++) {
        const TestSetProblem *p = &P[k];
        g_eps_vdp = 1e-6;
        Opts o = {p->rtol, p->atol, 0, 0, 0, 1000000};
        char name[128];
        double y[8];
        for (int m = 0; m < 3; m++) {
            memcpy(y, p->y0, sizeof y);
            int ok;
            const char *mn;
            if (m == 0) {
                ok = radau5(p->f, p->n, 0, p->t1, y, o).status > 0;
                mn = "radau5";
            } else if (m == 1) {
                ok = lsoda_run(p->f, p->n, 0, p->t1, y, o, 2, 0, 0).s.status > 0;
                mn = "lsoda";
            } else {
                ok = cvode(p->f, p->n, 0, p->t1, y, o, 0).status == CV_SUCCESS;
                mn = "cvode bdf";
            }
            double digits = scd(y, p->ref, p->n);
            snprintf(name, sizeof name, "test set %s, %s: scd >= %.0f", p->name, mn, p->min_scd);
            check(name, ok && digits >= p->min_scd, "scd %.1f", digits);
        }
    }
    /* E5: y1 and y4 fall to 1e-290, so the test set solves it with atol = 1.7e-24 and only y2,
     * y3 (about 8.9e-23) carry digits; at that atol they carry only a few. */
    {
        const double y0[4] = {1.76e-3, 0, 0, 0}, ref[2] = {8.867655517642120e-23, 8.854814626268838e-23};
        Opts o = {1e-8, 1.7e-24, 0, 0, 0, 1000000};
        double y[4];
        memcpy(y, y0, sizeof y);
        int ok = radau5(e5, 4, 0, 1e13, y, o).status > 0;
        check("test set E5 to 1e13, radau5: y2, y3 scd >= 2.5", ok && scd(y + 1, ref, 2) >= 2.5, "scd %.1f",
              scd(y + 1, ref, 2));
        memcpy(y, y0, sizeof y);
        ok = cvode(e5, 4, 0, 1e13, y, o, 0).status == CV_SUCCESS;
        check("test set E5 to 1e13, cvode bdf: y2, y3 scd >= 1.5", ok && scd(y + 1, ref, 2) >= 1.5, "scd %.1f",
              scd(y + 1, ref, 2));
    }
    /* Robertson invariants over the whole span (OrdinaryDiffEq FIRK, scipy): mass conservation
     * and no negative concentrations. */
    {
        double y[3] = {1, 0, 0};
        Opts o = {1e-8, 1e-14, 0, 0, 0, 1000000};
        int ok = radau5(robertson, 3, 0, 4e10, y, o).status > 0;
        const double ref[3] = {5.2083495894337328e-08, 2.0833399429795671e-13, 9.9999994791629776e-01};
        check("Robertson at 4e10 (SUNDIALS cvRoberts_dns check_ans), radau5", ok && rel_err(y, ref, 3) < 1e-4,
              "rel err %.1e", rel_err(y, ref, 3));
        check("Robertson: |sum(y) - 1| < 1e-8, y >= 0", fabs(y[0] + y[1] + y[2] - 1) < 1e-8 && y[1] >= 0,
              "sum-1 %.1e", y[0] + y[1] + y[2] - 1);
    }
}

/* SUNDIALS cv_test_kpr.out: CVODE BDF with a dense user Jacobian, rtol 1e-6, atol 1e-10, output
 * every 1 to t = 10. Compared like SUNDIALS' testRunner, which allows for platforms: 4
 * significant digits, statistics within 10% (exact on x86-64 Linux). */
static void sundials_kpr(void) {
    double y[2] = {sqrt(1.5), sqrt(3.0)}, t_out[10];
    for (int k = 0; k < 10; k++) t_out[k] = k + 1;
    g_jac = kpr_jac;
    Opts o = {1e-6, 1e-10, 0, 0, 1, 0};
    Stats s = cvode_run(kpr, 2, 0, t_out, 10, y, o, 0);
    const double ref[2] = {7.618838841024592e-01, 1.577079185839070e+00};
    check("SUNDIALS KPR: u(10), v(10) as in cv_test_kpr.out (4 digits)",
          s.status == CV_SUCCESS && rel_err(y, ref, 2) < 1e-4, "rel err %.1e", rel_err(y, ref, 2));
    check("SUNDIALS KPR: steps 1577, error test fails 147, RHS 2197, Jac 29 (+-10%)",
          stat_close(s.nstep, 1577) && stat_close(s.nreject, 147) && stat_close(s.nfev, 2197) &&
              stat_close(s.njac, 29),
          "steps %g", s.nstep);
    const double exact[2] = {sqrt(1 + 0.5 * cos(10.0)), sqrt(2 + cos(200.0))};
    check("SUNDIALS KPR: within 1e-4 of the exact solution", rel_err(y, exact, 2) < 1e-4, "rel err %.1e",
          rel_err(y, exact, 2));
    /* cv_kpr_cvode.init_step_0.01.out: the same with a first step of 0.01. */
    y[0] = sqrt(1.5); y[1] = sqrt(3.0);
    o.h0 = 0.01;
    s = cvode_run(kpr, 2, 0, t_out, 10, y, o, 0);
    check("SUNDIALS KPR, h0 = 0.01: u(10) = 7.618772869860075e-01, 1572 steps",
          rel_err(y, (double[]){7.618772869860075e-01}, 1) < 1e-4 && stat_close(s.nstep, 1572), "steps %g",
          s.nstep);
}

/* scipy gh-24933 (Jacobian orientation), ODEInterfaceDiffEq jac_tests (the user Jacobian is
 * called), OrdinaryDiffEq nsa_jacobian_reuse (it is reused). Robertson to t = 1 at rtol 1e-8
 * needs about 100 steps with the right Jacobian, ~20x more with its transpose. */
static void jacobian_orientation(void) {
    Opts o = {1e-8, 1e-10, 0, 0, 1, 100000};
    for (int m = 0; m < 3; m++) {
        const char *mn = m == 0 ? "radau5" : m == 1 ? "lsoda" : "cvode bdf";
        for (int transposed = 0; transposed < 2; transposed++) {
            double y[3] = {1, 0, 0};
            g_jac = transposed ? robertson_jac_transposed : robertson_jac;
            g_jac_calls = 0;
            Stats s = m == 0 ? radau5(robertson, 3, 0, 1, y, o)
                      : m == 1 ? lsoda_run(robertson, 3, 0, 1, y, o, 1, 0, 0).s
                               : cvode(robertson, 3, 0, 1, y, o, 0);
            char name[128];
            if (!transposed) {
                snprintf(name, sizeof name, "%s, user Jacobian: <= 150 steps, called and reused", mn);
                check(name, s.nstep <= 150 && g_jac_calls >= 1 && g_jac_calls < s.nstep, "steps %g",
                      (double)s.nstep);
            } else {
                snprintf(name, sizeof name, "%s, transposed Jacobian is noticeably worse (> 300 steps or fails)",
                         mn);
                check(name, s.nstep > 300 || s.status < 0, "steps %g", (double)s.nstep);
            }
        }
    }
}

/* ODEInterface.jl test_jacode2 / test_rhstimederiv1, scipy test_banded_ode_solvers: a
 * non-symmetric banded Jacobian and a time-dependent one, through every Jacobian mode. */
static void jacobian_modes(void) {
    const double T = 2.0;
    double exact[4];
    for (int k = 0; k < 4; k++) exact[k] = exp(T) * pow(T, k) / tgamma(k + 1.0);
    Opts o = {1e-11, 1e-12, 0, 0, 1, 0};
    const int jts[4] = {1, 2, 4, 5};
    const char *jtn[4] = {"user full", "internal full", "user banded", "internal banded"};
    for (int k = 0; k < 4; k++) {
        double y[4] = {1, 0, 0, 0};
        g_jac = chain_jac;
        LsodaRun r = lsoda_run(chain, 4, 0, T, y, o, jts[k], 1, 0);
        char name[128];
        snprintf(name, sizeof name, "lsoda %s Jacobian (ml=1, mu=0): bidiagonal chain", jtn[k]);
        check(name, r.s.status > 0 && rel_err(y, exact, 4) < 1e-9, "rel err %.1e", rel_err(y, exact, 4));
    }
    /* scipy test_banded_lsoda: on a stiff problem (so LSODA switches to BDF and uses the
     * Jacobian) every Jacobian mode agrees with a tight RADAU5 solution. */
    double ref[4] = {1, 1, 1, 1};
    Opts tight = {1e-13, 1e-15, 0, 0, 0, 0};
    radau5(stiff_chain, 4, 0, 5, ref, tight);
    for (int k = 0; k < 4; k++) {
        double y[4] = {1, 1, 1, 1};
        Opts os = {1e-10, 1e-12, 0, 0, 1, 0};
        g_jac = stiff_chain_jac;
        LsodaRun r = lsoda_run(stiff_chain, 4, 0, 5, y, os, jts[k], 1, 0);
        char name[128];
        snprintf(name, sizeof name, "lsoda %s Jacobian, stiff chain: BDF, matches radau5", jtn[k]);
        double err = mixed_err(y, ref, 4, 1e-12);
        check(name, r.s.status > 0 && r.last_method == 2 && r.s.njac > 0 && err < 1e-7,
              "rel err (floor 1e-12) %.1e", err);
    }
    double y[4] = {1, 0, 0, 0};
    g_jac = chain_jac;
    radau5(chain, 4, 0, T, y, o);
    check("radau5 user Jacobian: bidiagonal chain", rel_err(y, exact, 4) < 1e-9, "rel err %.1e",
          rel_err(y, exact, 4));
    y[0] = 1; y[1] = y[2] = y[3] = 0;
    cvode(chain, 4, 0, T, y, o, 0);
    check("cvode bdf user Jacobian: bidiagonal chain", rel_err(y, exact, 4) < 1e-8, "rel err %.1e",
          rel_err(y, exact, 4));

    const double t1 = 1.2, s2 = sinh(t1 * t1), c2 = cosh(t1 * t1);
    const double ex2[2] = {c2 + 0.5 * s2, c2 + 2 * s2};
    g_jac = tdep_jac;
    double z[2] = {1, 1};
    radau5(tdep, 2, 0, t1, z, o);
    check("radau5 time-dependent user Jacobian", rel_err(z, ex2, 2) < 1e-9, "rel err %.1e", rel_err(z, ex2, 2));
    z[0] = z[1] = 1;
    lsoda_run(tdep, 2, 0, t1, z, o, 1, 0, 0);
    check("lsoda time-dependent user Jacobian", rel_err(z, ex2, 2) < 1e-9, "rel err %.1e", rel_err(z, ex2, 2));
    z[0] = z[1] = 1;
    cvode(tdep, 2, 0, t1, z, o, 0);
    check("cvode bdf time-dependent user Jacobian", rel_err(z, ex2, 2) < 1e-8, "rel err %.1e", rel_err(z, ex2, 2));
}

/* scipy test_integration_stiff, OrdinaryDiffEqDefault: LSODA starts with Adams, switches to
 * BDF on a stiff problem and stays with Adams on a non-stiff one. */
static void stiffness_switching(void) {
    double y[3] = {1, 0, 0};
    Opts o = {1e-6, 1e-6, 0, 0, 0, 0};
    LsodaRun r = lsoda_run(robertson, 3, 0, 1e8, y, o, 2, 0, 0);
    check("lsoda Robertson to 1e8: starts Adams, ends BDF", r.s.status > 0 && r.first_method == 1 && r.last_method == 2,
          "switches %g", r.method_switches);
    check("lsoda Robertson to 1e8: nfev < 5000, njac < 200", r.s.nfev < 5000 && r.s.njac < 200, "nfev %g",
          (double)r.s.nfev);
    double z[2] = {0.5, 0.25};
    r = lsoda_run(linear_growth, 2, 0, 1, z, o, 2, 0, 0);
    check("lsoda u' = 1.01u: Adams throughout", r.s.status > 0 && r.first_method == 1 && r.last_method == 1 &&
                                                     r.method_switches == 0,
          "switches %g", r.method_switches);
}

/* OrdinaryDiffEq ode_firk_tests "test adaptivity": RADAU5 on very stiff Van der Pol, with the
 * Jacobian reused across steps and a bounded number of steps (OrdinaryDiffEq's RadauIIA5 needs
 * < 150..190; Hairer's step control here takes about twice as many, a regression guard). */
static void extreme_stiffness(void) {
    const double mus[3] = {1e6, 1e8, 1e10};
    for (int k = 0; k < 3; k++) {
        double y[2] = {sqrt(3.0), 0.0};
        g_mu_vdp = mus[k];
        Opts o = {1e-6, 1e-6, 0, 0, 0, 100000};
        Stats s = radau5(vdpol, 2, 0, 1, y, o);
        char name[128];
        snprintf(name, sizeof name, "radau5 Van der Pol mu=%.0e: < 500 steps, Jacobian reused", mus[k]);
        check(name, s.status > 0 && s.nstep < 500 && s.njac < s.nstep, "steps %g", (double)s.nstep);
        y[0] = sqrt(3.0); y[1] = 0;
        s = cvode(vdpol, 2, 0, 1, y, o, 0);
        snprintf(name, sizeof name, "cvode bdf Van der Pol mu=%.0e: succeeds", mus[k]);
        check(name, s.status == CV_SUCCESS, "steps %g", (double)s.nstep);
    }
}

/* Error proportional to the tolerance (B1): over rtol = 1e-4 .. 1e-10 the error falls, stays
 * within a constant of rtol, and the step count doesn't fall. HIRES against the test set. */
static void tolerance_proportionality(void) {
    const double y0[8] = {1, 0, 0, 0, 0, 0, 0, 0.0057};
    const double ref[8] = {7.371312573325668e-4, 1.442485726316185e-4, 5.888729740967575e-5, 1.175651343283149e-3,
                           2.386356198831331e-3, 6.238968252742796e-3, 2.849998395185769e-3, 2.850001604814231e-3};
    for (int m = 0; m < 3; m++) {
        const char *mn = m == 0 ? "radau5" : m == 1 ? "lsoda" : "cvode bdf";
        double prev_err = INFINITY, worst_ratio = 0;
        long prev_steps = 0;
        int monotone = 1;
        for (int e = 4; e <= 10; e += 2) {
            double rtol = pow(10.0, -e), y[8];
            memcpy(y, y0, sizeof y);
            Opts o = {rtol, rtol * 1e-2, 0, 0, 0, 1000000};
            Stats s = m == 0 ? radau5(hires, 8, 0, 321.8122, y, o)
                      : m == 1 ? lsoda_run(hires, 8, 0, 321.8122, y, o, 2, 0, 0).s
                               : cvode(hires, 8, 0, 321.8122, y, o, 0);
            double err = rel_err(y, ref, 8);
            if (!(err < prev_err) || s.nstep < prev_steps) monotone = 0;
            worst_ratio = fmax(worst_ratio, err / rtol);
            prev_err = err;
            prev_steps = s.nstep;
        }
        char name[128];
        snprintf(name, sizeof name, "%s HIRES: error falls with rtol 1e-4..1e-10, steps grow", mn);
        check(name, monotone, "last err %.1e", prev_err);
        /* CVODE's BDF is known to run a larger global error at a given tolerance. */
        const double limit = m == 2 ? 3000 : 1000;
        snprintf(name, sizeof name, "%s HIRES: error <= %.0f rtol", mn, limit);
        check(name, worst_ratio <= limit, "max err/rtol %.0f", worst_ratio);
    }
}

/* diffrax test_ode_order, OrdinaryDiffEq test_convergence: with the tolerance out of the way
 * (rtol = atol = 1) and the step forced by hmax, the global error falls with the method's
 * order. */
static void convergence_order(void) {
    const double T = 4.0;
    const double exact[2] = {exp(-0.1 * T) * cos(T), exp(-0.1 * T) * sin(T)};
    for (int m = 0; m < 3; m++) {
        const int order = m == 0 ? 5 : m == 1 ? 8 : 5;
        const char *mn = m == 0 ? "dopri5" : m == 1 ? "dop853" : "radau5";
        double hs[4], errs[4];
        for (int k = 0; k < 4; k++) {
            double h = (m == 1 ? 0.8 : 0.4) / pow(2.0, k), y[2] = {1, 0};
            Opts o = {1.0, 1.0, h, h, 0, 0};
            if (m == 2) radau5(rotation, 2, 0, T, y, o);
            else dopri(order, rotation, 2, 0, T, y, o, 0, -1);
            hs[k] = h;
            errs[k] = rel_err(y, exact, 2);
        }
        double slope = log(errs[0] / errs[3]) / log(hs[0] / hs[3]);
        char name[128];
        snprintf(name, sizeof name, "%s: convergence order %d +- 0.6 (fixed steps)", mn, order);
        check(name, fabs(slope - order) < 0.6, "slope %.2f", slope);
    }
}

/* OrdinaryDiffEq time_derivative_test: u' = -t is integrated exactly by the Runge-Kutta
 * methods (the multistep ones start at low order, so they aren't). */
static void polynomial_exactness(void) {
    const double exact[1] = {1.0 - 0.5 * 3.0 * 3.0};
    Opts o = {1e-6, 1e-6, 0, 0, 0, 0};
    double y[1];
    y[0] = 1; radau5(minus_t, 1, 0, 3, y, o);
    check("radau5 u' = -t exact", rel_err(y, exact, 1) < 1e-12, "rel err %.1e", rel_err(y, exact, 1));
    y[0] = 1; dopri(5, minus_t, 1, 0, 3, y, o, 0, -1);
    check("dopri5 u' = -t exact", rel_err(y, exact, 1) < 1e-12, "rel err %.1e", rel_err(y, exact, 1));
    y[0] = 1; dopri(8, minus_t, 1, 0, 3, y, o, 0, -1);
    check("dop853 u' = -t exact", rel_err(y, exact, 1) < 1e-12, "rel err %.1e", rel_err(y, exact, 1));
}

/* GlobalDiffEq.jl: atol = 0 (pure relative tolerance) on a solution crossing zero. Started at
 * omega = 0, every solver must either be accurate or refuse with an error, never silently
 * wrong; started off zero, the zero crossings along the way must not stop any of them. */
static void pure_relative_tolerance_from(double omega0);
static void pure_relative_tolerance(void) {
    for (int start = 0; start < 2; start++) pure_relative_tolerance_from(start ? 1e-3 : 0.0);
}
static void pure_relative_tolerance_from(double omega0) {
    double ref[2] = {0.01, omega0};
    Opts tight = {1e-14, 1e-16, 0, 0, 0, 0};
    dopri(8, pendulum, 2, 0, 10, ref, tight, 0, -1);
    Opts o = {1e-10, 0.0, 0, 0, 0, 1000000};
    for (int m = 0; m < 5; m++) {
        double y[2] = {0.01, omega0};
        const char *mn[5] = {"radau5", "dopri5", "dop853", "lsoda", "cvode bdf"};
        int status_ok;
        if (m == 0) status_ok = radau5(pendulum, 2, 0, 10, y, o).status > 0;
        else if (m <= 2) status_ok = dopri(m == 1 ? 5 : 8, pendulum, 2, 0, 10, y, o, 0, -1).status > 0;
        else if (m == 3) status_ok = lsoda_run(pendulum, 2, 0, 10, y, o, 2, 0, 0).s.status > 0;
        else status_ok = cvode(pendulum, 2, 0, 10, y, o, 0).status == CV_SUCCESS;
        char name[128];
        double err = rel_err(y, ref, 2);
        if (omega0 == 0) {
            snprintf(name, sizeof name, "%s atol = 0 from y = 0: accurate or an error", mn[m]);
            check(name, status_ok ? err < 1e-6 : 1, status_ok ? "rel err %.1e" : "refused (%g)",
                  status_ok ? err : 0.0);
        } else {
            /* RADAU5 refuses ATOL <= 0 as bad input (IDID -1) by design. */
            snprintf(name, sizeof name, "%s atol = 0, y crossing 0: accurate%s", mn[m],
                     m == 0 ? " (or refused)" : "");
            check(name, status_ok ? err < 1e-6 : m == 0, "rel err %.1e", err);
        }
    }
}

int main(void) {
    hairer_drivers();
    test_set();
    sundials_kpr();
    jacobian_orientation();
    jacobian_modes();
    stiffness_switching();
    extreme_stiffness();
    tolerance_proportionality();
    convergence_order();
    polynomial_exactness();
    pure_relative_tolerance();
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
