/* Contracts of the single-segment drivers in radau5_wrapper.c (run_radau5, run_radau5_fit,
 * run_dop853, run_dopri5, run_lsoda) and the output grid of solver_common.c.
 *
 * The contracts come from other ODE suites' tests of the same codes or the same wrapper
 * patterns: diffrax (test_saveat_solution, test_integrate), OrdinaryDiffEq.jl
 * (InterfaceI, check_error, time-reversal tests), ODEInterface.jl, scipy.integrate
 * (test_integrate.py, test_ivp.py). Errors are relative, max-norm. Build and run with
 * `make test`. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "solver_common.h"

typedef int (*RunFn)(int n, double t_start, double t_end, double *y, double *rpar, double rtol, double atol,
                     double h_init, int nmax);
extern int run_radau5(int, double, double, double *, double *, double, double, double, int);
extern int run_radau5_fit(int, double, double, double *, double *, double, double, double, int);
extern int run_dop853(int, double, double, double *, double *, double, double, double, int);
extern int run_dopri5(int, double, double, double *, double *, double, double, double, int);
extern int run_lsoda(int, double, double, double *, double *, double, double, double, int);

typedef struct {
    const char *name;
    RunFn run;
    int too_many_steps; /* the solver's own code for an exhausted step budget */
    int bad_input;
} Method;

static const Method METHODS[] = {
    {"radau5", run_radau5, -2, -1},  {"radau5_fit", run_radau5_fit, -2, -1}, {"dop853", run_dop853, -2, -1},
    {"dopri5", run_dopri5, -2, -1},  {"lsoda", run_lsoda, -1, -3},
};
#define N_METHODS ((int)(sizeof METHODS / sizeof METHODS[0]))

static int failures;
static void check(const char *method, const char *name, int ok, const char *fmt, double v) {
    printf("%-4s %-10s %-52s ", ok ? "ok" : "FAIL", method, name);
    printf(fmt, v);
    printf("\n");
    if (!ok) failures++;
}

/* ------------------------------------------------------------------ */
/* Models (ModelFn: n, t, y, dydt, pars)                               */
/* ------------------------------------------------------------------ */
static long nfev;
static double t_lo, t_hi; /* every evaluation must stay inside [t_lo, t_hi] */
static int out_of_span;

static void note(double t) {
    nfev++;
    if (t < t_lo || t > t_hi) out_of_span++;
}
static void decay(int n, double t, double *y, double *dy, double *p) { /* y' = -p0 y */
    note(t);
    for (int i = 0; i < n; i++) dy[i] = -p[0] * y[i];
}
static void growth(int n, double t, double *y, double *dy, double *p) { /* y' = +p0 y */
    note(t);
    for (int i = 0; i < n; i++) dy[i] = p[0] * y[i];
}
static void robertson(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)p;
    note(t);
    dy[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
    dy[2] = 3e7 * y[1] * y[1];
    dy[1] = -dy[0] - dy[2];
}
/* Mirror of robertson: g(u, t) = -f(u, -t), integrated over the negated span. */
static void robertson_mirror(int n, double t, double *y, double *dy, double *p) {
    robertson(n, -t, y, dy, p);
    for (int i = 0; i < 3; i++) dy[i] = -dy[i];
}
/* Non-autonomous, non-stiff (OrdinaryDiffEq bdf_time_reversal_tests). */
static void nonautonomous(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)p;
    note(t);
    dy[0] = cos(3 * t) * y[0] - 0.3 * y[1];
    dy[1] = sin(3 * t) * y[0];
}
static void nonautonomous_mirror(int n, double t, double *y, double *dy, double *p) {
    nonautonomous(n, -t, y, dy, p);
    dy[0] = -dy[0];
    dy[1] = -dy[1];
}
static void nan_after_half(int n, double t, double *y, double *dy, double *p) {
    (void)p;
    note(t);
    for (int i = 0; i < n; i++) dy[i] = t > 0.5 ? NAN : -y[i];
}
static void exp_blowup(int n, double t, double *y, double *dy, double *p) { /* u' = exp(u), singular at t = 1 */
    (void)n; (void)p;
    note(t);
    dy[0] = exp(y[0]);
}
static void square_blowup(int n, double t, double *y, double *dy, double *p) { /* x' = x^2, singular at t = 1 */
    (void)n; (void)p;
    note(t);
    dy[0] = y[0] * y[0];
}
/* OrdinaryDiffEq FIRK #4997: the third equation is NaN for c < 0, and its Jacobian is -Inf at c = 0. */
static void nan_region(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)p;
    note(t);
    dy[0] = -y[0];
    dy[1] = y[0] - y[1];
    dy[2] = -(y[1] < 0 ? NAN : pow(y[1], 0.1)) - y[2];
}
static void zero_rhs(int n, double t, double *y, double *dy, double *p) {
    (void)y; (void)p;
    note(t);
    for (int i = 0; i < n; i++) dy[i] = 0.0;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    int rc, n_out;
    double t[4096];
    double y[4096 * 3];
} Out;

static Out out;

/* One run. grid may be NULL (record every step). y is updated in place. */
static int solve(const Method *m, ModelFn f, int n, double t0, double t1, double *y, double *pars, double rtol,
                 double atol, double h_init, int nmax, const double *grid, int n_grid) {
    set_model_fn_ptr(f);
    init_output(16, n > 0 ? n : 1);
    if (grid) set_output_grid(grid, n_grid);
    nfev = 0;
    out_of_span = 0;
    t_lo = t0 < t1 ? t0 : t1;
    t_hi = t0 < t1 ? t1 : t0;
    out.rc = m->run(n, t0, t1, y, pars, rtol, atol, h_init, nmax);
    out.n_out = get_out_n();
    if (out.n_out > 4096) out.n_out = 4096;
    memcpy(out.t, get_out_t(), (size_t)out.n_out * sizeof(double));
    if (n > 0) memcpy(out.y, get_out_y(), (size_t)out.n_out * (size_t)n * sizeof(double));
    free_output();
    return out.rc;
}

static int success(const Method *m, int rc) { return m->run == run_lsoda ? rc == 1 : rc > 0; }

static double rel(double a, double b) { return fabs(a - b) / fmax(fabs(b), 1e-300); }

/* ------------------------------------------------------------------ */
/* Contracts                                                           */
/* ------------------------------------------------------------------ */

/* diffrax test_t0_eq_t1, scipy test_no_integration: a zero-length span returns y0 exactly. */
static void t0_eq_t1(const Method *m) {
    double p = 1.0, y[2] = {2.0, -3.0};
    const double grid[3] = {1.0, 1.0, 1.0};
    solve(m, decay, 2, 1.0, 1.0, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "t0 == t1: success, y0 unchanged", success(m, out.rc) && y[0] == 2.0 && y[1] == -3.0,
          "rc %g", out.rc);
    solve(m, decay, 2, 1.0, 1.0, y, &p, 1e-8, 1e-10, 0, 0, grid, 3);
    int ok = success(m, out.rc) && out.n_out == 3;
    for (int k = 0; ok && k < 3; k++) ok = out.t[k] == 1.0 && out.y[2 * k] == 2.0 && out.y[2 * k + 1] == -3.0;
    check(m->name, "t0 == t1 with grid [t0, t0, t0]: three rows of y0", ok, "n_out %g", out.n_out);
}

/* diffrax test_saveat_solution, OrdinaryDiffEq ode_saveat_tests: grid times come back exactly,
 * y(t0) is y0 bitwise, closely spaced points all appear, values are accurate. */
static void grid_contract(const Method *m) {
    double p = 0.5, y[1] = {2.1};
    const double grid[6] = {0.0, 0.125, 0.6, 0.61, 0.8, 1.0};
    solve(m, decay, 1, 0.0, 1.0, y, &p, 1e-10, 1e-12, 0, 0, grid, 6);
    int ok = success(m, out.rc) && out.n_out == 6;
    double err = 0;
    for (int k = 0; ok && k < 6; k++) {
        ok = out.t[k] == grid[k];
        err = fmax(err, rel(out.y[k], 2.1 * exp(-0.5 * grid[k])));
    }
    check(m->name, "grid times exact, all present", ok, "n_out %g", out.n_out);
    check(m->name, "grid y(t0) == y0 bitwise", out.n_out > 0 && out.y[0] == 2.1, "y %g", out.y[0]);
    check(m->name, "grid values (y' = -0.5y)", ok && err < 1e-8, "rel err %.1e", err);
    check(m->name, "final y is y(t1)", rel(y[0], 2.1 * exp(-0.5)) < 1e-8, "rel err %.1e",
          rel(y[0], 2.1 * exp(-0.5)));
}

/* ODEInterfaceRegression (calck): asking for a grid must not change how the solver steps,
 * so the final state is identical bitwise. */
static int implicit(const Method *m) { return m->run != run_dop853 && m->run != run_dopri5; }

static void grid_does_not_change_stepping(const Method *m) {
    double p[1] = {0}, y1[3] = {1, 0.5, 0}, y2[3] = {1, 0.5, 0};
    const double grid[4] = {0.1, 1.0, 1.7, 5.0};
    ModelFn f = implicit(m) ? robertson : nonautonomous;
    const int n = implicit(m) ? 3 : 2;
    if (implicit(m)) { y1[1] = y2[1] = 0; }
    solve(m, f, n, 0, 5, y1, p, 1e-8, 1e-10, 0, 0, NULL, 0);
    int rc1 = out.rc;
    solve(m, f, n, 0, 5, y2, p, 1e-8, 1e-10, 0, 0, grid, 4);
    double d = 0;
    for (int i = 0; i < 3; i++) d = fmax(d, fabs(y1[i] - y2[i]));
    check(m->name, "grid leaves the final state bitwise unchanged", success(m, rc1) && success(m, out.rc) && d == 0,
          "max abs diff %.1e", d);
}

/* diffrax test_reverse_time, scipy test_integration (backward span): backward integration of
 * y' = -y from 4 to 0.3 equals forward integration of the mirrored y' = y from -4 to -0.3,
 * both with and without a (decreasing) grid. */
static void backward(const Method *m) {
    double p = 1.0, yb[1] = {1.0}, yf[1] = {1.0};
    const double gb[3] = {4.0, 3.5, 0.7}, gf[3] = {-4.0, -3.5, -0.7};
    solve(m, decay, 1, 4.0, 0.3, yb, &p, 1e-8, 1e-10, 0, 0, gb, 3);
    Out b = out;
    solve(m, growth, 1, -4.0, -0.3, yf, &p, 1e-8, 1e-10, 0, 0, gf, 3);
    int ok = success(m, b.rc) && success(m, out.rc) && b.n_out == 3 && out.n_out == 3;
    double err = 0;
    for (int k = 0; ok && k < 3; k++) {
        ok = b.t[k] == gb[k];
        err = fmax(err, rel(b.y[k], exp(4.0 - gb[k])));
        err = fmax(err, rel(b.y[k], out.y[k]));
    }
    check(m->name, "backward span with a decreasing grid", ok && err < 1e-7, "rel err %.1e", err);
    check(m->name, "backward final state", rel(yb[0], exp(3.7)) < 1e-7, "rel err %.1e", rel(yb[0], exp(3.7)));
    check(m->name, "no evaluation outside the backward span", out_of_span == 0, "outside %g", out_of_span);
}

/* OrdinaryDiffEq firk/bdf time-reversal tests: integrating g(u, t) = -f(u, -t) over the
 * negated span takes bitwise the same step sizes. A sign bug in a finite-difference stencil,
 * an initial-step guess or an error norm breaks this. */
static void time_reversal_symmetry(const Method *m) {
    double p = 0, yf[3] = {1, 0, 0}, yb[3] = {1, 0, 0};
    int ok;
    if (implicit(m)) {
        solve(m, robertson, 3, -50, 50, yf, &p, 1e-8, 1e-8, 0, 0, NULL, 0);
        Out f = out;
        solve(m, robertson_mirror, 3, 50, -50, yb, &p, 1e-8, 1e-8, 0, 0, NULL, 0);
        ok = success(m, f.rc) && success(m, out.rc) && f.n_out == out.n_out;
        for (int k = 0; ok && k < f.n_out; k++) ok = f.t[k] == -out.t[k];
        check(m->name, "mirrored Robertson takes the mirrored steps", ok, "steps %g", f.n_out);
    }
    double zf[2] = {1, 0.5}, zb[2] = {1, 0.5};
    solve(m, nonautonomous, 2, -2.5, 2.5, zf, &p, 1e-8, 1e-8, 0, 0, NULL, 0);
    Out f = out;
    solve(m, nonautonomous_mirror, 2, 2.5, -2.5, zb, &p, 1e-8, 1e-8, 0, 0, NULL, 0);
    ok = success(m, f.rc) && success(m, out.rc) && f.n_out == out.n_out;
    for (int k = 0; ok && k < f.n_out; k++) ok = f.t[k] == -out.t[k];
    check(m->name, "mirrored non-autonomous problem: mirrored steps", ok, "steps %g", f.n_out);
}

/* OrdinaryDiffEq check_error / ode_initdt_tests, FIRK #4997, scipy: failures come back as a
 * negative code after a bounded amount of work, not after the whole step budget. */
static void failures_are_fast(const Method *m) {
    double p = 1.0, y[3];
    y[0] = 1.0;
    solve(m, nan_after_half, 1, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "NaN RHS fails fast", out.rc < 0 && nfev < 20000, "nfev %g", (double)nfev);
    y[0] = 0.0;
    solve(m, exp_blowup, 1, 0, 10, y, &p, 1e-8, 1e-8, 0, 0, NULL, 0);
    check(m->name, "u' = exp(u) blow-up fails fast", out.rc < 0 && nfev < 20000, "nfev %g", (double)nfev);
    y[0] = 1.0;
    solve(m, square_blowup, 1, 0, 2, y, &p, 1e-8, 1e-8, 0, 0, NULL, 0);
    check(m->name, "x' = x^2 blow-up fails fast", out.rc < 0 && nfev < 20000, "nfev %g", (double)nfev);
    y[0] = 1.0;
    solve(m, growth, 1, 0, 1000, y, &p, 1e-6, 1e-6, 0, 0, NULL, 0);
    double t_fail = out.n_out ? out.t[out.n_out - 1] : 0;
    check(m->name, "u' = u overflow fails, after t = 700", out.rc < 0 && t_fail >= 700 && nfev < 200000,
          "t_fail %g", t_fail);
    y[0] = 1.0; y[1] = 0.0; y[2] = 0.0;
    solve(m, nan_region, 3, 0, 10, y, &p, 1e-6, 1e-8, 0, 0, NULL, 0);
    int ok = out.rc < 0 || (success(m, out.rc) && rel(y[0], exp(-10)) < 1e-2);
    check(m->name, "NaN region: fails or stays right, never silently wrong", ok && nfev < 50000, "rc %g", out.rc);
}

/* DiffEqBase solve_error_handling, scipy test_initial_state_finiteness: bad input is refused
 * with the solver's bad-input code before any work. */
static void bad_input(const Method *m) {
    double p = 1.0, y[1] = {1.0};
    solve(m, decay, 1, NAN, 1, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "NaN t0 rejected", out.rc == m->bad_input && nfev == 0, "rc %g", out.rc);
    solve(m, decay, 1, 0, INFINITY, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "infinite t1 rejected", out.rc == m->bad_input && nfev == 0, "rc %g", out.rc);
    y[0] = NAN;
    solve(m, decay, 1, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "NaN y0 rejected", out.rc == m->bad_input && nfev == 0, "rc %g", out.rc);
    y[0] = 1.0;
    const double unsorted[3] = {0.0, 0.7, 0.3};
    solve(m, decay, 1, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, unsorted, 3);
    check(m->name, "non-monotone grid rejected", out.rc == m->bad_input && nfev == 0, "rc %g", out.rc);
    y[0] = 1.0;
    const double outside[2] = {0.5, 2.0};
    solve(m, decay, 1, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, outside, 2);
    check(m->name, "grid outside the span rejected", out.rc == m->bad_input && nfev == 0, "rc %g", out.rc);
    y[0] = 1.0;
    solve(m, decay, 1, 0, 1, y, &p, -1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "negative rtol rejected", out.rc == m->bad_input && nfev == 0, "rc %g", out.rc);
}

/* LSODA.jl maxiters=1, scipy ODECheckParameterUse: an exhausted budget gives the solver's
 * too-many-steps code; nmax <= 0 means the default budget. */
static void step_budget(const Method *m) {
    double p = 0, y[2] = {1, 0.5};
    solve(m, nonautonomous, 2, 0, 40, y, &p, 1e-8, 1e-10, 0, 5, NULL, 0);
    check(m->name, "nmax = 5 runs out of steps", out.rc == m->too_many_steps, "rc %g", out.rc);
    y[0] = 1; y[1] = 0.5;
    solve(m, nonautonomous, 2, 0, 40, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "nmax = 0 means the default", success(m, out.rc), "rc %g", out.rc);
    y[0] = 1; y[1] = 0.5;
    solve(m, nonautonomous, 2, 0, 40, y, &p, 1e-8, 1e-10, 0, -1, NULL, 0);
    check(m->name, "nmax < 0 means the default", success(m, out.rc), "rc %g", out.rc);
}

/* scipy test_first_step, ODEInterfaceRegression init_dt: the first step is h_init. */
static void first_step(const Method *m) {
    double p = 1.0, y[1] = {1.0};
    solve(m, decay, 1, 0, 1, y, &p, 1e-6, 1e-8, 1e-3, 0, NULL, 0);
    double h0 = out.n_out > 1 ? out.t[1] - out.t[0] : 0;
    check(m->name, "first step is h_init = 1e-3", success(m, out.rc) && rel(h0, 1e-3) < 1e-12, "h0 %.3e", h0);
    y[0] = 1.0;
    solve(m, decay, 1, 1, 0, y, &p, 1e-6, 1e-8, 1e-3, 0, NULL, 0);
    h0 = out.n_out > 1 ? out.t[0] - out.t[1] : 0;
    check(m->name, "backward: first step is |h_init|", success(m, out.rc) && rel(h0, 1e-3) < 1e-12, "h0 %.3e", h0);
}

/* scipy gh-17341/8848/9198: the RHS is never evaluated outside [t0, t1]. */
static void stays_in_span(const Method *m) {
    double p = 1.0, y[1] = {1.0};
    const double grid[2] = {1e-5, 5e-5};
    solve(m, decay, 1, 0, 1e-4, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    int bad = out_of_span;
    y[0] = 1.0;
    solve(m, decay, 1, -17, 2, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    bad += out_of_span;
    y[0] = 1.0;
    solve(m, zero_rhs, 1, 100, 200, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    bad += out_of_span;
    check(m->name, "f == 0 keeps y bitwise", y[0] == 1.0, "y %g", y[0]);
    y[0] = 1.0;
    solve(m, decay, 1, 0, 1e-4, y, &p, 1e-8, 1e-10, 0, 0, grid, 2); /* last grid point before t1 */
    bad += out_of_span;
    check(m->name, "RHS never evaluated outside [t0, t1]", bad == 0, "outside %g", bad);
}

/* scipy test_repeated_t_values (gh-8217): repeated grid times give identical rows. */
static void repeated_grid_times(const Method *m) {
    double p = 0.25, y[1] = {1.0};
    const double tau = 4 * log(2.0);
    double grid[13];
    for (int k = 0; k < 9; k++) grid[k] = 0.0;
    grid[9] = tau; grid[10] = 2 * tau; grid[11] = 2 * tau; grid[12] = 3 * tau;
    solve(m, decay, 1, 0, 3 * tau, y, &p, 1e-12, 1e-12, 0, 0, grid, 13);
    int ok = success(m, out.rc) && out.n_out == 13;
    double err = 0;
    for (int k = 0; ok && k < 13; k++) err = fmax(err, rel(out.y[k], exp(-0.25 * grid[k])));
    check(m->name, "repeated grid times (halvings)", ok && err < 1e-9 && out.y[10] == out.y[11], "rel err %.1e",
          err);
}

/* SUNDIALS iterators.jl / reinit, scipy test_concurrent_*: identical solves are bitwise
 * identical, even with a different solve in between (no state left over). */
static void deterministic(const Method *m) {
    double p = 0, a[3] = {1, 0, 0}, b[3] = {1, 0, 0}, q = 1.0, c[1] = {1.0};
    const double t1 = implicit(m) ? 40 : 0.1; /* Robertson is stiff */
    solve(m, robertson, 3, 0, t1, a, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    solve(m, decay, 1, 0, 3, c, &q, 1e-4, 1e-6, 0, 0, NULL, 0);
    solve(m, robertson, 3, 0, t1, b, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "repeated solves are bitwise identical", memcmp(a, b, sizeof a) == 0, "y1 %g", a[1]);
}

/* scipy test_empty, diffrax random_pytree: a state of size 0 is a no-op success. */
static void empty_state(const Method *m) {
    double p = 0, y[1] = {0};
    const double grid[2] = {0.0, 1.0};
    solve(m, decay, 0, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, grid, 2);
    check(m->name, "n = 0 succeeds, records the grid", success(m, out.rc) && out.n_out == 2, "rc %g", out.rc);
}

/* A grid set for one call must not leak into the next one. */
static void grid_is_consumed(const Method *m) {
    double p = 1.0, y[1] = {1.0};
    const double grid[2] = {0.25, 0.5};
    solve(m, decay, 1, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, grid, 2);
    y[0] = 1.0;
    solve(&METHODS[0], decay, 1, 0, 1, y, &p, 1e-8, 1e-10, 0, 0, NULL, 0);
    check(m->name, "the grid is cleared after the call", grid_t == NULL && out.n_out > 2, "n_out next %g",
          out.n_out);
}

int main(void) {
    for (int k = 0; k < N_METHODS; k++) {
        const Method *m = &METHODS[k];
        t0_eq_t1(m);
        grid_contract(m);
        grid_does_not_change_stepping(m);
        backward(m);
        time_reversal_symmetry(m);
        failures_are_fast(m);
        bad_input(m);
        step_budget(m);
        first_step(m);
        stays_in_span(m);
        repeated_grid_times(m);
        deterministic(m);
        empty_state(m);
        grid_is_consumed(m);
    }
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
