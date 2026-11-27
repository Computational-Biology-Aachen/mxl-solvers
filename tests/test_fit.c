/* Fit drivers (MINPACK + experiments) on problems with known optima. Build and run with
 * `make test`. The model functions follow ModelFn: (n, t, y, dydt, pars). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "fit_experiments.h"

int jacobian_init(int n_y, double *y0, int n_pars, double *pars, int n_theta, int *theta_idx,
                  int *log_flags, int n_targets, int *target_kind, int *target_index,
                  double *target_scale, int n_ovr, int *ovr_idx, int n_exp, int *exp_kind,
                  int *exp_n_seg, int *exp_n_points, int n_seg_total, double *seg_t_end,
                  double *seg_ovr, int n_points, double *data_t, double *data_y, int n_derived,
                  int solver_id, double rtol, double atol, double ss_tol, double ss_t_max,
                  double target_residual_norm, int progress_interval);
int jacobian_chunk(int maxfev);
int fit_init(int n_y, double *y0, int n_pars, double *pars, int n_fit, int *fit_idx, int *log_flags,
             int n_targets, int *target_kind, int *target_index, double *target_scale, int n_ovr,
             int *ovr_idx, int n_exp, int *exp_kind, int *exp_n_seg, int *exp_n_points,
             int n_seg_total, double *seg_t_end, double *seg_ovr, int n_points, double *data_t,
             double *data_y, int n_derived, int solver_id, double rtol, double atol, double ss_tol,
             double ss_t_max, double target_residual_norm, int progress_interval);
int fit_chunk(int maxfev);
void fit_get_params(double *out);
void fit_free(void);
void set_model_fn(intptr_t fn);
void jacobian_get_params(double *out);
double jacobian_get_residual_norm(void);
void jacobian_free(void);
void jacobian_set_plain_fn(intptr_t fn);
void jacobian_set_augmented_fn(intptr_t fn);
void set_forward_model_fn(intptr_t fn);
void set_adjoint_fn(intptr_t fn);
int adjoint_vjp(int n_y, double *y0, int n_pars, double *pars, int n_theta, int *theta_idx,
                int n_targets, int *target_kind, int *target_index, int n_ovr, int *ovr_idx,
                int n_exp, int *exp_kind, int *exp_n_seg, int *exp_n_points, int n_seg_total,
                double *seg_t_end, double *seg_ovr, int n_points, double *data_t,
                double *cotangent, int n_derived, int solver_id, double rtol, double atol,
                double ss_tol, double ss_t_max, double *out_pred, double *out_grad,
                double *out_lambda0);
int adjoint_init(int n_y, double *y0, int n_pars, double *pars, int n_theta, int *theta_idx,
                 int n_targets, int *target_kind, int *target_index, double *target_scale,
                 int n_ovr, int *ovr_idx, int n_exp, int *exp_kind, int *exp_n_seg,
                 int *exp_n_points, int n_seg_total, double *seg_t_end, double *seg_ovr,
                 int n_points, double *data_t, double *data_y, int n_derived, int solver_id,
                 double rtol, double atol, double ss_tol, double ss_t_max, double lr,
                 double beta1, double beta2, double eps, double target_residual_norm,
                 double grad_norm_tol, int plateau_patience, double plateau_min_delta,
                 int progress_interval);
int adjoint_chunk(int maxIterations);
double adjoint_get_residual_norm(void);
void adjoint_get_params(double *out);
void adjoint_free(void);
void set_derived_fn(intptr_t fn);
void set_derived_grad_fn(intptr_t fn);
double fit_get_residual_norm(void);
int fit_get_nfev(void);

static int failures;

static void check(const char *name, int ok, double err, double limit) {
    ok = ok && err <= limit;
    printf("%-40s %s  (err %.2e, limit %.0e)\n", name, ok ? "ok  " : "FAIL", err, limit);
    if (!ok) failures++;
}

/* x' = -k x */
static void decay(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)t;
    dy[0] = -p[0] * y[0];
}
/* Augmented with s = dx/dk: s' = -k s - x */
static void decay_aug(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)t;
    dy[0] = -p[0] * y[0];
    dy[1] = -p[0] * y[1] - y[0];
}

/* A <-> B with kf, kr; steady state A* = kr / (kf + kr) * (A0 + B0) */
static void isomer(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)t;
    double v = p[0] * y[0] - p[1] * y[1];
    dy[0] = -v;
    dy[1] = v;
}
/* Augmented: y = [A, B, dA/dkf, dB/dkf, dA/dkr, dB/dkr] (parameter-major, as the driver
 * reads sens[col * n_y + state]) */
static void isomer_aug(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)t;
    double v = p[0] * y[0] - p[1] * y[1];
    dy[0] = -v;
    dy[1] = v;
    for (int j = 0; j < 2; j++) {
        const double *s = y + 2 + 2 * j;
        double dv = p[0] * s[0] - p[1] * s[1] + (j == 0 ? y[0] : -y[1]);
        dy[2 + 2 * j] = -dv;
        dy[3 + 2 * j] = dv;
    }
}

/* Adjoint of decay: dlambda/dt = k lambda, -lambda^T df/dk = lambda x */
static void decay_adj(int n, double t, double *y, double *lam, double *p, double *dlam,
                      double *dtheta) {
    (void)n; (void)t;
    dlam[0] = p[0] * lam[0];
    dtheta[0] = lam[0] * y[0];
}

/* Derived target d = k x, gradient [dd/dx, dd/dk] */
static void decay_derived(int n, double t, double *y, double *out, double *p) {
    (void)n; (void)t;
    out[0] = p[0] * y[0];
}
static void decay_derived_grad(int n, double t, double *y, double *out, double *p) {
    (void)n; (void)t;
    out[0] = p[0];
    out[1] = y[0];
}

/* Adjoint of isomer, theta = (kf, kr) */
static void isomer_adj(int n, double t, double *y, double *lam, double *p, double *dlam,
                       double *dtheta) {
    (void)n; (void)t;
    /* f = (-v, v), v = kf A - kr B */
    double d = lam[1] - lam[0];
    dlam[0] = -p[0] * d;
    dlam[1] = p[1] * d;
    dtheta[0] = -d * y[0];
    dtheta[1] = d * y[1];
}

/* sum_j w_j x(t_j) and k x(t_j) for x = 2 exp(-k t), with every solver */
static void test_vjp_time_course(const char *name, int solver) {
    const double k = 0.7, x0 = 2.0;
    double t[4] = {0.5, 1.0, 2.0, 4.0}, w[8], pred[8], grad[1], lam0[1];
    double exact_k = 0.0, exact_x0 = 0.0;
    for (int j = 0; j < 4; j++) {
        double e = exp(-k * t[j]), wx = 1.0 + j, wd = 0.5 - j;
        w[j] = wx;     /* target 0: x */
        w[4 + j] = wd; /* target 1: d = k x */
        exact_k += wx * (-x0 * t[j] * e) + wd * (x0 * e - k * x0 * t[j] * e);
        exact_x0 += wx * e + wd * k * e;
    }
    double y0[1] = {x0}, pars[1] = {k}, seg_t_end[1] = {4.0};
    int theta[1] = {0}, kind[2] = {TARGET_STATE, TARGET_DERIVED}, idx[2] = {0, 0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {4};
    set_forward_model_fn((intptr_t)decay);
    set_adjoint_fn((intptr_t)decay_adj);
    set_derived_fn((intptr_t)decay_derived);
    set_derived_grad_fn((intptr_t)decay_derived_grad);
    int rc = adjoint_vjp(1, y0, 1, pars, 1, theta, 2, kind, idx, 0, NULL, 1, exp_kind, n_seg,
                         n_pts, 1, seg_t_end, NULL, 4, t, w, 1, solver, 1e-10, 1e-12, 1e-10, 1e6,
                         pred, grad, lam0);
    double pred_err = fabs(pred[3] - x0 * exp(-k * 4.0)) + fabs(pred[7] - k * x0 * exp(-k * 4.0));
    double err = fabs(grad[0] - exact_k) / fabs(exact_k);
    err = fmax(err, fabs(lam0[0] - exact_x0) / fabs(exact_x0));
    check(name, rc == 0 && pred_err < 1e-7, err, 1e-6);
}

static void test_vjp_steady_state(void) {
    /* A* = kr / (kf + kr) (A0 + B0), so d A* / d kf = -kr / (kf + kr)^2 */
    double y0[2] = {1.0, 0.0}, pars[2] = {2.0, 1.0}, data_t[1] = {0.0}, seg_t_end[1] = {0.0};
    double w[1] = {1.0}, pred[1], grad[2], lam0[2];
    int theta[2] = {0, 1}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_STEADY_STATE}, n_seg[1] = {1}, n_pts[1] = {1};
    set_forward_model_fn((intptr_t)isomer);
    set_adjoint_fn((intptr_t)isomer_adj);
    int rc = adjoint_vjp(2, y0, 2, pars, 2, theta, 1, kind, idx, 0, NULL, 1, exp_kind, n_seg,
                         n_pts, 1, seg_t_end, NULL, 1, data_t, w, 0, SOLVER_RADAU5, 1e-10, 1e-12,
                         1e-12, 1e6, pred, grad, lam0);
    double err = fabs(grad[0] + 1.0 / 9.0) * 9.0;
    err = fmax(err, fabs(grad[1] - 2.0 / 9.0) * 4.5);
    /* lambda(0): d A* / d A0 = d A* / d B0 = kr / (kf + kr) */
    err = fmax(err, fabs(lam0[0] - 1.0 / 3.0) * 3.0 + fabs(lam0[1] - 1.0 / 3.0) * 3.0);
    check("adjoint vjp steady state", rc == 0, err, 1e-6);
}

/* x' = -k x + u, u overridden per protocol segment; theta = k */
static void decay_input(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)t;
    dy[0] = -p[0] * y[0] + p[1];
}
static void decay_input_adj(int n, double t, double *y, double *lam, double *p, double *dlam,
                            double *dtheta) {
    (void)n; (void)t; (void)p;
    dlam[0] = p[0] * lam[0];
    dtheta[0] = lam[0] * y[0];
}

/* Protocol with an override, a data point on a segment boundary and a second
 * experiment: the gradient against central differences of the predictions.
 * Small cotangents (1e-6) check that accuracy doesn't depend on their size. */
static void test_vjp_protocol(void) {
    double y0[1] = {1.0}, t[6] = {0.5, 1.0, 1.5, 2.5, 1.0, 2.0};
    double w[6] = {1e-6, -2e-6, 1.5e-6, 1e-6, 0.5e-6, 1e-6};
    double seg_t_end[4] = {1.0, 2.0, 3.0, 2.0}, seg_ovr[4] = {0.0, 2.0, 0.5, NAN};
    int theta[1] = {0}, kind[1] = {TARGET_STATE}, idx[1] = {0}, ovr[1] = {1};
    int exp_kind[2] = {EXP_TIME_COURSE, EXP_TIME_COURSE}, n_seg[2] = {3, 1}, n_pts[2] = {4, 2};
    double pred[6], grad[1], lam0[2];
    set_forward_model_fn((intptr_t)decay_input);
    set_adjoint_fn((intptr_t)decay_input_adj);
    double pars[2] = {0.8, 0.3};
    int rc = adjoint_vjp(1, y0, 2, pars, 1, theta, 1, kind, idx, 1, ovr, 2, exp_kind, n_seg,
                         n_pts, 4, seg_t_end, seg_ovr, 6, t, w, 0, SOLVER_RADAU5, 1e-12, 1e-14,
                         1e-10, 1e6, pred, grad, lam0);
    double fd = 0.0, h = 1e-4;
    for (int sgn = -1; sgn <= 1; sgn += 2) {
        double p2[2] = {0.8 + sgn * h, 0.3}, g2[1], l2[2], pr[6];
        adjoint_vjp(1, y0, 2, p2, 1, theta, 1, kind, idx, 1, ovr, 2, exp_kind, n_seg, n_pts, 4,
                    seg_t_end, seg_ovr, 6, t, w, 0, SOLVER_RADAU5, 1e-12, 1e-14, 1e-10, 1e6, pr,
                    g2, l2);
        for (int j = 0; j < 6; j++) fd += sgn * w[j] * pr[j] / (2 * h);
    }
    check("adjoint vjp protocol + experiments", rc == 0, fabs(grad[0] - fd) / fabs(fd), 1e-6);
}

/* x' = x^2 blows up at t = 1/x0: the failed solve must not look like a setup error */
static void blowup(int n, double t, double *y, double *dy, double *p) {
    (void)n; (void)t; (void)p;
    dy[0] = y[0] * y[0];
}
static void blowup_adj(int n, double t, double *y, double *lam, double *p, double *dlam,
                       double *dtheta) {
    (void)n; (void)t; (void)p;
    dlam[0] = -2.0 * y[0] * lam[0];
    dtheta[0] = 0.0;
}

static void test_vjp_solve_failure(void) {
    double y0[1] = {1.0}, pars[1] = {0.0}, t[1] = {4.0}, w[1] = {1.0}, seg_t_end[1] = {4.0};
    double pred[1], grad[1], lam0[1];
    int theta[1] = {0}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {1};
    set_forward_model_fn((intptr_t)blowup);
    set_adjoint_fn((intptr_t)blowup_adj);
    int rc = adjoint_vjp(1, y0, 1, pars, 1, theta, 1, kind, idx, 0, NULL, 1, exp_kind, n_seg,
                         n_pts, 1, seg_t_end, NULL, 1, t, w, 0, SOLVER_RADAU5, 1e-8, 1e-10,
                         1e-10, 1e6, pred, grad, lam0);
    printf("%-40s %s  (rc %d)\n", "adjoint vjp solve failure code", rc <= -1000 ? "ok  " : "FAIL",
           rc);
    if (rc > -1000) failures++;
}

static void test_adam(void) {
    double k_true = 0.7, t[6], data[6];
    for (int j = 0; j < 6; j++) {
        t[j] = 0.5 * (j + 1);
        data[j] = 2.0 * exp(-k_true * t[j]);
    }
    double y0[1] = {2.0}, pars[1] = {1.2}, scale[1] = {1.0}, seg_t_end[1] = {3.0};
    int theta[1] = {0}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {6};
    set_forward_model_fn((intptr_t)decay);
    set_adjoint_fn((intptr_t)decay_adj);
    int rc = adjoint_init(1, y0, 1, pars, 1, theta, 1, kind, idx, scale, 0, NULL, 1, exp_kind,
                          n_seg, n_pts, 1, seg_t_end, NULL, 6, t, data, 0, SOLVER_RADAU5, 1e-10,
                          1e-12, 1e-10, 1e6, 0.02, 0.9, 0.999, 1e-8, -1.0, 1e-9, 0, 0.0, 0);
    double before = adjoint_get_residual_norm();
    if (rc == 0) adjoint_chunk(2000);
    double out[1];
    adjoint_get_params(out);
    check("adjoint adam time course", rc == 0 && adjoint_get_residual_norm() < before,
          fabs(out[0] - k_true) / k_true, 1e-3);
    adjoint_free();
}


static int run(int (*chunk)(int));

/* ------------------------------------------------------------------ */
/* Contracts from NonlinearSolve.jl (least_squares_tests,             */
/* nonfinite_objective), optimistix (test_nonfinite_input) and         */
/* SciMLSensitivity / SUNDIALS (adjoint references)                    */
/* ------------------------------------------------------------------ */

static void check_rc(const char *name, int ok, int rc) {
    printf("%-40s %s  (rc %d)\n", name, ok ? "ok  " : "FAIL", rc);
    if (!ok) failures++;
}

/* fit_init / jacobian_init on x' = -k x with one time course of 6 points, with one argument
 * spoiled at a time. */
static int init_decay(int lmder, double k0, int n_fit, int *fit_idx, double scale, int target_index,
                      int n_points) {
    double t[6] = {0.5, 1, 1.5, 2, 2.5, 3}, data[6], y0[1] = {2.0}, pars[2] = {k0, 1.0};
    for (int j = 0; j < 6; j++) data[j] = 2.0 * exp(-0.7 * t[j]);
    double sc[1] = {scale}, seg_t_end[1] = {3.0};
    int logf[2] = {0, 0}, kind[1] = {TARGET_STATE}, idx[1] = {target_index};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {n_points};
    set_model_fn((intptr_t)decay);
    jacobian_set_plain_fn((intptr_t)decay);
    jacobian_set_augmented_fn((intptr_t)decay_aug);
    int rc = (lmder ? jacobian_init : fit_init)(1, y0, 2, pars, n_fit, fit_idx, logf, 1, kind, idx, sc, 0, NULL,
                                                1, exp_kind, n_seg, n_pts, 1, seg_t_end, NULL, n_points, t, data,
                                                0, SOLVER_RADAU5, 1e-10, 1e-12, 1e-10, 1e6, -1.0, 0);
    if (lmder) jacobian_free();
    else fit_free();
    return rc;
}

static void test_init_validation(void) {
    for (int lmder = 0; lmder < 2; lmder++) {
        int one[1] = {0}, out_of_range[1] = {5}, dup[2] = {0, 0}, both[2] = {0, 1};
        char name[64];
        const char *d = lmder ? "jacobian_init" : "fit_init";
        snprintf(name, sizeof name, "%s: valid input", d);
        check_rc(name, init_decay(lmder, 1.5, 1, one, 1.0, 0, 6) == 0, 0);
        int rc = init_decay(lmder, 1.5, 1, one, 0.0, 0, 6);
        snprintf(name, sizeof name, "%s: zero target scale -> -3", d);
        check_rc(name, rc == -3, rc);
        rc = init_decay(lmder, 1.5, 1, one, NAN, 0, 6);
        snprintf(name, sizeof name, "%s: NaN target scale -> -3", d);
        check_rc(name, rc == -3, rc);
        rc = init_decay(lmder, 1.5, 1, one, 1.0, 3, 6);
        snprintf(name, sizeof name, "%s: target index past n_y -> -3", d);
        check_rc(name, rc == -3, rc);
        rc = init_decay(lmder, 1.5, 1, out_of_range, 1.0, 0, 6);
        snprintf(name, sizeof name, "%s: fit index out of range -> -6", d);
        check_rc(name, rc == -6, rc);
        rc = init_decay(lmder, 1.5, 2, dup, 1.0, 0, 6);
        snprintf(name, sizeof name, "%s: repeated fit index -> -6", d);
        check_rc(name, rc == -6, rc);
        rc = init_decay(lmder, 1.5, 2, both, 1.0, 0, 1);
        snprintf(name, sizeof name, "%s: 2 parameters, 1 residual -> -7", d);
        check_rc(name, rc == -7, rc);
        rc = init_decay(lmder, NAN, 1, one, 1.0, 0, 6);
        snprintf(name, sizeof name, "%s: NaN initial value -> -2", d);
        check_rc(name, rc == -2, rc);
    }
}

/* A budget smaller than one lmdif iteration (n + 2 evaluations) still runs one, reporting 5
 * (budget used); it used to return 0, "improper input", without evaluating anything. */
static void test_small_budget(void) {
    int one[1] = {0};
    double t[6] = {0.5, 1, 1.5, 2, 2.5, 3}, data[6], y0[1] = {2.0}, pars[1] = {1.5};
    for (int j = 0; j < 6; j++) data[j] = 2.0 * exp(-0.7 * t[j]);
    double sc[1] = {1.0}, seg_t_end[1] = {3.0};
    int logf[1] = {0}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {6};
    set_model_fn((intptr_t)decay);
    fit_init(1, y0, 1, pars, 1, one, logf, 1, kind, idx, sc, 0, NULL, 1, exp_kind, n_seg, n_pts, 1, seg_t_end,
             NULL, 6, t, data, 0, SOLVER_RADAU5, 1e-10, 1e-12, 1e-10, 1e6, -1.0, 0);
    int info = fit_chunk(1);
    check_rc("fit_chunk(1): evaluates, reports 5 (not 0)", info == 5 && fit_get_nfev() >= 3, info);
    fit_free();
}

/* A fit starting exactly at the optimum (zero residual) converges where it is. */
static void test_zero_residual_start(void) {
    int one[1] = {0};
    double t[6] = {0.5, 1, 1.5, 2, 2.5, 3}, data[6], y0[1] = {2.0}, pars[1] = {0.7};
    double sc[1] = {1.0}, seg_t_end[1] = {3.0};
    int logf[1] = {1}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {6};
    set_model_fn((intptr_t)decay);
    /* data from the same solver and tolerance, so the residual is ~0 */
    for (int j = 0; j < 6; j++) data[j] = 2.0 * exp(-0.7 * t[j]);
    fit_init(1, y0, 1, pars, 1, one, logf, 1, kind, idx, sc, 0, NULL, 1, exp_kind, n_seg, n_pts, 1, seg_t_end,
             NULL, 6, t, data, 0, SOLVER_RADAU5, 1e-12, 1e-14, 1e-10, 1e6, -1.0, 0);
    int info = run(fit_chunk);
    double out[1];
    fit_get_params(out);
    check("lmdif from the optimum stays there", info >= 1 && info <= 4, fabs(out[0] - 0.7) / 0.7, 1e-8);
    fit_free();
}

/* NonlinearSolveBase nonfinite_objective: a derived target that is NaN for part of the parameter
 * range must give a finite (penalised) residual, never NaN. d = sqrt(k - 1) x. */
static void sqrt_derived(int n, double t, double *y, double *out, double *p) {
    (void)n; (void)t;
    out[0] = sqrt(p[0] - 1.0) * y[0];
}
static void test_nonfinite_residual(void) {
    int one[1] = {0};
    double t[3] = {0.5, 1, 1.5}, data[3] = {1, 1, 1}, y0[1] = {2.0}, pars[1] = {0.5}; /* k < 1 */
    double sc[1] = {1.0}, seg_t_end[1] = {1.5};
    int logf[1] = {0}, kind[1] = {TARGET_DERIVED}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {3};
    set_model_fn((intptr_t)decay);
    set_derived_fn((intptr_t)sqrt_derived);
    int rc = fit_init(1, y0, 1, pars, 1, one, logf, 1, kind, idx, sc, 0, NULL, 1, exp_kind, n_seg, n_pts, 1,
                      seg_t_end, NULL, 3, t, data, 1, SOLVER_RADAU5, 1e-8, 1e-10, 1e-10, 1e6, -1.0, 0);
    double r = fit_get_residual_norm();
    printf("%-40s %s  (residual %.3g)\n", "NaN derived target -> finite penalty", rc == 0 && isfinite(r) && r > 100 ? "ok  " : "FAIL", r);
    if (!(rc == 0 && isfinite(r) && r > 100)) failures++;
    fit_free();
    set_derived_fn((intptr_t)0);
}

/* optimistix: a non-finite gradient must stop Adam without corrupting its parameters. */
static void nan_adj(int n, double t, double *y, double *lam, double *p, double *dlam, double *dtheta) {
    (void)n; (void)t; (void)y; (void)p;
    dlam[0] = p[0] * lam[0];
    dtheta[0] = NAN;
}
static void test_adam_nonfinite(void) {
    double t[3] = {0.5, 1, 1.5}, data[3], y0[1] = {2.0}, pars[1] = {1.2}, scale[1] = {1.0}, seg_t_end[1] = {1.5};
    for (int j = 0; j < 3; j++) data[j] = 2.0 * exp(-0.7 * t[j]);
    int theta[1] = {0}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {3};
    set_forward_model_fn((intptr_t)decay);
    set_adjoint_fn((intptr_t)nan_adj);
    int rc = adjoint_init(1, y0, 1, pars, 1, theta, 1, kind, idx, scale, 0, NULL, 1, exp_kind, n_seg, n_pts, 1,
                          seg_t_end, NULL, 3, t, data, 0, SOLVER_RADAU5, 1e-10, 1e-12, 1e-10, 1e6, 0.02, 0.9, 0.999,
                          1e-8, -1.0, 1e-9, 0, 0.0, 0);
    /* The first step runs inside adjoint_init: it must refuse (-6) instead of moving k to NaN. */
    double out[1] = {0};
    adjoint_get_params(out);
    check_rc("Adam NaN gradient: init refuses", rc == -6, rc);
    adjoint_free();
}

/* SUNDIALS cvsLotkaVolterra_ASA.out: p = [1.5, 1, 3, 1], u0 = [1, 1], tf = 10,
 * g = 1/2 |1 - u(tf)|^2. Golden dg/du0 and dg/dp from CVODES at rtol 1e-10, atol 1e-14. */
static void lv(int n, double t, double *u, double *du, double *p) {
    (void)n; (void)t;
    du[0] = p[0] * u[0] - p[1] * u[0] * u[1];
    du[1] = -p[2] * u[1] + p[3] * u[0] * u[1];
}
static void lv_adj(int n, double t, double *u, double *lam, double *p, double *dlam, double *dtheta) {
    (void)n; (void)t;
    /* dlam = -J^T lam, dtheta = -lam^T df/dp */
    const double J00 = p[0] - p[1] * u[1], J01 = -p[1] * u[0], J10 = p[3] * u[1], J11 = -p[2] + p[3] * u[0];
    dlam[0] = -(J00 * lam[0] + J10 * lam[1]);
    dlam[1] = -(J01 * lam[0] + J11 * lam[1]);
    dtheta[0] = -lam[0] * u[0];
    dtheta[1] = lam[0] * u[0] * u[1];
    dtheta[2] = lam[1] * u[1];
    dtheta[3] = -lam[1] * u[0] * u[1];
}
static void test_vjp_sundials_lotka_volterra(int solver, const char *name) {
    double y0[2] = {1, 1}, pars[4] = {1.5, 1, 3, 1}, t[1] = {10.0}, seg_t_end[1] = {10.0};
    int theta[4] = {0, 1, 2, 3}, kind[2] = {TARGET_STATE, TARGET_STATE}, idx[2] = {0, 1};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {1};
    double w[2] = {0, 0}, pred[2], grad[4], lam0[2];
    set_forward_model_fn((intptr_t)lv);
    set_adjoint_fn((intptr_t)lv_adj);
    adjoint_vjp(2, y0, 4, pars, 4, theta, 2, kind, idx, 0, NULL, 1, exp_kind, n_seg, n_pts, 1, seg_t_end, NULL, 1,
                t, w, 0, solver, 1e-10, 1e-14, 1e-10, 1e6, pred, grad, lam0);
    w[0] = pred[0] - 1.0; /* dg/du(tf) = u(tf) - 1 */
    w[1] = pred[1] - 1.0;
    int rc = adjoint_vjp(2, y0, 4, pars, 4, theta, 2, kind, idx, 0, NULL, 1, exp_kind, n_seg, n_pts, 1, seg_t_end,
                         NULL, 1, t, w, 0, solver, 1e-10, 1e-14, 1e-10, 1e6, pred, grad, lam0);
    const double u_ref[2] = {1.026344801589377, 0.9096910639243128};
    const double du0_ref[2] = {0.2995996056786226, -0.01415181416922305};
    const double dp_ref[4] = {0.6219617362820864, 0.06800142032161892, 0.1691765074712130, 0.2725607529639176};
    double err = 0;
    for (int i = 0; i < 2; i++) err = fmax(err, fabs(pred[i] - u_ref[i]) / fabs(u_ref[i]));
    for (int i = 0; i < 2; i++) err = fmax(err, fabs(lam0[i] - du0_ref[i]) / fabs(du0_ref[i]));
    for (int i = 0; i < 4; i++) err = fmax(err, fabs(grad[i] - dp_ref[i]) / fabs(dp_ref[i]));
    /* LSODA's forward solution is the least accurate at this tolerance. */
    check(name, rc == 0, err, solver == SOLVER_LSODA ? 1e-5 : 1e-6);
}

/* SciMLSensitivity Core6 steady_state.jl: f = [p1 + p2 u1, p3 u1 + p4 u2], p = [2, -2, 1, -4],
 * u* = [-p1/p2, p1 p3 / (p2 p4)] = [1, 1/4], whatever u0 is: d u* / d u0 = 0. */
static void ss_lin(int n, double t, double *u, double *du, double *p) {
    (void)n; (void)t;
    du[0] = p[0] + p[1] * u[0];
    du[1] = p[2] * u[0] + p[3] * u[1];
}
static void ss_lin_adj(int n, double t, double *u, double *lam, double *p, double *dlam, double *dtheta) {
    (void)n; (void)t;
    dlam[0] = -(p[1] * lam[0] + p[2] * lam[1]);
    dlam[1] = -(p[3] * lam[1]);
    dtheta[0] = -lam[0];
    dtheta[1] = -lam[0] * u[0];
    dtheta[2] = -lam[1] * u[0];
    dtheta[3] = -lam[1] * u[1];
}
static void test_vjp_steady_state_linear(void) {
    double y0[2] = {0, 0}, pars[4] = {2, -2, 1, -4}, data_t[1] = {0}, seg_t_end[1] = {0};
    double w[2] = {1.0, 1.0}, pred[2], grad[4], lam0[2];
    int theta[4] = {0, 1, 2, 3}, kind[2] = {TARGET_STATE, TARGET_STATE}, idx[2] = {0, 1};
    int exp_kind[1] = {EXP_STEADY_STATE}, n_seg[1] = {1}, n_pts[1] = {1};
    set_forward_model_fn((intptr_t)ss_lin);
    set_adjoint_fn((intptr_t)ss_lin_adj);
    int rc = adjoint_vjp(2, y0, 4, pars, 4, theta, 2, kind, idx, 0, NULL, 1, exp_kind, n_seg, n_pts, 1, seg_t_end,
                         NULL, 1, data_t, w, 0, SOLVER_RADAU5, 1e-10, 1e-12, 1e-12, 1e6, pred, grad, lam0);
    /* d(u1* + u2*)/dp = (0.5 + 0.125, 0.5 + 0.125, 0.25, 0.0625) */
    const double ref[4] = {0.625, 0.625, 0.25, 0.0625};
    double err = fabs(pred[0] - 1.0) + fabs(pred[1] - 0.25) * 4;
    for (int i = 0; i < 4; i++) err = fmax(err, fabs(grad[i] - ref[i]) / ref[i]);
    check("adjoint vjp steady state (analytic dg/dp)", rc == 0, err, 1e-6);
    check("adjoint vjp steady state: du*/du0 = 0", rc == 0, fabs(lam0[0]) + fabs(lam0[1]), 1e-7);
}

/* SciMLSensitivity physical_ode_regression: a falling mass u = [x, v], x' = v, v' = -g; loss
 * sum_t x(t) over t = 0, 0.05, ..., 2 (41 points, the first at t0). d/dg = -1/2 sum t^2 =
 * -27.675, and the unused mass parameter has a gradient of exactly 0. */
static void fall(int n, double t, double *u, double *du, double *p) {
    (void)n; (void)t; (void)p;
    du[0] = u[1];
    du[1] = -p[0];
}
static void fall_adj(int n, double t, double *u, double *lam, double *p, double *dlam, double *dtheta) {
    (void)n; (void)t; (void)u; (void)p;
    dlam[0] = 0.0;      /* -J^T lam with J = [[0, 1], [0, 0]] */
    dlam[1] = -lam[0];
    dtheta[0] = lam[1]; /* -lam^T df/dg, df/dg = (0, -1) */
    dtheta[1] = 0.0;    /* the mass doesn't appear */
}
static void test_vjp_falling_mass(int solver, const char *name) {
    double t[41], w[41], pred[41], grad[2], lam0[2], y0[2] = {1, 0}, pars[2] = {9.81, 1.0}, seg_t_end[1] = {2.0};
    for (int j = 0; j < 41; j++) {
        t[j] = 0.05 * j;
        w[j] = 1.0;
    }
    int theta[2] = {0, 1}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {41};
    set_forward_model_fn((intptr_t)fall);
    set_adjoint_fn((intptr_t)fall_adj);
    int rc = adjoint_vjp(2, y0, 2, pars, 2, theta, 1, kind, idx, 0, NULL, 1, exp_kind, n_seg, n_pts, 1, seg_t_end,
                         NULL, 41, t, w, 0, solver, 1e-12, 1e-14, 1e-10, 1e6, pred, grad, lam0);
    double err = fabs(grad[0] + 27.675) / 27.675;
    err = fmax(err, fabs(pred[0] - 1.0)); /* the point at t0 is y0 */
    check(name, rc == 0 && grad[1] == 0.0, err, 1e-8);
    /* d loss / d x0 = 41 points, d loss / d v0 = sum t = 41 */
    check("  ... and lambda(0) = (41, sum t)", rc == 0, fabs(lam0[0] - 41) / 41 + fabs(lam0[1] - 41) / 41, 1e-8);
}

static int run(int (*chunk)(int)) {
    int info = 0;
    for (int i = 0; i < 20 && (info == 0 || info == 5); i++) info = chunk(50);
    return info;
}

/* lmder (forward sensitivities) or lmdif (finite differences) with RADAU5 or LSODA */
static void test_time_course(const char *name, int lmder, int solver, int log_space) {
    double k_true = 0.7, t[6], data[6];
    for (int j = 0; j < 6; j++) {
        t[j] = 0.5 * (j + 1);
        data[j] = 2.0 * exp(-k_true * t[j]);
    }
    double y0[1] = {2.0}, pars[1] = {1.5}, scale[1] = {1.0}, seg_t_end[1] = {3.0};
    int theta[1] = {0}, logf[1] = {log_space}, kind[1] = {TARGET_STATE}, idx[1] = {0};
    int exp_kind[1] = {EXP_TIME_COURSE}, n_seg[1] = {1}, n_pts[1] = {6};
    int (*init)(int, double *, int, double *, int, int *, int *, int, int *, int *, double *, int,
                int *, int, int *, int *, int *, int, double *, double *, int, double *, double *,
                int, int, double, double, double, double, double, int) =
        lmder ? jacobian_init : fit_init;
    set_model_fn((intptr_t)decay);
    jacobian_set_plain_fn((intptr_t)decay);
    jacobian_set_augmented_fn((intptr_t)decay_aug);
    int rc = init(1, y0, 1, pars, 1, theta, logf, 1, kind, idx, scale, 0, NULL, 1, exp_kind, n_seg,
                  n_pts, 1, seg_t_end, NULL, 6, t, data, 0, solver, 1e-10, 1e-12, 1e-10, 1e6, -1.0, 0);
    int info = rc == 0 ? run(lmder ? jacobian_chunk : fit_chunk) : -1;
    double out[1];
    if (lmder) jacobian_get_params(out);
    else fit_get_params(out);
    /* lmdif's finite differences limit its precision */
    check(name, rc == 0 && info >= 1 && info <= 4, fabs(out[0] - k_true) / k_true, lmder ? 1e-8 : 1e-6);
    if (lmder) jacobian_free();
    else fit_free();
}

static void test_steady_state(void) {
    /* Steady state of A (target 0) and B (target 1) for kf = 2, kr = 1 */
    double y0[2] = {1.0, 0.0}, pars[2] = {1.0, 2.0}, scale[2] = {1.0, 1.0};
    double data_t[1] = {0.0}, data[2] = {1.0 / 3.0, 2.0 / 3.0}, seg_t_end[1] = {0.0};
    int theta[2] = {0, 1}, logf[2] = {1, 1}, kind[2] = {TARGET_STATE, TARGET_STATE}, idx[2] = {0, 1};
    int exp_kind[1] = {EXP_STEADY_STATE}, n_seg[1] = {1}, n_pts[1] = {1};
    jacobian_set_plain_fn((intptr_t)isomer);
    jacobian_set_augmented_fn((intptr_t)isomer_aug);
    int rc = jacobian_init(2, y0, 2, pars, 2, theta, logf, 2, kind, idx, scale, 0, NULL, 1, exp_kind,
                           n_seg, n_pts, 1, seg_t_end, NULL, 1, data_t, data, 0, SOLVER_RADAU5,
                           1e-10, 1e-12, 1e-10, 1e6, -1.0, 0);
    int info = rc == 0 ? run(jacobian_chunk) : -1;
    double out[2];
    jacobian_get_params(out);
    /* Only the ratio kf / kr is identifiable from a steady state */
    check("lmder steady state (ratio)", rc == 0 && info >= 1 && info <= 4,
          fabs(out[0] / out[1] - 2.0) / 2.0, 1e-6);
    check("lmder steady state (residual)", rc == 0, jacobian_get_residual_norm(), 1e-8);
    jacobian_free();
}

int main(void) {
    test_time_course("lmder radau5 time course (linear)", 1, SOLVER_RADAU5, 0);
    test_time_course("lmder radau5 time course (log)", 1, SOLVER_RADAU5, 1);
    test_time_course("lmder lsoda time course (log)", 1, SOLVER_LSODA, 1);
    test_time_course("lmdif radau5 time course (log)", 0, SOLVER_RADAU5, 1);
    test_time_course("lmdif lsoda time course (log)", 0, SOLVER_LSODA, 1);
    test_steady_state();
    test_vjp_time_course("adjoint vjp radau5 time course", SOLVER_RADAU5);
    test_vjp_time_course("adjoint vjp lsoda time course", SOLVER_LSODA);
    test_vjp_time_course("adjoint vjp dopri5 time course", SOLVER_DOPRI5);
    test_vjp_steady_state();
    test_vjp_protocol();
    test_vjp_solve_failure();
    test_adam();
    test_init_validation();
    test_small_budget();
    test_zero_residual_start();
    test_nonfinite_residual();
    test_adam_nonfinite();
    test_vjp_sundials_lotka_volterra(SOLVER_RADAU5, "adjoint vjp radau5 = CVODES LV ASA");
    test_vjp_sundials_lotka_volterra(SOLVER_LSODA, "adjoint vjp lsoda = CVODES LV ASA");
    test_vjp_steady_state_linear();
    test_vjp_falling_mass(SOLVER_RADAU5, "adjoint vjp falling mass radau5");
    test_vjp_falling_mass(SOLVER_DOP853, "adjoint vjp falling mass dop853");
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
