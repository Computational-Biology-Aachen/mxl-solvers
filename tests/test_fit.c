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
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
