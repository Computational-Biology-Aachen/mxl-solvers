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
    printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
