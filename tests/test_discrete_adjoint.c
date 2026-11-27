/* Discrete adjoint (discrete_adjoint.c): the gradient is that of the computed solution with frozen
 * step sizes (as diffrax's), so it is checked against central differences of da_replay, which
 * re-solves the tape's steps at perturbed parameters, at loose and tight tolerances. Errors are
 * relative, max-norm over the gradient. Build and run with `make test`. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "discrete_adjoint.h"
#include "finite_jacobian.h"

static int failures;
static int g_method = DA_METHOD_RADAU5;
static double g_newton = 1.0; /* newton_scale */
static const char *METHOD_NAMES[] = {"radau5", "lsoda", "cvode_bdf", "cvode_adams"};

#define N 3
#define NP 5 /* k1..k4 and an input u (overridden per protocol segment) */
#define NT 4 /* theta = k1..k4 */

/* y0' = -k1 y0 + k2 y1 y2 + u, y1' = k1 y0 - k2 y1 y2 - k3 y1^2, y2' = k3 y1^2 - k4 y2 */
static void model(int n, double t, double *y, double *f, double *p) {
  (void)n;
  (void)t;
  f[0] = -p[0] * y[0] + p[1] * y[1] * y[2] + p[4];
  f[1] = p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] * y[1];
  f[2] = p[2] * y[1] * y[1] - p[3] * y[2];
}

static void jac(int n, double t, double *y, double *J, double *p) {
  (void)t;
#define JJ(i, j) J[(i) + (j) * n]
  JJ(0, 0) = -p[0];
  JJ(0, 1) = p[1] * y[2];
  JJ(0, 2) = p[1] * y[1];
  JJ(1, 0) = p[0];
  JJ(1, 1) = -p[1] * y[2] - 2 * p[2] * y[1];
  JJ(1, 2) = -p[1] * y[1];
  JJ(2, 0) = 0;
  JJ(2, 1) = 2 * p[2] * y[1];
  JJ(2, 2) = -p[3];
#undef JJ
}

static void adjoint(int n, double t, double *y, double *lam, double *p, double *dlam,
                    double *dtheta) {
  double J[N * N];
  jac(n, t, y, J, p);
  for (int j = 0; j < n; j++) {
    double v = 0;
    for (int i = 0; i < n; i++) v += J[i + j * n] * lam[i];
    dlam[j] = -v;
  }
  dtheta[0] = -(lam[0] * -y[0] + lam[1] * y[0]);
  dtheta[1] = -(lam[0] * y[1] * y[2] - lam[1] * y[1] * y[2]);
  dtheta[2] = -(-lam[1] * y[1] * y[1] + lam[2] * y[1] * y[1]);
  dtheta[3] = -(-lam[2] * y[2]);
}

/* d = y0 + 2 k4 y1 */
static void derived(int n, double t, double *y, double *out, double *p) {
  (void)n;
  (void)t;
  out[0] = y[0] + 2 * p[3] * y[1];
}

static void derived_grad(int n, double t, double *y, double *out, double *p) {
  (void)n;
  (void)t;
  out[0] = 1;
  out[1] = 2 * p[3];
  out[2] = 0;
  out[3] = 0;
  out[4] = 0;
  out[5] = 0;
  out[6] = 2 * y[1];
}

#define NSEG 2
#define NPTS 7
#define NTGT 3
/* Two problems with the same equations: a mild one, and Robertson's stiff kinetics
 * (k1 = 0.04, k2 = 1e4, k3 = 3e7) with a slow decay of y2 */
typedef struct {
  const char *name;
  double seg_end[NSEG], u[NSEG], t_points[NPTS], k[4], y0[N];
} Problem;
static const Problem PROBLEMS[2] = {
    /* t = 0 (initial), dense points, a segment boundary and the end (step ends) */
    {"mild", {2.0, 5.0}, {0.0, 0.3}, {0.0, 0.37, 1.1, 2.0, 2.71, 3.3, 5.0}, {0.8, 2.0, 1.5, 0.4},
     {1.0, 0.2, 0.1}},
    {"stiff", {10.0, 400.0}, {0.0, 1e-3}, {0.0, 0.3, 4.0, 10.0, 30.0, 70.0, 400.0},
     {0.04, 1e4, 3e7, 1e-2}, {1.0, 0.0, 0.0}},
};
static const Problem *P = &PROBLEMS[0];
static const int KIND[NTGT] = {0, 0, 1};
static const int INDEX[NTGT] = {0, 2, 0};

static void fill_pars(const double *k, double *seg_pars) {
  for (int g = 0; g < NSEG; g++) {
    memcpy(seg_pars + g * NP, k, 4 * sizeof(double));
    seg_pars[g * NP + 4] = P->u[g];
  }
}

/* sum(cot * pred) for parameters k and initial state y0; *steps = tape length. */
static double loss(const double *k, const double *y0, const double *cot, double rtol, int *steps,
                   DaTape **keep) {
  double seg_pars[NSEG * NP], pred[NPTS * NTGT];
  fill_pars(k, seg_pars);
  DaTape *tape = NULL;
  int st = da_forward((intptr_t)model, (intptr_t)jac, (intptr_t)derived, g_method, N, y0,
                      NP, NSEG, P->seg_end, seg_pars, NPTS, P->t_points, NTGT, KIND, INDEX, 1, rtol,
                      rtol * 1e-2, 100000, DA_FORWARD_JAC, g_newton, 0.0, pred, &tape);
  if (st != 0) {
    printf("da_forward failed: %d\n", st);
    failures++;
    return NAN;
  }
  *steps = da_tape_steps(tape);
  double v = 0;
  for (int i = 0; i < NPTS * NTGT; i++) v += cot[i] * pred[i];
  if (keep) {
    *keep = tape;
  } else {
    da_free(tape);
  }
  return v;
}

static double dot(const double *a, const double *b, int m) {
  double v = 0;
  for (int i = 0; i < m; i++) v += a[i] * b[i];
  return v;
}

/* sum(cot * pred) of the replayed (frozen-step) solve. */
static double replay_loss(const DaTape *tape, const double *k, const double *y0,
                          const double *cot) {
  double seg_pars[NSEG * NP], pred[NPTS * NTGT];
  fill_pars(k, seg_pars);
  int st = da_replay(tape, seg_pars, y0, pred);
  if (st != 0) {
    printf("da_replay failed: %d\n", st);
    failures++;
    return NAN;
  }
  return dot(cot, pred, NPTS * NTGT);
}

static void check_gradient(double rtol, double limit) {
  const double *k = P->k, *y0 = P->y0;
  double cot[NPTS * NTGT];
  srand(1);
  for (int i = 0; i < NPTS * NTGT; i++) cot[i] = 2.0 * rand() / RAND_MAX - 1.0;

  int steps = 0;
  DaTape *tape = NULL;
  double l0 = loss(k, y0, cot, rtol, &steps, &tape);
  if (!tape) return;
  double grad[NT], lam0[N];
  int st = da_backward(tape, (intptr_t)adjoint, (intptr_t)derived_grad, NT, cot, grad, lam0);
  if (st != 0) {
    printf("da_backward failed: %d\n", st);
    failures++;
    da_free(tape);
    return;
  }

  /* (a) the replay reproduces the recorded solve (up to RADAU5's Newton tolerance) */
  double replay_err = fabs(replay_loss(tape, k, y0, cot) - l0) / fmax(fabs(l0), 1e-300);

  /* (c) central differences of the frozen-step solve; for information also of the
   * adaptive one, whose step sizes move with the parameters */
  double fd[NT + N], fd_adaptive[NT + N], got[NT + N], scale = 0, err = 0, err_adaptive = 0;
  for (int q = 0; q < NT + N; q++) {
    double kp[4], km[4], yp[N], ym[N];
    memcpy(kp, k, sizeof kp);
    memcpy(km, k, sizeof km);
    memcpy(yp, y0, sizeof yp);
    memcpy(ym, y0, sizeof ym);
    double *v = q < NT ? &kp[q] : &yp[q - NT];
    double *w = q < NT ? &km[q] : &ym[q - NT];
    double h = 1e-6 * fmax(1.0, fabs(*v));
    *v += h;
    *w -= h;
    fd[q] = (replay_loss(tape, kp, yp, cot) - replay_loss(tape, km, ym, cot)) / (2 * h);
    int s1 = 0, s2 = 0;
    fd_adaptive[q] =
        (loss(kp, yp, cot, rtol, &s1, NULL) - loss(km, ym, cot, rtol, &s2, NULL)) / (2 * h);
    got[q] = q < NT ? grad[q] : lam0[q - NT];
    scale = fmax(scale, fabs(fd[q]));
  }
  for (int q = 0; q < NT + N; q++) {
    err = fmax(err, fabs(got[q] - fd[q]));
    err_adaptive = fmax(err_adaptive, fabs(got[q] - fd_adaptive[q]));
  }
  err /= scale;
  err_adaptive /= scale;
  printf("%s %s, newton x%.0e, rtol %.0e (%d steps): rel. error vs frozen-step FD %.2e "
         "(vs adaptive FD %.2e), replay vs forward %.2e\n",
         METHOD_NAMES[g_method], P->name, g_newton, rtol, steps, err, err_adaptive, replay_err);
  if (g_method != DA_METHOD_RADAU5 && rtol == 1e-6 && g_newton == 1.0) {
    int c[15];
    da_tape_op_counts(tape, c, 15);
    printf("  ops: init %d deriv %d scale %d predict %d retract %d correct %d update %d set_acor %d "
           "set_zero %d set_col %d axpy %d\n",
           c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]);
  }
  /* The gradient is exact for converged stage/corrector equations, the forward solve stops
   * iterating earlier: allow the difference that makes (replay vs forward). With the solvers'
   * own Newton tolerance that estimate is platform-dependent (2.7e-6 vs 2.8e-8 for LSODA stiff
   * at rtol 1e-6 on macOS), so allow 10 rtol too; the 1000x tighter Newton runs check exactness */
  limit = fmax(limit, 10 * replay_err);
  if (g_newton == 1.0) limit = fmax(limit, 10 * rtol);
  if (!(err < limit)) {
    printf("  FAIL: above %.0e\n", limit);
    for (int q = 0; q < NT + N; q++) printf("    %d: %.10e vs %.10e\n", q, got[q], fd[q]);
    failures++;
  }
  if (!(replay_err < 10 * rtol)) {
    printf("  FAIL: the replay doesn't reproduce the forward solve\n");
    failures++;
  }
  da_free(tape);
}

/* The tape guard: a limit below one step's record must fail cleanly. */
static void check_tape_limit(void) {
  const double *k = P->k, *y0 = P->y0;
  double seg_pars[NSEG * NP], pred[NPTS * NTGT];
  fill_pars(k, seg_pars);
  DaTape *tape = NULL;
  int st = da_forward((intptr_t)model, (intptr_t)jac, (intptr_t)derived, g_method, N, y0,
                      NP, NSEG, P->seg_end, seg_pars, NPTS, P->t_points, NTGT, KIND, INDEX, 1, 1e-6,
                      1e-8, 100000, 0, 1.0, 100.0, pred, &tape);
  printf("tape limit: status %d\n", st);
  if (st != DA_TAPE_TOO_LARGE || tape) {
    printf("  FAIL: expected DA_TAPE_TOO_LARGE\n");
    failures++;
  }
}

/* A Jacobian that can't evaluate column 1 (as a symbolic derivative at an overflow):
 * finite_jacobian replaces it by differences, so the gradient matches the exact one. */
static void jac_nan(int n, double t, double *y, double *J, double *p) {
  jac(n, t, y, J, p);
  J[1 * n + 0] = NAN;
}

static void gradient_with(intptr_t jacobian, double *grad) {
  double seg_pars[NSEG * NP], pred[NPTS * NTGT], cot[NPTS * NTGT], lam0[N];
  for (int i = 0; i < NPTS * NTGT; i++) cot[i] = 1.0;
  fill_pars(P->k, seg_pars);
  DaTape *tape = NULL;
  int st = da_forward((intptr_t)model, jacobian, (intptr_t)derived, g_method, N, P->y0, NP, NSEG,
                      P->seg_end, seg_pars, NPTS, P->t_points, NTGT, KIND, INDEX, 1, 1e-8, 1e-10,
                      100000, DA_FORWARD_JAC, 1.0, 0.0, pred, &tape);
  if (st == 0) st = da_backward(tape, (intptr_t)adjoint, (intptr_t)derived_grad, NT, cot, grad, lam0);
  if (st != 0) {
    for (int q = 0; q < NT; q++) grad[q] = NAN;
  }
  da_free(tape);
}

static void check_nan_jacobian(void) {
  double J[N * N], expected[N * N];
  double y[N] = {0.7, 0.3, 0.2}, k[NP] = {0.8, 2.0, 1.5, 0.4, 0.1};
  jac(N, 0.0, y, expected, k);
  int replaced = finite_jacobian(model, jac_nan, N, 0.0, y, J, k);
  double err = 0, scale = 0;
  for (int i = 0; i < N * N; i++) {
    double d = fabs(J[i] - expected[i]);
    err = isnan(d) ? INFINITY : fmax(err, d); /* fmax alone would drop a NaN */
    scale = fmax(scale, fabs(expected[i]));
  }
  err /= scale;
  printf("finite_jacobian: %d column replaced, rel. error %.2e\n", replaced, err);
  if (replaced != 1 || !(err < 1e-8)) {
    printf("  FAIL\n");
    failures++;
  }
  P = &PROBLEMS[0];
  for (g_method = DA_METHOD_RADAU5; g_method <= DA_METHOD_CVODE_BDF; g_method++) {
    double exact[NT], fallback[NT];
    gradient_with((intptr_t)jac, exact);
    gradient_with((intptr_t)jac_nan, fallback);
    double e = 0, s = 0;
    for (int q = 0; q < NT; q++) {
      double d = fabs(fallback[q] - exact[q]);
      e = isnan(d) ? INFINITY : fmax(e, d);
      s = fmax(s, fabs(exact[q]));
    }
    e /= s;
    printf("%s with a NaN Jacobian column: rel. gradient difference %.2e\n",
           METHOD_NAMES[g_method], e);
    if (!(e < 1e-6)) {
      printf("  FAIL\n");
      failures++;
    }
  }
}

int main(void) {
  for (g_method = DA_METHOD_RADAU5; g_method <= DA_METHOD_CVODE_ADAMS; g_method++) {
    /* The solvers' own Newton tolerance, and one 1000x tighter, where the gradient agrees
     * with the frozen-step differences to the limits without the replay allowance */
    const double newton[2] = {1.0, 1e-3};
    for (int k = 0; k < 2; k++) {
      g_newton = newton[k];
      for (int p = 0; p < 2; p++) {
        P = &PROBLEMS[p];
        /* Adams with fixed-point iteration can't integrate stiff problems */
        if (g_method == DA_METHOD_CVODE_ADAMS && p == 1) continue;
        check_gradient(1e-3, 1e-5);
        check_gradient(1e-6, 1e-7);
        check_gradient(1e-10, 1e-8);
      }
    }
    check_tape_limit();
  }
  check_nan_jacobian();
  if (failures) {
    printf("%d FAILED\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
