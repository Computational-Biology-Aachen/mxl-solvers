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

static int failures;

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
static const double SEG_END[NSEG] = {2.0, 5.0};
static const double U[NSEG] = {0.0, 0.3};
/* t = 0 (initial), dense points, a segment boundary and the end (step ends) */
static const double T_POINTS[NPTS] = {0.0, 0.37, 1.1, 2.0, 2.71, 3.3, 5.0};
static const int KIND[NTGT] = {0, 0, 1};
static const int INDEX[NTGT] = {0, 2, 0};

static void fill_pars(const double *k, double *seg_pars) {
  for (int g = 0; g < NSEG; g++) {
    memcpy(seg_pars + g * NP, k, 4 * sizeof(double));
    seg_pars[g * NP + 4] = U[g];
  }
}

/* sum(cot * pred) for parameters k and initial state y0; *steps = tape length. */
static double loss(const double *k, const double *y0, const double *cot, double rtol, int *steps,
                   DaTape **keep) {
  double seg_pars[NSEG * NP], pred[NPTS * NTGT];
  fill_pars(k, seg_pars);
  DaTape *tape = NULL;
  int st = da_forward((intptr_t)model, (intptr_t)jac, (intptr_t)derived, DA_METHOD_RADAU5, N, y0,
                      NP, NSEG, SEG_END, seg_pars, NPTS, T_POINTS, NTGT, KIND, INDEX, 1, rtol,
                      rtol * 1e-2, 100000, DA_FORWARD_JAC, 0.0, pred, &tape);
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
  const double k[4] = {0.8, 2.0, 1.5, 0.4};
  const double y0[N] = {1.0, 0.2, 0.1};
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
  printf("radau5 discrete adjoint, rtol %.0e (%d steps): rel. error vs frozen-step FD %.2e "
         "(vs adaptive FD %.2e), replay vs forward %.2e\n",
         rtol, steps, err, err_adaptive, replay_err);
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
  const double k[4] = {0.8, 2.0, 1.5, 0.4};
  const double y0[N] = {1.0, 0.2, 0.1};
  double seg_pars[NSEG * NP], pred[NPTS * NTGT];
  fill_pars(k, seg_pars);
  DaTape *tape = NULL;
  int st = da_forward((intptr_t)model, (intptr_t)jac, (intptr_t)derived, DA_METHOD_RADAU5, N, y0,
                      NP, NSEG, SEG_END, seg_pars, NPTS, T_POINTS, NTGT, KIND, INDEX, 1, 1e-6,
                      1e-8, 100000, 0, 100.0, pred, &tape);
  printf("tape limit: status %d\n", st);
  if (st != DA_TAPE_TOO_LARGE || tape) {
    printf("  FAIL: expected DA_TAPE_TOO_LARGE\n");
    failures++;
  }
}

int main(void) {
  check_gradient(1e-3, 1e-5); /* limited by RADAU5's Newton tolerance (replay vs forward) */
  check_gradient(1e-6, 1e-7);
  check_gradient(1e-10, 1e-8);
  check_tape_limit();
  if (failures) {
    printf("%d FAILED\n", failures);
    return 1;
  }
  printf("all passed\n");
  return 0;
}
