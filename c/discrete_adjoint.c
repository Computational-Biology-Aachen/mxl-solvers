/*
 * discrete_adjoint.c — see discrete_adjoint.h.
 *
 * RADAU5 (Radau IIA, 3 stages, order 5)
 * -------------------------------------
 * A step from (x, y) with size h solves the collocation equations
 *     Z_i = h sum_j a_ij f(x + c_j h, y + Z_j),   i = 1..3,
 * for the stage increments Z (Newton, simplified), then y_new = y + Z_3. The
 * tape keeps x, h, y and Z of every accepted step. Z isn't passed to SOLOUT,
 * but its CONT array (the collocation polynomial, radau5.c just before the
 * SOLOUT call) is linear in Z and y_new, so Z is recovered from it and
 * radau5.c stays untouched.
 *
 * Backward, for a step with cotangent ybar_new on y_new and cotangents w on
 * dense outputs inside the step (CONTR5 is linear in y and Z):
 *     Zbar = e_3 ybar_new + (dense contributions)
 *     solve (I - h diag(J_j^T) (A^T x I)) mu = Zbar    (3n x 3n, J_j = df/dy at stage j)
 *     nu_j = h sum_i a_ij mu_i
 *     ybar = ybar_new + (dense contributions) + sum_j J_j^T nu_j
 *     thetabar += sum_j (df/dtheta)_j^T nu_j
 * i.e. the implicit function theorem on the converged stage equations, with
 * the step sizes frozen. The products with J_j^T and df/dtheta come from the
 * generated adjoint function, J_j itself (for the stage matrix) from the
 * generated Jacobian.
 */
#include "discrete_adjoint.h"

#include "f2c.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef void (*DaModelFn)(int, double, double *, double *, double *);
typedef void (*DaAdjointFn)(int, double, double *, double *, double *, double *, double *);

extern int radau5_(integer *n, U_fp fcn, doublereal *x, doublereal *y, doublereal *xend,
                   doublereal *h, doublereal *rtol, doublereal *atol, integer *itol, U_fp jac,
                   integer *ijac, integer *mljac, integer *mujac, U_fp mas, integer *imas,
                   integer *mlmas, integer *mumas, U_fp solout, integer *iout,
                   doublereal *work, integer *lwork, integer *iwork, integer *liwork,
                   doublereal *rpar, integer *ipar, integer *idid);
extern double radau5_jac_growth;
extern int dec_(integer *n, integer *ndim, doublereal *a, integer *ip, integer *ier);
extern int sol_(integer *n, integer *ndim, doublereal *a, doublereal *b, integer *ip);

/* ------------------------------------------------------------------ */
/* Radau IIA coefficients                                              */
/* ------------------------------------------------------------------ */
#define SQ6 2.449489742783178098197284074705891391965947480656670128432692567
static const double RC[3] = {(4.0 - SQ6) / 10.0, (4.0 + SQ6) / 10.0, 1.0};
static const double RA[3][3] = {
    {(88.0 - 7.0 * SQ6) / 360.0, (296.0 - 169.0 * SQ6) / 1800.0, (-2.0 + 3.0 * SQ6) / 225.0},
    {(296.0 + 169.0 * SQ6) / 1800.0, (88.0 + 7.0 * SQ6) / 360.0, (-2.0 - 3.0 * SQ6) / 225.0},
    {(16.0 - SQ6) / 36.0, (16.0 + SQ6) / 36.0, 1.0 / 9.0}};

/* ------------------------------------------------------------------ */
/* Tape                                                                */
/* ------------------------------------------------------------------ */

/* Where a data point's state comes from. */
enum { PT_INITIAL = 0, PT_STEP_END = 1, PT_DENSE = 2 };

typedef struct {
  int step; /* tape step (PT_STEP_END, PT_DENSE) */
  int kind;
  int seg;
  double t;
} DaPoint;

struct DaTape {
  int method;
  int n, n_pars, n_seg;
  DaModelFn model, jac, derived;
  double *seg_pars; /* n_seg * n_pars */
  double *y0;
  int n_targets, n_derived;
  int *target_kind, *target_index;

  /* Steps: rec doubles each. RADAU5: x, h, y (n), Z (3n). */
  int rec;
  int n_steps, cap_steps;
  double *steps;
  int *step_seg;

  int n_points;
  DaPoint *points;
};

static double tape_bytes(const DaTape *T, int steps) {
  return (double)steps * ((double)T->rec * sizeof(double) + sizeof(int));
}

int da_tape_steps(const DaTape *tape) { return tape ? tape->n_steps : 0; }
double da_tape_bytes(const DaTape *tape) {
  return tape ? tape_bytes(tape, tape->cap_steps) +
                    (double)tape->n_points * sizeof(DaPoint)
              : 0.0;
}

void da_free(DaTape *T) {
  if (!T) return;
  free(T->seg_pars);
  free(T->y0);
  free(T->target_kind);
  free(T->target_index);
  free(T->steps);
  free(T->step_seg);
  free(T->points);
  free(T);
}

/* A new step record, NULL if out of memory or over the limit. */
static double *push_step(DaTape *T, int seg, double max_bytes, int *status) {
  if (T->n_steps == T->cap_steps) {
    int cap = T->cap_steps > 0 ? 2 * T->cap_steps : 256;
    if (tape_bytes(T, cap) > max_bytes) {
      cap = T->n_steps + 1;
      if (tape_bytes(T, cap) > max_bytes) {
        *status = DA_TAPE_TOO_LARGE;
        return NULL;
      }
    }
    double *s = (double *)realloc(T->steps, (size_t)cap * (size_t)T->rec * sizeof(double));
    if (!s) {
      *status = DA_NO_MEMORY;
      return NULL;
    }
    T->steps = s;
    int *g = (int *)realloc(T->step_seg, (size_t)cap * sizeof(int));
    if (!g) {
      *status = DA_NO_MEMORY;
      return NULL;
    }
    T->step_seg = g;
    T->cap_steps = cap;
  }
  T->step_seg[T->n_steps] = seg;
  return T->steps + (size_t)(T->n_steps++) * (size_t)T->rec;
}

/* ------------------------------------------------------------------ */
/* RADAU5 forward                                                      */
/* ------------------------------------------------------------------ */

/* The solve in progress (RADAU5 isn't reentrant anyway). */
static struct {
  DaTape *T;
  double *pars;
  int seg;
  const double *t_points;
  int n_points, next; /* next data point to record */
  double t_start;     /* segment start */
  double *y_prev;     /* state at the start of the step SOLOUT reports */
  double max_bytes;
  int status;
} R;

static int rd_fcn(integer *n, doublereal *x, doublereal *y, doublereal *f, doublereal *rpar,
                  integer *ipar) {
  (void)ipar;
  R.T->model((int)*n, *x, y, f, rpar);
  return 0;
}

static int rd_jac(integer *n, doublereal *x, doublereal *y, doublereal *dfy, integer *ldfy,
                  doublereal *rpar, integer *ipar) {
  (void)ipar;
  (void)ldfy; /* full Jacobian: ldfy == n */
  R.T->jac((int)*n, *x, y, dfy, rpar);
  return 0;
}

static int rd_dummy(void) { return 0; }

static void add_point(DaTape *T, int step, int kind, double t) {
  DaPoint *p = T->points + R.next++;
  p->step = step;
  p->kind = kind;
  p->seg = R.seg;
  p->t = t;
}

/* Hairer's CONT holds y_new, then the divided differences of the collocation
 * polynomial; invert radau5.c's construction for Z. */
static void z_from_cont(int n, const double *y_old, const double *cont, double *z) {
  const double c1 = RC[0], c2 = RC[1], c1m1 = c1 - 1.0, c2m1 = c2 - 1.0, c1mc2 = c1 - c2;
  for (int i = 0; i < n; i++) {
    double z3 = cont[i] - y_old[i];
    double z2 = z3 + c2m1 * cont[i + n];
    double ak = cont[i + n] + c1m1 * cont[i + 2 * n];
    double z1 = z2 + c1mc2 * ak;
    z[i] = z1;
    z[i + n] = z2;
    z[i + 2 * n] = z3;
  }
}

static int rd_solout(integer *nr, doublereal *xold, doublereal *x, doublereal *y,
                     doublereal *cont, integer *lrc, integer *n_, doublereal *rpar,
                     integer *ipar, integer *irtrn) {
  (void)lrc;
  (void)rpar;
  (void)ipar;
  DaTape *T = R.T;
  const int n = (int)*n_;
  *irtrn = 1;
  if (*nr < 2) {
    /* The initial point: only t = 0 can be recorded here, a segment's
     * start point belongs to the previous segment. */
    while (R.next < R.n_points && R.t_points[R.next] <= *x && R.seg == 0) {
      add_point(T, -1, PT_INITIAL, R.t_points[R.next]);
    }
    memcpy(R.y_prev, y, (size_t)n * sizeof(double));
    return 0;
  }
  double *s = push_step(T, R.seg, R.max_bytes, &R.status);
  if (!s) {
    *irtrn = -1;
    return 0;
  }
  s[0] = *xold;
  s[1] = *x - *xold;
  memcpy(s + 2, R.y_prev, (size_t)n * sizeof(double));
  z_from_cont(n, R.y_prev, cont, s + 2 + n);
  const int step = T->n_steps - 1;
  while (R.next < R.n_points && R.t_points[R.next] <= *x) {
    double t = R.t_points[R.next];
    add_point(T, step, t == *x ? PT_STEP_END : PT_DENSE, t);
  }
  memcpy(R.y_prev, y, (size_t)n * sizeof(double));
  return 0;
}

/* CONTR5 from a tape step: y(t) = y_new + s (C1 + (s - c2m1)(C2 + (s - c1m1) C3)),
 * s = (t - x_new) / h, with the CONT coefficients C rebuilt from Z. */
static void rd_dense(int n, const double *rec, double t, double *out) {
  const double c1 = RC[0], c2 = RC[1], c1m1 = c1 - 1.0, c2m1 = c2 - 1.0, c1mc2 = c1 - c2;
  const double x = rec[0], h = rec[1];
  const double *y = rec + 2, *z1 = rec + 2 + n, *z2 = z1 + n, *z3 = z2 + n;
  const double s = (t - (x + h)) / h;
  for (int i = 0; i < n; i++) {
    double k1 = (z2[i] - z3[i]) / c2m1;
    double ak = (z1[i] - z2[i]) / c1mc2;
    double a3 = (ak - z1[i] / c1) / c2;
    double k2 = (ak - k1) / c1m1;
    double k3 = k2 - a3;
    out[i] = (y[i] + z3[i]) + s * (k1 + (s - c2m1) * (k2 + (s - c1m1) * k3));
  }
}

/* Transposed: adds the cotangent w of rd_dense(t) to ybar_old and zbar. */
static void rd_dense_vjp(int n, const double *rec, double t, const double *w, double *ybar,
                         double *zbar) {
  const double c1 = RC[0], c2 = RC[1], c1m1 = c1 - 1.0, c2m1 = c2 - 1.0, c1mc2 = c1 - c2;
  const double x = rec[0], h = rec[1];
  const double s = (t - (x + h)) / h;
  const double b1 = s, b2 = s * (s - c2m1), b3 = b2 * (s - c1m1);
  double *z1b = zbar, *z2b = zbar + n, *z3b = zbar + 2 * n;
  for (int i = 0; i < n; i++) {
    double k1b = b1 * w[i], k2b = b2 * w[i], k3b = b3 * w[i];
    /* k3 = k2 - a3 */
    k2b += k3b;
    double a3b = -k3b;
    /* a3 = (ak - z1 / c1) / c2 */
    double akb = a3b / c2;
    z1b[i] += -a3b / (c1 * c2);
    /* k2 = (ak - k1) / c1m1 */
    akb += k2b / c1m1;
    k1b += -k2b / c1m1;
    /* ak = (z1 - z2) / c1mc2 */
    z1b[i] += akb / c1mc2;
    z2b[i] += -akb / c1mc2;
    /* k1 = (z2 - z3) / c2m1 */
    z2b[i] += k1b / c2m1;
    z3b[i] += -k1b / c2m1;
    /* y_new = y_old + z3 */
    z3b[i] += w[i];
    ybar[i] += w[i];
  }
}

static int rd_forward(DaTape *T, double *y, const double *seg_t_end, const double *t_points,
                      int n_points, double rtol, double atol, int nmax, int options) {
  const int n = T->n;
  integer N = n, itol = 0, ijac = (options & DA_FORWARD_JAC) ? 1 : 0, mljac = n, mujac = n;
  integer imas = 0, mlmas = n, mumas = n, iout = 1, idid = 1, ipar = 0;
  integer lwork = 4 * N * N + 12 * N + 20, liwork = 3 * N + 20;
  doublereal *work = (doublereal *)malloc((size_t)lwork * sizeof(doublereal));
  integer *iwork = (integer *)malloc((size_t)liwork * sizeof(integer));
  R.y_prev = (double *)malloc((size_t)n * sizeof(double));
  if (!work || !iwork || !R.y_prev) {
    free(work);
    free(iwork);
    free(R.y_prev);
    return DA_NO_MEMORY;
  }
  double t = 0.0;
  for (int g = 0; g < T->n_seg && idid > 0; g++) {
    memset(work, 0, (size_t)lwork * sizeof(doublereal));
    memset(iwork, 0, (size_t)liwork * sizeof(integer));
    iwork[0] = 1;    /* Hessenberg form */
    iwork[1] = nmax; /* maximal number of steps */
    radau5_jac_growth = 0.0;
    if (options & DA_REUSE_JACOBIAN) {
      /* Same settings as mxn_radau's reuse_jacobian */
      iwork[2] = 10;
      work[2] = 0.05;
      work[5] = 1.5;
      radau5_jac_growth = 8.0;
    }
    R.seg = g;
    R.pars = T->seg_pars + (size_t)g * (size_t)T->n_pars;
    R.t_start = t;
    doublereal x = t, xend = seg_t_end[g], h = 0.0, rt = rtol, at = atol;
    if (xend > x) {
      radau5_(&N, (U_fp)rd_fcn, &x, y, &xend, &h, &rt, &at, &itol, (U_fp)rd_jac, &ijac,
              &mljac, &mujac, (U_fp)rd_dummy, &imas, &mlmas, &mumas, (U_fp)rd_solout, &iout,
              work, &lwork, iwork, &liwork, R.pars, &ipar, &idid);
    } else if (g == 0) {
      /* Only t = 0 */
      while (R.next < n_points && t_points[R.next] <= x) add_point(T, -1, PT_INITIAL, 0.0);
    }
    t = xend;
  }
  free(work);
  free(iwork);
  free(R.y_prev);
  R.y_prev = NULL;
  if (R.status != 0) return R.status;
  if (idid < 0) return (int)idid;
  if (R.next != n_points) return DA_BAD_INPUT;
  return 0;
}

/* ------------------------------------------------------------------ */
/* Shared forward driver                                               */
/* ------------------------------------------------------------------ */

static double *dup_doubles(const double *src, size_t count) {
  double *d = (double *)malloc((count > 0 ? count : 1) * sizeof(double));
  if (d && src && count > 0) memcpy(d, src, count * sizeof(double));
  return d;
}

static int *dup_ints(const int *src, size_t count) {
  int *d = (int *)malloc((count > 0 ? count : 1) * sizeof(int));
  if (d && src && count > 0) memcpy(d, src, count * sizeof(int));
  return d;
}

/* State of data point p on the steps (scratch: n doubles). */
static const double *point_state_on(const DaTape *T, const double *steps, const double *y0,
                                    const DaPoint *p, double *scratch) {
  const int n = T->n;
  if (p->kind == PT_INITIAL) return y0;
  const double *rec = steps + (size_t)p->step * (size_t)T->rec;
  if (p->kind == PT_STEP_END) {
    const double *y = rec + 2, *z3 = rec + 2 + 3 * n;
    for (int i = 0; i < n; i++) scratch[i] = y[i] + z3[i];
    return scratch;
  }
  rd_dense(n, rec, p->t, scratch);
  return scratch;
}

static const double *point_state(const DaTape *T, const DaPoint *p, double *scratch) {
  return point_state_on(T, T->steps, T->y0, p, scratch);
}

/* The targets at every data point into out_pred. Returns 0 or DA_NOT_FINITE. */
static int predictions(const DaTape *T, const double *steps, const double *y0,
                       const double *seg_pars, double *out_pred) {
  const int n = T->n;
  double *scratch = (double *)malloc((size_t)n * sizeof(double));
  double *dvals =
      (double *)malloc((size_t)(T->n_derived > 0 ? T->n_derived : 1) * sizeof(double));
  if (!scratch || !dvals) {
    free(scratch);
    free(dvals);
    return DA_NO_MEMORY;
  }
  int status = 0;
  for (int j = 0; j < T->n_points; j++) {
    const DaPoint *p = T->points + j;
    const double *x = point_state_on(T, steps, y0, p, scratch);
    const double *pars = seg_pars + (size_t)p->seg * (size_t)T->n_pars;
    if (T->n_derived > 0) T->derived(n, p->t, (double *)x, dvals, (double *)pars);
    for (int k = 0; k < T->n_targets; k++) {
      double v = T->target_kind[k] == 0 ? x[T->target_index[k]] : dvals[T->target_index[k]];
      if (!isfinite(v)) status = DA_NOT_FINITE;
      out_pred[(size_t)j * (size_t)T->n_targets + (size_t)k] = v;
    }
  }
  free(scratch);
  free(dvals);
  return status;
}

int da_forward(intptr_t model, intptr_t jac, intptr_t derived, int method, int n_y,
               const double *y0, int n_pars, int n_seg, const double *seg_t_end,
               const double *seg_pars, int n_points, const double *t_points, int n_targets,
               const int *target_kind, const int *target_index, int n_derived, double rtol,
               double atol, int nmax, int options, double max_tape_bytes, double *out_pred,
               DaTape **out_tape) {
  *out_tape = NULL;
  if (method != DA_METHOD_RADAU5 || n_y < 1 || n_seg < 1 || !model || !jac ||
      (n_derived > 0 && !derived)) {
    return DA_BAD_INPUT;
  }
  for (int j = 0; j < n_points; j++) {
    if (t_points[j] < 0 || t_points[j] > seg_t_end[n_seg - 1] ||
        (j > 0 && t_points[j] < t_points[j - 1])) {
      return DA_BAD_INPUT;
    }
  }
  for (int k = 0; k < n_targets; k++) {
    int bound = target_kind[k] == 0 ? n_y : n_derived;
    if (target_index[k] < 0 || target_index[k] >= bound) return DA_BAD_INPUT;
  }
  DaTape *T = (DaTape *)calloc(1, sizeof(DaTape));
  if (!T) return DA_NO_MEMORY;
  T->method = method;
  T->n = n_y;
  T->n_pars = n_pars;
  T->n_seg = n_seg;
  T->model = (DaModelFn)model;
  T->jac = (DaModelFn)jac;
  T->derived = (DaModelFn)derived;
  T->seg_pars = dup_doubles(seg_pars, (size_t)n_seg * (size_t)n_pars);
  T->y0 = dup_doubles(y0, (size_t)n_y);
  T->n_targets = n_targets;
  T->n_derived = n_derived;
  T->target_kind = dup_ints(target_kind, (size_t)n_targets);
  T->target_index = dup_ints(target_index, (size_t)n_targets);
  T->rec = 2 + 4 * n_y;
  T->n_points = n_points;
  T->points = (DaPoint *)malloc((size_t)(n_points > 0 ? n_points : 1) * sizeof(DaPoint));
  double *y = dup_doubles(y0, (size_t)n_y);
  if (!T->seg_pars || !T->y0 || !T->target_kind || !T->target_index || !T->points || !y) {
    free(y);
    da_free(T);
    return DA_NO_MEMORY;
  }

  memset(&R, 0, sizeof(R));
  R.T = T;
  R.t_points = t_points;
  R.n_points = n_points;
  R.max_bytes = max_tape_bytes > 0 ? max_tape_bytes : INFINITY;
  int status = rd_forward(T, y, seg_t_end, t_points, n_points, rtol, atol, nmax, options);
  memset(&R, 0, sizeof(R));
  free(y);
  if (status == 0) status = predictions(T, T->steps, T->y0, T->seg_pars, out_pred);
  if (status != 0) {
    da_free(T);
    return status;
  }
  *out_tape = T;
  return 0;
}

/* ------------------------------------------------------------------ */
/* Backward                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
  DaAdjointFn adjoint;
  DaModelFn derived_grad;
  int n_theta;
  double *lam_scratch, *dlam, *dtheta, *dgrad, *state;
} DaBack;

/* The state cotangent of point j (into w, n) and its direct parameter
 * contribution (added to grad). Returns 0 if the cotangent is all zero. */
static int point_cotangent(const DaTape *T, DaBack *B, int j, const double *cotangent,
                           double *w, double *grad) {
  const int n = T->n, nt = B->n_theta;
  const double *c = cotangent + (size_t)j * (size_t)T->n_targets;
  int any = 0, derived = 0;
  memset(w, 0, (size_t)n * sizeof(double));
  for (int k = 0; k < T->n_targets; k++) {
    if (c[k] == 0.0) continue;
    any = 1;
    if (T->target_kind[k] == 0) {
      w[T->target_index[k]] += c[k];
    } else {
      derived = 1;
    }
  }
  if (derived) {
    const DaPoint *p = T->points + j;
    const double *x = point_state(T, p, B->state);
    double *pars = T->seg_pars + (size_t)p->seg * (size_t)T->n_pars;
    B->derived_grad(n, p->t, (double *)x, B->dgrad, pars);
    for (int k = 0; k < T->n_targets; k++) {
      if (c[k] == 0.0 || T->target_kind[k] == 0) continue;
      const double *row = B->dgrad + (size_t)T->target_index[k] * (size_t)(n + nt);
      for (int i = 0; i < n; i++) w[i] += c[k] * row[i];
      for (int q = 0; q < nt; q++) grad[q] += c[k] * row[n + q];
    }
  }
  return any;
}

/* RADAU5: one step backwards. ybar holds the cotangent of y_new on entry
 * (dense points of this step already in ybar_dense / zbar) and that of y_old
 * on return. */
static int rd_step_vjp(const DaTape *T, DaBack *B, int step, double *ybar, double *zbar,
                       double *M, double *J, integer *ip, double *grad) {
  const int n = T->n, n3 = 3 * n, nt = B->n_theta;
  const double *rec = T->steps + (size_t)step * (size_t)T->rec;
  const double x = rec[0], h = rec[1], *y = rec + 2, *z = rec + 2 + n;
  double *pars = T->seg_pars + (size_t)T->step_seg[step] * (size_t)T->n_pars;
  double *mu = zbar; /* solved in place */
  double *stage = B->state;

  for (int i = 0; i < n; i++) zbar[2 * n + i] += ybar[i];

  /* M = I - h diag(J_j^T) (A^T x I), column-major 3n x 3n:
   * block (row j, column i) = delta_ji I - h a_ij J_j^T */
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < n; i++) stage[i] = y[i] + z[j * n + i];
    T->jac(n, x + RC[j] * h, stage, J, pars);
    for (int i = 0; i < 3; i++) {
      const double f = -h * RA[i][j];
      for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
          double v = f * J[c + (size_t)r * n]; /* J_j^T (r, c) */
          if (i == j && r == c) v += 1.0;
          M[(size_t)(j * n + r) + (size_t)(i * n + c) * (size_t)n3] = v;
        }
      }
    }
  }
  integer N3 = n3, ier = 0;
  dec_(&N3, &N3, M, ip, &ier);
  if (ier != 0) return DA_SINGULAR;
  sol_(&N3, &N3, M, mu, ip);

  /* nu_j = h sum_i a_ij mu_i; ybar += J_j^T nu_j; grad += (df/dtheta)_j^T nu_j */
  for (int j = 0; j < 3; j++) {
    double *nu = B->lam_scratch;
    for (int r = 0; r < n; r++) {
      double v = 0.0;
      for (int i = 0; i < 3; i++) v += RA[i][j] * mu[i * n + r];
      nu[r] = h * v;
    }
    for (int i = 0; i < n; i++) stage[i] = y[i] + z[j * n + i];
    B->adjoint(n, x + RC[j] * h, stage, nu, pars, B->dlam, B->dtheta);
    for (int i = 0; i < n; i++) ybar[i] -= B->dlam[i];
    for (int q = 0; q < nt; q++) grad[q] -= B->dtheta[q];
  }
  return 0;
}

int da_backward(DaTape *T, intptr_t adjoint, intptr_t derived_grad, int n_theta,
                const double *cotangent, double *out_grad, double *out_lambda0) {
  if (!T || !adjoint || n_theta < 0) return DA_BAD_INPUT;
  const int n = T->n, n3 = 3 * n;
  DaBack B = {0};
  B.adjoint = (DaAdjointFn)adjoint;
  B.derived_grad = (DaModelFn)derived_grad;
  B.n_theta = n_theta;
  if (T->n_derived > 0 && !derived_grad) {
    for (int j = 0; j < T->n_points; j++) {
      for (int k = 0; k < T->n_targets; k++) {
        if (T->target_kind[k] != 0 && cotangent[(size_t)j * T->n_targets + k] != 0.0) {
          return DA_BAD_INPUT;
        }
      }
    }
  }
  const size_t nd = (size_t)(T->n_derived > 0 ? T->n_derived : 1);
  B.lam_scratch = (double *)malloc((size_t)n * sizeof(double));
  B.dlam = (double *)malloc((size_t)n * sizeof(double));
  B.dtheta = (double *)malloc((size_t)(n_theta > 0 ? n_theta : 1) * sizeof(double));
  B.dgrad = (double *)malloc(nd * (size_t)(n + n_theta) * sizeof(double));
  B.state = (double *)malloc((size_t)n * sizeof(double));
  double *ybar = (double *)calloc((size_t)n, sizeof(double));
  double *w = (double *)malloc((size_t)n * sizeof(double));
  double *zbar = (double *)malloc((size_t)n3 * sizeof(double));
  double *M = (double *)malloc((size_t)n3 * (size_t)n3 * sizeof(double));
  double *J = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  integer *ip = (integer *)malloc((size_t)n3 * sizeof(integer));
  int status = 0;
  if (!B.lam_scratch || !B.dlam || !B.dtheta || !B.dgrad || !B.state || !ybar || !w || !zbar ||
      !M || !J || !ip) {
    status = DA_NO_MEMORY;
  }
  memset(out_grad, 0, (size_t)n_theta * sizeof(double));

  int j = T->n_points - 1;
  for (int s = T->n_steps - 1; status == 0 && s >= 0; s--) {
    memset(zbar, 0, (size_t)n3 * sizeof(double));
    double *dense_ybar = B.lam_scratch; /* reused before rd_step_vjp needs it */
    memset(dense_ybar, 0, (size_t)n * sizeof(double));
    const double *rec = T->steps + (size_t)s * (size_t)T->rec;
    for (; j >= 0 && T->points[j].step == s; j--) {
      if (!point_cotangent(T, &B, j, cotangent, w, out_grad)) continue;
      if (T->points[j].kind == PT_STEP_END) {
        for (int i = 0; i < n; i++) ybar[i] += w[i];
      } else {
        rd_dense_vjp(n, rec, T->points[j].t, w, dense_ybar, zbar);
      }
    }
    for (int i = 0; i < n; i++) w[i] = dense_ybar[i];
    status = rd_step_vjp(T, &B, s, ybar, zbar, M, J, ip, out_grad);
    for (int i = 0; i < n; i++) ybar[i] += w[i];
  }
  for (; status == 0 && j >= 0; j--) {
    /* PT_INITIAL */
    if (point_cotangent(T, &B, j, cotangent, w, out_grad)) {
      for (int i = 0; i < n; i++) ybar[i] += w[i];
    }
  }
  if (status == 0) {
    for (int i = 0; i < n; i++) {
      if (!isfinite(ybar[i])) status = DA_NOT_FINITE;
    }
    for (int q = 0; q < n_theta; q++) {
      if (!isfinite(out_grad[q])) status = DA_NOT_FINITE;
    }
    memcpy(out_lambda0, ybar, (size_t)n * sizeof(double));
  }
  free(B.lam_scratch);
  free(B.dlam);
  free(B.dtheta);
  free(B.dgrad);
  free(B.state);
  free(ybar);
  free(w);
  free(zbar);
  free(M);
  free(J);
  free(ip);
  return status;
}

/* ------------------------------------------------------------------ */
/* Replay                                                              */
/* ------------------------------------------------------------------ */

/* Solves step rec's stage equations from y (rec's y is overwritten with it)
 * by full Newton, starting from rec's Z. */
static int rd_replay_step(const DaTape *T, double *rec, const double *y, const double *pars,
                          double *M, double *J, double *G, double *F, double *stage,
                          integer *ip) {
  const int n = T->n, n3 = 3 * n;
  const double x = rec[0], h = rec[1];
  double *z = rec + 2 + n;
  memcpy(rec + 2, y, (size_t)n * sizeof(double));
  for (int it = 0; it < 50; it++) {
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < n; i++) stage[i] = y[i] + z[j * n + i];
      T->model(n, x + RC[j] * h, stage, F + j * n, (double *)pars);
      T->jac(n, x + RC[j] * h, stage, J, (double *)pars);
      for (int i = 0; i < 3; i++) {
        const double f = -h * RA[i][j];
        for (int c = 0; c < n; c++) {
          for (int r = 0; r < n; r++) {
            double v = f * J[r + (size_t)c * n];
            if (i == j && r == c) v += 1.0;
            M[(size_t)(i * n + r) + (size_t)(j * n + c) * (size_t)n3] = v;
          }
        }
      }
    }
    double norm = 0.0, znorm = 0.0;
    for (int i = 0; i < 3; i++) {
      for (int r = 0; r < n; r++) {
        double v = z[i * n + r];
        for (int j = 0; j < 3; j++) v -= h * RA[i][j] * F[j * n + r];
        G[i * n + r] = -v;
      }
    }
    integer N3 = n3, ier = 0;
    dec_(&N3, &N3, M, ip, &ier);
    if (ier != 0) return DA_SINGULAR;
    sol_(&N3, &N3, M, G, ip);
    for (int k = 0; k < n3; k++) {
      z[k] += G[k];
      norm = fmax(norm, fabs(G[k]));
      znorm = fmax(znorm, fabs(z[k]));
    }
    if (!isfinite(norm)) return DA_NOT_FINITE;
    if (norm <= 1e-15 * (1.0 + znorm)) return 0;
  }
  return 0; /* rounding-level stagnation: as converged as it gets */
}

int da_replay(const DaTape *T, const double *seg_pars, const double *y0, double *out_pred) {
  if (!T) return DA_BAD_INPUT;
  const int n = T->n, n3 = 3 * n;
  const size_t bytes = (size_t)T->n_steps * (size_t)T->rec * sizeof(double);
  double *steps = (double *)malloc(bytes > 0 ? bytes : 1);
  double *y = dup_doubles(y0, (size_t)n);
  double *M = (double *)malloc((size_t)n3 * (size_t)n3 * sizeof(double));
  double *J = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  double *G = (double *)malloc((size_t)n3 * sizeof(double));
  double *F = (double *)malloc((size_t)n3 * sizeof(double));
  double *stage = (double *)malloc((size_t)n * sizeof(double));
  integer *ip = (integer *)malloc((size_t)n3 * sizeof(integer));
  int status = 0;
  if (!steps || !y || !M || !J || !G || !F || !stage || !ip) status = DA_NO_MEMORY;
  if (status == 0 && bytes > 0) memcpy(steps, T->steps, bytes);
  for (int s = 0; status == 0 && s < T->n_steps; s++) {
    double *rec = steps + (size_t)s * (size_t)T->rec;
    const double *pars = seg_pars + (size_t)T->step_seg[s] * (size_t)T->n_pars;
    status = rd_replay_step(T, rec, y, pars, M, J, G, F, stage, ip);
    for (int i = 0; i < n; i++) y[i] = rec[2 + i] + rec[2 + 3 * n + i];
  }
  if (status == 0) status = predictions(T, steps, y0, seg_pars, out_pred);
  free(steps);
  free(y);
  free(M);
  free(J);
  free(G);
  free(F);
  free(stage);
  free(ip);
  return status;
}
