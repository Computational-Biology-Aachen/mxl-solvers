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
 *
 * Nordsieck solvers (LSODA, CVODE)
 * --------------------------------
 * Their state between steps is the Nordsieck array Z (column j = h^j y^(j)/j!),
 * which they rescale, predict, retract after failures, update with each
 * step's correction and reshape on order and method changes. Rather than
 * modelling those rules here, the solvers report every operation on Z
 * through nordsieck_tape.h's hooks, and the tape is that operation log, with
 * all scalars (step sizes, coefficients) as recorded. Every operation is
 * linear in Z except the start (Z1 = h f(t, Z0)) and the correction, which
 * solves acor + d Z1 - b f(t, Z0 + c acor) = 0 and is differentiated with the
 * implicit function theorem (a transposed n x n solve with J at the converged
 * state). Failed attempts stay on the log; their corrections get no
 * cotangent, so they cost nothing backwards. Outputs are the solver's own
 * interpolation sum_j s^j Z_j, s = (t - tn) / h.
 */
#include "discrete_adjoint.h"

#include "f2c.h"
#include "lsoda.h"
#include "nordsieck_tape.h"
#include "sundials/src/cvode/cvode_impl.h"
#include <cvode/cvode.h>
#include <math.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>
#include <stdlib.h>
#include <string.h>

/* sundials/src/cvode/cvode.c (mxl-solvers patch) */
int CVodeSetNordsieckTape(void *cvode_mem, const NtHooks *hooks);

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

/* RADAU5: where a data point's state comes from. */
enum { PT_INITIAL = 0, PT_STEP_END = 1, PT_DENSE = 2 };

typedef struct {
  int step; /* RADAU5: tape step (PT_STEP_END, PT_DENSE) */
  int kind;
  int seg;
  double t;
} DaPoint;

/* Nordsieck operations, as stored on the log (all fields doubles):
 *   NT_INIT       seg t h y[n]           Z0 = carry, Z1 = h f(t, Z0)
 *   NT_DERIV      seg t h y[n]           Z1 = h f(t, Z0), y = Z0
 *   NT_SCALE      q eta
 *   NT_PREDICT    q
 *   NT_RETRACT    q
 *   NT_CORRECT    seg t d b c acor[n] y[n]
 *   NT_UPDATE     q coef[q+1]
 *   NT_SET_ACOR   col r
 *   NT_SET_ZERO   col
 *   NT_SET_COL    dst src r
 *   NT_AXPY_MULTI src j0 count coef[count]
 *   NT_OUTPUT     j s q                  point j = sum_k s^k Z_k
 *   NT_OUTPUT_Y   j                      point j = carry (the start state)
 *   NT_CARRY      s q                    carry = sum_k s^k Z_k (next segment's start)
 */
enum {
  NT_INIT = 1,
  NT_DERIV,
  NT_SCALE,
  NT_PREDICT,
  NT_RETRACT,
  NT_CORRECT,
  NT_UPDATE,
  NT_SET_ACOR,
  NT_SET_ZERO,
  NT_SET_COL,
  NT_AXPY_MULTI,
  NT_OUTPUT,
  NT_OUTPUT_Y,
  NT_CARRY
};

struct DaTape {
  int method;
  int n, n_pars, n_seg;
  DaModelFn model, jac, derived;
  double *seg_pars; /* n_seg * n_pars */
  double *y0;
  int n_targets, n_derived;
  int *target_kind, *target_index;

  int n_points;
  DaPoint *points;
  double *pt_states; /* n_points * n: the state at every data point */

  /* RADAU5 steps: rec doubles each, x, h, y (n), Z (3n). n_steps counts
   * the accepted steps of Nordsieck solvers too. */
  int rec;
  int n_steps, cap_steps;
  double *steps;
  int *step_seg;

  /* Nordsieck operation log */
  double *ops;
  size_t n_ops_d, cap_ops_d; /* doubles */
  int *op_off;               /* start of every operation */
  int n_ops, cap_ops;

  double max_bytes;
  double bytes;
};

int da_tape_steps(const DaTape *tape) { return tape ? tape->n_steps : 0; }

void da_tape_op_counts(const DaTape *tape, int *counts, int n_counts) {
  memset(counts, 0, (size_t)n_counts * sizeof(int));
  if (!tape) return;
  for (int o = 0; o < tape->n_ops; o++) {
    int code = (int)tape->ops[tape->op_off[o]];
    if (code >= 0 && code < n_counts) counts[code]++;
  }
}
double da_tape_bytes(const DaTape *tape) { return tape ? tape->bytes : 0.0; }

void da_free(DaTape *T) {
  if (!T) return;
  free(T->seg_pars);
  free(T->y0);
  free(T->target_kind);
  free(T->target_index);
  free(T->points);
  free(T->pt_states);
  free(T->steps);
  free(T->step_seg);
  free(T->ops);
  free(T->op_off);
  free(T);
}

/* Grows *buf to hold `need` elements of `size` bytes, counting the bytes on
 * the tape. Returns 0, DA_NO_MEMORY or DA_TAPE_TOO_LARGE. */
static int grow(DaTape *T, void **buf, size_t *cap, size_t need, size_t size) {
  if (need <= *cap) return 0;
  size_t new_cap = *cap > 0 ? 2 * *cap : 256;
  while (new_cap < need) new_cap *= 2;
  double extra = (double)(new_cap - *cap) * (double)size;
  if (T->bytes + extra > T->max_bytes) {
    new_cap = need;
    extra = (double)(new_cap - *cap) * (double)size;
    if (T->bytes + extra > T->max_bytes) return DA_TAPE_TOO_LARGE;
  }
  void *p = realloc(*buf, new_cap * size);
  if (!p) return DA_NO_MEMORY;
  *buf = p;
  *cap = new_cap;
  T->bytes += extra;
  return 0;
}

/* A new RADAU5 step record, NULL (and *status set) on failure. */
static double *push_step(DaTape *T, int seg, int *status) {
  size_t cap = (size_t)T->cap_steps, cap2 = cap;
  int st = grow(T, (void **)&T->steps, &cap, (size_t)T->n_steps + 1,
                (size_t)T->rec * sizeof(double));
  if (st == 0) st = grow(T, (void **)&T->step_seg, &cap2, cap, sizeof(int));
  if (st != 0) {
    *status = st;
    return NULL;
  }
  T->cap_steps = (int)cap;
  T->step_seg[T->n_steps] = seg;
  return T->steps + (size_t)(T->n_steps++) * (size_t)T->rec;
}

/* Appends one operation of `count` doubles; NULL (and *status) on failure. */
static double *push_op(DaTape *T, int code, size_t count, int *status) {
  if (*status != 0) return NULL;
  size_t cap_off = (size_t)T->cap_ops;
  int st = grow(T, (void **)&T->ops, &T->cap_ops_d, T->n_ops_d + 1 + count, sizeof(double));
  if (st == 0) st = grow(T, (void **)&T->op_off, &cap_off, (size_t)T->n_ops + 1, sizeof(int));
  if (st != 0) {
    *status = st;
    return NULL;
  }
  T->cap_ops = (int)cap_off;
  T->op_off[T->n_ops++] = (int)T->n_ops_d;
  double *op = T->ops + T->n_ops_d;
  op[0] = (double)code;
  T->n_ops_d += 1 + count;
  return op + 1;
}

/* ------------------------------------------------------------------ */
/* Small dense helpers                                                 */
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

static double binom(int k, int j) {
  double v = 1.0;
  for (int i = 1; i <= j; i++) v = v * (double)(k - j + i) / (double)i;
  return v;
}

/* Horner, as LSODA's intdy and CVodeGetDky: sum_{k<=q} s^k Z_k. */
static void nordsieck_eval(int n, const double *Z, int q, double s, double *out) {
  for (int i = 0; i < n; i++) out[i] = Z[i + (size_t)q * n];
  for (int k = q - 1; k >= 0; k--) {
    for (int i = 0; i < n; i++) out[i] = Z[i + (size_t)k * n] + s * out[i];
  }
}

/* ------------------------------------------------------------------ */
/* Recording state (one solve at a time; the solvers aren't reentrant) */
/* ------------------------------------------------------------------ */
static struct {
  DaTape *T;
  double *pars;
  int seg;
  const double *t_points;
  int n_points, next; /* next data point to record */
  double *y_prev;     /* RADAU5: state at the start of the step SOLOUT reports */
  int status;
} R;

static void add_point(DaTape *T, int step, int kind, double t, const double *y) {
  DaPoint *p = T->points + R.next;
  p->step = step;
  p->kind = kind;
  p->seg = R.seg;
  p->t = t;
  double *dst = T->pt_states + (size_t)R.next * (size_t)T->n;
  if (dst != y) memcpy(dst, y, (size_t)T->n * sizeof(double));
  R.next++;
}

/* ------------------------------------------------------------------ */
/* RADAU5 forward                                                      */
/* ------------------------------------------------------------------ */

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
    while (R.seg == 0 && R.next < R.n_points && R.t_points[R.next] <= *x) {
      add_point(T, -1, PT_INITIAL, R.t_points[R.next], y);
    }
    memcpy(R.y_prev, y, (size_t)n * sizeof(double));
    return 0;
  }
  double *s = push_step(T, R.seg, &R.status);
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
    if (t == *x) {
      add_point(T, step, PT_STEP_END, t, y);
    } else {
      double *state = T->pt_states + (size_t)R.next * (size_t)n;
      rd_dense(n, s, t, state);
      add_point(T, step, PT_DENSE, t, state);
    }
  }
  memcpy(R.y_prev, y, (size_t)n * sizeof(double));
  return 0;
}

static int rd_forward(DaTape *T, double *y, const double *seg_t_end, double rtol,
                      double atol, int nmax, int options, double newton_scale) {
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
  for (int g = 0; g < T->n_seg && idid > 0 && R.status == 0; g++) {
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
    if (newton_scale != 1.0) {
      /* RADAU5's default FNEWT for its internally transformed tolerance */
      const double uround = 1e-16, tolst = 0.1 * pow(rtol, 2.0 / 3.0);
      const double fnewt = fmax(10.0 * uround / tolst, fmin(0.03, sqrt(tolst)));
      work[3] = fmax(fnewt * newton_scale, 2.0 * uround / tolst);
    }
    R.seg = g;
    R.pars = T->seg_pars + (size_t)g * (size_t)T->n_pars;
    doublereal x = t, xend = seg_t_end[g], h = 0.0, rt = rtol, at = atol;
    if (xend > x) {
      radau5_(&N, (U_fp)rd_fcn, &x, y, &xend, &h, &rt, &at, &itol, (U_fp)rd_jac, &ijac,
              &mljac, &mujac, (U_fp)rd_dummy, &imas, &mlmas, &mumas, (U_fp)rd_solout, &iout,
              work, &lwork, iwork, &liwork, R.pars, &ipar, &idid);
    } else if (g == 0) {
      /* Only t = 0 */
      while (R.next < R.n_points && R.t_points[R.next] <= x) {
        add_point(T, -1, PT_INITIAL, 0.0, y);
      }
    }
    t = xend;
  }
  free(work);
  free(iwork);
  free(R.y_prev);
  R.y_prev = NULL;
  if (R.status != 0) return R.status;
  if (idid < 0) return (int)idid;
  return 0;
}

/* ------------------------------------------------------------------ */
/* Nordsieck recording hooks                                           */
/* ------------------------------------------------------------------ */

static void nt_init(void *ctx, double t, double h, const double *y) {
  (void)ctx;
  const int n = R.T->n;
  double *op = push_op(R.T, NT_INIT, 3 + (size_t)n, &R.status);
  if (!op) return;
  op[0] = R.seg;
  op[1] = t;
  op[2] = h;
  memcpy(op + 3, y, (size_t)n * sizeof(double));
}

static void nt_deriv(void *ctx, double t, double h, const double *y) {
  (void)ctx;
  const int n = R.T->n;
  double *op = push_op(R.T, NT_DERIV, 3 + (size_t)n, &R.status);
  if (!op) return;
  op[0] = R.seg;
  op[1] = t;
  op[2] = h;
  memcpy(op + 3, y, (size_t)n * sizeof(double));
}

static void nt_scale(void *ctx, int q, double eta) {
  (void)ctx;
  double *op = push_op(R.T, NT_SCALE, 2, &R.status);
  if (!op) return;
  op[0] = q;
  op[1] = eta;
}

static void nt_predict(void *ctx, int q) {
  (void)ctx;
  double *op = push_op(R.T, NT_PREDICT, 1, &R.status);
  if (op) op[0] = q;
}

static void nt_retract(void *ctx, int q) {
  (void)ctx;
  double *op = push_op(R.T, NT_RETRACT, 1, &R.status);
  if (op) op[0] = q;
}

static void nt_correct(void *ctx, double t, double d, double b, double c, const double *acor,
                       const double *y) {
  (void)ctx;
  const int n = R.T->n;
  double *op = push_op(R.T, NT_CORRECT, 5 + 2 * (size_t)n, &R.status);
  if (!op) return;
  op[0] = R.seg;
  op[1] = t;
  op[2] = d;
  op[3] = b;
  op[4] = c;
  memcpy(op + 5, acor, (size_t)n * sizeof(double));
  memcpy(op + 5 + n, y, (size_t)n * sizeof(double));
}

static void nt_update(void *ctx, int q, const double *coef) {
  (void)ctx;
  double *op = push_op(R.T, NT_UPDATE, 2 + (size_t)q, &R.status);
  if (!op) return;
  R.T->n_steps++; /* one per accepted step */
  op[0] = q;
  memcpy(op + 1, coef, (size_t)(q + 1) * sizeof(double));
}

static void nt_set_acor(void *ctx, int col, double r) {
  (void)ctx;
  double *op = push_op(R.T, NT_SET_ACOR, 2, &R.status);
  if (!op) return;
  op[0] = col;
  op[1] = r;
}

static void nt_set_zero(void *ctx, int col) {
  (void)ctx;
  double *op = push_op(R.T, NT_SET_ZERO, 1, &R.status);
  if (op) op[0] = col;
}

static void nt_set_col(void *ctx, int dst, int src, double r) {
  (void)ctx;
  double *op = push_op(R.T, NT_SET_COL, 3, &R.status);
  if (!op) return;
  op[0] = dst;
  op[1] = src;
  op[2] = r;
}

static void nt_axpy_multi(void *ctx, int src, int j0, int count, const double *coef) {
  (void)ctx;
  double *op = push_op(R.T, NT_AXPY_MULTI, 3 + (size_t)count, &R.status);
  if (!op) return;
  op[0] = src;
  op[1] = j0;
  op[2] = count;
  memcpy(op + 3, coef, (size_t)count * sizeof(double));
}

static const NtHooks NT_HOOKS = {NULL,       nt_init,    nt_deriv,    nt_scale,
                                 nt_predict, nt_retract, nt_correct,  nt_update,
                                 nt_set_acor, nt_set_zero, nt_set_col, nt_axpy_multi};

/* Records data point R.next (in segment R.seg) from the solver's array Z. */
static void nt_output(DaTape *T, const double *Z, int q, double s) {
  double *op = push_op(T, NT_OUTPUT, 3, &R.status);
  if (!op) return;
  op[0] = R.next;
  op[1] = s;
  op[2] = q;
  double *state = T->pt_states + (size_t)R.next * (size_t)T->n;
  nordsieck_eval(T->n, Z, q, s, state);
  add_point(T, -1, 0, R.t_points[R.next], state);
}

/* ------------------------------------------------------------------ */
/* LSODA forward                                                       */
/* ------------------------------------------------------------------ */

static void ls_fcn(int *n, double *t, double *y, double *dydt) {
  R.T->model(*n, *t, y, dydt, R.pars);
}

static void ls_jac(int *n, double *t, double *y, int *ml, int *mu, double *pd, int *nrowpd) {
  (void)ml;
  (void)mu;
  (void)nrowpd; /* full Jacobian: nrowpd == n */
  R.T->jac(*n, *t, y, pd, R.pars);
}

/* Like mxn_lsoda (itask 1, one call per output time, its step budget), one
 * restart per protocol segment. */
static int ls_forward(DaTape *T, double *y, const double *seg_t_end, double rtol, double atol,
                      int nmax, int options, double newton_scale) {
  const int n = T->n;
  const int lrn = 20 + n * 13 + 3 * n, lrs = 20 + n * 6 + 3 * n + n * n + 2;
  const int lrw = lrn > lrs ? lrn : lrs, liw = 20 + n;
  double *rwork = (double *)malloc((size_t)lrw * sizeof(double));
  int *iwork = (int *)malloc((size_t)liw * sizeof(int));
  if (!rwork || !iwork) {
    free(rwork);
    free(iwork);
    return DA_NO_MEMORY;
  }
  int status = 0;
  double t = 0.0;
  /* t = 0 points: the initial state */
  R.seg = 0;
  while (R.next < R.n_points && R.t_points[R.next] <= 0.0) {
    double *op = push_op(T, NT_OUTPUT_Y, 1, &R.status);
    if (op) op[0] = R.next;
    add_point(T, -1, 0, R.t_points[R.next], y);
  }
  for (int g = 0; g < T->n_seg && status == 0 && R.status == 0; g++) {
    const double end = seg_t_end[g];
    if (!(end > t)) continue;
    memset(rwork, 0, (size_t)lrw * sizeof(double));
    memset(iwork, 0, (size_t)liw * sizeof(int));
    iwork[5] = nmax; /* IWORK(6): steps allowed per output point */
    lsoda_common_struct_t S;
    memset(&S, 0, sizeof(S));
    S.tape = &NT_HOOKS;
    S.conit_scale = newton_scale;
    int itask = 1, istate = 1, iopt = 1;
    R.seg = g;
    R.pars = T->seg_pars + (size_t)g * (size_t)T->n_pars;
    /* Every data point in (t, end], then end itself */
    for (;;) {
      int is_point = R.next < R.n_points && R.t_points[R.next] <= end;
      double tout = is_point ? R.t_points[R.next] : end;
      if (tout != t) {
        lsoda(ls_fcn, n, y, &t, &tout, 1, &rtol, &atol, &itask, &istate, &iopt, rwork, lrw,
              iwork, liw, (options & DA_FORWARD_JAC) ? ls_jac : NULL,
              (options & DA_FORWARD_JAC) ? 1 : 2, &S);
        if (istate > 0 && iwork[10] > nmax) istate = -1; /* IWORK(11): steps so far */
        if (istate < 0) {
          status = istate;
          break;
        }
        if (R.status != 0) break;
      }
      const double *Z = rwork + S.lyh;
      const double s = (tout - S.tn) / S.h;
      if (is_point) {
        nt_output(T, Z, S.nq, s);
      }
      if (!is_point || tout == end) {
        /* The state the next segment starts from */
        double *op = push_op(T, NT_CARRY, 2, &R.status);
        if (op) {
          op[0] = s;
          op[1] = S.nq;
        }
        nordsieck_eval(n, Z, S.nq, s, y);
        if (!is_point || R.next >= R.n_points || R.t_points[R.next] > end) break;
      }
    }
    t = end;
  }
  free(rwork);
  free(iwork);
  if (R.status != 0) return R.status;
  return status;
}

/* ------------------------------------------------------------------ */
/* CVODE forward                                                       */
/* ------------------------------------------------------------------ */

static int cv_rhs(sunrealtype t, N_Vector y, N_Vector dydt, void *user_data) {
  (void)user_data;
  int n = (int)N_VGetLength(y);
  double *f = N_VGetArrayPointer(dydt);
  R.T->model(n, t, N_VGetArrayPointer(y), f, R.pars);
  for (int i = 0; i < n; i++) {
    if (!isfinite(f[i])) return 1; /* recoverable: CVODE retries with a smaller step */
  }
  return 0;
}

static int cv_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
                  N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  (void)fy;
  (void)user_data;
  (void)tmp1;
  (void)tmp2;
  (void)tmp3;
  /* SUNDenseMatrix data is column-major n x n */
  R.T->jac((int)N_VGetLength(y), t, N_VGetArrayPointer(y), SM_DATA_D(J), R.pars);
  return 0;
}

/* CVODE's Nordsieck columns 0..q copied into the contiguous Z. */
static void cv_columns(CVodeMem mem, int n, double *Z) {
  for (int k = 0; k <= mem->cv_q; k++) {
    memcpy(Z + (size_t)k * (size_t)n, N_VGetArrayPointer(mem->cv_zn[k]),
           (size_t)n * sizeof(double));
  }
}

/* Like mxn_cvode (CV_NORMAL, one call per output time, a stop time at the
 * segment end, its step budget), one restart per protocol segment. */
static int cv_forward(DaTape *T, double *y, const double *seg_t_end, double rtol, double atol,
                      int nmax, int options, double newton_scale, int adams) {
  const int n = T->n;
  double *Z = (double *)calloc((size_t)n * NT_MAX_COLS, sizeof(double));
  SUNContext ctx = NULL;
  if (!Z || SUNContext_Create(SUN_COMM_NULL, &ctx) != 0) {
    free(Z);
    return DA_NO_MEMORY;
  }
  int flag = CV_SUCCESS;
  long int steps_before = 0;
  double t = 0.0;
  R.seg = 0;
  while (R.next < R.n_points && R.t_points[R.next] <= 0.0) {
    double *op = push_op(T, NT_OUTPUT_Y, 1, &R.status);
    if (op) op[0] = R.next;
    add_point(T, -1, 0, R.t_points[R.next], y);
  }
  for (int g = 0; g < T->n_seg && flag >= 0 && R.status == 0; g++) {
    const double end = seg_t_end[g];
    if (!(end > t)) continue;
    R.seg = g;
    R.pars = T->seg_pars + (size_t)g * (size_t)T->n_pars;
    N_Vector v = N_VMake_Serial(n, y, ctx); /* works on y in place */
    void *mem = CVodeCreate(adams ? CV_ADAMS : CV_BDF, ctx);
    SUNMatrix A = NULL;
    SUNLinearSolver ls = NULL;
    SUNNonlinearSolver nls = NULL;
    flag = (v && mem) ? CV_SUCCESS : DA_NO_MEMORY;
    if (flag == CV_SUCCESS) flag = CVodeInit(mem, cv_rhs, t, v);
    if (flag == CV_SUCCESS) flag = CVodeSStolerances(mem, rtol, atol);
    if (flag == CV_SUCCESS) flag = CVodeSetStopTime(mem, end);
    if (flag == CV_SUCCESS && adams) {
      nls = SUNNonlinSol_FixedPoint(v, 0, ctx);
      flag = nls ? CVodeSetNonlinearSolver(mem, nls) : DA_NO_MEMORY;
    } else if (flag == CV_SUCCESS) {
      A = SUNDenseMatrix(n, n, ctx);
      ls = A ? SUNLinSol_Dense(v, A, ctx) : NULL;
      flag = ls ? CVodeSetLinearSolver(mem, ls, A) : DA_NO_MEMORY;
      if (flag == CV_SUCCESS && (options & DA_FORWARD_JAC)) flag = CVodeSetJacFn(mem, cv_jac);
    }
    if (flag == CV_SUCCESS && newton_scale != 1.0) {
      flag = CVodeSetNonlinConvCoef(mem, 0.1 * newton_scale); /* CVODE's default is 0.1 */
    }
    if (flag == CV_SUCCESS) flag = CVodeSetNordsieckTape(mem, &NT_HOOKS);
    while (flag >= 0 && R.status == 0) {
      int is_point = R.next < R.n_points && R.t_points[R.next] <= end;
      double tout = is_point ? R.t_points[R.next] : end, tret = t;
      if (tout != t) {
        long int steps = 0;
        CVodeGetNumSteps(mem, &steps);
        if (steps_before + steps >= nmax) {
          flag = CV_TOO_MUCH_WORK;
          break;
        }
        CVodeSetMaxNumSteps(mem, nmax - steps_before - steps);
        flag = CVode(mem, tout, v, &tret, CV_NORMAL);
        if (flag < 0) break;
        t = tout;
      }
      CVodeMem cm = (CVodeMem)mem;
      cv_columns(cm, n, Z);
      const double s = (tout - cm->cv_tn) / cm->cv_h;
      if (is_point) nt_output(T, Z, cm->cv_q, s);
      if (!is_point || tout == end) {
        double *op = push_op(T, NT_CARRY, 2, &R.status);
        if (op) {
          op[0] = s;
          op[1] = cm->cv_q;
        }
        nordsieck_eval(n, Z, cm->cv_q, s, y);
        if (!is_point || R.next >= R.n_points || R.t_points[R.next] > end) break;
      }
    }
    long int steps = 0;
    if (mem) CVodeGetNumSteps(mem, &steps);
    steps_before += steps;
    CVodeFree(&mem);
    SUNNonlinSolFree(nls);
    SUNLinSolFree(ls);
    SUNMatDestroy(A);
    N_VDestroy(v);
    t = end;
  }
  SUNContext_Free(&ctx);
  free(Z);
  if (R.status != 0) return R.status;
  return flag < 0 ? flag : 0;
}

/* ------------------------------------------------------------------ */
/* Shared forward driver                                               */
/* ------------------------------------------------------------------ */

/* The targets at every data point (states pt_states) into out_pred.
 * Returns 0, DA_NOT_FINITE or DA_NO_MEMORY. */
static int predictions(const DaTape *T, const double *pt_states, const double *seg_pars,
                       double *out_pred) {
  const int n = T->n;
  double *dvals =
      (double *)malloc((size_t)(T->n_derived > 0 ? T->n_derived : 1) * sizeof(double));
  if (!dvals) return DA_NO_MEMORY;
  int status = 0;
  for (int j = 0; j < T->n_points; j++) {
    const DaPoint *p = T->points + j;
    const double *x = pt_states + (size_t)j * (size_t)n;
    const double *pars = seg_pars + (size_t)p->seg * (size_t)T->n_pars;
    if (T->n_derived > 0) T->derived(n, p->t, (double *)x, dvals, (double *)pars);
    for (int k = 0; k < T->n_targets; k++) {
      double v = T->target_kind[k] == 0 ? x[T->target_index[k]] : dvals[T->target_index[k]];
      if (!isfinite(v)) status = DA_NOT_FINITE;
      out_pred[(size_t)j * (size_t)T->n_targets + (size_t)k] = v;
    }
  }
  free(dvals);
  return status;
}

int da_forward(intptr_t model, intptr_t jac, intptr_t derived, int method, int n_y,
               const double *y0, int n_pars, int n_seg, const double *seg_t_end,
               const double *seg_pars, int n_points, const double *t_points, int n_targets,
               const int *target_kind, const int *target_index, int n_derived, double rtol,
               double atol, int nmax, int options, double newton_scale, double max_tape_bytes,
               double *out_pred,
               DaTape **out_tape) {
  *out_tape = NULL;
  if (!(newton_scale > 0.0)) newton_scale = 1.0;
  if (method < DA_METHOD_RADAU5 || method > DA_METHOD_CVODE_ADAMS || n_y < 1 || n_seg < 1 ||
      !model || !jac || (n_derived > 0 && !derived)) {
    return DA_BAD_INPUT;
  }
  for (int g = 0; g < n_seg; g++) {
    if (seg_t_end[g] < (g > 0 ? seg_t_end[g - 1] : 0.0)) return DA_BAD_INPUT;
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
  T->pt_states = (double *)malloc((size_t)(n_points > 0 ? n_points : 1) * (size_t)n_y *
                                  sizeof(double));
  T->max_bytes = max_tape_bytes > 0 ? max_tape_bytes : INFINITY;
  T->bytes = (double)n_points * ((double)sizeof(DaPoint) + (double)n_y * sizeof(double));
  double *y = dup_doubles(y0, (size_t)n_y);
  if (!T->seg_pars || !T->y0 || !T->target_kind || !T->target_index || !T->points ||
      !T->pt_states || !y) {
    free(y);
    da_free(T);
    return DA_NO_MEMORY;
  }

  memset(&R, 0, sizeof(R));
  R.T = T;
  R.t_points = t_points;
  R.n_points = n_points;
  int status;
  switch (method) {
  case DA_METHOD_RADAU5:
    status = rd_forward(T, y, seg_t_end, rtol, atol, nmax, options, newton_scale);
    break;
  case DA_METHOD_LSODA:
    status = ls_forward(T, y, seg_t_end, rtol, atol, nmax, options, newton_scale);
    break;
  default:
    status = cv_forward(T, y, seg_t_end, rtol, atol, nmax, options, newton_scale,
                        method == DA_METHOD_CVODE_ADAMS);
  }
  if (status == 0 && R.next != n_points) status = DA_BAD_INPUT;
  memset(&R, 0, sizeof(R));
  free(y);
  if (status == 0) status = predictions(T, T->pt_states, T->seg_pars, out_pred);
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
  double *dlam, *dtheta, *dgrad, *nu, *stage;
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
    double *x = T->pt_states + (size_t)j * (size_t)n;
    double *pars = T->seg_pars + (size_t)p->seg * (size_t)T->n_pars;
    B->derived_grad(n, p->t, x, B->dgrad, pars);
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
 * and that of y_old (without the step's dense points) on return; zbar holds
 * the dense points' stage cotangents on entry. */
static int rd_step_vjp(const DaTape *T, DaBack *B, int step, double *ybar, double *zbar,
                       double *M, double *J, integer *ip, double *grad) {
  const int n = T->n, n3 = 3 * n, nt = B->n_theta;
  const double *rec = T->steps + (size_t)step * (size_t)T->rec;
  const double x = rec[0], h = rec[1], *y = rec + 2, *z = rec + 2 + n;
  double *pars = T->seg_pars + (size_t)T->step_seg[step] * (size_t)T->n_pars;
  double *mu = zbar; /* solved in place */
  double *stage = B->stage;

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
    double *nu = B->nu;
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

static int rd_backward(const DaTape *T, DaBack *B, const double *cotangent, double *grad,
                       double *ybar) {
  const int n = T->n, n3 = 3 * n;
  double *w = (double *)malloc((size_t)n * sizeof(double));
  double *dense = (double *)malloc((size_t)n * sizeof(double));
  double *zbar = (double *)malloc((size_t)n3 * sizeof(double));
  double *M = (double *)malloc((size_t)n3 * (size_t)n3 * sizeof(double));
  double *J = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  integer *ip = (integer *)malloc((size_t)n3 * sizeof(integer));
  int status = (w && dense && zbar && M && J && ip) ? 0 : DA_NO_MEMORY;
  int j = T->n_points - 1;
  for (int s = T->n_steps - 1; status == 0 && s >= 0; s--) {
    memset(zbar, 0, (size_t)n3 * sizeof(double));
    memset(dense, 0, (size_t)n * sizeof(double));
    const double *rec = T->steps + (size_t)s * (size_t)T->rec;
    for (; j >= 0 && T->points[j].step == s; j--) {
      if (!point_cotangent(T, B, j, cotangent, w, grad)) continue;
      if (T->points[j].kind == PT_STEP_END) {
        for (int i = 0; i < n; i++) ybar[i] += w[i];
      } else {
        rd_dense_vjp(n, rec, T->points[j].t, w, dense, zbar);
      }
    }
    status = rd_step_vjp(T, B, s, ybar, zbar, M, J, ip, grad);
    for (int i = 0; i < n; i++) ybar[i] += dense[i];
  }
  for (; status == 0 && j >= 0; j--) {
    /* PT_INITIAL */
    if (point_cotangent(T, B, j, cotangent, w, grad)) {
      for (int i = 0; i < n; i++) ybar[i] += w[i];
    }
  }
  free(w);
  free(dense);
  free(zbar);
  free(M);
  free(J);
  free(ip);
  return status;
}

/* Nordsieck: the operation log backwards. cbar (the carry's cotangent) ends
 * as the initial state's. */
static int nt_backward(const DaTape *T, DaBack *B, const double *cotangent, double *grad,
                       double *cbar) {
  const int n = T->n, nt = B->n_theta;
  const size_t nc = NT_MAX_COLS;
  double *Zb = (double *)calloc((size_t)n * nc, sizeof(double));
  double *tmp = (double *)calloc((size_t)n * nc, sizeof(double));
  double *ab = (double *)calloc((size_t)n, sizeof(double));
  double *w = (double *)malloc((size_t)n * sizeof(double));
  double *M = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  double *J = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  integer *ip = (integer *)malloc((size_t)n * sizeof(integer));
  int status = (Zb && tmp && ab && w && M && J && ip) ? 0 : DA_NO_MEMORY;
  int ab_nonzero = 0;
#define ZB(k) (Zb + (size_t)(k) * (size_t)n)
  for (int o = T->n_ops - 1; status == 0 && o >= 0; o--) {
    const double *op = T->ops + T->op_off[o];
    const int code = (int)op[0];
    const double *a = op + 1;
    switch (code) {
    case NT_OUTPUT: {
      const int j = (int)a[0], q = (int)a[2];
      const double s = a[1];
      if (!point_cotangent(T, B, j, cotangent, w, grad)) break;
      double sk = 1.0;
      for (int k = 0; k <= q; k++, sk *= s) {
        for (int i = 0; i < n; i++) ZB(k)[i] += sk * w[i];
      }
      break;
    }
    case NT_OUTPUT_Y: {
      if (!point_cotangent(T, B, (int)a[0], cotangent, w, grad)) break;
      for (int i = 0; i < n; i++) cbar[i] += w[i];
      break;
    }
    case NT_CARRY: {
      const int q = (int)a[1];
      const double s = a[0];
      double sk = 1.0;
      for (int k = 0; k <= q; k++, sk *= s) {
        for (int i = 0; i < n; i++) ZB(k)[i] += sk * cbar[i];
      }
      memset(cbar, 0, (size_t)n * sizeof(double));
      break;
    }
    case NT_UPDATE: {
      const int q = (int)a[0];
      for (int k = 0; k <= q; k++) {
        const double c = a[1 + k];
        if (c == 0.0) continue;
        for (int i = 0; i < n; i++) ab[i] += c * ZB(k)[i];
      }
      ab_nonzero = 1;
      break;
    }
    case NT_SET_ACOR: {
      const int col = (int)a[0];
      for (int i = 0; i < n; i++) ab[i] += a[1] * ZB(col)[i];
      memset(ZB(col), 0, (size_t)n * sizeof(double));
      ab_nonzero = 1;
      break;
    }
    case NT_SET_ZERO:
      memset(ZB((int)a[0]), 0, (size_t)n * sizeof(double));
      break;
    case NT_SET_COL: {
      const int dst = (int)a[0], src = (int)a[1];
      if (dst == src) {
        /* In place (CVODE's order increase to qmax): Z[dst] *= r */
        for (int i = 0; i < n; i++) ZB(dst)[i] *= a[2];
      } else {
        for (int i = 0; i < n; i++) ZB(src)[i] += a[2] * ZB(dst)[i];
        memset(ZB(dst), 0, (size_t)n * sizeof(double));
      }
      break;
    }
    case NT_AXPY_MULTI: {
      const int src = (int)a[0], j0 = (int)a[1], count = (int)a[2];
      for (int k = 0; k < count; k++) {
        for (int i = 0; i < n; i++) ZB(src)[i] += a[3 + k] * ZB(j0 + k)[i];
      }
      break;
    }
    case NT_SCALE: {
      const int q = (int)a[0];
      double r = 1.0;
      for (int k = 1; k <= q; k++) {
        r *= a[1];
        for (int i = 0; i < n; i++) ZB(k)[i] *= r;
      }
      break;
    }
    case NT_PREDICT:
    case NT_RETRACT: {
      /* Zp_j = sum_{k>=j} C(k,j) Z_k (predict); the retraction is the inverse
       * with signs (-1)^(k-j). Transposed: Zbar_k = sum_{j<=k} c_kj Zbar_j. */
      const int q = (int)a[0];
      const double sign = code == NT_PREDICT ? 1.0 : -1.0;
      memset(tmp, 0, (size_t)n * (size_t)(q + 1) * sizeof(double));
      for (int k = 0; k <= q; k++) {
        for (int j = 0; j <= k; j++) {
          double c = binom(k, j) * (((k - j) % 2 == 1) ? sign : 1.0);
          for (int i = 0; i < n; i++) tmp[i + (size_t)k * n] += c * ZB(j)[i];
        }
      }
      memcpy(Zb, tmp, (size_t)n * (size_t)(q + 1) * sizeof(double));
      break;
    }
    case NT_CORRECT: {
      if (!ab_nonzero) break;
      const int seg = (int)a[0];
      const double t = a[1], d = a[2], b = a[3], c = a[4];
      double *y = (double *)(a + 5 + n);
      double *pars = T->seg_pars + (size_t)seg * (size_t)T->n_pars;
      /* (I - b c J^T) mu = acor_bar */
      T->jac(n, t, y, J, pars);
      for (int col = 0; col < n; col++) {
        for (int r = 0; r < n; r++) {
          double v = -b * c * J[col + (size_t)r * n]; /* J^T (r, col) */
          if (r == col) v += 1.0;
          M[r + (size_t)col * n] = v;
        }
      }
      integer N = n, ier = 0;
      dec_(&N, &N, M, ip, &ier);
      if (ier != 0) {
        status = DA_SINGULAR;
        break;
      }
      sol_(&N, &N, M, ab, ip);
      B->adjoint(n, t, y, ab, pars, B->dlam, B->dtheta);
      for (int i = 0; i < n; i++) {
        ZB(0)[i] -= b * B->dlam[i];
        ZB(1)[i] -= d * ab[i];
      }
      for (int q = 0; q < nt; q++) grad[q] -= b * B->dtheta[q];
      memset(ab, 0, (size_t)n * sizeof(double));
      ab_nonzero = 0;
      break;
    }
    case NT_DERIV:
    case NT_INIT: {
      const int seg = (int)a[0];
      const double t = a[1], h = a[2];
      double *y = (double *)(a + 3);
      double *pars = T->seg_pars + (size_t)seg * (size_t)T->n_pars;
      B->adjoint(n, t, y, ZB(1), pars, B->dlam, B->dtheta);
      double *dst = code == NT_INIT ? cbar : ZB(0);
      for (int i = 0; i < n; i++) dst[i] -= h * B->dlam[i];
      for (int q = 0; q < nt; q++) grad[q] -= h * B->dtheta[q];
      if (code == NT_INIT) {
        for (int i = 0; i < n; i++) cbar[i] += ZB(0)[i];
        memset(Zb, 0, (size_t)n * nc * sizeof(double));
      } else {
        memset(ZB(1), 0, (size_t)n * sizeof(double));
      }
      break;
    }
    default:
      status = DA_BAD_INPUT;
    }
  }
#undef ZB
  free(Zb);
  free(tmp);
  free(ab);
  free(w);
  free(M);
  free(J);
  free(ip);
  return status;
}

int da_backward(DaTape *T, intptr_t adjoint, intptr_t derived_grad, int n_theta,
                const double *cotangent, double *out_grad, double *out_lambda0) {
  if (!T || !adjoint || n_theta < 0) return DA_BAD_INPUT;
  const int n = T->n;
  if (T->n_derived > 0 && !derived_grad) {
    for (int j = 0; j < T->n_points; j++) {
      for (int k = 0; k < T->n_targets; k++) {
        if (T->target_kind[k] != 0 && cotangent[(size_t)j * T->n_targets + k] != 0.0) {
          return DA_BAD_INPUT;
        }
      }
    }
  }
  DaBack B = {0};
  B.adjoint = (DaAdjointFn)adjoint;
  B.derived_grad = (DaModelFn)derived_grad;
  B.n_theta = n_theta;
  const size_t nd = (size_t)(T->n_derived > 0 ? T->n_derived : 1);
  B.dlam = (double *)malloc((size_t)n * sizeof(double));
  B.dtheta = (double *)malloc((size_t)(n_theta > 0 ? n_theta : 1) * sizeof(double));
  B.dgrad = (double *)malloc(nd * (size_t)(n + n_theta) * sizeof(double));
  B.nu = (double *)malloc((size_t)n * sizeof(double));
  B.stage = (double *)malloc((size_t)n * sizeof(double));
  double *ybar = (double *)calloc((size_t)n, sizeof(double));
  int status = (B.dlam && B.dtheta && B.dgrad && B.nu && B.stage && ybar) ? 0 : DA_NO_MEMORY;
  memset(out_grad, 0, (size_t)n_theta * sizeof(double));
  if (status == 0) {
    status = T->method == DA_METHOD_RADAU5 ? rd_backward(T, &B, cotangent, out_grad, ybar)
                                           : nt_backward(T, &B, cotangent, out_grad, ybar);
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
  free(B.dlam);
  free(B.dtheta);
  free(B.dgrad);
  free(B.nu);
  free(B.stage);
  free(ybar);
  return status;
}

/* ------------------------------------------------------------------ */
/* Replay                                                              */
/* ------------------------------------------------------------------ */

/* Solves the dense n x n system M x = b in place (M overwritten). */
static int solve_dense(int n, double *M, double *b, integer *ip) {
  integer N = n, ier = 0;
  dec_(&N, &N, M, ip, &ier);
  if (ier != 0) return DA_SINGULAR;
  sol_(&N, &N, M, b, ip);
  return 0;
}

/* RADAU5: solves step rec's stage equations from y (rec's y is overwritten
 * with it) by full Newton, starting from rec's Z. */
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
    for (int i = 0; i < 3; i++) {
      for (int r = 0; r < n; r++) {
        double v = z[i * n + r];
        for (int j = 0; j < 3; j++) v -= h * RA[i][j] * F[j * n + r];
        G[i * n + r] = -v;
      }
    }
    if (solve_dense(n3, M, G, ip) != 0) return DA_SINGULAR;
    double norm = 0.0, znorm = 0.0;
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

static int rd_replay(const DaTape *T, const double *seg_pars, const double *y0,
                     double *states) {
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
  int status = (steps && y && M && J && G && F && stage && ip) ? 0 : DA_NO_MEMORY;
  if (status == 0 && bytes > 0) memcpy(steps, T->steps, bytes);
  for (int s = 0; status == 0 && s < T->n_steps; s++) {
    double *rec = steps + (size_t)s * (size_t)T->rec;
    const double *pars = seg_pars + (size_t)T->step_seg[s] * (size_t)T->n_pars;
    status = rd_replay_step(T, rec, y, pars, M, J, G, F, stage, ip);
    for (int i = 0; i < n; i++) y[i] = rec[2 + i] + rec[2 + 3 * n + i];
  }
  for (int j = 0; status == 0 && j < T->n_points; j++) {
    const DaPoint *p = T->points + j;
    double *out = states + (size_t)j * (size_t)n;
    if (p->kind == PT_INITIAL) {
      memcpy(out, y0, (size_t)n * sizeof(double));
    } else {
      const double *rec = steps + (size_t)p->step * (size_t)T->rec;
      if (p->kind == PT_STEP_END) {
        for (int i = 0; i < n; i++) out[i] = rec[2 + i] + rec[2 + 3 * n + i];
      } else {
        rd_dense(n, rec, p->t, out);
      }
    }
  }
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

/* Nordsieck: the operation log forwards, re-solving every correction. */
static int nt_replay(const DaTape *T, const double *seg_pars, const double *y0,
                     double *states) {
  const int n = T->n;
  const size_t nc = NT_MAX_COLS;
  double *Z = (double *)calloc((size_t)n * nc, sizeof(double));
  double *tmp = (double *)calloc((size_t)n * nc, sizeof(double));
  double *acor = (double *)calloc((size_t)n, sizeof(double));
  double *carry = dup_doubles(y0, (size_t)n);
  double *y = (double *)malloc((size_t)n * sizeof(double));
  double *f = (double *)malloc((size_t)n * sizeof(double));
  double *G = (double *)malloc((size_t)n * sizeof(double));
  double *M = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  double *J = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  integer *ip = (integer *)malloc((size_t)n * sizeof(integer));
  int status = (Z && tmp && acor && carry && y && f && G && M && J && ip) ? 0 : DA_NO_MEMORY;
#define ZC(k) (Z + (size_t)(k) * (size_t)n)
  for (int o = 0; status == 0 && o < T->n_ops; o++) {
    const double *op = T->ops + T->op_off[o];
    const int code = (int)op[0];
    const double *a = op + 1;
    switch (code) {
    case NT_INIT:
    case NT_DERIV: {
      const double *pars = seg_pars + (size_t)a[0] * (size_t)T->n_pars;
      if (code == NT_INIT) {
        memset(Z, 0, (size_t)n * nc * sizeof(double));
        memcpy(ZC(0), carry, (size_t)n * sizeof(double));
      }
      T->model(n, a[1], ZC(0), f, (double *)pars);
      for (int i = 0; i < n; i++) ZC(1)[i] = a[2] * f[i];
      break;
    }
    case NT_SCALE: {
      double r = 1.0;
      for (int k = 1; k <= (int)a[0]; k++) {
        r *= a[1];
        for (int i = 0; i < n; i++) ZC(k)[i] *= r;
      }
      break;
    }
    case NT_PREDICT:
    case NT_RETRACT: {
      const int q = (int)a[0];
      const double sign = code == NT_PREDICT ? 1.0 : -1.0;
      memset(tmp, 0, (size_t)n * (size_t)(q + 1) * sizeof(double));
      for (int j = 0; j <= q; j++) {
        for (int k = j; k <= q; k++) {
          double c = binom(k, j) * (((k - j) % 2 == 1) ? sign : 1.0);
          for (int i = 0; i < n; i++) tmp[i + (size_t)j * n] += c * ZC(k)[i];
        }
      }
      memcpy(Z, tmp, (size_t)n * (size_t)(q + 1) * sizeof(double));
      break;
    }
    case NT_CORRECT: {
      const double *pars = seg_pars + (size_t)a[0] * (size_t)T->n_pars;
      const double t = a[1], d = a[2], b = a[3], c = a[4];
      memcpy(acor, a + 5, (size_t)n * sizeof(double));
      for (int it = 0; it < 50; it++) {
        for (int i = 0; i < n; i++) y[i] = ZC(0)[i] + c * acor[i];
        T->model(n, t, y, f, (double *)pars);
        T->jac(n, t, y, J, (double *)pars);
        for (int i = 0; i < n; i++) G[i] = -(acor[i] + d * ZC(1)[i] - b * f[i]);
        for (int k = 0; k < n * n; k++) M[k] = -b * c * J[k];
        for (int i = 0; i < n; i++) M[i + (size_t)i * n] += 1.0;
        if (solve_dense(n, M, G, ip) != 0) {
          status = DA_SINGULAR;
          break;
        }
        double norm = 0.0, anorm = 0.0;
        for (int i = 0; i < n; i++) {
          acor[i] += G[i];
          norm = fmax(norm, fabs(G[i]));
          anorm = fmax(anorm, fabs(acor[i]));
        }
        if (!isfinite(norm)) {
          status = DA_NOT_FINITE;
          break;
        }
        if (norm <= 1e-15 * (1.0 + anorm)) break;
      }
      break;
    }
    case NT_UPDATE:
      for (int k = 0; k <= (int)a[0]; k++) {
        for (int i = 0; i < n; i++) ZC(k)[i] += a[1 + k] * acor[i];
      }
      break;
    case NT_SET_ACOR:
      for (int i = 0; i < n; i++) ZC((int)a[0])[i] = a[1] * acor[i];
      break;
    case NT_SET_ZERO:
      memset(ZC((int)a[0]), 0, (size_t)n * sizeof(double));
      break;
    case NT_SET_COL:
      for (int i = 0; i < n; i++) ZC((int)a[0])[i] = a[2] * ZC((int)a[1])[i];
      break;
    case NT_AXPY_MULTI:
      for (int k = 0; k < (int)a[2]; k++) {
        for (int i = 0; i < n; i++) ZC((int)a[1] + k)[i] += a[3 + k] * ZC((int)a[0])[i];
      }
      break;
    case NT_OUTPUT:
      nordsieck_eval(n, Z, (int)a[2], a[1], states + (size_t)a[0] * (size_t)n);
      break;
    case NT_OUTPUT_Y:
      memcpy(states + (size_t)a[0] * (size_t)n, carry, (size_t)n * sizeof(double));
      break;
    case NT_CARRY:
      nordsieck_eval(n, Z, (int)a[1], a[0], carry);
      break;
    default:
      status = DA_BAD_INPUT;
    }
  }
#undef ZC
  free(Z);
  free(tmp);
  free(acor);
  free(carry);
  free(y);
  free(f);
  free(G);
  free(M);
  free(J);
  free(ip);
  return status;
}

int da_replay(const DaTape *T, const double *seg_pars, const double *y0, double *out_pred) {
  if (!T) return DA_BAD_INPUT;
  const int n = T->n;
  double *states = (double *)malloc((size_t)(T->n_points > 0 ? T->n_points : 1) * (size_t)n *
                                    sizeof(double));
  if (!states) return DA_NO_MEMORY;
  int status = T->method == DA_METHOD_RADAU5 ? rd_replay(T, seg_pars, y0, states)
                                             : nt_replay(T, seg_pars, y0, states);
  if (status == 0) status = predictions(T, states, seg_pars, out_pred);
  free(states);
  return status;
}
