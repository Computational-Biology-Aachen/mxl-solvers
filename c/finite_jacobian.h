/*
 * finite_jacobian.h — evaluate a model's compiled Jacobian, with finite
 * differences for the columns it can't evaluate.
 *
 * A Jacobian derived symbolically can be NaN or inf where the model itself is
 * finite: d/dx of 1 / (1 + exp(x)) is exp(x) / (1 + exp(x))^2 = inf / inf once
 * exp(x) overflows, d/dx of x^a with 0 < a < 1 is inf at x = 0. Solvers would
 * fail on such a Jacobian although finite differences of the model work, so
 * every column holding a non-finite entry is replaced by central differences
 * of the model. Columns that stay non-finite are left for the solver to fail on.
 *
 * Header-only, so each translation unit that calls compiled Jacobians can use
 * it without linking another object.
 */
#ifndef FINITE_JACOBIAN_H
#define FINITE_JACOBIAN_H

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef void (*FjModelFn)(int, double, double *, double *, double *);

/* Writes df/dy at (t, y) to J (column-major n x n) with jac, then replaces the
 * columns with a non-finite entry by central differences of model. y isn't
 * modified. Returns the number of replaced columns, or -1 if their scratch
 * space couldn't be allocated (J then holds jac's values). */
static inline int finite_jacobian(FjModelFn model, FjModelFn jac, int n, double t,
                                  const double *y, double *J, double *pars) {
  jac(n, t, (double *)y, J, pars);
  int first_bad = -1;
  for (int k = 0; k < n * n && first_bad < 0; k++) {
    if (!isfinite(J[k])) first_bad = k / n;
  }
  if (first_bad < 0) return 0;
  double *work = (double *)malloc((size_t)3 * (size_t)n * sizeof(double));
  if (!work) return -1;
  double *yh = work, *up = work + n, *down = work + 2 * n;
  memcpy(yh, y, (size_t)n * sizeof(double));
  const double rel = cbrt(DBL_EPSILON);
  int replaced = 0;
  for (int j = first_bad; j < n; j++) {
    double *col = J + (size_t)j * n;
    int finite = 1;
    for (int i = 0; i < n && finite; i++) finite = isfinite(col[i]);
    if (finite) continue;
    const double h = rel * fmax(fabs(y[j]), 1e-5);
    yh[j] = y[j] + h;
    model(n, t, yh, up, pars);
    yh[j] = y[j] - h;
    model(n, t, yh, down, pars);
    yh[j] = y[j];
    for (int i = 0; i < n; i++) col[i] = (up[i] - down[i]) / (2.0 * h);
    replaced++;
  }
  free(work);
  return replaced;
}

#endif /* FINITE_JACOBIAN_H */
