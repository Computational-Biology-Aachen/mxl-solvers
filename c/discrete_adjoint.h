/*
 * discrete_adjoint.h — discrete ("discretise-then-optimise") adjoint gradients.
 *
 * da_forward integrates a time course (piecewise-constant parameters per
 * protocol segment) and records every accepted step on a tape;
 * da_backward then returns the exact gradient of sum(cotangent * prediction)
 * of that computed solution, for any number of cotangents, by differentiating
 * the solver's own steps backwards. Step sizes (and orders) are frozen, the
 * implicit stage equations are differentiated with the implicit function
 * theorem, dense output at the data times is differentiated exactly. See
 * discrete_adjoint.c for the per-method details.
 *
 * Unlike adjoint_wrapper.c (continuous adjoint), the forward solve and the
 * backward pass are separate calls, so a caller can evaluate a loss on the
 * predictions before choosing the cotangents, and reuse one tape for several
 * cotangents.
 *
 * Functions are function pointers natively and table indices in wasm:
 *   model:        void(int n, double t, double* y, double* dydt, double* pars)
 *   jac:          same signature, writes the n x n Jacobian df/dy column-major
 *                 (out[i + j*n] = df_i/dy_j)
 *   derived:      same signature, writes the n_derived derived targets
 *   adjoint:      void(int n, double t, double* y, double* lam, double* pars,
 *                      double* dlam, double* dtheta), writes
 *                 dlam = -(df/dy)^T lam and dtheta = -lam^T df/dtheta for the
 *                 n_theta parameters it was generated for (adjoint_wrapper.c)
 *   derived_grad: same signature as model, per derived target
 *                 [dd/dy (n), dd/dtheta (n_theta)]
 */
#ifndef DISCRETE_ADJOINT_H
#define DISCRETE_ADJOINT_H

#include <stdint.h>

#define DA_METHOD_RADAU5 0
#define DA_METHOD_LSODA 1
#define DA_METHOD_CVODE_BDF 2
#define DA_METHOD_CVODE_ADAMS 3

/* Bits of da_forward's options. */
#define DA_REUSE_JACOBIAN 1 /* RADAU5's tuned Jacobian reuse (mxn_radau's reuse_jacobian) */
#define DA_FORWARD_JAC 2    /* the forward solver uses jac instead of finite differences */

/* da_forward / da_backward status codes, besides the solver's negative IDID. */
#define DA_NO_MEMORY (-1001)
#define DA_NOT_FINITE (-1002)
#define DA_TAPE_TOO_LARGE (-1003)
#define DA_BAD_INPUT (-1004)
#define DA_SINGULAR (-1005)

typedef struct DaTape DaTape;

/*
 * Integrates from t = 0 through n_seg segments (segment k ends at
 * seg_t_end[k] and uses the parameters seg_pars[k * n_pars ..]), carrying the
 * state across boundaries, and writes the targets at every data point into
 * out_pred (n_points x n_targets, point-major). A point on a segment boundary
 * belongs to the earlier segment; t_points must be ascending, within
 * [0, seg_t_end[n_seg - 1]].
 * Targets: target_kind[k] 0 = state target_index[k], 1 = derived target
 * target_index[k] (derived must then be set).
 * newton_scale (<= 0: 1) multiplies the solver's convergence tolerance of its
 * Newton / corrector iterations (RADAU5's FNEWT, LSODA's conit, CVODE's
 * nonlinear convergence coefficient): the gradient is exact for converged
 * iterations, so a smaller one makes it agree better with the predictions,
 * at the cost of a few more iterations.
 * On success *out_tape receives a tape the caller frees with da_free.
 * Returns 0, a negative solver IDID or one of the DA_* codes.
 */
int da_forward(intptr_t model, intptr_t jac, intptr_t derived, int method, int n_y,
               const double *y0, int n_pars, int n_seg, const double *seg_t_end,
               const double *seg_pars, int n_points, const double *t_points, int n_targets,
               const int *target_kind, const int *target_index, int n_derived, double rtol,
               double atol, int nmax, int options, double newton_scale, double max_tape_bytes,
               double *out_pred,
               DaTape **out_tape);

/*
 * Gradient of sum(cotangent * prediction) (cotangent laid out like
 * out_pred) w.r.t. the n_theta parameters adjoint was generated for, into
 * out_grad, and w.r.t. the initial state into out_lambda0 (n_y).
 * derived_grad may be 0 if no derived target has a nonzero cotangent.
 * Returns 0 or a DA_* code.
 */
int da_backward(DaTape *tape, intptr_t adjoint, intptr_t derived_grad, int n_theta,
                const double *cotangent, double *out_grad, double *out_lambda0);

/*
 * Replays the tape's steps (same times and step sizes) with the parameters
 * seg_pars and initial state y0 (laid out as in da_forward), solving every
 * step's stage equations to convergence, and writes the predictions into
 * out_pred. With the recorded parameters this reproduces da_forward's
 * predictions up to its Newton tolerance; da_backward's gradient is the
 * exact derivative of this function. For tests and gradient checks.
 * Returns 0, DA_SINGULAR if a Newton iteration fails, or another DA_* code.
 */
int da_replay(const DaTape *tape, const double *seg_pars, const double *y0, double *out_pred);

void da_free(DaTape *tape);

/* Diagnostics: operations of each kind on a Nordsieck tape's log (counts[code]
 * for codes 1..14, see discrete_adjoint.c), 0 for RADAU5 tapes. */
void da_tape_op_counts(const DaTape *tape, int *counts, int n_counts);

/* Accepted steps on the tape and the bytes it holds. */
int da_tape_steps(const DaTape *tape);
double da_tape_bytes(const DaTape *tape);

#endif
