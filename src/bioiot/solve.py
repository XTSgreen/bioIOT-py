# -*- coding: utf-8 -*-
"""Core solver: semi-relaxed OT plan with implicit gradients.

Solves the row-marginal-hard / column-marginal-KL-soft problem

    min_P <C, P> - eps * H(P) + mu * KL(col(P) || b),   s.t.  P 1 = a

Forward: Anderson-accelerated damped fixed-point iteration (pure numpy,
machine-precision convergence, no autograd through the solve).
Backward: implicit differentiation of the fixed point via the implicit
function theorem, which stays numerically stable where unrolled
backpropagation diverges (experiments: auto-grad ~1e292 vs finite
difference ~1e5).

Numerical core is a faithful packaging of the paper's self-contained
solver (``script/uot_fitter.py`` of the new_treatment_resistance_IOT
project).
"""

import numpy as np
import torch

_TORCH_F64 = torch.float64
_EPS_T = 1e-300

__all__ = ["soft_sinkhorn", "row_conditional", "log"]


def log(*a):
    """Print diagnostics even on non-UTF-8 consoles (e.g. GBK stdout)."""
    try:
        print(*a, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(x) for x in a)
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


def to_f64_tensor(x, name="input", ndim=None):
    """Coerce array-like/tensor to a float64 torch tensor with checks.

    Torch tensors keep their autograd graph (no detach) so gradients can
    flow through :func:`soft_sinkhorn` to ``C``.
    """
    if isinstance(x, torch.Tensor):
        t = x.to(_TORCH_F64).cpu()
    else:
        t = torch.as_tensor(np.asarray(x, dtype=np.float64))
    if ndim is not None and t.dim() != ndim:
        raise ValueError(f"`{name}` must be {ndim}-D, got shape {tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError(f"`{name}` contains non-finite values")
    return t


def to_prob_vector(x, name="marginal", size=None):
    """Coerce to float64 1-D non-negative vector auto-normalized to sum 1."""
    t = to_f64_tensor(x, name, ndim=1)
    if size is not None and t.numel() != size:
        raise ValueError(f"`{name}` must have length {size}, got {t.numel()}")
    if (t < 0).any():
        raise ValueError(f"`{name}` must be non-negative")
    s = float(t.sum())
    if s <= 0:
        raise ValueError(f"`{name}` must have positive total mass")
    return t / s


def _plan_from_col(col, C, a, b, mu, eps):
    """Rebuild the plan P from a converged column marginal (col = P^T 1)."""
    logits = -C / eps + (mu / eps) * np.log(b[None, :] / (col[None, :] + 1e-300))
    mlog = logits - logits.max(1, keepdims=True)
    e = np.exp(mlog)
    s = e / e.sum(1, keepdims=True)
    return a[:, None] * s


def _anderson_col(col0, C, a, b, mu, eps, iters, damp, m=6, tol=1e-12):
    """Anderson-accelerated fixed-point solve for the column marginal.

    f_j(col) = sum_i a_i * softmax_j(-C_ij/eps + (mu/eps)*log(b_j/col_j)).
    The accelerated candidate is only accepted when it lowers the residual;
    otherwise the update falls back to the damped iteration. Falls back to
    damped iteration entirely on LAPACK non-convergence.
    """
    def fcol(col):
        logits = -C / eps + (mu / eps) * np.log(b[None, :] / (col[None, :] + 1e-300))
        mlog = logits - logits.max(1, keepdims=True)
        e = np.exp(mlog)
        s = e / e.sum(1, keepdims=True)
        return s.T @ a

    col = col0.copy()
    xs, gs = [], []
    best_col, best_r = col, np.inf
    for it in range(iters):
        fx = fcol(col)
        g = fx - col
        r = float(np.abs(g).max())
        if r < best_r:
            best_r, best_col = r, fx.copy()
        if r < tol:
            return fx, _plan_from_col(fx, C, a, b, mu, eps)
        xs.append(col.copy())
        gs.append(g.copy())
        if len(xs) > m:
            xs.pop(0)
            gs.pop(0)
        col_dmp = damp * col + (1.0 - damp) * fx
        if len(xs) >= 2:
            try:
                gm = gs[-1]
                xm = xs[-1]
                G = np.column_stack([g_ - gm for g_ in gs[:-1]])
                c = -np.linalg.lstsq(G, gm, rcond=None)[0]
                X = np.column_stack(
                    [(x_ - xm) + (g_ - gm) for x_, g_ in zip(xs[:-1], gs[:-1])]
                )
                col_and = xm + gm + X @ c
                if np.all(np.isfinite(col_and)):
                    col_and = np.maximum(col_and, 1e-12)
                    col_and = col_and / col_and.sum()
                    fx_and = fcol(col_and)
                    r_and = float(np.abs(fx_and - col_and).max())
                    r_dmp = float(np.abs(fcol(col_dmp) - col_dmp).max())
                    col = col_and if r_and < r_dmp else col_dmp
                else:
                    col = col_dmp
            except np.linalg.LinAlgError:
                col = col_dmp
        else:
            col = col_dmp
    if best_r > 1e-4:
        log(f"[bioiot] warning: fixed-point residual {best_r:.2e} after iters={iters}")
    P = _plan_from_col(best_col, C, a, b, mu, eps)
    return best_col, P


class _UOTPlanFn(torch.autograd.Function):
    """Forward: fixed-point solve (no grad); backward: implicit function theorem."""

    @staticmethod
    def forward(ctx, C, a, b, mu, eps, iters, damp):
        C_np = C.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        col, P = _anderson_col(
            b_np.copy(), C_np, a_np, b_np, float(mu), float(eps), int(iters), float(damp)
        )
        ctx.save_for_backward(
            C, a, b,
            torch.as_tensor(col, dtype=_TORCH_F64),
            torch.as_tensor(P, dtype=_TORCH_F64),
        )
        ctx.mu = mu
        ctx.eps = eps
        ctx.damp = damp
        return torch.as_tensor(P, dtype=_TORCH_F64)

    @staticmethod
    def backward(ctx, grad_P):
        C, a, b, col, P = ctx.saved_tensors
        mu, eps, damp = ctx.mu, ctx.eps, ctx.damp
        K = C.shape[0]
        g = grad_P
        s = P / (a[:, None] + _EPS_T)
        Tmat = P.t() @ s
        invcol = 1.0 / (col + _EPS_T)
        A_mat = damp * torch.eye(K, dtype=_TORCH_F64) - (1.0 - damp) * (mu / eps) * \
            (torch.diag(col) - Tmat) * invcol[None, :]
        rowsum_g = (g * s).sum(1)
        direct = -(a / eps)[:, None] * s * (g - rowsum_g[:, None])
        term1 = (g * P).sum(0)
        rowsum_gP = (g * P).sum(1)
        term2 = s.t() @ rowsum_gP
        w = -(mu / eps) * invcol * (term1 - term2)
        lhs = torch.eye(K, dtype=_TORCH_F64) - A_mat.t() + 1e-12 * torch.eye(K, dtype=_TORCH_F64)
        adj = torch.linalg.solve(lhs, w[:, None]).squeeze(1)
        adj_rowsum = (adj[None, :] * s).sum(1)
        corr = -(1.0 - damp) * (a / eps)[:, None] * s * (adj[None, :] - adj_rowsum[:, None])
        dLdC = direct + corr
        return dLdC, None, None, None, None, None, None


def soft_sinkhorn(C, a, b, mu=0.5, eps=1.0, iters=1000, damp=0.5):
    """Solve ``min_P <C,P> - eps H(P) + mu KL(col(P)||b)  s.t. P1 = a``.

    Parameters
    ----------
    C : (K, K) array-like
        Cost matrix (numpy array or torch tensor, any float dtype).
    a, b : (K,) array-like
        Source / target marginal weights. Non-negative; auto-normalized to
        sum to 1, so raw count vectors are accepted as-is.
    mu : float
        Soft-marginal strength. ``mu -> inf`` recovers hard-marginal OT
        (pure column features unidentifiable); ``mu -> 0`` recovers a plain
        row-softmax (no target composition anchoring).
    eps : float
        Entropic regularization (> 0).
    iters : int
        Fixed-point iteration cap.
    damp : float
        Damping factor of the base iteration.

    Returns
    -------
    torch.Tensor (float64, K x K)
        Transport plan ``P`` with ``P @ 1 = a``; differentiable in ``C``
        through exact implicit differentiation.
    """
    C = to_f64_tensor(C, "C", ndim=2)
    K = C.shape[0]
    if C.shape[1] != K:
        raise ValueError(f"`C` must be square, got shape {tuple(C.shape)}")
    a = to_prob_vector(a, "a", size=K)
    b = to_prob_vector(b, "b", size=K)
    if eps <= 0:
        raise ValueError("`eps` must be > 0")
    if mu < 0:
        raise ValueError("`mu` must be >= 0")
    return _UOTPlanFn.apply(C, a, b, mu, eps, iters, damp)


def row_conditional(P, a, eps_t=_EPS_T):
    """Row-conditional transition matrix Q(i,:) = P(i,:) / a_i."""
    P = to_f64_tensor(P, "P", ndim=2)
    a = to_f64_tensor(a, "a", ndim=1)
    return P / (a[:, None] + eps_t)
