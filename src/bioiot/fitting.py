# -*- coding: utf-8 -*-
"""Parameter fitting: two-stage debiased l1 fitting with multi-restart.

``fit_uot`` is the paper's main entry: semi-relaxed IOT with l1-regularized
Adam descent (implicit gradients), then a Buehlmann-style two-stage debias
refit on the selected support. ``fit_hard`` is the hard-marginal OT control
arm used in the identifiability comparison.
"""

import numpy as np
import torch

from .features import make_cost, row_ce_loss
from .solve import _TORCH_F64, log, soft_sinkhorn, to_f64_tensor

__all__ = ["fit_once", "fit_uot", "fit_hard"]

EPS = 1.0
LAM = 0.05
MU = 0.5
N_RESTART = 4
SEED = 1
UOT_ITERS = 1000


def _check_scenarios(phil, a_l, b_l, T_l):
    """Validate scenario lists.

    Scenarios may differ in the number of states K (e.g. empty clusters
    at some timepoints); the number of features F must be shared.
    """
    if not (len(phil) == len(a_l) == len(b_l) == len(T_l)):
        raise ValueError("phil, a_l, b_l, T_l must have the same number of scenarios")
    phi0 = np.asarray(phil[0], dtype=np.float64)
    if phi0.ndim != 3:
        raise ValueError(f"each `phi` must be (K, K, F), got shape {phi0.shape}")
    F = phi0.shape[-1]
    for i, (ph, a, b, T) in enumerate(zip(phil, a_l, b_l, T_l)):
        ph = np.asarray(ph, dtype=np.float64)
        if ph.ndim != 3 or ph.shape[2] != F:
            raise ValueError(
                f"scenario {i}: `phi` must be (K, K, {F}), got shape {ph.shape}"
            )
        if ph.shape[0] != ph.shape[1]:
            raise ValueError(f"scenario {i}: `phi` must be square (K, K, F)")
        if not np.isfinite(ph).all():
            raise ValueError(f"scenario {i}: `phi` contains non-finite values")
        K = ph.shape[0]
        a = to_f64_tensor(a, "a", ndim=1)
        b = to_f64_tensor(b, "b", ndim=1)
        T = to_f64_tensor(T, "T", ndim=2)
        if a.numel() != K or b.numel() != K or T.shape != (K, K):
            raise ValueError(
                f"scenario {i}: a/b length and T shape must match phi K={K}"
            )
        if (np.asarray(T) < 0).any():
            raise ValueError(f"scenario {i}: `T` must be non-negative")
    return F


def _adam_fit(theta, ce, lam, epochs, lr, warm_frac):
    """Shared Adam loop: l1 after warm-up, grad clip, finite-state restart."""
    opt = torch.optim.Adam([theta], lr=lr)
    warm = int(epochs * warm_frac)
    trace = []
    for it in range(epochs):
        opt.zero_grad()
        loss = ce(theta) + (lam if it >= warm else 0.0) * theta.abs().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
        with torch.no_grad():
            theta.clamp_(-10, 10)
            if not torch.isfinite(theta).all():
                theta.data.copy_(torch.randn_like(theta) * 0.05)
        if it % 100 == 0 or it == epochs - 1:
            trace.append(float(loss.detach()))
    return trace


def fit_once(phil, a_l, b_l, T_l, mu=MU, lam=LAM, eps=EPS, epochs=300, lr=1e-2,
             two_stage=True, iters=UOT_ITERS, seed=0, verbose=False):
    """Single semi-relaxed IOT fit (two-stage debias).

    Returns ``(theta1, support, theta_final, loss_final)`` where ``theta1``
    is the l1-regularized estimate, ``support`` its boolean support
    (|theta| > 1e-3), ``theta_final`` the debiased refit, and ``loss_final``
    the unregularized row-CE of ``theta_final``.
    """
    _check_scenarios(phil, a_l, b_l, T_l)
    torch.manual_seed(seed)
    F = phil[0].shape[-1]
    phi_t = [torch.as_tensor(np.asarray(ph, dtype=np.float64)) for ph in phil]
    a_t = [to_f64_tensor(aa, "a") for aa in a_l]
    b_t = [to_f64_tensor(bb, "b") for bb in b_l]
    T_t = [to_f64_tensor(T, "T") for T in T_l]

    def ce(theta):
        P_l = [soft_sinkhorn(make_cost(ph, theta), a, b, mu=mu, eps=eps, iters=iters)
               for ph, a, b in zip(phi_t, a_t, b_t)]
        return row_ce_loss(T_t, P_l, a_t)

    theta = (torch.randn(F) * 0.05).requires_grad_(True)
    trace = _adam_fit(theta, ce, lam, epochs, lr, 0.4)
    if verbose:
        for k, lv in enumerate(trace):
            log(f"    [fit] epoch ~{k * 100 + 1} loss={lv:.4f}")
    theta1 = theta.detach().clone()
    support = theta1.abs() > 1e-3
    theta_final = theta1
    if two_stage and support.any():
        theta2 = theta1.clone().requires_grad_(True)
        with torch.no_grad():
            theta2.masked_fill_(~support, 0.0)
        _adam_fit(theta2, ce, 0.0, int(epochs * 0.75), lr * 2.0, 0.0)
        theta_final = theta2.detach().clone()
    with torch.no_grad():
        loss_f = float(ce(theta_final))
    return theta1, support, theta_final, loss_f


def fit_uot(phil, a_l, b_l, T_l, mu=MU, lam=LAM, eps=EPS, n_restart=N_RESTART,
            seed=SEED, **kw):
    """Multi-restart semi-relaxed IOT fit.

    Returns ``(theta_best, support, theta_final, all_theta1, all_final)``
    where the best restart minimizes the final unregularized loss.
    """
    _check_scenarios(phil, a_l, b_l, T_l)
    best = None
    all_l1, all_fin = [], []
    for r in range(int(n_restart)):
        out = fit_once(phil, a_l, b_l, T_l, mu=mu, lam=lam, eps=eps,
                       seed=seed * 100 + r, **kw)
        all_l1.append(out[0].numpy())
        all_fin.append(out[2].numpy())
        if best is None or out[3] < best[3]:
            best = out
    return best[0], best[1], best[2], np.array(all_l1), np.array(all_fin)


# ---------------- hard-marginal OT control arm ----------------
def _hard_sinkhorn(cost, a, b, eps, iters=50):
    """Entropic hard-marginal Sinkhorn (column marginal strictly = b)."""
    K = torch.exp((-cost / eps).clamp(-30, 30))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iters):
        v = b / (K.t() @ u + 1e-12)
        u = a / (K @ v + 1e-12)
    return u[:, None] * K * v[None, :]


def _hard_fit_once(phil, a_l, b_l, T_l, lam, eps=0.1, lr=1e-2, epochs=300,
                   two_stage=True, seed=0):
    _check_scenarios(phil, a_l, b_l, T_l)
    torch.manual_seed(seed)
    F = phil[0].shape[-1]
    phi_t = [torch.as_tensor(np.asarray(ph, dtype=np.float64)) for ph in phil]
    a_t = [to_f64_tensor(aa, "a") for aa in a_l]
    b_t = [to_f64_tensor(bb, "b") for bb in b_l]
    T_t = [to_f64_tensor(T, "T") for T in T_l]

    def ce(theta):
        tot = torch.tensor(0.0, dtype=_TORCH_F64)
        for T, a, b, ph in zip(T_t, a_t, b_t, phi_t):
            pi = _hard_sinkhorn(make_cost(ph, theta), a, b, eps)
            Q = pi / (a[:, None] + 1e-12)
            tot = tot - (T * torch.log(Q + 1e-12)).sum()
        return tot

    theta = (torch.randn(F) * 0.05).requires_grad_(True)
    _adam_fit(theta, ce, lam, epochs, lr, 0.4)
    theta1 = theta.detach().clone()
    support = theta1.abs() > 1e-3
    theta_final = theta1
    if two_stage and support.any():
        theta2 = theta1.clone().requires_grad_(True)
        with torch.no_grad():
            theta2.masked_fill_(~support, 0.0)
        _adam_fit(theta2, ce, 0.0, int(epochs * 0.5), lr, 0.0)
        theta_final = theta2.detach().clone()
    with torch.no_grad():
        loss_f = float(ce(theta_final))
    return theta1, support, theta_final, loss_f


def fit_hard(phil, a_l, b_l, T_l, lam=LAM, eps=0.1, n_restart=N_RESTART, seed=0, **kw):
    """Multi-restart hard-marginal OT fit (control arm).

    Returns the same tuple layout as :func:`fit_uot`.
    """
    _check_scenarios(phil, a_l, b_l, T_l)
    best = None
    all_l1, all_fin = [], []
    for r in range(int(n_restart)):
        out = _hard_fit_once(phil, a_l, b_l, T_l, lam, eps=eps, seed=seed * 100 + r, **kw)
        all_l1.append(out[0].numpy())
        all_fin.append(out[2].numpy())
        if best is None or out[3] < best[3]:
            best = out
    return best[0], best[1], best[2], np.array(all_l1), np.array(all_fin)
