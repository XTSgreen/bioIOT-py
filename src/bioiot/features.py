# -*- coding: utf-8 -*-
"""Feature/cost construction, standardization and likelihood terms."""

import numpy as np
import torch

from .solve import _TORCH_F64, row_conditional, to_f64_tensor

__all__ = ["make_cost", "uot_plan", "row_ce_loss", "zscore_phi"]


def _coerce(x):
    """float64 torch tensor; torch tensors keep their autograd graph."""
    if isinstance(x, torch.Tensor):
        return x.to(_TORCH_F64)
    return torch.as_tensor(np.asarray(x, dtype=np.float64))


def make_cost(phi, theta):
    """Linear feature cost ``C = -einsum('ijk,k->ij', phi, theta)``.

    A 2-D ``phi`` (K, K) with scalar ``theta`` is promoted to a single
    feature for convenience. ``theta`` may require grad; the cost stays
    differentiable in it.
    """
    phi_t = _coerce(phi)
    theta_t = _coerce(theta)
    if phi_t.dim() == 2:
        phi_t = phi_t[:, :, None]
        theta_t = theta_t.reshape(1)
    if phi_t.dim() != 3:
        raise ValueError(f"`phi` must be (K, K, F), got shape {tuple(phi_t.shape)}")
    if theta_t.dim() != 1 or theta_t.numel() != phi_t.shape[2]:
        raise ValueError(
            f"`theta` must have length {phi_t.shape[2]} to match `phi` features"
        )
    if not torch.isfinite(phi_t).all() or not torch.isfinite(theta_t).all():
        raise ValueError("`phi`/`theta` contain non-finite values")
    return -torch.einsum("ijk,k->ij", phi_t, theta_t)


def uot_plan(phi, theta, a, b, mu=0.5, eps=1.0, iters=1000, damp=0.5):
    """Convenience wrapper: build the cost from (phi, theta) and solve."""
    from .solve import soft_sinkhorn

    return soft_sinkhorn(make_cost(phi, theta), a, b, mu=mu, eps=eps, iters=iters, damp=damp)


def row_ce_loss(T_l, P_l, a_l, eps_t=1e-12):
    """Cross-entropy between observed row-conditional T-hat and model Q = P/a.

    Zero-mass source states (a_i = 0) are masked out: their rows are
    unobservable placeholders and only inject constants. The log floor
    1e-12 (not 1e-300) keeps the gradient T/Q finite as Q -> 0.
    """
    if not (len(T_l) == len(P_l) == len(a_l)):
        raise ValueError("T_l, P_l, a_l must have the same number of scenarios")
    loss = torch.tensor(0.0, dtype=_TORCH_F64)
    for T, P, a in zip(T_l, P_l, a_l):
        a = to_f64_tensor(a, "a", ndim=1)
        T = to_f64_tensor(T, "T", ndim=2)
        P = to_f64_tensor(P, "P", ndim=2)
        Q = row_conditional(P, a, eps_t=eps_t)
        mask = (a > eps_t).double()
        loss = loss - (mask[:, None] * T * torch.log(Q + eps_t)).sum()
    return loss


def zscore_phi(phi):
    """Z-score each feature across all (i, j) entries.

    Returns ``(phi_z, meta)`` where ``meta[k] = (mean_k, sd_k)`` allows the
    identical transform to be re-applied to validation scenarios.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 3:
        raise ValueError(f"`phi` must be (K, K, F), got shape {phi.shape}")
    if not np.isfinite(phi).all():
        raise ValueError("`phi` contains non-finite values")
    K = phi.shape[0]
    p = phi.reshape(K * K, phi.shape[-1])
    mu = p.mean(0)
    sd = p.std(0)
    sd[sd < 1e-12] = 1.0
    pz = (p - mu) / sd
    return pz.reshape(K, K, phi.shape[-1]), [(float(m), float(s)) for m, s in zip(mu, sd)]
