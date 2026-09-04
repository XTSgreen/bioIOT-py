# -*- coding: utf-8 -*-
"""Trajectory layer: transition matrices, random-walk pseudotime, features."""

import numpy as np

from .features import make_cost, zscore_phi
from .solve import row_conditional, soft_sinkhorn, to_prob_vector

__all__ = [
    "transition_matrix",
    "pseudotime_from_transition",
    "build_state_features",
]


def transition_matrix(model, which=0, phi=None, a=None, b=None):
    """Row-conditional state-transition matrix at the fitted theta.

    Parameters
    ----------
    model : bioiot.IOTModel
        A fitted model.
    which : int
        Scenario index (default 0). Scenarios may differ in K.
    phi, a, b : optional
        User-supplied scenario overriding the stored one; ``phi`` is
        standardized with the transform stored at fit time (scenario 0).

    Returns
    -------
    numpy.ndarray (K, K)
        Row-stochastic transition matrix Q = P / a.
    """
    if not hasattr(model, "scenarios_") or not hasattr(model, "theta_"):
        raise TypeError("`model` must be a fitted bioiot.IOTModel")
    if phi is None:
        sc = model.scenarios_[which]
        phi_pre, a_raw, b_raw = sc["phi"], sc["a"], sc["b"]
    else:
        phi_pre = model._prepare(phi)
        a_raw = np.asarray(a, dtype=np.float64)
        b_raw = np.asarray(b, dtype=np.float64)
    a_n = a_raw / a_raw.sum()
    b_n = b_raw / b_raw.sum()
    C = make_cost(phi_pre, model.theta_)
    P = soft_sinkhorn(C, a_n, b_n, mu=model.mu, eps=model.eps,
                      iters=model.iters, damp=model.damp)
    return np.asarray(row_conditional(P, a_n))


def pseudotime_from_transition(Q, root):
    """Random-walk pseudotime: expected hitting times to the root state.

    States with no outgoing mass (e.g. zero-mass source rows of an IOT
    plan, or terminal states) are bounced back along their inflow
    distribution so the walk stays ergodic and terminal states land late
    instead of degenerate.

    Parameters
    ----------
    Q : (K, K) array-like
        Row-stochastic transition matrix.
    root : int or str
        Root state index or label (labels come from a pandas-backed
        index when available, else "S1".."SK").

    Returns
    -------
    dict
        ``{label: expected_steps}``; the root has value 0.0.
    """
    Q = np.array(Q, dtype=np.float64, copy=True)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("`Q` must be square (K, K)")
    if not np.isfinite(Q).all() or (Q < 0).any():
        raise ValueError("`Q` must be finite and non-negative")
    K = Q.shape[0]
    nms = [f"S{i + 1}" for i in range(K)]
    try:  # pandas index, if the caller passed a DataFrame
        idx = getattr(Q, "index", None)
        if idx is not None:
            nms = [str(x) for x in idx]
    except Exception:
        pass
    if isinstance(root, str):
        if root not in nms:
            raise ValueError(f"`root` label {root!r} not found in state labels")
        root = nms.index(root)
    if not (0 <= root < K):
        raise ValueError("`root` must be an index in [0, K) or a state label")

    zero = Q.sum(axis=1) < 1e-12
    if zero.any():
        cs = Q.sum(axis=0)
        for j in np.where(zero)[0]:
            r = Q[:, j] / max(cs[j], 1e-12)  # inflow distribution into j
            if r.sum() <= 0 or not np.isfinite(r.sum()):
                r = np.ones(K)
            Q[j, :] = r / r.sum()

    free = [i for i in range(K) if i != root]
    if not free:
        return {nms[root]: 0.0}
    Qf = Q[np.ix_(free, free)]
    h = np.linalg.solve(np.eye(len(free)) - Qf, np.ones(len(free)))
    pt = np.zeros(K)
    pt[free] = h
    return {nms[i]: float(pt[i]) for i in range(K)}


def build_state_features(u):
    """IOT state-transition features from state centroids.

    Builds the paper's feature library: per-dimension pure-column
    target-state features plus the state-similarity interaction block
    ``phi[:, :, D] = u @ u.T``.

    Parameters
    ----------
    u : (K, D) array-like
        State feature vectors (e.g. cluster centroids in a reduced space).

    Returns
    -------
    numpy.ndarray (K, K, D + 1)
        Feature array; the last slice is the "sim" interaction.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"`u` must be (K, D), got shape {u.shape}")
    if not np.isfinite(u).all():
        raise ValueError("`u` contains non-finite values")
    K, D = u.shape
    phi = np.zeros((K, K, D + 1))
    for d in range(D):
        phi[:, :, d] = np.repeat(u[None, :, d], K, axis=0)  # phi[i, j, d] = u[j, d]
    phi[:, :, D] = u @ u.T
    return phi


def standardize_like(phi, meta):
    """Re-apply a z-score transform (mean/sd per feature) stored at fit time."""
    p = np.asarray(phi, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"`phi` must be (K, K, F), got shape {p.shape}")
    K = p.shape[0]
    flat = p.reshape(K * K, p.shape[-1])
    mu = np.array([m for m, _ in meta])
    sd = np.array([s for _, s in meta])
    return ((flat - mu) / sd).reshape(K, K, p.shape[-1])
