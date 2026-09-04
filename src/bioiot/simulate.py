# -*- coding: utf-8 -*-
"""Reproducible synthetic demo data for bioIOT."""

import numpy as np

from .features import make_cost, zscore_phi
from .solve import row_conditional, soft_sinkhorn
from .transitions import build_state_features

__all__ = ["simulate_iot_states"]


def simulate_iot_states(K=6, F_emb=2, n_cells=50, seed=1,
                        theta_true=(0.9, -0.7, 1.1), mu=0.5, eps=1.0):
    """Simulate a single-cell-like state-transition dataset with truth.

    Returns a dict with state centroids ``u``, features ``phi``,
    normalized masses ``a``/``b``, the true plan ``P_true`` and
    row-conditional ``T_true``, true weights ``theta_true``, a 2-D
    ``embedding`` for plots, and cell-level metadata (``cell_embedding``,
    ``cell_state``, ``cell_time`` with two timepoints "t0"/"t1") for the
    :func:`run_iot` single-cell interface.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.3, 3.0, size=(K, F_emb))
    phi = build_state_features(u)
    phi_z, _ = zscore_phi(phi)
    theta_true = np.asarray(theta_true, dtype=np.float64)
    if theta_true.shape[0] != phi.shape[2]:
        raise ValueError(
            f"`theta_true` must have length {phi.shape[2]} (F_emb + 1)")
    a = np.ones(K) / K
    b_raw = 1.0 + 0.5 * (np.arange(K) % 2)  # non-uniform target composition
    b = b_raw / b_raw.sum()
    C = make_cost(phi_z, theta_true)
    P = soft_sinkhorn(C, a, b, mu=mu, eps=eps)
    T_true = np.asarray(row_conditional(P.detach(), a))

    n2 = min(2, F_emb)
    embedding = u[:, :n2] + rng.uniform(-0.15, 0.15, size=(K, n2))

    D = F_emb
    blocks = []
    for k in range(K):
        m1 = np.concatenate([u[k, :], np.zeros(max(0, D - F_emb))])[:D]
        blocks.append(rng.normal(m1, 0.25, size=(n_cells * 2, D)))
    cell_embedding = np.vstack(blocks)
    cell_state = np.repeat([f"S{k + 1}" for k in range(K)], n_cells * 2)
    cell_time = np.tile(np.repeat(["t0", "t1"], n_cells), K)

    return {
        "u": u,
        "phi": phi,
        "a": a,
        "b": b,
        "P_true": np.asarray(P.detach()),
        "T_true": T_true,
        "theta_true": theta_true,
        "embedding": embedding,
        "K": K,
        "F_emb": F_emb,
        "cell_embedding": cell_embedding,
        "cell_state": cell_state,
        "cell_time": cell_time,
    }
