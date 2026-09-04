# -*- coding: utf-8 -*-
"""Single-cell interface: cell embeddings -> states -> IOT -> pseudotime.

``run_iot`` is the matrix-level entry (cells x dims embedding + state
labels + two timepoint masks). ``run_iot_adata`` is the AnnData/scanpy
adapter (soft-gated).
"""

import numpy as np

from .features import make_cost, zscore_phi
from .model import IOTModel
from .solve import row_conditional, soft_sinkhorn
from .transitions import build_state_features, pseudotime_from_transition

__all__ = ["run_iot", "run_iot_adata"]


def _as_index(x, n):
    x = np.asarray(x)
    if x.dtype == bool:
        if x.shape[0] != n:
            raise ValueError("boolean mask length must match number of cells")
        return np.where(x)[0]
    return np.asarray(x, dtype=int)


def _pairwise_min_idx(src_emb, tgt_emb):
    """For each target cell, index of its nearest source cell."""
    # (T, S) squared distance via broadcasting; fine for demo-scale data
    d2 = ((tgt_emb[:, None, :] - src_emb[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1)


def run_iot(emb, state, from_mask, to_mask, root=None, n_dim=None,
            T_obs=None, mu=0.5, eps=1.0, lam=0.05, epochs=300, lr=1e-2,
            n_restart=4, seed=1, two_stage=True, **fit_kw):
    """Run the bioIOT pipeline on cell-level data.

    Aggregates cells into states, builds state-transition features from
    the cell embedding, optionally fits weights against an observed
    transition matrix ``T_obs`` (e.g. from lineage/clone data), solves the
    plan, and returns the state transition matrix with optional
    random-walk pseudotime.

    Parameters
    ----------
    emb : (n_cells, D) array-like
        Cell embedding (e.g. PCA space).
    state : (n_cells,) array-like
        State / cluster label per cell.
    from_mask, to_mask : boolean mask or index array
        Cells of the source / target timepoint.
    root : optional
        Root state label or index for pseudotime.
    n_dim : optional
        Number of embedding dims used as state features (default all).
    T_obs : optional (K, K) array-like
        Observed row-conditional transitions; when given, feature weights
        are fitted, otherwise the plan uses uniform weights.

    Returns
    -------
    dict
        ``fit`` (IOTModel or None), ``Q`` (K, K), ``u`` (K, D),
        ``a``, ``b``, ``theta``, ``states``, ``pseudotime`` (dict or None).
    """
    emb = np.asarray(emb, dtype=np.float64)
    state = np.array([str(s) for s in state])
    if emb.shape[0] != state.shape[0]:
        raise ValueError("`state` must have one label per cell (row)")
    src = _as_index(from_mask, emb.shape[0])
    tgt = _as_index(to_mask, emb.shape[0])
    states = sorted(set(state[src]) | set(state[tgt]))
    K = len(states)
    D = emb.shape[1] if n_dim is None else min(n_dim, emb.shape[1])
    E = emb[:, :D]

    # state centroids over ALL cells (timepoint-independent definition)
    u = np.vstack([E[state == s].mean(axis=0) for s in states])
    a_raw = np.array([(state[src] == s).sum() for s in states], dtype=float)
    b_raw = np.array([(state[tgt] == s).sum() for s in states], dtype=float)

    phi = build_state_features(u)
    phi_z, meta = zscore_phi(phi)

    if T_obs is not None:
        T_obs = np.asarray(T_obs, dtype=np.float64)
        if T_obs.shape != (K, K):
            raise ValueError(f"`T_obs` must be ({K}, {K})")
        model = IOTModel(mu=mu, eps=eps, lam=lam, epochs=epochs, lr=lr,
                         n_restart=n_restart, seed=seed,
                         two_stage=two_stage, standardize=False, **fit_kw)
        model.fit([phi_z], [a_raw], [b_raw], [T_obs])
        theta = model.theta_
    else:
        model = None
        theta = np.ones(phi_z.shape[2])

    C = make_cost(phi_z, theta)
    a_n = a_raw / a_raw.sum()
    b_n = b_raw / b_raw.sum()
    P = soft_sinkhorn(C, a_n, b_n, mu=mu, eps=eps)
    Q = np.asarray(row_conditional(P.detach(), a_n))

    out = {
        "fit": model,
        "Q": Q,
        "u": u,
        "a": a_n,
        "b": b_n,
        "theta": theta,
        "states": states,
        "phi_meta": meta,
    }
    if root is not None:
        out["pseudotime"] = pseudotime_from_transition(Q, root=root)
    return out


def run_iot_adata(adata, state_col, time_col, from_key, to_key,
                  use_rep="X_pca", root=None, **kw):
    """AnnData/scanpy adapter for :func:`run_iot`.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with a reduced dimension in ``obsm[use_rep]`` and state /
        timepoint columns in ``obs``.
    state_col, time_col : str
        Column names in ``adata.obs``.
    from_key, to_key : str
        Values of ``time_col`` marking source / target timepoints.
    use_rep : str
        Key of the embedding in ``adata.obsm`` (default "X_pca").

    Returns
    -------
    dict
        Same layout as :func:`run_iot`.
    """
    try:
        import anndata  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "anndata is required for run_iot_adata: pip install anndata"
        ) from e
    if use_rep not in adata.obsm:
        raise KeyError(f"`{use_rep}` not found in adata.obsm")
    emb = np.asarray(adata.obsm[use_rep])
    if state_col not in adata.obs or time_col not in adata.obs:
        raise KeyError("`state_col`/`time_col` must be columns of adata.obs")
    state = adata.obs[state_col].astype(str).values
    timev = adata.obs[time_col].astype(str).values
    return run_iot(emb, state, timev == str(from_key), timev == str(to_key),
                   root=root, **kw)
