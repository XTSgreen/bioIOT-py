# -*- coding: utf-8 -*-
"""Plug-and-play estimator API for bioIOT.

``IOTModel`` wraps the functional core in a scikit-learn-style object::

    import bioiot
    model = bioiot.IOTModel().fit(phi, a, b, T)   # one scenario or a list
    model.theta_        # debiased feature coefficients
    model.support_      # selected-feature mask
    P = model.plan(phi, a, b)           # transport plan under fitted theta
    Q = model.row_conditional(phi, a)   # state-transition matrix
"""

import numpy as np
import torch

from .features import make_cost, row_ce_loss, zscore_phi
from .fitting import fit_uot
from .solve import row_conditional, soft_sinkhorn, to_prob_vector

__all__ = ["IOTModel", "fit"]


def _as_scenario_list(phi, a, b, T):
    """Accept one scenario (K, K, F arrays) or parallel lists of scenarios."""
    if isinstance(phi, (list, tuple)):
        scen = [(np.asarray(p, dtype=np.float64), aa, bb, tt)
                for p, aa, bb, tt in zip(phi, a, b, T)]
    else:
        scen = [(np.asarray(phi, dtype=np.float64), a, b, T)]
    if not scen:
        raise ValueError("no scenarios provided")
    return scen


def _standardize_with_meta(p, meta):
    """Re-apply a z-score transform stored during fit."""
    K = p.shape[0]
    flat = p.reshape(K * K, p.shape[-1])
    mu = np.array([m for m, _ in meta])
    sd = np.array([s for _, s in meta])
    return ((flat - mu) / sd).reshape(K, K, p.shape[-1])


class IOTModel:
    """Semi-relaxed inverse optimal transport estimator.

    Solves for feature coefficients ``theta`` such that the soft-marginal OT
    plan induced by the linear cost ``C = -einsum(phi, theta)`` reproduces
    the observed row-conditional transitions ``T``.

    Parameters
    ----------
    mu : float
        Soft-marginal strength (KL anchor on the target composition).
        Default 0.5, the synthetic-calibration working point of the paper.
    eps : float
        Entropic regularization. Default 1.0.
    lam : float
        l1 strength on theta in stage 1. Default 0.05.
    epochs : int
        Adam steps per stage. Default 300.
    lr : float
        Adam learning rate. Default 1e-2.
    n_restart : int
        Random restarts; lowest final unregularized loss wins. Default 4.
    seed : int
        Base random seed. Default 1.
    two_stage : bool
        Buehlmann-style debias refit on the selected support. Default True.
    standardize : bool
        Z-score each scenario's features across all (i, j) entries before
        fitting (paper pipeline). Default True.
    iters, damp :
        Forward fixed-point solver controls.

    Attributes (after ``fit``)
    --------------------------
    theta_ : ndarray (F,)
        Debiased coefficients (standardized feature space when
        ``standardize=True``).
    theta1_ : ndarray (F,)
        Stage-1 l1-regularized coefficients.
    support_ : ndarray (F,) of bool
        Selected features (|theta1| > 1e-3).
    loss_ : float
        Unregularized row cross-entropy of ``theta_`` on the training data.
    restart_losses_ : list of float
        Loss of each restart's debiased theta.
    """

    def __init__(self, mu=0.5, eps=1.0, lam=0.05, epochs=300, lr=1e-2,
                 n_restart=4, seed=1, two_stage=True, standardize=True,
                 iters=1000, damp=0.5, verbose=False):
        self.mu = mu
        self.eps = eps
        self.lam = lam
        self.epochs = epochs
        self.lr = lr
        self.n_restart = n_restart
        self.seed = seed
        self.two_stage = two_stage
        self.standardize = standardize
        self.iters = iters
        self.damp = damp
        self.verbose = verbose

    # ---------------- fitting ----------------
    def fit(self, phi, a, b, T):
        """Fit theta from observed scenarios.

        Parameters
        ----------
        phi : (K, K, F) array or list of such arrays
            Scenario feature tensors (``phi[:, :, k] = u[:, k][None, :]``
            for a pure-column feature k, plus interaction blocks).
        a, b : (K,) array or list of arrays
            Source / target state masses (any positive scale; normalized
            internally). Zero-mass source states are allowed and masked in
            the likelihood.
        T : (K, K) array or list of arrays
            Observed row-conditional transition matrices (placeholder rows
            on zero-mass source states are ignored).

        Returns
        -------
        self
        """
        scen = _as_scenario_list(phi, a, b, T)
        phis, a_l, b_l, T_l = [], [], [], []
        self.feature_meta_ = []
        for p, aa, bb, tt in scen:
            if p.ndim != 3:
                raise ValueError(f"`phi` must be (K, K, F), got shape {p.shape}")
            if self.standardize:
                p, meta = zscore_phi(p)
                self.feature_meta_.append(meta)
            phis.append(p)
            a_l.append(np.asarray(aa, dtype=np.float64))
            b_l.append(np.asarray(bb, dtype=np.float64))
            T_l.append(np.asarray(tt, dtype=np.float64))

        if self.verbose:
            from .solve import log
            log(f"[bioiot] fit: {len(phis)} scenario(s), K={phis[0].shape[0]}, "
                f"F={phis[0].shape[2]}, n_restart={self.n_restart}")

        theta1, support, theta_final, all_l1, all_fin = fit_uot(
            phis, a_l, b_l, T_l,
            mu=self.mu, lam=self.lam, eps=self.eps,
            n_restart=self.n_restart, seed=self.seed,
            epochs=self.epochs, lr=self.lr, two_stage=self.two_stage,
            iters=self.iters, verbose=self.verbose,
        )
        self.theta1_ = np.asarray(theta1)
        self.support_ = np.asarray(support, dtype=bool)
        self.theta_ = np.asarray(theta_final)
        # store the (post-standardization) scenarios so transition_matrix()
        # can re-solve plans without the user re-passing inputs
        self.scenarios_ = [
            {"phi": p, "a": np.asarray(aa, dtype=np.float64),
             "b": np.asarray(bb, dtype=np.float64)}
            for p, aa, bb in zip(phis, a_l, b_l)
        ]

        T_t = [torch.as_tensor(x, dtype=torch.float64) for x in T_l]
        a_t = [torch.as_tensor(x, dtype=torch.float64) for x in a_l]
        restart_losses = []
        for th in np.asarray(all_fin):
            P_l = [self._plan_from_theta(th, p, aa, bb)
                   for p, aa, bb in zip(phis, a_l, b_l)]
            restart_losses.append(float(row_ce_loss(T_t, P_l, a_t)))
        self.restart_losses_ = restart_losses
        self.loss_ = min(restart_losses)
        self.n_scenarios_ = len(phis)
        self.K_ = np.array([p.shape[0] for p in phis])  # scenarios may differ in K
        self.F_ = phis[0].shape[2]
        return self

    # ---------------- inference ----------------
    def _prepare(self, phi):
        """Standardize a new scenario with the meta stored at fit time."""
        p = np.asarray(phi, dtype=np.float64)
        if p.ndim != 3:
            raise ValueError(f"`phi` must be (K, K, F), got shape {p.shape}")
        if self.standardize and getattr(self, "feature_meta_", None):
            p = _standardize_with_meta(p, self.feature_meta_[0])
        return p

    def _plan_from_theta(self, theta, phi_prepped, a, b):
        a = to_prob_vector(a, "a")
        b = to_prob_vector(b, "b")
        C = make_cost(phi_prepped, np.asarray(theta, dtype=np.float64))
        return soft_sinkhorn(C, a, b, mu=self.mu, eps=self.eps,
                             iters=self.iters, damp=self.damp)

    def cost(self, phi):
        """Linear cost matrix C = -einsum(phi, theta_) under the fitted theta."""
        return make_cost(self._prepare(phi), self.theta_)

    def plan(self, phi, a, b):
        """Transport plan P for a scenario under the fitted theta.

        Returns a (K, K) float64 torch tensor with ``P @ 1 = a``.
        """
        a = to_prob_vector(a, "a")
        b = to_prob_vector(b, "b")
        C = self.cost(phi)
        return soft_sinkhorn(C, a, b, mu=self.mu, eps=self.eps,
                             iters=self.iters, damp=self.damp)

    def row_conditional(self, phi, a):
        """Row-conditional transition matrix Q = P / a under the fitted theta."""
        a = np.asarray(a, dtype=np.float64)
        P = self.plan(phi, a, np.ones_like(a))
        return row_conditional(P, a)

    def score(self, phi, a, b, T):
        """Unregularized row cross-entropy of ``theta_`` (lower is better)."""
        scen = _as_scenario_list(phi, a, b, T)
        a_l, T_l, P_l = [], [], []
        for p, aa, bb, tt in scen:
            aa = np.asarray(aa, dtype=np.float64)
            P_l.append(self._plan_from_theta(self.theta_, self._prepare(p), aa, bb))
            a_l.append(aa)
            T_l.append(np.asarray(tt, dtype=np.float64))
        T_t = [torch.as_tensor(x, dtype=torch.float64) for x in T_l]
        a_t = [torch.as_tensor(x, dtype=torch.float64) for x in a_l]
        return float(row_ce_loss(T_t, P_l, a_t))


def fit(phi, a, b, T, **kwargs):
    """One-liner: ``bioiot.fit(phi, a, b, T)`` == ``IOTModel(**kwargs).fit(...)``."""
    return IOTModel(**kwargs).fit(phi, a, b, T)
