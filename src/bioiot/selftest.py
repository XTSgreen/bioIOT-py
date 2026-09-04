# -*- coding: utf-8 -*-
"""Synthetic self-test: implicit-gradient check + theta recovery."""

import time

import numpy as np
import torch

from .features import make_cost, row_ce_loss, zscore_phi
from .fitting import EPS, MU, fit_uot
from .solve import log, row_conditional, soft_sinkhorn
from .solve import _TORCH_F64

__all__ = ["self_test"]


def self_test():
    """Run the paper's synthetic checks and return a result dict.

    1. Implicit gradient (autograd) vs central finite differences at a
       non-stationary point 0.5 * theta*.
    2. Multi-feature recovery: T generated at theta* must be recovered by
       the multi-restart fitter (correlation > 0.7, max error < 1.0).
    3. Single pure-column feature recovery (the hard identifiability case).
    """
    rng = np.random.default_rng(0)
    K = 6
    a = np.ones(K) / K
    b = np.ones(K) / K
    u = rng.uniform(0.3, 3.0, size=(K, 5))
    sim = rng.uniform(-1, 1, size=(K, K))
    phi = np.zeros((K, K, 6))
    for k in range(5):
        phi[:, :, k] = u[:, k][None, :]
    phi[:, :, 5] = sim
    phi_z, _ = zscore_phi(phi)
    theta_star = np.array([0.8, 1.5, -1.0, 2.0, 0.6, 1.2])
    mu = MU

    P_true0 = soft_sinkhorn(make_cost(phi_z, theta_star), a, b, mu, eps=EPS)
    T = np.asarray(row_conditional(P_true0.detach(), a).numpy())
    th_chk = 0.5 * theta_star

    def L(th_np):
        Pp = soft_sinkhorn(make_cost(phi_z, th_np), a, b, mu, eps=EPS)
        return float(row_ce_loss([torch.as_tensor(T)], [Pp], [torch.as_tensor(a)]).detach())

    thg = torch.tensor(th_chk, dtype=_TORCH_F64, requires_grad=True)
    P0 = soft_sinkhorn(make_cost(phi_z, thg), a, b, mu, eps=EPS)
    loss0 = row_ce_loss([torch.as_tensor(T)], [P0], [torch.as_tensor(a)])
    loss0.backward()
    g_auto = thg.grad.numpy()
    h = 1e-4
    g_fd = np.array([(L(th_chk + h * np.eye(6)[k]) - L(th_chk - h * np.eye(6)[k])) / (2 * h)
                     for k in range(6)])
    log("=== bioiot self_test: implicit gradient (autograd vs finite diff) ===")
    log(f"  auto : {np.round(g_auto, 4)}")
    log(f"  fd   : {np.round(g_fd, 4)}")
    log(f"  ratio: {np.round(g_auto / (g_fd + 1e-300), 3)}")
    grad_ok = bool(np.allclose(g_auto, g_fd, rtol=1e-3, atol=1e-3))

    P_true = soft_sinkhorn(make_cost(phi_z, theta_star), a, b, mu)
    T_true = np.asarray(row_conditional(P_true, a).detach().numpy())
    log("\n=== bioiot self_test: multi-feature recovery ===")
    t0 = time.time()
    th1, sup, th_fin, all_l1, all_fin = fit_uot(
        [phi_z], [a], [b], [T_true], mu=mu, n_restart=3, seed=1, epochs=300)
    th_hat = th_fin
    corr = float(np.corrcoef(th_hat, theta_star)[0, 1])
    err = float(np.abs(th_hat - theta_star).max())
    log(f"  theta* = {np.round(theta_star, 3).tolist()}")
    log(f"  theta^ = {np.round(th_hat, 3).tolist()}")
    log(f"  corr={corr:.3f}  max|dtheta|={err:.3f}  elapsed {time.time() - t0:.0f}s")

    log("\n=== bioiot self_test: single pure-column feature recovery ===")
    _, _, thf, _, _ = fit_uot(
        [phi_z[:, :, :1]], [a], [b],
        [np.asarray(row_conditional(
            soft_sinkhorn(make_cost(phi_z[:, :, :1], np.array([2.0])), a, b, mu), a).numpy())],
        mu=mu, n_restart=3, seed=1, epochs=300)
    log(f"  theta^ = {float(thf[0]):.3f} (truth 2.0)")

    verdict = "PASS" if grad_ok and corr > 0.7 and err < 1.0 else "FAIL"
    log(f"\n  verdict: {verdict}")
    return dict(grad_ok=grad_ok, corr=corr, max_err=err, verdict=verdict,
                theta_star=theta_star.tolist(), theta_hat=th_hat.tolist())
