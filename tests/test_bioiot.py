# -*- coding: utf-8 -*-
"""bioIOT test suite (unittest, CPU, fast)."""

import unittest

import numpy as np
import torch

import bioiot
from bioiot import IOTModel, make_cost, row_ce_loss, row_conditional, soft_sinkhorn, zscore_phi
from bioiot.fitting import fit_once


def _random_cost(K, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(K, K))


class TestSoftSinkhorn(unittest.TestCase):
    def test_row_marginal_exact_and_nonneg(self):
        K = 8
        a = np.ones(K) / K
        b = np.ones(K) / K
        P = soft_sinkhorn(_random_cost(K), a, b, mu=0.5, eps=1.0)
        Pnp = P.detach().numpy()
        self.assertTrue(np.all(np.isfinite(Pnp)))
        self.assertGreaterEqual(float(Pnp.min()), 0.0)
        np.testing.assert_allclose(Pnp.sum(1), a, atol=1e-8)

    def test_unnormalized_marginals_accepted(self):
        K = 5
        a_raw = np.full(K, 37.0)          # arbitrary positive scale
        b_raw = np.arange(1, K + 1) * 11.0
        a = a_raw / a_raw.sum()
        b = b_raw / b_raw.sum()
        P1 = soft_sinkhorn(_random_cost(K, 1), a, b, mu=0.5)
        P2 = soft_sinkhorn(_random_cost(K, 1), a_raw, b_raw, mu=0.5)
        np.testing.assert_allclose(P1.numpy(), P2.numpy(), atol=1e-12)
        np.testing.assert_allclose(P2.numpy().sum(1), a, atol=1e-8)

    def test_mu_interpolates_toward_hard_marginal(self):
        K = 6
        a = np.ones(K) / K
        b = np.exp(np.linspace(0, 0.7, K)); b /= b.sum()
        C = _random_cost(K, 2)
        dev = []
        for mu in (0.05, 5.0):
            P = soft_sinkhorn(C, a, b, mu=mu, iters=5000).detach().numpy()
            col = P.sum(0)
            dev.append(float(np.abs(col - b).max()))
        self.assertLess(dev[1], dev[0] + 1e-12,
                        "larger mu must anchor col(P) closer to b")

    def test_torch_input_and_grad_flows_to_C(self):
        K = 4
        rng = np.random.default_rng(9)
        C = torch.randn(K, K, dtype=torch.float64, requires_grad=True)
        a = torch.ones(K) / K
        # a random linear functional of P depends on C (P.sum() does not:
        # the hard row marginal pins the total mass regardless of C)
        W = torch.as_tensor(rng.random((K, K)), dtype=torch.float64)
        P = soft_sinkhorn(C, a, np.ones(K) / K, mu=0.5)
        (P * W).sum().backward()
        self.assertTrue(torch.isfinite(C.grad).all())
        self.assertGreater(float(C.grad.abs().sum()), 0.0)

    def test_input_validation(self):
        K = 4
        with self.assertRaises(ValueError):
            soft_sinkhorn(np.zeros((K, K + 1)), np.ones(K) / K, np.ones(K) / K)
        with self.assertRaises(ValueError):
            soft_sinkhorn(_random_cost(K), -np.ones(K), np.ones(K) / K)
        with self.assertRaises(ValueError):
            soft_sinkhorn(_random_cost(K), np.zeros(K), np.ones(K) / K)
        with self.assertRaises(ValueError):
            soft_sinkhorn(_random_cost(K), np.ones(K) / K, np.ones(K) / K, eps=0.0)


class TestImplicitGradient(unittest.TestCase):
    def test_autograd_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        K, F = 5, 2
        a = np.ones(K) / K
        b = np.ones(K) / K
        phi = rng.normal(size=(K, K, F))
        theta_star = np.array([0.9, -1.3])
        T = np.asarray(row_conditional(
            soft_sinkhorn(make_cost(phi, theta_star), a, b, mu=0.5).detach(), a).numpy())
        th0 = np.array([0.4, -0.5])

        def L(th):
            Pp = soft_sinkhorn(make_cost(phi, th), a, b, mu=0.5)
            return float(row_ce_loss([torch.as_tensor(T)], [Pp], [torch.as_tensor(a)]))

        thg = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
        P0 = soft_sinkhorn(make_cost(phi, thg), a, b, mu=0.5)
        row_ce_loss([torch.as_tensor(T)], [P0], [torch.as_tensor(a)]).backward()
        g_auto = thg.grad.numpy()
        h = 1e-4
        g_fd = np.array([(L(th0 + h * np.eye(F)[k]) - L(th0 - h * np.eye(F)[k])) / (2 * h)
                         for k in range(F)])
        np.testing.assert_allclose(g_auto, g_fd, rtol=1e-3, atol=1e-3)


class TestFeaturesAndLoss(unittest.TestCase):
    def test_zscore_roundtrip(self):
        rng = np.random.default_rng(4)
        phi = rng.normal(size=(5, 5, 3)) * 7 + 2
        pz, meta = zscore_phi(phi)
        self.assertEqual(pz.shape, phi.shape)
        self.assertAlmostEqual(float(pz.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(pz.std()), 1.0, places=10)
        # identical transform re-applied to new data via stored meta
        phi2 = rng.normal(size=(5, 5, 3))
        flat = phi2.reshape(25, 3)
        mu = np.array([m for m, _ in meta])
        sd = np.array([s for _, s in meta])
        pz2 = ((flat - mu) / sd).reshape(5, 5, 3)
        self.assertEqual(pz2.shape, phi2.shape)

    def test_make_cost_2d_promotion(self):
        K = 4
        phi = np.arange(K * K, dtype=float).reshape(K, K)
        C = make_cost(phi, 2.0)
        self.assertEqual(tuple(C.shape), (K, K))
        np.testing.assert_allclose(C.numpy(), -2.0 * phi)

    def test_row_ce_masks_zero_mass_rows(self):
        K = 4
        a = np.array([0.25, 0.25, 0.5, 0.0])
        rng = np.random.default_rng(5)
        P = torch.as_tensor(rng.random((K, K)) + 0.01, dtype=torch.float64)
        P = P / P.sum(1, keepdims=True)
        T1 = rng.random((K, K))
        T2 = T1.copy()
        T2[3, :] = 999.0            # placeholder row on zero-mass source
        l1 = float(row_ce_loss([torch.as_tensor(T1)], [P], [torch.as_tensor(a)]))
        l2 = float(row_ce_loss([torch.as_tensor(T2)], [P], [torch.as_tensor(a)]))
        self.assertAlmostEqual(l1, l2, places=10)


class TestFitRecovery(unittest.TestCase):
    def test_single_feature_recovery(self):
        K = 6
        a = np.ones(K) / K
        b = np.ones(K) / K
        rng = np.random.default_rng(6)
        u = rng.uniform(0.3, 3.0, size=K)
        phi = np.repeat(u[None, :, None], K, axis=0)   # (K, K, 1) pure-column
        phi_z, _ = zscore_phi(phi)
        T = np.asarray(row_conditional(
            soft_sinkhorn(make_cost(phi_z, np.array([2.0])), a, b, mu=0.5).detach(),
            a).numpy())
        th1, support, th_fin, loss_f = fit_once(
            [phi_z], [a], [b], [T], mu=0.5, seed=1, epochs=200)
        self.assertTrue(np.isfinite(float(th_fin[0])))
        self.assertGreater(float(th_fin[0]), 0.5, "coefficient must recover sign+scale")
        self.assertLess(float(th_fin[0]), 4.0)

    def test_fit_once_shapes(self):
        K, F = 4, 2
        a = np.ones(K) / K
        b = np.ones(K) / K
        rng = np.random.default_rng(7)
        phi = rng.normal(size=(K, K, F))
        T = rng.random((K, K)); T /= T.sum(1, keepdims=True)
        th1, support, th_fin, loss_f = fit_once(
            [phi], [a], [b], [T], epochs=5, seed=0)
        self.assertEqual(th1.shape, (F,))
        self.assertEqual(support.shape, (F,))
        self.assertTrue(isinstance(loss_f, float))


class TestIOTModel(unittest.TestCase):
    def _toy_scenario(self, seed=8):
        K, F = 6, 2
        a = np.ones(K) / K
        b = np.ones(K) / K
        rng = np.random.default_rng(seed)
        u = rng.uniform(0.3, 3.0, size=(K, F))
        phi = np.zeros((K, K, F + 1))
        for k in range(F):
            phi[:, :, k] = u[:, k][None, :]
        phi[:, :, F] = rng.uniform(-1, 1, size=(K, K))
        phi_z, _ = zscore_phi(phi)
        theta = np.array([0.9, -0.7, 1.1])
        T = np.asarray(row_conditional(
            soft_sinkhorn(make_cost(phi_z, theta), a, b, mu=0.5).detach(), a).numpy())
        return phi, a, b, T

    def test_estimator_workflow(self):
        phi, a, b, T = self._toy_scenario()
        model = IOTModel(epochs=60, n_restart=1, seed=1).fit(phi, a, b, T)
        self.assertEqual(model.theta_.shape, (3,))
        self.assertEqual(model.support_.shape, (3,))
        self.assertTrue(np.isfinite(model.loss_))
        self.assertEqual(len(model.restart_losses_), 1)
        # inference helpers
        P = model.plan(phi, a, b)
        Q = model.row_conditional(phi, a)
        np.testing.assert_allclose(P.numpy().sum(1), np.asarray(a), atol=1e-8)
        self.assertEqual(tuple(Q.shape), (6, 6))
        s = model.score(phi, a, b, T)
        self.assertTrue(np.isfinite(s))
        # standardize=False path
        model2 = IOTModel(epochs=20, n_restart=1, seed=1, standardize=False).fit(phi, a, b, T)
        self.assertTrue(np.isfinite(model2.theta_).all())

    def test_one_liner_fit(self):
        phi, a, b, T = self._toy_scenario(seed=9)
        model = bioiot.fit(phi, a, b, T, epochs=10, n_restart=1, seed=0)
        self.assertTrue(np.isfinite(model.theta_).all())

    def test_multi_scenario_list_input(self):
        s1 = self._toy_scenario(10)
        s2 = self._toy_scenario(11)
        model = IOTModel(epochs=15, n_restart=1, seed=0).fit(
            [s1[0], s2[0]], [s1[1], s2[1]], [s1[2], s2[2]], [s1[3], s2[3]])
        self.assertEqual(model.n_scenarios_, 2)


class TestSelfTestEntrypoint(unittest.TestCase):
    def test_version_exported(self):
        self.assertIsInstance(bioiot.__version__, str)
        self.assertTrue(callable(bioiot.self_test))


if __name__ == "__main__":
    unittest.main()
