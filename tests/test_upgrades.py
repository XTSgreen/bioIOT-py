# -*- coding: utf-8 -*-
"""bioIOT 0.2.0 upgrades: trajectory layer, single-cell, viz, simulate."""

import unittest

import numpy as np

import bioiot
from bioiot import (
    IOTModel,
    build_state_features,
    pseudotime_from_transition,
    run_iot,
    simulate_iot_states,
    transition_matrix,
)
from bioiot import standardize_like


class TestMultiK(unittest.TestCase):
    def test_scenarios_may_differ_in_K(self):
        rng = np.random.default_rng(11)
        phi1 = rng.normal(size=(5, 5, 2))
        phi2 = rng.normal(size=(6, 6, 2))
        a1 = np.ones(5) / 5
        a2 = np.ones(6) / 6
        T1 = rng.random((5, 5)); T1 /= T1.sum(1, keepdims=True)
        T2 = rng.random((6, 6)); T2 /= T2.sum(1, keepdims=True)
        model = IOTModel(epochs=10, n_restart=1, seed=0).fit(
            [phi1, phi2], [a1, a2], [a1, a2], [T1, T2])
        np.testing.assert_array_equal(model.K_, [5, 6])
        self.assertEqual(len(model.scenarios_), 2)
        Q2 = transition_matrix(model, which=1)
        self.assertEqual(Q2.shape, (6, 6))
        np.testing.assert_allclose(Q2.sum(1), 1.0, atol=1e-8)  # row-conditional
        with self.assertRaises(ValueError):
            IOTModel(epochs=2, n_restart=1).fit(
                [phi1, rng.normal(size=(4, 4, 3))], [a1, a1], [a1, a1],
                [T1, T1])


class TestTrajectoryLayer(unittest.TestCase):
    def _fit(self):
        sim = simulate_iot_states(K=5, seed=1)
        model = IOTModel(epochs=80, n_restart=1, seed=1).fit(
            sim["phi"], sim["a"], sim["b"], sim["T_true"])
        return sim, model

    def test_transition_matrix_from_model_and_override(self):
        sim, model = self._fit()
        Q = transition_matrix(model)
        self.assertEqual(Q.shape, (5, 5))
        np.testing.assert_allclose(Q.sum(1), 1.0, atol=1e-8)  # row-conditional
        Q2 = transition_matrix(model, phi=sim["phi"], a=sim["a"], b=sim["b"])
        self.assertEqual(Q2.shape, (5, 5))
        with self.assertRaises(TypeError):
            transition_matrix(object())

    def test_pseudotime_root_zero_others_positive(self):
        sim, model = self._fit()
        Q = transition_matrix(model)
        pt = pseudotime_from_transition(Q, root="S1")
        self.assertEqual(pt["S1"], 0.0)
        self.assertTrue(all(v > 0 for k, v in pt.items() if k != "S1"))
        with self.assertRaises(ValueError):
            pseudotime_from_transition(Q, root="nope")

    def test_pseudotime_zero_outflow_terminal_state_is_late(self):
        Q = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.1, 0.1, 0.4, 0.4],
            [0.0, 0.0, 0.0, 0.0],   # S4 terminal: zero outflow
        ])
        pt = pseudotime_from_transition(Q, root=0)
        self.assertEqual(pt["S1"], 0.0)
        self.assertTrue(all(np.isfinite(v) for v in pt.values()))
        self.assertGreater(pt["S4"], min(v for k, v in pt.items() if k != "S1"))

    def test_build_state_features(self):
        u = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        phi = build_state_features(u)
        self.assertEqual(phi.shape, (3, 3, 3))
        np.testing.assert_allclose(phi[0, :, 0], u[:, 0])  # pure column
        np.testing.assert_allclose(phi[:, :, 2], u @ u.T)  # similarity
        with self.assertRaises(ValueError):
            build_state_features(np.array([np.nan, 1.0]))

    def test_standardize_like(self):
        rng = np.random.default_rng(4)
        phi = rng.normal(size=(4, 4, 3)) * 7 + 2
        pz, meta = bioiot.zscore_phi(phi)
        pz2 = standardize_like(phi, meta)
        np.testing.assert_allclose(pz, pz2, atol=1e-12)


class TestSimulate(unittest.TestCase):
    def test_deterministic_shapes(self):
        s1 = simulate_iot_states(K=5, n_cells=20, seed=1)
        s2 = simulate_iot_states(K=5, n_cells=20, seed=1)
        np.testing.assert_allclose(s1["P_true"], s2["P_true"], atol=1e-12)
        self.assertEqual(s1["phi"].shape, (5, 5, 3))
        self.assertEqual(s1["cell_embedding"].shape, (200, 2))
        self.assertEqual(len(s1["cell_state"]), 200)
        np.testing.assert_allclose(s1["T_true"].sum(1), 1.0, atol=1e-8)


class TestRunIOT(unittest.TestCase):
    def test_unsupervised_plan_and_pseudotime(self):
        sim = simulate_iot_states(K=5, n_cells=40, seed=1)
        res = run_iot(sim["cell_embedding"], sim["cell_state"],
                      from_mask=sim["cell_time"] == "t0",
                      to_mask=sim["cell_time"] == "t1",
                      root="S1")
        self.assertIsNone(res["fit"])
        self.assertEqual(res["Q"].shape, (5, 5))
        np.testing.assert_allclose(res["Q"].sum(1), 1.0, atol=1e-8)
        self.assertEqual(res["pseudotime"]["S1"], 0.0)
        self.assertTrue(all(v > 0 for k, v in res["pseudotime"].items()
                            if k != "S1"))

    def test_supervised_mode_with_T_obs(self):
        sim = simulate_iot_states(K=5, n_cells=40, seed=2)
        res = run_iot(sim["cell_embedding"], sim["cell_state"],
                      from_mask=sim["cell_time"] == "t0",
                      to_mask=sim["cell_time"] == "t1",
                      T_obs=sim["T_true"], n_restart=1, epochs=80, seed=1)
        self.assertIsInstance(res["fit"], IOTModel)
        self.assertTrue(np.isfinite(res["theta"]).all())

    def test_input_validation(self):
        sim = simulate_iot_states(K=4, n_cells=10, seed=3)
        with self.assertRaises(ValueError):
            run_iot(sim["cell_embedding"], sim["cell_state"][:-1],
                    from_mask=sim["cell_time"] == "t0",
                    to_mask=sim["cell_time"] == "t1")
        with self.assertRaises(ValueError):
            run_iot(sim["cell_embedding"], sim["cell_state"],
                    from_mask=sim["cell_time"] == "t0",
                    to_mask=sim["cell_time"] == "t1", T_obs=np.zeros((3, 3)))

    def test_anndata_adapter(self):
        try:
            import anndata  # noqa: F401
        except ImportError:
            self.skipTest("anndata not installed")
        import anndata as ad
        import pandas as pd
        sim = simulate_iot_states(K=4, n_cells=15, seed=3)
        adata = ad.AnnData(
            X=np.random.default_rng(0).normal(size=(120, 20)),
            obs=pd.DataFrame({"state": sim["cell_state"],
                              "time": sim["cell_time"]},
                             index=[f"c{i}" for i in range(120)]),
        )
        adata.obsm["X_pca"] = sim["cell_embedding"]
        res = bioiot.run_iot_adata(adata, state_col="state", time_col="time",
                                   from_key="t0", to_key="t1",
                                   use_rep="X_pca", root="S1", n_dim=2)
        self.assertEqual(res["Q"].shape, (4, 4))
        with self.assertRaises(KeyError):
            bioiot.run_iot_adata(adata, state_col="nope", time_col="time",
                                 from_key="t0", to_key="t1")


class TestViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cls.plt = plt

    def test_plots_return_axes(self):
        import matplotlib.axes

        sim = simulate_iot_states(K=5, seed=1)
        model = IOTModel(epochs=60, n_restart=1, seed=1).fit(
            sim["phi"], sim["a"], sim["b"], sim["T_true"])
        Q = transition_matrix(model)
        ax1 = bioiot.plot_transition_heatmap(Q)
        self.assertIsInstance(ax1, matplotlib.axes.Axes)
        ax2 = bioiot.plot_transition_flow(Q, sim["embedding"], threshold=0.05)
        self.assertIsInstance(ax2, matplotlib.axes.Axes)
        ax3 = bioiot.plot_theta(model)
        self.assertIsInstance(ax3, matplotlib.axes.Axes)
        ax4 = bioiot.plot_theta(np.array([0.5, -1.0]))
        self.assertIsInstance(ax4, matplotlib.axes.Axes)
        with self.assertRaises(ValueError):
            bioiot.plot_transition_flow(Q, sim["embedding"][:, :1])
        self.plt.close("all")


if __name__ == "__main__":
    unittest.main()
