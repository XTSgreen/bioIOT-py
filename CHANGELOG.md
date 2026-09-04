# Changelog

## 0.2.0 (2026-09-04)

* New trajectory layer: `transition_matrix()`, `pseudotime_from_transition()`
  (random-walk pseudotime with absorbing-closure fix for zero-outflow
  terminal states), `build_state_features()`, `standardize_like()`.
* New single-cell interface: `run_iot()` (cell embedding matrix) and
  `run_iot_adata()` (AnnData/scanpy adapter, soft-gated).
* New visualisation (matplotlib, soft-gated): `plot_transition_heatmap()`,
  `plot_transition_flow()`, `plot_theta()`.
* New reproducible demo generator: `simulate_iot_states()`.
* `IOTModel.fit()` now accepts scenarios with different K (state counts);
  `IOTModel.scenarios_` / `K_` expose the stored scenarios.
* Packaging: author Han Dong (XTSgreen) <dh411424@163.com>, project URLs.
* Test suite grown to 27 cases (multi-K, trajectory layer, single-cell,
  plotting, simulation).

## 0.1.0

* Initial release: semi-relaxed IOT core (`soft_sinkhorn` with Anderson
  acceleration and exact implicit gradients), two-stage debiased fitting
  (`fit_once` / `fit_uot` / `fit_hard`), scikit-learn-style `IOTModel`
  estimator, synthetic self-test.
