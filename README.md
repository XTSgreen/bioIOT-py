<h1 align="center">bioIOT</h1>

<p align="center">
  <b>Plug-and-play semi-relaxed inverse optimal transport for single-cell state transitions</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/bioIOT/"><img src="https://img.shields.io/pypi/v/bioIOT" alt="PyPI"></a>
  <a href="https://github.com/XTSgreen/bioIOT-py/actions/workflows/ci.yml"><img src="https://github.com/XTSgreen/bioIOT-py/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://img.shields.io/pypi/pyversions/bioIOT"><img src="https://img.shields.io/pypi/pyversions/bioIOT" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/bioIOT" alt="License"></a>
</p>

---

<p align="center"><a href="README.md"><b>English</b></a> | <a href="README.zh-CN.md">简体中文</a></p>

**bioIOT** solves the semi-relaxed inverse optimal transport problem: given
state-transition features, source/target state masses and observed
transitions, it learns feature weights θ such that the soft-marginal
optimal-transport plan induced by the linear cost `C = -einsum(φ, θ)`
reproduces the data — then turns the fit into state transition matrices,
random-walk pseudotime and publication-ready plots.

The solver was developed and validated as part of a research project on
treatment-resistance state transitions, and is packaged here for general
use.

## Why bioIOT?

- **Identifiable by construction.** Hard-marginal OT makes pure column
  features unidentifiable; bioIOT's KL-soft column anchoring restores
  identifiability while preserving the target composition constraint.
- **Exact implicit gradients.** The forward solve is an Anderson-accelerated
  fixed-point iteration; gradients come from the implicit function theorem —
  numerically stable where unrolled backpropagation diverges.
- **Denoises finite-sampling noise.** In a 50-replicate benchmark with 30
  cells per state, bioIOT recovers the true transition matrix ~3× more
  accurately than using the observed transitions raw
  (mean per-row L1 0.013 vs 0.039), with feature-weight correlation 0.96.
- **Truly plug-and-play.** numpy in, numpy out; inputs are validated and
  normalized automatically; one line to fit, one line to plot.

## Installation

```bash
pip install bioIOT
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `torch` (CPU is fine).
Optional: `matplotlib` for plots, `anndata` for the scanpy adapter.

<details>
<summary>From source</summary>

```bash
pip install git+https://github.com/XTSgreen/bioIOT-py.git
```

</details>

## Quick start

```python
import bioiot

# 1) Generate (or load) scenarios: phi (K, K, F) features, a/b (K,) masses,
#    T (K, K) observed row-conditional transitions
sim = bioiot.simulate_iot_states(K=6, seed=1)   # synthetic ground truth

# 2) Fit feature weights — multi-restart + two-stage debiasing, one line
model = bioiot.IOTModel().fit(sim["phi"], sim["a"], sim["b"], sim["T_true"])
model.theta_        # debiased coefficients
model.support_      # selected-feature mask

# 3) Trajectory layer
Q  = bioiot.transition_matrix(model)              # (K, K) transitions
pt = bioiot.pseudotime_from_transition(Q, root="S1")

# 4) Straight from cell-level data (embedding + cluster labels + 2 timepoints)
res = bioiot.run_iot(sim["cell_embedding"], sim["cell_state"],
                     from_mask=sim["cell_time"] == "t0",
                     to_mask=sim["cell_time"] == "t1",
                     root="S1")
res["Q"]; res["pseudotime"]
# AnnData/scanpy: bioiot.run_iot_adata(adata, "state", "time", "t0", "t1")

# 5) Plots (matplotlib)
ax = bioiot.plot_transition_heatmap(Q)
ax = bioiot.plot_transition_flow(Q, sim["embedding"], threshold=0.05)
ax = bioiot.plot_theta(model)
```

## Functional core

For users who want the low-level solver (notation follows the paper):

```python
C = bioiot.make_cost(phi, theta)                  # C = -einsum(phi, theta)
P = bioiot.soft_sinkhorn(C, a, b, mu=0.5, eps=1.0)  # (K, K) plan, P @ 1 = a
P = bioiot.uot_plan(phi, theta, a, b)             # cost + solve in one call
loss = bioiot.row_ce_loss([T], [P], [a])
phi_z, meta = bioiot.zscore_phi(phi)
```

`soft_sinkhorn` accepts numpy arrays or torch tensors; the returned plan is
differentiable in `C` through exact implicit differentiation.

## How it works

bioIOT solves

```text
min_P  <C, P> − eps·H(P) + mu·KL(col(P) ‖ b)    s.t.  P·1 = a
```

with a hard source-side row marginal and a KL-anchored column marginal:

- `mu → ∞` recovers hard-marginal OT (pure column features unidentifiable);
- `mu → 0` recovers a plain row-softmax (no target-composition anchoring);
- finite `mu` interpolates the two — the paper's working point is
  `mu = 0.5, eps = 1.0, lam = 0.05`.

Fitting (`IOTModel`, `fit_iot`) uses Adam with exact implicit gradients,
l1 selection followed by a Bühlmann-style debias refit on the selected
support, and multiple random restarts.

## Testing

```bash
python -m unittest discover -s tests -v   # from the repository root
python -c "import bioiot; bioiot.self_test()"   # numerical self-check
```

## Citation

If you use bioIOT, please cite:

```bibtex
@misc{dong2026bioiot,
  author       = {Dong, Han},
  title        = {bioIOT: Plug-and-Play Semi-Relaxed Inverse Optimal
                  Transport for Single-Cell State Transitions},
  year         = {2026},
  howpublished = {\url{https://github.com/XTSgreen/bioIOT-py}},
  note         = {Python package version 0.2.0}
}
```

## License

[MIT](LICENSE) © 2026 Han Dong (XTSgreen)
