# -*- coding: utf-8 -*-
"""bioIOT: plug-and-play semi-relaxed inverse optimal transport.

Quickstart::

    import bioiot

    # estimator API
    model = bioiot.IOTModel().fit(phi, a, b, T)   # one scenario or lists
    model.theta_                                  # debiased coefficients
    Q = bioiot.transition_matrix(model)           # state transitions
    pt = bioiot.pseudotime_from_transition(Q, root="S1")

    # one-liner on cell-level data
    res = bioiot.run_iot(emb, state, from_mask, to_mask, root="S1")

    # low-level functional API
    C = bioiot.make_cost(phi, theta)
    P = bioiot.soft_sinkhorn(C, a, b)

    # synthetic demo data and plots
    sim = bioiot.simulate_iot_states(seed=1)
    ax = bioiot.plot_transition_heatmap(Q)

    # installation self-check
    bioiot.self_test()
"""

from .features import make_cost, row_ce_loss, uot_plan, zscore_phi
from .fitting import (
    EPS,
    LAM,
    MU,
    N_RESTART,
    SEED,
    UOT_ITERS,
    fit_hard,
    fit_once,
    fit_uot,
)
from .model import IOTModel, fit
from .selftest import self_test
from .simulate import simulate_iot_states
from .singlecell import run_iot, run_iot_adata
from .solve import log, row_conditional, soft_sinkhorn
from .transitions import (
    build_state_features,
    pseudotime_from_transition,
    standardize_like,
    transition_matrix,
)
from .viz import plot_theta, plot_transition_flow, plot_transition_heatmap

__version__ = "0.2.0"

__all__ = [
    # estimator
    "IOTModel",
    "fit",
    # trajectory layer
    "transition_matrix",
    "pseudotime_from_transition",
    "build_state_features",
    "standardize_like",
    # single-cell interface
    "run_iot",
    "run_iot_adata",
    # visualisation
    "plot_transition_heatmap",
    "plot_transition_flow",
    "plot_theta",
    # data
    "simulate_iot_states",
    # functional core
    "soft_sinkhorn",
    "row_conditional",
    "make_cost",
    "uot_plan",
    "row_ce_loss",
    "zscore_phi",
    "fit_once",
    "fit_uot",
    "fit_hard",
    "self_test",
    "log",
    "EPS",
    "LAM",
    "MU",
    "N_RESTART",
    "SEED",
    "UOT_ITERS",
    "__version__",
]
