# -*- coding: utf-8 -*-
"""Matplotlib visualisation for bioIOT (soft-gated)."""

import numpy as np

__all__ = [
    "plot_transition_heatmap",
    "plot_transition_flow",
    "plot_theta",
]


def _labels(Q, labels):
    if labels is not None:
        return [str(x) for x in labels]
    idx = getattr(Q, "index", None)
    if idx is not None:
        try:
            return [str(x) for x in idx]
        except Exception:
            pass
    return [f"S{i + 1}" for i in range(Q.shape[0])]


def plot_transition_heatmap(Q, labels=None, ax=None, cmap="Reds"):
    """Annotated transition-matrix heatmap. Returns a matplotlib Axes."""
    import matplotlib.pyplot as plt

    Q = np.asarray(Q, dtype=np.float64)
    K = Q.shape[0]
    labels = _labels(Q, labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(Q, cmap=cmap, vmin=0.0, aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(K))
    ax.set_yticklabels(labels)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{Q[i, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if Q[i, j] > 0.6 * max(Q.max(), 1e-12)
                    else "black")
    ax.set_xlabel("Target state")
    ax.set_ylabel("Source state")
    ax.set_title("IOT state transitions")
    ax.figure.colorbar(im, ax=ax, label="Q(i->j)")
    return ax


def plot_transition_flow(Q, embedding, labels=None, threshold=0.05,
                         ax=None, scale=5.0):
    """CellRank-style transition arrows on a 2-D state embedding."""
    import matplotlib.pyplot as plt

    Q = np.asarray(Q, dtype=np.float64)
    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError("`embedding` needs >= 2 columns")
    K = Q.shape[0]
    labels = _labels(Q, labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    mass = Q.sum(axis=1)
    ax.scatter(emb[:, 0], emb[:, 1], s=80 + 400 * mass / max(mass.max(), 1e-12),
               color="#B2182B", alpha=0.9, zorder=2)
    for i in range(K):
        for j in range(K):
            if i != j and Q[i, j] > threshold:
                ax.annotate(
                    "",
                    xy=emb[j], xytext=emb[i],
                    arrowprops=dict(arrowstyle="-|>", color="#2166AC",
                                    alpha=0.85, lw=0.5 + scale * Q[i, j],
                                    shrinkA=10, shrinkB=10),
                    zorder=1,
                )
    for i in range(K):
        ax.annotate(labels[i], emb[i], textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, zorder=3)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title("IOT transition flow")
    return ax


def plot_theta(model_or_theta, labels=None, ax=None):
    """Barplot of fitted feature weights; selected support in red."""
    import matplotlib.pyplot as plt

    from .model import IOTModel

    if isinstance(model_or_theta, IOTModel):
        theta = np.asarray(model_or_theta.theta_)
        support = np.asarray(model_or_theta.support_, dtype=bool)
    else:
        theta = np.asarray(model_or_theta, dtype=np.float64)
        support = getattr(model_or_theta, "support", None)
        if support is None:
            support = np.ones(theta.shape[0], dtype=bool)
        support = np.asarray(support, dtype=bool)
    F = theta.shape[0]
    if labels is None:
        labels = [f"f{i + 1}" for i in range(F)]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = ["#B2182B" if s else "grey" for s in support]
    ax.barh(range(F), theta, color=colors, height=0.65)
    ax.set_yticks(range(F))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#4d4d4d", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("theta")
    ax.set_title("IOT feature weights (red = selected support)")
    return ax
