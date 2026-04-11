"""Scatter plot helpers for projections and torsion analysis."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from mdpp.analysis.decomposition import PCAResult, TICAResult, TorsionFeatures
from mdpp.plots.utils import get_axis


def plot_projection(
    result: PCAResult | TICAResult,
    *,
    x_index: int = 0,
    y_index: int = 1,
    color_by: ArrayLike | None = None,
    cmap: str = "viridis",
    ax: Axes | None = None,
    s: float = 5.0,
    alpha: float = 0.6,
    add_colorbar: bool = True,
) -> Axes:
    """Plot a 2D scatter of projected coordinates (PCA or TICA).

    Args:
        result: PCAResult or TICAResult containing projections.
        x_index: Component index for the x-axis.
        y_index: Component index for the y-axis.
        color_by: Optional array to color points by (e.g. time or cluster
            labels). Must have length equal to the number of samples.
        cmap: Colormap name when ``color_by`` is provided.
        ax: Optional matplotlib axis.
        s: Marker size.
        alpha: Marker transparency.
        add_colorbar: Whether to add a colorbar when ``color_by`` is used.

    Returns:
        The matplotlib axis with the scatter plot.
    """
    axis = get_axis(ax)
    projections = result.projections
    x = projections[:, x_index]
    y = projections[:, y_index]

    if color_by is not None:
        c = np.asarray(color_by)
        scatter = axis.scatter(x, y, c=c, cmap=cmap, s=s, alpha=alpha, edgecolors="none")
        if add_colorbar:
            axis.figure.colorbar(scatter, ax=axis)
    else:
        axis.scatter(x, y, s=s, alpha=alpha, edgecolors="none")

    if isinstance(result, PCAResult):
        axis.set_xlabel(f"PC{x_index + 1}")
        axis.set_ylabel(f"PC{y_index + 1}")
    else:
        axis.set_xlabel(f"IC{x_index + 1}")
        axis.set_ylabel(f"IC{y_index + 1}")

    return axis


def plot_ramachandran(
    torsions: TorsionFeatures,
    *,
    ax: Axes | None = None,
    s: float = 3.0,
    alpha: float = 0.3,
    color: str = "tab:blue",
) -> Axes:
    """Plot a Ramachandran diagram from backbone torsion features.

    Expects ``torsions`` to have been computed with ``sincos_embedding=False``
    so that raw phi/psi angles (in radians) are available.

    Args:
        torsions: TorsionFeatures from ``featurize_backbone_torsions``
            with ``sincos_embedding=False``.
        ax: Optional matplotlib axis.
        s: Marker size.
        alpha: Marker transparency.
        color: Marker color.

    Returns:
        The matplotlib axis with the Ramachandran plot.

    Raises:
        ValueError: If phi and psi angles cannot be identified from labels.
    """
    axis = get_axis(ax)

    phi_cols = [i for i, label in enumerate(torsions.labels) if label.startswith("phi_")]
    psi_cols = [i for i, label in enumerate(torsions.labels) if label.startswith("psi_")]

    if not phi_cols or not psi_cols:
        raise ValueError(
            "Ramachandran plot requires raw phi/psi angles. "
            "Use featurize_backbone_torsions(sincos_embedding=False)."
        )

    phi = torsions.values[:, phi_cols].ravel()
    psi = torsions.values[:, psi_cols].ravel()

    axis.scatter(np.degrees(phi), np.degrees(psi), s=s, alpha=alpha, color=color, edgecolors="none")
    axis.set_xlabel("φ (degrees)")
    axis.set_ylabel("ψ (degrees)")
    axis.set_xlim(-180, 180)
    axis.set_ylim(-180, 180)
    axis.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axis.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    axis.set_aspect("equal")
    return axis


def plot_pca_scree(
    *results: PCAResult,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a scree plot (variance explained per principal component).

    Multiple PCA results can be overlaid as grouped bars for comparison.

    Args:
        *results: One or more PCAResult objects.
        labels: Optional legend labels, one per result.
        colors: Optional bar colors, one per result.
        ax: Optional matplotlib axis.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    n_results = len(results)
    bar_width = 0.8 / max(n_results, 1)

    for idx, result in enumerate(results):
        n_pcs = result.explained_variance_ratio.size
        pcs = np.arange(1, n_pcs + 1, dtype=np.float64)
        offset = (idx - (n_results - 1) / 2) * bar_width
        label = labels[idx] if labels is not None and idx < len(labels) else None
        color = colors[idx] if colors is not None and idx < len(colors) else None
        axis.bar(
            pcs + offset,
            result.explained_variance_ratio * 100,
            width=bar_width,
            label=label,
            color=color,
        )

    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Variance Explained (%)")
    if n_results > 0:
        axis.set_xticks(np.arange(1, results[0].explained_variance_ratio.size + 1))
    if labels is not None:
        axis.legend()
    return axis


def plot_pca_cumulative_variance(
    *results: PCAResult,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    thresholds: list[float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot cumulative explained variance as a function of the number of PCs.

    Args:
        *results: One or more PCAResult objects.
        labels: Optional legend labels, one per result.
        colors: Optional line colors, one per result.
        thresholds: Optional horizontal reference lines (e.g. ``[80, 90]``
            for 80% and 90% thresholds). Drawn as dashed gray lines.
        ax: Optional matplotlib axis.

    Returns:
        The matplotlib axis.
    """
    if thresholds is None:
        thresholds = [80, 90]

    axis = get_axis(ax)

    for idx, result in enumerate(results):
        n_pcs = result.explained_variance_ratio.size
        pcs = np.arange(1, n_pcs + 1, dtype=np.float64)
        cumulative = np.cumsum(result.explained_variance_ratio) * 100
        label = labels[idx] if labels is not None and idx < len(labels) else None
        color = colors[idx] if colors is not None and idx < len(colors) else None
        axis.plot(pcs, cumulative, "o-", label=label, color=color, markersize=4)

    for threshold in thresholds:
        axis.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)

    axis.set_xlabel("Number of PCs")
    axis.set_ylabel("Cumulative Variance (%)")
    axis.set_ylim(0, 100)
    if n_pcs := (results[0].explained_variance_ratio.size if results else 0):
        axis.set_xticks(np.arange(1, n_pcs + 1))
    if labels is not None:
        axis.legend()
    return axis
