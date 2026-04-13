"""Clustering visualization helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from mdpp.analysis.clustering import ClusteringResult, FeatureClusteringResult
from mdpp.analysis.decomposition import PCAResult, TICAResult
from mdpp.plots.utils import get_axis


def plot_feature_clustering(
    result: FeatureClusteringResult,
    projections: PCAResult | TICAResult | NDArray[np.floating],
    *,
    x_index: int = 0,
    y_index: int = 1,
    cmap: str = "tab10",
    center_color: str = "black",
    center_marker: str = "x",
    center_size: float = 80.0,
    center_linewidths: float = 2.0,
    s: float = 2.0,
    alpha: float = 0.4,
    rasterized: bool = True,
    show_centers: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Scatter plot of projections colored by cluster labels with centers.

    Args:
        result: Feature clustering result (KMeans, MiniBatchKMeans, or
            RegularSpace).
        projections: 2-D projection coordinates. Accepts a ``PCAResult``,
            ``TICAResult``, or a raw ``(n_samples, n_features)`` array.
        x_index: Column index for the x-axis.
        y_index: Column index for the y-axis.
        cmap: Colormap for cluster labels.
        center_color: Color for cluster center markers.
        center_marker: Marker style for cluster centers.
        center_size: Marker size for cluster centers.
        center_linewidths: Line width for cluster center markers.
        s: Marker size for data points.
        alpha: Marker transparency for data points.
        rasterized: Whether to rasterize the scatter layer (recommended
            for large trajectories to keep file sizes small).
        show_centers: Whether to overlay cluster center markers.
        ax: Optional matplotlib axis.

    Returns:
        The matplotlib axis with the scatter plot.
    """
    axis = get_axis(ax)

    if isinstance(projections, PCAResult | TICAResult):
        coords = projections.projections
    else:
        coords = np.asarray(projections)

    x = coords[:, x_index]
    y = coords[:, y_index]

    axis.scatter(
        x,
        y,
        c=result.labels,
        cmap=cmap,
        s=s,
        alpha=alpha,
        edgecolors="none",
        rasterized=rasterized,
    )

    if show_centers:
        axis.scatter(
            result.cluster_centers[:, x_index],
            result.cluster_centers[:, y_index],
            c=center_color,
            marker=center_marker,
            s=center_size,
            linewidths=center_linewidths,
            zorder=5,
        )

    if isinstance(projections, PCAResult):
        axis.set_xlabel(f"PC{x_index + 1}")
        axis.set_ylabel(f"PC{y_index + 1}")
    elif isinstance(projections, TICAResult):
        axis.set_xlabel(f"IC{x_index + 1}")
        axis.set_ylabel(f"IC{y_index + 1}")
    else:
        axis.set_xlabel(f"Component {x_index + 1}")
        axis.set_ylabel(f"Component {y_index + 1}")

    return axis


def plot_cluster_populations(
    result: ClusteringResult | FeatureClusteringResult,
    *,
    top_k: int = 20,
    color: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of cluster populations (frame counts per cluster).

    Args:
        result: Clustering result from any method.
        top_k: Maximum number of clusters to show (largest first).
        color: Bar color. If ``None``, uses the default color cycle.
        ax: Optional matplotlib axis.

    Returns:
        The matplotlib axis with the bar chart.
    """
    axis = get_axis(ax)

    valid = result.labels[result.labels >= 0]
    if len(valid) > 0:
        counts = np.bincount(valid)
        ranked = np.argsort(counts)[::-1]
        n_show = min(top_k, len(counts))
        ranked_counts = counts[ranked[:n_show]]
        if color is not None:
            axis.bar(range(n_show), ranked_counts, color=color)
        else:
            axis.bar(range(n_show), ranked_counts)

    axis.set_xlabel("Cluster (ranked)")
    axis.set_ylabel("Frames")

    return axis
