"""Matrix-like plotting helpers."""

from __future__ import annotations

from matplotlib.axes import Axes

from mdpp.analysis.metrics import DCCMResult
from mdpp.plots.utils import get_axis


def plot_dccm(
    result: DCCMResult,
    *,
    ax: Axes | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    add_colorbar: bool = True,
) -> Axes:
    """Plot a dynamic cross-correlation matrix as a heatmap.

    Args:
        result: DCCMResult from ``compute_dccm``.
        ax: Optional matplotlib axis.
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.
        cmap: Colormap name.
        add_colorbar: Whether to add a colorbar.

    Returns:
        The matplotlib axis with the DCCM heatmap.
    """
    axis = get_axis(ax)
    extent: tuple[float, float, float, float] | None = None
    if result.residue_ids is not None and result.residue_ids.size > 0:
        residue_start = float(result.residue_ids[0])
        residue_end = float(result.residue_ids[-1])
        extent = (residue_start, residue_end, residue_start, residue_end)

    image = axis.imshow(
        result.correlation,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )
    axis.set_xlabel("Residue ID")
    axis.set_ylabel("Residue ID")
    if add_colorbar:
        axis.figure.colorbar(image, ax=axis, label="Cross-correlation")
    return axis
