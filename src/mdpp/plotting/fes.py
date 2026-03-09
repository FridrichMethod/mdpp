"""Free-energy surface plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from mdpp.analysis.fes import FES2DResult
from mdpp.plotting._common import get_axis


def plot_fes(
    result: FES2DResult,
    *,
    ax: Axes | None = None,
    cmap: str = "coolwarm",
    add_colorbar: bool = True,
    add_contours: bool = True,
    contour_levels: int = 12,
) -> Axes:
    """Plot a 2D free-energy surface."""
    axis = get_axis(ax)
    image = axis.imshow(
        result.free_energy_kj_mol.T,
        origin="lower",
        extent=(
            float(result.x_edges[0]),
            float(result.x_edges[-1]),
            float(result.y_edges[0]),
            float(result.y_edges[-1]),
        ),
        cmap=cmap,
        aspect="auto",
    )

    finite_values = result.free_energy_kj_mol[np.isfinite(result.free_energy_kj_mol)]
    if add_contours and finite_values.size > 1:
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        if value_max > value_min:
            levels = np.linspace(value_min, value_max, contour_levels)
            grid_x, grid_y = np.meshgrid(result.x_centers, result.y_centers)
            axis.contour(
                grid_x,
                grid_y,
                np.ma.masked_invalid(result.free_energy_kj_mol.T),
                levels=levels,
                colors="k",
                linewidths=0.4,
                alpha=0.65,
            )

    axis.set_xlabel("CV 1")
    axis.set_ylabel("CV 2")
    if add_colorbar:
        axis.figure.colorbar(image, ax=axis, label="ΔG (kJ/mol)")
    return axis
