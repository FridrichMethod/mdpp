"""Contact map visualization helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from mdpp.plots.utils import get_axis


def plot_contact_map(
    frequency: NDArray[np.floating],
    *,
    residue_ids: NDArray[np.int_] | None = None,
    ax: Axes | None = None,
    cmap: str = "hot_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    add_colorbar: bool = True,
) -> Axes:
    """Plot a residue-residue contact frequency heatmap.

    Args:
        frequency: Symmetric contact frequency matrix of shape ``(n_res, n_res)``
            with values in ``[0, 1]``.
        residue_ids: Optional residue IDs for axis labels.
        ax: Optional matplotlib axis.
        cmap: Colormap name.
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.
        add_colorbar: Whether to add a colorbar.

    Returns:
        The matplotlib axis with the contact map.
    """
    axis = get_axis(ax)

    extent: tuple[float, float, float, float] | None = None
    if residue_ids is not None and residue_ids.size > 0:
        r_start = float(residue_ids[0])
        r_end = float(residue_ids[-1])
        extent = (r_start, r_end, r_start, r_end)

    image = axis.imshow(
        frequency,
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
        axis.figure.colorbar(image, ax=axis, label="Contact Frequency")
    return axis


def contact_frequency_to_matrix(
    frequency: NDArray[np.floating],
    residue_pairs: NDArray[np.int_],
    n_residues: int,
) -> NDArray[np.floating]:
    """Convert per-pair contact frequencies to a symmetric matrix.

    Args:
        frequency: Per-pair frequencies of shape ``(n_pairs,)``.
        residue_pairs: Residue index pairs of shape ``(n_pairs, 2)``.
        n_residues: Total number of residues for the output matrix.

    Returns:
        Symmetric matrix of shape ``(n_residues, n_residues)`` in the
        same floating dtype as ``frequency`` (float32 by default).
    """
    matrix = np.zeros((n_residues, n_residues), dtype=frequency.dtype)
    for pair_index in range(residue_pairs.shape[0]):
        i, j = residue_pairs[pair_index]
        matrix[i, j] = frequency[pair_index]
        matrix[j, i] = frequency[pair_index]
    return matrix
