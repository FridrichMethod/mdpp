"""Time-series and per-residue plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from mdpp.analysis.hbond import HBondResult
from mdpp.analysis.metrics import RMSDResult, RMSFResult, SASAResult
from mdpp.plotting._common import get_axis


def plot_rmsd(
    result: RMSDResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    linewidth: float = 1.5,
) -> Axes:
    """Plot RMSD as a function of time."""
    axis = get_axis(ax)
    axis.plot(result.time_ns, result.rmsd_angstrom, label=label, linewidth=linewidth)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("RMSD (Å)")
    if label is not None:
        axis.legend()
    return axis


def plot_rmsf(
    result: RMSFResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    linewidth: float = 1.5,
) -> Axes:
    """Plot per-atom RMSF."""
    axis = get_axis(ax)
    x_values = (
        np.asarray(result.residue_ids, dtype=np.float64)
        if result.residue_ids is not None
        else np.arange(result.rmsf_nm.size, dtype=np.float64) + 1.0
    )
    axis.plot(x_values, result.rmsf_angstrom, label=label, linewidth=linewidth)
    axis.set_xlabel("Residue ID")
    axis.set_ylabel("RMSF (Å)")
    if label is not None:
        axis.legend()
    return axis


def plot_sasa(
    result: SASAResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    aggregate: str = "sum",
    linewidth: float = 1.5,
) -> Axes:
    """Plot SASA over time.

    Args:
        result: SASA results.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        aggregate: ``"sum"``, ``"mean"``, or ``"none"``.
        linewidth: Line width for plotted traces.
    """
    axis = get_axis(ax)
    if aggregate == "sum":
        y_values = np.sum(result.values_nm2, axis=1)
        axis.plot(result.time_ns, y_values, label=label, linewidth=linewidth)
        axis.set_ylabel("SASA (nm²)")
    elif aggregate == "mean":
        y_values = np.mean(result.values_nm2, axis=1)
        axis.plot(result.time_ns, y_values, label=label, linewidth=linewidth)
        axis.set_ylabel("Mean SASA (nm²)")
    elif aggregate == "none":
        for index in range(result.values_nm2.shape[1]):
            axis.plot(
                result.time_ns,
                result.values_nm2[:, index],
                linewidth=linewidth,
                alpha=0.6,
            )
        axis.set_ylabel("SASA (nm²)")
    else:
        raise ValueError("aggregate must be one of {'sum', 'mean', 'none'}.")

    axis.set_xlabel("Time (ns)")
    if label is not None and aggregate != "none":
        axis.legend()
    return axis


def plot_hbond_counts(
    result: HBondResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    linewidth: float = 1.5,
) -> Axes:
    """Plot hydrogen-bond count over time."""
    axis = get_axis(ax)
    axis.plot(result.time_ns, result.count_per_frame, label=label, linewidth=linewidth)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Hydrogen Bond Count")
    if label is not None:
        axis.legend()
    return axis


def plot_hbond_occupancy(
    result: HBondResult,
    *,
    ax: Axes | None = None,
    labels: list[str] | None = None,
    top_n: int | None = None,
) -> Axes:
    """Plot per-bond occupancy as a bar plot.

    Args:
        result: Hydrogen-bond analysis result.
        ax: Optional matplotlib axis.
        labels: Optional labels corresponding to ``result.triplets``.
        top_n: Optional number of highest-occupancy bonds to display.
    """
    axis = get_axis(ax)
    occupancy = result.occupancy
    if top_n is not None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1.")
        order = np.argsort(occupancy)[::-1][:top_n]
    else:
        order = np.arange(occupancy.shape[0], dtype=np.int_)

    selected = occupancy[order] * 100.0
    x_values = np.arange(selected.shape[0], dtype=np.float64)
    axis.bar(x_values, selected, width=0.8)

    if labels is not None:
        selected_labels = [labels[int(index)] for index in order]
        axis.set_xticks(x_values)
        axis.set_xticklabels(selected_labels, rotation=90)
    else:
        axis.set_xlabel("Hydrogen Bond Index")
    axis.set_ylabel("Occupancy (%)")
    return axis
