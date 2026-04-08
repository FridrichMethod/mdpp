"""Time-series and per-residue plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from mdpp.analysis.contacts import NativeContactResult
from mdpp.analysis.distance import DistanceResult
from mdpp.analysis.hbond import HBondResult
from mdpp.analysis.metrics import (
    DeltaRMSFResult,
    RadiusOfGyrationResult,
    RMSDResult,
    RMSFResult,
    SASAResult,
)
from mdpp.plots.utils import get_axis


def plot_rmsd(
    result: RMSDResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    color: str | None = None,
    moving_average: int | None = None,
    ma_linewidth: float = 2.0,
) -> Axes:
    """Plot RMSD as a function of time.

    Args:
        result: RMSDResult from ``compute_rmsd``.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        linewidth: Line width for the raw trace.
        alpha: Opacity of the raw trace (0.0 -- 1.0).
        color: Optional line color. Defaults to matplotlib's next color in
            the property cycle.
        moving_average: If set, overlay a moving-average line computed with
            a centered window of this many frames.
        ma_linewidth: Line width for the moving-average overlay.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    use_ma = moving_average is not None and moving_average > 1
    (line,) = axis.plot(
        result.time_ns,
        result.rmsd_angstrom,
        label=None if use_ma else label,
        linewidth=linewidth,
        alpha=alpha,
        color=color,
    )
    if use_ma and moving_average is not None:
        kernel = np.ones(moving_average) / moving_average
        smoothed = np.convolve(result.rmsd_angstrom, kernel, mode="same")
        axis.plot(
            result.time_ns,
            smoothed,
            color=line.get_color(),
            linewidth=ma_linewidth,
            label=label,
        )
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
    alpha: float = 1.0,
    color: str | None = None,
) -> Axes:
    """Plot per-atom RMSF.

    Args:
        result: RMSFResult from ``compute_rmsf``.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        linewidth: Line width.
        alpha: Opacity of the trace (0.0 -- 1.0).
        color: Optional line color. Defaults to matplotlib's next color in
            the property cycle.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    x_values = (
        np.asarray(result.residue_ids, dtype=np.float64)
        if result.residue_ids is not None
        else np.arange(result.rmsf_nm.size, dtype=np.float64) + 1.0
    )
    axis.plot(
        x_values, result.rmsf_angstrom, label=label, linewidth=linewidth, alpha=alpha, color=color
    )
    axis.set_xlabel("Residue ID")
    axis.set_ylabel("RMSF (Å)")
    if label is not None:
        axis.legend()
    return axis


def plot_rmsf_average(
    results: list[RMSFResult],
    *,
    ax: Axes | None = None,
    label: str = "average",
    linewidth: float = 2.0,
    color: str | None = None,
    show_sem: bool = True,
    sem_alpha: float = 0.2,
) -> Axes:
    """Plot the average RMSF from multiple replicas with optional SEM band.

    Averaging is done in MSF (mean-square fluctuation) space: the per-residue
    RMSF^2 values are averaged across replicas, then the square root is taken.
    This is the physically correct way to combine RMSF values because RMSF is
    the square root of a variance-like quantity.

    The error band is propagated consistently through the same transformation.
    Given ``avg_rmsf = sqrt(mean(MSF))``, the SEM on the MSF is propagated
    through the square root via ``sem_rmsf = sem_msf / (2 * avg_rmsf)``.

    Args:
        results: List of RMSFResult objects (one per replica). All must have
            the same number of atoms.
        ax: Optional matplotlib axis.
        label: Legend label for the average line.
        linewidth: Line width.
        color: Optional line color. Defaults to matplotlib's next color in
            the property cycle.
        show_sem: If ``True``, draw a transparent band showing +/- 1 SEM
            around the average. Requires at least 2 replicas.
        sem_alpha: Opacity of the SEM band (0.0 -- 1.0).

    Returns:
        The matplotlib axis.

    Raises:
        ValueError: If ``results`` is empty or RMSF arrays differ in length.
    """
    if not results:
        raise ValueError("results must not be empty.")

    sizes = {r.rmsf_nm.size for r in results}
    if len(sizes) > 1:
        raise ValueError(f"All RMSFResult arrays must have the same length, got sizes {sizes}.")

    msf_stack = np.stack([r.rmsf_nm**2 for r in results])
    avg_msf = np.mean(msf_stack, axis=0)
    avg_rmsf_angstrom = np.sqrt(avg_msf) * 10.0

    axis = get_axis(ax)
    ref = results[0]
    x_values = (
        np.asarray(ref.residue_ids, dtype=np.float64)
        if ref.residue_ids is not None
        else np.arange(ref.rmsf_nm.size, dtype=np.float64) + 1.0
    )
    (line,) = axis.plot(x_values, avg_rmsf_angstrom, label=label, linewidth=linewidth, color=color)
    if show_sem and len(results) >= 2:
        n_replicas = len(results)
        sem_msf = np.std(msf_stack, axis=0, ddof=1) / np.sqrt(n_replicas)
        # Propagate SEM through sqrt: d/d(MSF) sqrt(MSF) = 1 / (2 * sqrt(MSF))
        avg_rmsf_nm = np.sqrt(avg_msf)
        sem = np.where(avg_rmsf_nm > 0, sem_msf / (2.0 * avg_rmsf_nm), 0.0) * 10.0
        axis.fill_between(
            x_values,
            avg_rmsf_angstrom - sem,
            avg_rmsf_angstrom + sem,
            alpha=sem_alpha,
            color=line.get_color(),
        )
    axis.set_xlabel("Residue ID")
    axis.set_ylabel("RMSF (Å)")
    axis.legend()
    return axis


def plot_sasa(
    result: SASAResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    aggregate: str = "sum",
    linewidth: float = 1.5,
    color: str | None = None,
) -> Axes:
    """Plot SASA over time.

    Args:
        result: SASAResult from ``compute_sasa``.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        aggregate: Aggregation mode: ``"sum"``, ``"mean"``, or ``"none"``.
            When ``"none"``, each atom/residue trace uses the property cycle
            and the ``color`` parameter is ignored.
        linewidth: Line width for plotted traces.
        color: Optional line color for ``"sum"`` and ``"mean"`` aggregation.
            Ignored when ``aggregate="none"`` (multiple traces use the
            property cycle). Defaults to matplotlib's next color.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    if aggregate == "sum":
        y_values = np.sum(result.values_nm2, axis=1)
        axis.plot(result.time_ns, y_values, label=label, linewidth=linewidth, color=color)
        axis.set_ylabel("SASA (nm²)")
    elif aggregate == "mean":
        y_values = np.mean(result.values_nm2, axis=1)
        axis.plot(result.time_ns, y_values, label=label, linewidth=linewidth, color=color)
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
    color: str | None = None,
) -> Axes:
    """Plot hydrogen-bond count over time.

    Args:
        result: HBondResult from ``compute_hbonds``.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        linewidth: Line width.
        color: Optional line color. Defaults to matplotlib's next color in
            the property cycle.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    axis.plot(result.time_ns, result.count_per_frame, label=label, linewidth=linewidth, color=color)
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
        result: HBondResult from ``compute_hbonds``.
        ax: Optional matplotlib axis.
        labels: Optional labels corresponding to ``result.triplets``.
        top_n: Optional number of highest-occupancy bonds to display.

    Returns:
        The matplotlib axis.
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


def plot_radius_of_gyration(
    result: RadiusOfGyrationResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    linewidth: float = 1.5,
    color: str | None = None,
) -> Axes:
    """Plot radius of gyration over time.

    Args:
        result: RadiusOfGyrationResult from ``compute_radius_of_gyration``.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        linewidth: Line width.
        color: Optional line color. Defaults to matplotlib's next color in
            the property cycle.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    axis.plot(
        result.time_ns,
        result.radius_gyration_angstrom,
        label=label,
        linewidth=linewidth,
        color=color,
    )
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Radius of Gyration (Å)")
    if label is not None:
        axis.legend()
    return axis


def plot_distances(
    result: DistanceResult,
    *,
    ax: Axes | None = None,
    pair_labels: list[str] | None = None,
    linewidth: float = 1.5,
) -> Axes:
    """Plot pairwise distances over time.

    Args:
        result: DistanceResult from ``compute_distances`` or
            ``compute_minimum_distance``.
        ax: Optional matplotlib axis.
        pair_labels: Optional labels for each distance pair.
        linewidth: Line width.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    n_pairs = result.distances_nm.shape[1]
    for i in range(n_pairs):
        label = pair_labels[i] if pair_labels and i < len(pair_labels) else None
        axis.plot(result.time_ns, result.distances_angstrom[:, i], label=label, linewidth=linewidth)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Distance (Å)")
    if pair_labels:
        axis.legend()
    return axis


def plot_native_contacts(
    result: NativeContactResult,
    *,
    ax: Axes | None = None,
    label: str | None = None,
    linewidth: float = 1.5,
    color: str | None = None,
) -> Axes:
    """Plot fraction of native contacts Q(t) over time.

    Args:
        result: NativeContactResult from ``compute_native_contacts``.
        ax: Optional matplotlib axis.
        label: Optional legend label.
        linewidth: Line width.
        color: Optional line color. Defaults to matplotlib's next color in
            the property cycle.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    axis.plot(result.time_ns, result.fraction, label=label, linewidth=linewidth, color=color)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Q(t)")
    axis.set_ylim(-0.05, 1.05)
    if label is not None:
        axis.legend()
    return axis


def plot_energy(
    data: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    time_column: str | None = None,
    ax: Axes | None = None,
    linewidth: float = 1.5,
) -> Axes:
    """Plot energy terms from a parsed XVG or EDR DataFrame.

    Args:
        data: DataFrame as returned by ``read_xvg`` or ``read_edr``.
        columns: Column names to plot. If ``None``, plots all non-time
            columns.
        time_column: Name of the time column. Auto-detected if ``None``
            (looks for ``"Time"`` or the first column).
        ax: Optional matplotlib axis.
        linewidth: Line width.

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)

    if time_column is None:
        for candidate in ("Time", "time", data.columns[0]):
            if candidate in data.columns:
                time_column = candidate
                break
        else:
            time_column = data.columns[0]

    time_values = data[time_column]

    if columns is None:
        columns = [col for col in data.columns if col != time_column]

    for col in columns:
        axis.plot(time_values, data[col], label=col, linewidth=linewidth)

    axis.set_xlabel(time_column)
    axis.set_ylabel("Energy")
    if len(columns) > 1:
        axis.legend()
    elif len(columns) == 1:
        axis.set_ylabel(columns[0])
    return axis


def plot_delta_rmsf(
    result: DeltaRMSFResult,
    *,
    ax: Axes | None = None,
    labels: tuple[str, str] = ("system A", "system B"),
    positive_color: str = "crimson",
    negative_color: str = "steelblue",
    linewidth: float = 1.5,
    sem_alpha: float = 0.3,
) -> Axes:
    """Plot per-residue RMSF difference as a line with SEM error band.

    The delta-RMSF line is drawn in black. When SEM data is available
    (requires >= 2 replicas per system), the +/- 1 SEM band is drawn
    with ``positive_color`` where delta > 0 (system B more flexible)
    and ``negative_color`` where delta < 0 (system A more flexible).

    Args:
        result: DeltaRMSFResult from ``compute_delta_rmsf``.
        ax: Optional matplotlib axis.
        labels: Tuple of ``(system_a_name, system_b_name)`` used in the
            legend (e.g. ``("BirA", "TurboID")``).
        positive_color: SEM band color for residues where system B is
            more flexible (delta > 0).
        negative_color: SEM band color for residues where system A is
            more flexible (delta < 0).
        linewidth: Line width for the delta-RMSF trace.
        sem_alpha: Opacity of the SEM band (0.0 -- 1.0).

    Returns:
        The matplotlib axis.
    """
    axis = get_axis(ax)
    drmsf = result.delta_rmsf_angstrom
    x_values = (
        np.asarray(result.residue_ids, dtype=np.float64)
        if result.residue_ids is not None
        else np.arange(drmsf.size, dtype=np.float64) + 1.0
    )

    axis.plot(x_values, drmsf, color="black", linewidth=linewidth)

    sem = result.sem_angstrom
    if sem is not None:
        upper = drmsf + sem
        lower = drmsf - sem
        positive = (drmsf > 0).tolist()
        negative = (drmsf < 0).tolist()
        axis.fill_between(
            x_values,
            lower,
            upper,
            where=positive,
            alpha=sem_alpha,
            color=positive_color,
            label=f"{labels[1]} more flexible",
        )
        axis.fill_between(
            x_values,
            lower,
            upper,
            where=negative,
            alpha=sem_alpha,
            color=negative_color,
            label=f"{labels[0]} more flexible",
        )

    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Residue ID")
    axis.set_ylabel(r"$\Delta$RMSF (Å)")
    if sem is not None:
        axis.legend()
    return axis
