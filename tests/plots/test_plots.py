"""Smoke tests for plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mdpp.analysis.fes import compute_fes_2d
from mdpp.analysis.hbond import compute_hbonds
from mdpp.analysis.metrics import (
    RMSDResult,
    compute_dccm,
    compute_rmsd,
    compute_rmsf,
    compute_sasa,
)
from mdpp.plots import (
    plot_dccm,
    plot_fes,
    plot_hbond_counts,
    plot_hbond_occupancy,
    plot_rmsd,
    plot_rmsf,
    plot_sasa,
)


def test_plot_helpers_return_axes(
    two_atom_trajectory,
    correlated_ca_trajectory,
    hbond_trajectory,
) -> None:
    """Each plotting helper should return a valid axis with expected labels."""
    rmsd_result = compute_rmsd(two_atom_trajectory, atom_selection="name CA")
    rmsf_result = compute_rmsf(two_atom_trajectory, atom_selection="name CA")
    dccm_result = compute_dccm(correlated_ca_trajectory, atom_selection="name CA")
    sasa_result = compute_sasa(correlated_ca_trajectory, atom_selection=None, mode="residue")
    hbond_result = compute_hbonds(
        hbond_trajectory,
        method="baker_hubbard",
        freq=0.0,
        periodic=False,
    )

    rng = np.random.default_rng(4)
    fes_result = compute_fes_2d(rng.normal(size=3000), rng.normal(size=3000), bins=20)

    figure, axes = plt.subplots(2, 4, figsize=(16, 6))
    axis_rmsd = plot_rmsd(rmsd_result, ax=axes[0, 0])
    axis_rmsf = plot_rmsf(rmsf_result, ax=axes[0, 1])
    axis_dccm = plot_dccm(dccm_result, ax=axes[0, 2], add_colorbar=False)
    axis_hbond_counts = plot_hbond_counts(hbond_result, ax=axes[0, 3])
    axis_sasa = plot_sasa(sasa_result, ax=axes[1, 0], aggregate="mean")
    axis_fes = plot_fes(fes_result, ax=axes[1, 1], add_colorbar=False)
    axis_hbond_occ = plot_hbond_occupancy(hbond_result, ax=axes[1, 2])

    assert axis_rmsd.get_ylabel() == r"RMSD ($\mathrm{\AA}$)"
    assert axis_rmsf.get_ylabel() == r"RMSF ($\mathrm{\AA}$)"
    assert axis_dccm.get_xlabel() == "Residue ID"
    assert axis_hbond_counts.get_ylabel() == "Hydrogen Bond Count"
    assert axis_sasa.get_xlabel() == "Time (ns)"
    assert axis_fes.get_ylabel() == "CV 2"
    assert axis_hbond_occ.get_ylabel() == "Occupancy (%)"
    plt.close(figure)


def test_plot_rmsd_moving_average_not_edge_biased() -> None:
    """A constant RMSD trace must stay flat under the moving-average overlay.

    The previous ``np.convolve(mode="same")`` box kernel zero-padded the edges
    and pulled the smoothed line toward zero; count-normalization keeps it flat.
    """
    rmsd_nm = np.full(50, 0.3, dtype=np.float64)  # constant 3.0 Angstrom
    result = RMSDResult(
        time_ps=np.arange(50.0) * 10.0,
        rmsd_nm=rmsd_nm,
        atom_indices=np.array([0], dtype=np.int_),
    )
    figure, axis = plt.subplots()
    plot_rmsd(result, ax=axis, moving_average=11)
    # Line 0 is the raw trace; line 1 is the moving-average overlay.
    smoothed = np.asarray(axis.lines[1].get_ydata(), dtype=float)
    np.testing.assert_allclose(smoothed, 3.0, atol=1e-6)
    plt.close(figure)


def test_plot_dccm_extent_padded_to_cell_edges(correlated_ca_trajectory) -> None:
    """Imshow extent must sit on outer cell edges, not residue-id cell centers."""
    dccm_result = compute_dccm(correlated_ca_trajectory, atom_selection="name CA")
    figure, axis = plt.subplots()
    plot_dccm(dccm_result, ax=axis, add_colorbar=False)
    # residue_ids = [1, 2, 3], n = 3 -> half-cell = 0.5 -> edges at 0.5 and 3.5
    # (cell centers, the old buggy behavior, would be (1, 3, 1, 3)).
    assert tuple(axis.images[0].get_extent()) == (0.5, 3.5, 0.5, 3.5)
    plt.close(figure)
