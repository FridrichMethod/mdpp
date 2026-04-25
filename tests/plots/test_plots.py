"""Smoke tests for plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mdpp.analysis.fes import compute_fes_2d
from mdpp.analysis.hbond import compute_hbonds
from mdpp.analysis.metrics import (
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
