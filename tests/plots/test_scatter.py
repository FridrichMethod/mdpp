"""Regression tests for residue-aware torsion scatter plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdpp.analysis.decomposition import TorsionFeatures
from mdpp.plots.scatter import plot_ramachandran


def test_ramachandran_pairs_angles_from_same_residue() -> None:
    """Terminal phi/psi columns must be omitted instead of shifting all pairs."""
    torsions = TorsionFeatures(
        values=np.deg2rad([[11, 12, 14, 15, 20, 21, 24], [31, 32, 34, 35, 40, 41, 44]]),
        labels=["phi_1", "phi_2", "phi_4", "phi_5", "psi_0", "psi_1", "psi_4"],
    )
    fig, ax = plt.subplots()
    try:
        plot_ramachandran(torsions, ax=ax)
        np.testing.assert_allclose(
            np.asarray(ax.collections[0].get_offsets(), dtype=np.float64),
            [[11, 21], [14, 24], [31, 41], [34, 44]],
        )
    finally:
        plt.close(fig)


def test_ramachandran_rejects_unpaired_angles() -> None:
    """A dipeptide with no residue having both angles has no Ramachandran pairs."""
    torsions = TorsionFeatures(values=np.zeros((3, 2)), labels=["phi_1", "psi_0"])
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="both phi and psi"):
            plot_ramachandran(torsions, ax=ax)
    finally:
        plt.close(fig)
