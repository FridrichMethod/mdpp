"""Tests for core MD trajectory metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis import (
    compute_dccm,
    compute_radius_of_gyration,
    compute_rmsd,
    compute_rmsf,
    compute_sasa,
)


def test_compute_rmsd_reference_frame_is_zero(two_atom_trajectory) -> None:
    """RMSD should be zero at the reference frame."""
    result = compute_rmsd(
        two_atom_trajectory,
        atom_selection="name CA",
        reference_frame=0,
        align=False,
    )

    assert result.rmsd_nm.shape == (two_atom_trajectory.n_frames,)
    assert result.rmsd_nm[0] == pytest.approx(0.0)
    assert result.time_ns[-1] == pytest.approx(0.02)


def test_compute_rmsf_matches_expected_fluctuation(two_atom_trajectory) -> None:
    """RMSF should match a known two-atom fluctuation example."""
    result = compute_rmsf(two_atom_trajectory, atom_selection="name CA", align=False)
    expected_atom_2_nm = np.sqrt((0.0**2 + 0.1**2 + (-0.1) ** 2) / 3.0)

    assert result.rmsf_nm[0] == pytest.approx(0.0, abs=1e-8)
    assert result.rmsf_nm[1] == pytest.approx(expected_atom_2_nm, rel=1e-6)
    assert result.rmsf_angstrom[1] == pytest.approx(expected_atom_2_nm * 10.0, rel=1e-6)
    assert result.residue_ids is not None
    assert np.array_equal(result.residue_ids, np.array([1, 2], dtype=np.int_))


def test_compute_dccm_detects_correlation_patterns(correlated_ca_trajectory) -> None:
    """DCCM should capture correlated and anti-correlated atom pairs."""
    result = compute_dccm(correlated_ca_trajectory, atom_selection="name CA", align=False)

    assert result.correlation.shape == (3, 3)
    assert np.allclose(np.diag(result.correlation), 1.0)
    assert result.correlation[0, 1] > 0.99
    assert result.correlation[0, 2] < -0.99


def test_compute_sasa_atom_and_residue_modes(correlated_ca_trajectory) -> None:
    """SASA output shapes should match atom and residue modes."""
    atom_result = compute_sasa(
        correlated_ca_trajectory,
        atom_selection=None,
        mode="atom",
        n_sphere_points=120,
    )
    residue_result = compute_sasa(
        correlated_ca_trajectory,
        atom_selection=None,
        mode="residue",
        n_sphere_points=120,
    )

    assert atom_result.values_nm2.shape == (
        correlated_ca_trajectory.n_frames,
        correlated_ca_trajectory.n_atoms,
    )
    assert residue_result.values_nm2.shape == (
        correlated_ca_trajectory.n_frames,
        correlated_ca_trajectory.n_residues,
    )
    assert np.all(atom_result.total_nm2 > 0.0)


def test_compute_radius_of_gyration_returns_timeseries(correlated_ca_trajectory) -> None:
    """Radius of gyration should be available per frame."""
    result = compute_radius_of_gyration(
        correlated_ca_trajectory,
        atom_selection="name CA",
    )

    assert result.radius_gyration_nm.shape == (correlated_ca_trajectory.n_frames,)
    assert np.all(result.radius_gyration_nm > 0.0)
    assert result.time_ns.shape == (correlated_ca_trajectory.n_frames,)
