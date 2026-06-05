"""Tests for mdpp.analysis.dssp."""

from __future__ import annotations

import numpy as np

from mdpp.analysis.dssp import compute_dssp


class TestComputeDSSP:
    def test_shapes(self, correlated_ca_trajectory):
        result = compute_dssp(correlated_ca_trajectory)
        n_frames = correlated_ca_trajectory.n_frames
        n_residues = correlated_ca_trajectory.topology.n_residues
        assert result.assignments.shape == (n_frames, n_residues)
        assert result.residue_ids.shape == (n_residues,)
        assert result.frequency.shape[0] == n_residues

    def test_time_axis_matches_trajectory(self, correlated_ca_trajectory):
        result = compute_dssp(correlated_ca_trajectory)
        assert result.time_ps.shape == (correlated_ca_trajectory.n_frames,)
        np.testing.assert_allclose(result.time_ps, correlated_ca_trajectory.time, rtol=1e-6)

    def test_time_ns_property(self, correlated_ca_trajectory):
        result = compute_dssp(correlated_ca_trajectory)
        np.testing.assert_allclose(result.time_ns, result.time_ps / 1000.0, rtol=1e-6)

    def test_timestep_override(self, correlated_ca_trajectory):
        result = compute_dssp(correlated_ca_trajectory, timestep_ps=2.0)
        expected = np.arange(correlated_ca_trajectory.n_frames) * 2.0
        np.testing.assert_allclose(result.time_ps, expected, rtol=1e-6)
