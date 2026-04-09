"""Tests for featurize_ca_distances."""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from mdpp.analysis.decomposition import DistanceFeatures, featurize_ca_distances


@pytest.fixture()
def ca_trajectory() -> md.Trajectory:
    """5-frame trajectory with 4 CA atoms."""
    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(1, 5):
        res = topology.add_residue("ALA", chain, resSeq=i)
        topology.add_atom("CA", md.element.carbon, res)
        topology.add_atom("CB", md.element.carbon, res)

    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(5, 8, 3)).astype(np.float32) * 0.1
    return md.Trajectory(xyz=xyz, topology=topology)


class TestFeaturizeCaDistances:
    def test_returns_distance_features(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        assert isinstance(result, DistanceFeatures)

    def test_shape(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        n_ca = 4
        n_pairs = n_ca * (n_ca - 1) // 2  # 6
        assert result.values.shape == (5, n_pairs)
        assert result.pairs.shape == (n_pairs, 2)
        assert result.atom_indices.size == n_ca

    def test_pairs_are_upper_triangle(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        for i, j in result.pairs:
            assert i < j

    def test_distances_positive(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        assert np.all(result.values >= 0)

    def test_custom_selection(self, ca_trajectory: md.Trajectory) -> None:
        """Selecting CA from first 2 residues gives 1 pair."""
        result = featurize_ca_distances(ca_trajectory, atom_selection="name CA and resSeq 1 2")
        assert result.values.shape[1] == 1
        assert result.atom_indices.size == 2

    def test_matches_mdtraj_distances(self, ca_trajectory: md.Trajectory) -> None:
        """Values should match mdtraj's compute_distances."""
        result = featurize_ca_distances(ca_trajectory)
        ca_idx = ca_trajectory.topology.select("name CA")
        sliced = ca_trajectory.atom_slice(ca_idx)
        expected = md.compute_distances(sliced, result.pairs)
        np.testing.assert_allclose(result.values, expected, atol=1e-6)

    def test_single_atom_raises(self) -> None:
        topology = md.Topology()
        chain = topology.add_chain()
        res = topology.add_residue("ALA", chain, resSeq=1)
        topology.add_atom("CA", md.element.carbon, res)
        traj = md.Trajectory(xyz=np.zeros((3, 1, 3), dtype=np.float32), topology=topology)
        with pytest.raises(ValueError, match="At least 2 atoms"):
            featurize_ca_distances(traj)

    def test_dtype(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        assert result.values.dtype == np.float64

    def test_matches_mdtraj_compute_distances(self, ca_trajectory: md.Trajectory) -> None:
        """Numba kernel output must match mdtraj's compute_distances exactly."""
        result = featurize_ca_distances(ca_trajectory)
        ca_idx = ca_trajectory.topology.select("name CA")
        sliced = ca_trajectory.atom_slice(ca_idx)
        mdtraj_distances = md.compute_distances(sliced, result.pairs)
        np.testing.assert_allclose(result.values, mdtraj_distances, atol=1e-5)


# --- Benchmark: Numba vs mdtraj ---


@pytest.fixture()
def benchmark_traj() -> md.Trajectory:
    """1000-frame, 100-CA-atom trajectory for benchmarking."""
    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(1, 101):
        res = topology.add_residue("ALA", chain, resSeq=i)
        topology.add_atom("CA", md.element.carbon, res)

    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(1000, 100, 3)).astype(np.float32) * 0.1
    return md.Trajectory(xyz=xyz, topology=topology)


def test_benchmark_numba_vs_mdtraj(benchmark_traj: md.Trajectory) -> None:
    """Numba parallel kernel should be faster than mdtraj for non-periodic distances.

    Also verifies numerical correctness against mdtraj's output.
    """
    import time

    from mdpp.analysis.decomposition import _pairwise_distances_numba

    n_atoms = benchmark_traj.n_atoms
    pairs = np.array(
        [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)],
        dtype=np.int_,
    )
    n_pairs = pairs.shape[0]
    n_frames = benchmark_traj.n_frames

    # Warm up JIT compilation
    _pairwise_distances_numba(benchmark_traj.xyz[:2], pairs[:10])

    # Benchmark: Numba
    t0 = time.perf_counter()
    numba_result = _pairwise_distances_numba(benchmark_traj.xyz, pairs)
    t_numba = time.perf_counter() - t0

    # Benchmark: mdtraj
    t0 = time.perf_counter()
    mdtraj_result = np.asarray(
        md.compute_distances(benchmark_traj, pairs, periodic=False),
        dtype=np.float64,
    )
    t_mdtraj = time.perf_counter() - t0

    # Correctness
    assert numba_result.shape == (n_frames, n_pairs)
    np.testing.assert_allclose(numba_result, mdtraj_result, atol=1e-5)

    print(f"\n  {n_frames} frames x {n_atoms} atoms ({n_pairs} pairs)")
    print(f"  Numba:  {t_numba:.4f}s")
    print(f"  mdtraj: {t_mdtraj:.4f}s")
    print(f"  Speedup: {t_mdtraj / t_numba:.1f}x")
