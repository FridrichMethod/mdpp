"""Tests for featurize_ca_distances and distance backends."""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from mdpp.analysis.decomposition import (
    DistanceFeatures,
    _pairwise_distances_mdtraj,
    _pairwise_distances_numba,
    featurize_ca_distances,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ca_trajectory() -> md.Trajectory:
    """5-frame trajectory with 4 residues (CA + CB each = 8 atoms)."""
    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(1, 5):
        res = topology.add_residue("ALA", chain, resSeq=i)
        topology.add_atom("CA", md.element.carbon, res)
        topology.add_atom("CB", md.element.carbon, res)

    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(5, 8, 3)).astype(np.float32) * 0.1
    return md.Trajectory(xyz=xyz, topology=topology)


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


# ---------------------------------------------------------------------------
# featurize_ca_distances -- basic behavior
# ---------------------------------------------------------------------------


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
        result = featurize_ca_distances(ca_trajectory, atom_selection="name CA and resSeq 1 2")
        assert result.values.shape[1] == 1
        assert result.atom_indices.size == 2

    def test_default_dtype_float32(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        assert result.values.dtype == np.float32

    def test_explicit_dtype_float64(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, dtype=np.float64)
        assert result.values.dtype == np.float64

    def test_single_atom_raises(self) -> None:
        topology = md.Topology()
        chain = topology.add_chain()
        res = topology.add_residue("ALA", chain, resSeq=1)
        topology.add_atom("CA", md.element.carbon, res)
        traj = md.Trajectory(xyz=np.zeros((3, 1, 3), dtype=np.float32), topology=topology)
        with pytest.raises(ValueError, match="At least 2 atoms"):
            featurize_ca_distances(traj)

    def test_unknown_backend_raises(self, ca_trajectory: md.Trajectory) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            featurize_ca_distances(ca_trajectory, backend="invalid")

    def test_residue_ids_preserved(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        traj = ca_trajectory.atom_slice(result.atom_indices)
        res_ids = [r.resSeq for r in traj.topology.residues]
        assert res_ids == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Backend equivalence
# ---------------------------------------------------------------------------


class TestBackendEquivalence:
    """Numba and mdtraj backends should produce identical results."""

    def test_numba_matches_mdtraj(self, ca_trajectory: md.Trajectory) -> None:
        numba_result = featurize_ca_distances(ca_trajectory, backend="numba")
        mdtraj_result = featurize_ca_distances(ca_trajectory, backend="mdtraj")
        np.testing.assert_allclose(numba_result.values, mdtraj_result.values, atol=1e-5)

    def test_numba_matches_mdtraj_large(self, benchmark_traj: md.Trajectory) -> None:
        """Equivalence on a larger trajectory."""
        numba_result = featurize_ca_distances(benchmark_traj, backend="numba")
        mdtraj_result = featurize_ca_distances(benchmark_traj, backend="mdtraj")
        np.testing.assert_allclose(numba_result.values, mdtraj_result.values, atol=1e-5)

    def test_backends_same_pairs(self, ca_trajectory: md.Trajectory) -> None:
        numba_result = featurize_ca_distances(ca_trajectory, backend="numba")
        mdtraj_result = featurize_ca_distances(ca_trajectory, backend="mdtraj")
        np.testing.assert_array_equal(numba_result.pairs, mdtraj_result.pairs)

    def test_backends_same_atom_indices(self, ca_trajectory: md.Trajectory) -> None:
        numba_result = featurize_ca_distances(ca_trajectory, backend="numba")
        mdtraj_result = featurize_ca_distances(ca_trajectory, backend="mdtraj")
        np.testing.assert_array_equal(numba_result.atom_indices, mdtraj_result.atom_indices)


# ---------------------------------------------------------------------------
# Low-level kernel tests
# ---------------------------------------------------------------------------


class TestNumbaKernel:
    """Direct tests on _pairwise_distances_numba."""

    def test_known_distance(self) -> None:
        """Two atoms 1 nm apart along x-axis."""
        xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_numba(xyz, pairs)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_self_distance_is_zero(self) -> None:
        """Distance of an atom to itself should be zero."""
        xyz = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
        pairs = np.array([[0, 0]], dtype=np.int_)
        result = _pairwise_distances_numba(xyz, pairs)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_3d_distance(self) -> None:
        """Distance in 3D: sqrt(1 + 4 + 9) = sqrt(14)."""
        xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]], dtype=np.float32)
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_numba(xyz, pairs)
        assert result[0, 0] == pytest.approx(np.sqrt(14.0), abs=1e-5)

    def test_multi_frame(self) -> None:
        """Distances should vary across frames."""
        xyz = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_numba(xyz, pairs)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(2.0, abs=1e-6)

    def test_output_dtype_float64(self) -> None:
        xyz = np.zeros((2, 2, 3), dtype=np.float32)
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_numba(xyz, pairs)
        assert result.dtype == np.float64

    def test_out_of_range_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[0, 5]], dtype=np.int_)  # 5 >= 3 atoms
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            _pairwise_distances_numba(xyz, pairs)

    def test_negative_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[-1, 1]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            _pairwise_distances_numba(xyz, pairs)

    def test_empty_pairs(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.empty((0, 2), dtype=np.int_)
        result = _pairwise_distances_numba(xyz, pairs)
        assert result.shape == (2, 0)


class TestMdtrajKernel:
    """Direct tests on _pairwise_distances_mdtraj."""

    def test_known_distance(self) -> None:
        topology = md.Topology()
        chain = topology.add_chain()
        for _ in range(2):
            res = topology.add_residue("ALA", chain)
            topology.add_atom("CA", md.element.carbon, res)
        xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
        traj = md.Trajectory(xyz=xyz, topology=topology)
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_mdtraj(traj, pairs, periodic=False)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_periodic_flag_accepted(self) -> None:
        """periodic=True should not error even without unit-cell (mdtraj falls back)."""
        topology = md.Topology()
        chain = topology.add_chain()
        for _ in range(2):
            res = topology.add_residue("ALA", chain)
            topology.add_atom("CA", md.element.carbon, res)
        xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
        traj = md.Trajectory(xyz=xyz, topology=topology)
        pairs = np.array([[0, 1]], dtype=np.int_)
        # No unit-cell: periodic=True falls back to non-periodic in mdtraj
        result = _pairwise_distances_mdtraj(traj, pairs, periodic=True)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_default_dtype_float32(self) -> None:
        topology = md.Topology()
        chain = topology.add_chain()
        for _ in range(2):
            res = topology.add_residue("ALA", chain)
            topology.add_atom("CA", md.element.carbon, res)
        traj = md.Trajectory(xyz=np.zeros((2, 2, 3), dtype=np.float32), topology=topology)
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_mdtraj(traj, pairs, periodic=False)
        assert result.dtype == np.float32

    def test_explicit_dtype_float64(self) -> None:
        topology = md.Topology()
        chain = topology.add_chain()
        for _ in range(2):
            res = topology.add_residue("ALA", chain)
            topology.add_atom("CA", md.element.carbon, res)
        traj = md.Trajectory(xyz=np.zeros((2, 2, 3), dtype=np.float32), topology=topology)
        pairs = np.array([[0, 1]], dtype=np.int_)
        result = _pairwise_distances_mdtraj(traj, pairs, periodic=False, dtype=np.float64)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_numba_vs_mdtraj(benchmark_traj: md.Trajectory) -> None:
    """Numba kernel should be faster and numerically equivalent to mdtraj."""
    import time

    n_atoms = benchmark_traj.n_atoms
    pairs = np.array(
        [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)],
        dtype=np.int_,
    )
    n_pairs = pairs.shape[0]
    n_frames = benchmark_traj.n_frames

    # Warm up JIT
    _pairwise_distances_numba(benchmark_traj.xyz[:2], pairs[:10])

    # Numba
    t0 = time.perf_counter()
    numba_result = _pairwise_distances_numba(benchmark_traj.xyz, pairs)
    t_numba = time.perf_counter() - t0

    # mdtraj
    t0 = time.perf_counter()
    mdtraj_result = _pairwise_distances_mdtraj(benchmark_traj, pairs, periodic=False)
    t_mdtraj = time.perf_counter() - t0

    # Correctness
    assert numba_result.shape == (n_frames, n_pairs)
    np.testing.assert_allclose(numba_result, mdtraj_result, atol=1e-5)

    # Memory: numba returns float64, mdtraj float32 -> numba is 2x larger per value
    # but avoids the intermediate allocations mdtraj uses internally

    print(f"\n  {n_frames} frames x {n_atoms} atoms ({n_pairs} pairs)")
    print(f"  Numba:  {t_numba:.4f}s (float64)")
    print(f"  mdtraj: {t_mdtraj:.4f}s (float32 -> float64)")
    print(f"  Speedup: {t_mdtraj / t_numba:.1f}x")
