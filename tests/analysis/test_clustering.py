"""Tests for conformational clustering (RMSD matrix + GROMOS)."""

from __future__ import annotations

import time

import mdtraj as md
import numpy as np
import pytest

from mdpp.analysis.clustering import (
    ClusteringResult,
    RMSDMatrixResult,
    cluster_conformations,
    compute_rmsd_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backbone_trajectory() -> md.Trajectory:
    """Return a small trajectory with backbone-like atoms and known geometry."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for res_idx in range(1, 6):
        residue = topology.add_residue("ALA", chain, resSeq=res_idx)
        n = topology.add_atom("N", md.element.nitrogen, residue)
        ca = topology.add_atom("CA", md.element.carbon, residue)
        c = topology.add_atom("C", md.element.carbon, residue)
        atoms.extend([n, ca, c])
        topology.add_bond(n, ca)
        topology.add_bond(ca, c)
        if res_idx > 1:
            topology.add_bond(atoms[-6], c)  # previous C -> this N

    rng = np.random.RandomState(42)
    n_frames = 30
    n_atoms = len(atoms)
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    xyz = base + perturbation
    time_ps = np.arange(n_frames, dtype=np.float64) * 10.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


@pytest.fixture()
def large_trajectory() -> md.Trajectory:
    """Return a larger trajectory for benchmark tests."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for res_idx in range(1, 51):
        residue = topology.add_residue("ALA", chain, resSeq=res_idx)
        n = topology.add_atom("N", md.element.nitrogen, residue)
        ca = topology.add_atom("CA", md.element.carbon, residue)
        c = topology.add_atom("C", md.element.carbon, residue)
        atoms.extend([n, ca, c])
        topology.add_bond(n, ca)
        topology.add_bond(ca, c)
        if res_idx > 1:
            topology.add_bond(atoms[-6], c)

    rng = np.random.RandomState(99)
    n_frames = 200
    n_atoms = len(atoms)
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    xyz = base + perturbation
    time_ps = np.arange(n_frames, dtype=np.float64) * 10.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


# ---------------------------------------------------------------------------
# RMSD matrix tests
# ---------------------------------------------------------------------------


class TestComputeRmsdMatrix:
    """Tests for ``compute_rmsd_matrix``."""

    def test_result_shape_and_type(self, backbone_trajectory: md.Trajectory) -> None:
        """Result should have the correct shape and be a frozen dataclass."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        n = backbone_trajectory.n_frames

        assert isinstance(result, RMSDMatrixResult)
        assert result.rmsd_matrix_nm.shape == (n, n)
        assert result.rmsd_matrix_angstrom.shape == (n, n)

    def test_diagonal_is_zero(self, backbone_trajectory: md.Trajectory) -> None:
        """Self-RMSD (diagonal) should be zero."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        np.testing.assert_allclose(np.diag(result.rmsd_matrix_nm), 0.0, atol=1e-6)

    def test_matrix_is_symmetric(self, backbone_trajectory: md.Trajectory) -> None:
        """Pairwise RMSD matrix should be symmetric."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        np.testing.assert_allclose(result.rmsd_matrix_nm, result.rmsd_matrix_nm.T, atol=1e-6)

    def test_values_are_nonnegative(self, backbone_trajectory: md.Trajectory) -> None:
        """All RMSD values should be >= 0."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        assert np.all(result.rmsd_matrix_nm >= 0.0)

    def test_angstrom_conversion(self, backbone_trajectory: md.Trajectory) -> None:
        """Angstrom property should be 10x the nm values."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        np.testing.assert_allclose(result.rmsd_matrix_angstrom, result.rmsd_matrix_nm * 10.0)

    def test_invalid_backend_raises(self, backbone_trajectory: md.Trajectory) -> None:
        """An unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="bogus")

    def test_numba_and_mdtraj_agree(self, backbone_trajectory: md.Trajectory) -> None:
        """Numba QCP and mdtraj backends should produce the same matrix."""
        result_numba = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", backend="numba"
        )
        result_mdtraj = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", backend="mdtraj"
        )
        # Symmetrise mdtraj result (it is not inherently symmetric)
        mdtraj_sym = (result_mdtraj.rmsd_matrix_nm + result_mdtraj.rmsd_matrix_nm.T) / 2.0

        np.testing.assert_allclose(result_numba.rmsd_matrix_nm, mdtraj_sym, atol=5e-5)

    def test_backends_agree_on_atom_selection(self, backbone_trajectory: md.Trajectory) -> None:
        """Both backends should respect atom_selection and agree on the result."""
        result_numba = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="name CA", backend="numba"
        )
        result_mdtraj = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="name CA", backend="mdtraj"
        )
        mdtraj_sym = (result_mdtraj.rmsd_matrix_nm + result_mdtraj.rmsd_matrix_nm.T) / 2.0

        assert len(result_numba.atom_indices) == 5  # 5 residues, 1 CA each
        np.testing.assert_array_equal(result_numba.atom_indices, result_mdtraj.atom_indices)
        np.testing.assert_allclose(result_numba.rmsd_matrix_nm, mdtraj_sym, atol=5e-5)


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------


class TestClusterConformations:
    """Tests for ``cluster_conformations``."""

    def test_all_frames_assigned(self, backbone_trajectory: md.Trajectory) -> None:
        """Every frame should receive a cluster label >= 0."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = cluster_conformations(result.rmsd_matrix_nm, cutoff_nm=0.5)

        assert isinstance(clustering, ClusteringResult)
        assert clustering.labels.shape == (backbone_trajectory.n_frames,)
        assert np.all(clustering.labels >= 0)

    def test_cluster_count_matches_labels(self, backbone_trajectory: md.Trajectory) -> None:
        """n_clusters should equal the number of unique labels."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = cluster_conformations(result.rmsd_matrix_nm, cutoff_nm=0.5)

        assert clustering.n_clusters == len(np.unique(clustering.labels))

    def test_medoid_count_matches_clusters(self, backbone_trajectory: md.Trajectory) -> None:
        """There should be one medoid per cluster."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = cluster_conformations(result.rmsd_matrix_nm, cutoff_nm=0.5)

        assert len(clustering.medoid_frames) == clustering.n_clusters

    def test_tight_cutoff_gives_more_clusters(self, backbone_trajectory: md.Trajectory) -> None:
        """A tighter cutoff should produce at least as many clusters."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        loose = cluster_conformations(result.rmsd_matrix_nm, cutoff_nm=1.0)
        tight = cluster_conformations(result.rmsd_matrix_nm, cutoff_nm=0.01)

        assert tight.n_clusters >= loose.n_clusters

    def test_invalid_method_raises(self) -> None:
        """An unsupported method should raise ValueError."""
        dummy_matrix = np.zeros((3, 3))
        with pytest.raises(ValueError, match="Unsupported clustering method"):
            cluster_conformations(dummy_matrix, method="kmeans")


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestRmsdMatrixBenchmark:
    """Performance comparison between numba and mdtraj backends.

    Uses a synthetic trajectory (50 residues x 150 backbone atoms,
    200 frames -> 19 900 unique pairs) to measure wall-clock time of
    the pairwise RMSD matrix computation.

    Marked ``@pytest.mark.slow`` -- skipped by default in fast CI runs.
    Run explicitly with ``pytest -m slow`` or ``pytest -k Benchmark``.
    """

    @pytest.mark.slow()
    def test_numba_not_slower_than_mdtraj(self, large_trajectory: md.Trajectory) -> None:
        """Numba QCP backend should not be slower than the mdtraj loop.

        Methodology:
            1. **JIT warmup** -- a throwaway 3-frame call compiles the
               Numba kernels so compilation time is excluded.
            2. **Timing** -- each backend is timed 5 times on the full
               200-frame trajectory.  Runs alternate (numba, mdtraj,
               numba, mdtraj, ...) to reduce systematic bias from
               thermal throttling or background load.
            3. **Comparison** -- the *median* of each backend's 5 runs
               is used (robust to outliers).  The test asserts that
               Numba's median is less than 2x the mdtraj median.  The
               2x ceiling is deliberately generous so the test stays
               green on single-core CI runners where Numba's ``prange``
               cannot exploit parallelism; on multi-core machines the
               Numba kernel is typically 4-50x faster.
            4. **Printed report** -- median times and the speedup ratio
               are printed to stdout (visible with ``pytest -s``).
        """
        # JIT warmup
        compute_rmsd_matrix(large_trajectory[:3], atom_selection="all", backend="numba")

        n_runs = 5
        times_numba: list[float] = []
        times_mdtraj: list[float] = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            compute_rmsd_matrix(large_trajectory, atom_selection="all", backend="numba")
            times_numba.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            compute_rmsd_matrix(large_trajectory, atom_selection="all", backend="mdtraj")
            times_mdtraj.append(time.perf_counter() - t0)

        median_numba = float(np.median(times_numba))
        median_mdtraj = float(np.median(times_mdtraj))

        print(
            f"\n  numba:  {median_numba:.4f}s"
            f"\n  mdtraj: {median_mdtraj:.4f}s"
            f"\n  ratio:  {median_mdtraj / median_numba:.1f}x"
        )

        # Numba should be no more than 2x slower (generous margin for CI).
        assert median_numba < median_mdtraj * 2.0, (
            f"Numba ({median_numba:.3f}s) more than 2x slower than mdtraj ({median_mdtraj:.3f}s)"
        )
