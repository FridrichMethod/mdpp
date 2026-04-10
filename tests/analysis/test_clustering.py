"""Tests for conformational clustering (RMSD matrix + GROMOS)."""

from __future__ import annotations

import os
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
# Optional backend availability
# ---------------------------------------------------------------------------

try:
    import cupy  # noqa: F401

    _has_cupy = True
except ImportError:
    _has_cupy = False

try:
    import torch  # noqa: F401

    _has_torch = True
except ImportError:
    _has_torch = False

try:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax  # noqa: F401

    _has_jax = True
except ImportError:
    _has_jax = False

requires_cupy = pytest.mark.skipif(not _has_cupy, reason="CuPy not installed")
requires_torch = pytest.mark.skipif(not _has_torch, reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not _has_jax, reason="JAX not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backbone_trajectory() -> md.Trajectory:
    """Return a small trajectory with backbone-like atoms and known geometry.

    5 ALA residues x 3 backbone atoms (N, CA, C) = 15 atoms, 30 frames.
    Coordinates are a shared base structure plus small random perturbations
    (0.02 nm), giving pairwise RMSDs in the 0.01-0.06 nm range.
    """
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
            topology.add_bond(atoms[-6], c)

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
    """Return a larger trajectory for benchmark tests.

    50 ALA residues x 3 backbone atoms = 150 atoms, 200 frames.
    Gives 19 900 unique pairs for the pairwise RMSD matrix.
    """
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
# GPU / optional backend agreement tests
# ---------------------------------------------------------------------------


class TestGpuBackendAgreement:
    """Verify that torch, jax, and cupy backends agree with numba.

    Each test computes the pairwise RMSD matrix with the GPU backend and
    the numba reference, then asserts element-wise agreement within
    ``atol=5e-5 nm`` (~0.0005 Angstrom).  The numba result is itself
    validated against mdtraj in ``TestComputeRmsdMatrix``.

    Tests are skipped when the corresponding package is not installed.
    """

    @requires_torch
    def test_torch_agrees_with_numba(self, backbone_trajectory: md.Trajectory) -> None:
        """PyTorch backend should match numba QCP within 5e-5 nm."""
        ref = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="numba")
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="torch")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    @requires_jax
    def test_jax_agrees_with_numba(self, backbone_trajectory: md.Trajectory) -> None:
        """JAX backend should match numba QCP within 5e-5 nm."""
        ref = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="numba")
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="jax")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    @requires_cupy
    def test_cupy_agrees_with_numba(self, backbone_trajectory: md.Trajectory) -> None:
        """CuPy backend should match numba QCP within 5e-5 nm."""
        ref = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="numba")
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="cupy")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    @requires_torch
    @requires_jax
    @requires_cupy
    def test_all_gpu_backends_agree(self, backbone_trajectory: md.Trajectory) -> None:
        """All three GPU backends should agree with each other within 5e-5 nm."""
        results = {
            name: compute_rmsd_matrix(
                backbone_trajectory, atom_selection="all", backend=name
            ).rmsd_matrix_nm
            for name in ("torch", "jax", "cupy")
        }
        for a_name, a_mat in results.items():
            for b_name, b_mat in results.items():
                if a_name >= b_name:
                    continue
                np.testing.assert_allclose(
                    a_mat,
                    b_mat,
                    atol=5e-5,
                    err_msg=f"{a_name} vs {b_name}",
                )


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


def _run_rmsd_benchmark(traj: md.Trajectory) -> None:
    """Run all available RMSD matrix backends and print a comparison table.

    Each backend is warmed up (JIT/CUDA context/XLA compilation) with a
    3-frame slice before the timed run on the full trajectory.  The numba
    result is used as the correctness reference (atol=5e-5 nm).
    """
    ref = compute_rmsd_matrix(traj, atom_selection="all", backend="numba")
    ref_mat = ref.rmsd_matrix_nm

    backends: list[tuple[str, bool]] = [
        ("mdtraj", True),
        ("numba", True),
        ("cupy", _has_cupy),
        ("torch", _has_torch),
        ("jax", _has_jax),
    ]

    timings: dict[str, float] = {}
    for name, available in backends:
        if not available:
            continue
        # Warmup
        compute_rmsd_matrix(traj[:3], atom_selection="all", backend=name)
        t0 = time.perf_counter()
        result = compute_rmsd_matrix(traj, atom_selection="all", backend=name)
        timings[name] = time.perf_counter() - t0
        if name != "numba":
            np.testing.assert_allclose(result.rmsd_matrix_nm, ref_mat, atol=5e-5)

    n = traj.n_frames
    n_pairs = n * (n - 1) // 2
    print(f"\n  RMSD matrix benchmark: {n} frames, {len(ref.atom_indices)} atoms ({n_pairs} pairs)")
    print(f"  {'Backend':<10s} {'Time (s)':>10s} {'vs mdtraj':>10s}")
    print(f"  {'-' * 32}")
    t_mdtraj = timings.get("mdtraj", 1.0)
    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        speedup = t_mdtraj / t
        print(f"  {name:<10s} {t:>10.4f} {speedup:>9.1f}x")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "n_frames",
    [
        pytest.param(100, id="small-100"),
        pytest.param(200, id="medium-200"),
    ],
)
def test_benchmark_rmsd_backends(n_frames: int) -> None:
    """Benchmark all available RMSD matrix backends at different scales.

    Builds a synthetic trajectory (50 residues x 150 backbone atoms) and
    times each installed backend.  Verifies correctness of every backend
    against the numba reference (atol=5e-5 nm) as a side effect.

    Run only benchmarks:  ``pytest -m benchmark``
    Skip benchmarks:      ``pytest -m "not benchmark"``
    """
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for res_idx in range(1, 51):
        residue = topology.add_residue("ALA", chain, resSeq=res_idx)
        n_atom = topology.add_atom("N", md.element.nitrogen, residue)
        ca = topology.add_atom("CA", md.element.carbon, residue)
        c = topology.add_atom("C", md.element.carbon, residue)
        atoms.extend([n_atom, ca, c])
        topology.add_bond(n_atom, ca)
        topology.add_bond(ca, c)
        if res_idx > 1:
            topology.add_bond(atoms[-6], c)

    rng = np.random.RandomState(99)
    n_atoms = len(atoms)
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    xyz = base + perturbation
    traj = md.Trajectory(xyz=xyz, topology=topology)

    _run_rmsd_benchmark(traj)
