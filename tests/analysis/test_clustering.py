"""Tests for conformational clustering (RMSD matrix + GROMOS)."""

from __future__ import annotations

import time

import mdtraj as md
import numpy as np
import pytest
from numpy.typing import NDArray

from mdpp.analysis._backends import has_cupy, has_jax, has_torch
from mdpp.analysis.clustering import (
    ClusteringResult,
    RMSDMatrixResult,
    cluster_conformations,
    compute_rmsd_matrix,
)

requires_cupy = pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
requires_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not has_jax, reason="JAX not installed")

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
        with pytest.raises(ValueError, match="Unknown backend"):
            compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="bogus")  # type: ignore[arg-type]

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


@pytest.mark.gpu
class TestGpuBackendAgreement:
    """Verify that torch, jax, and cupy backends agree with numba.

    Each test computes the pairwise RMSD matrix with the GPU backend and
    the numba reference, then asserts element-wise agreement within
    ``atol=5e-5 nm`` (~0.0005 Angstrom).  The numba result is itself
    validated against mdtraj in ``TestComputeRmsdMatrix``.

    Tests are skipped when the corresponding package is not installed
    or when the ``gpu`` marker is deselected (``-m "not gpu"``).
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
                backbone_trajectory,
                atom_selection="all",
                backend=name,  # type: ignore[arg-type]
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

    def test_float32_matrix_matches_float64(self, backbone_trajectory: md.Trajectory) -> None:
        """Clustering must accept float32 matrices and give the same result as float64."""
        result_f32 = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", dtype=np.float32
        )
        result_f64 = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", dtype=np.float64
        )
        assert result_f32.rmsd_matrix_nm.dtype == np.float32
        assert result_f64.rmsd_matrix_nm.dtype == np.float64

        clust_f32 = cluster_conformations(result_f32.rmsd_matrix_nm, cutoff_nm=0.2)
        clust_f64 = cluster_conformations(result_f64.rmsd_matrix_nm, cutoff_nm=0.2)

        # Labels may not be identical if two centres tie, but the number
        # of clusters should match at a safe cutoff above the matrix noise.
        assert clust_f32.n_clusters == clust_f64.n_clusters
        # Every member of cluster k in f32 should share the same label in f64
        # (modulo the label id -- use a canonical remapping).
        mapping: dict[int, int] = {}
        for a, b in zip(clust_f32.labels, clust_f64.labels, strict=True):
            mapping.setdefault(int(a), int(b))
            assert mapping[int(a)] == int(b)

    def test_known_greedy_result_small(self) -> None:
        """Hand-checked GROMOS result on a 6x6 matrix.

        Frames 0-2 are within cutoff of each other; frames 3-4 form a
        second tight pair; frame 5 is an isolated singleton.  GROMOS
        picks the largest cluster first (3 members), then the pair, then
        the singleton.
        """
        rmsd = np.array(
            [
                [0.00, 0.05, 0.07, 0.80, 0.82, 1.50],
                [0.05, 0.00, 0.06, 0.81, 0.83, 1.51],
                [0.07, 0.06, 0.00, 0.79, 0.80, 1.49],
                [0.80, 0.81, 0.79, 0.00, 0.10, 1.00],
                [0.82, 0.83, 0.80, 0.10, 0.00, 0.99],
                [1.50, 1.51, 1.49, 1.00, 0.99, 0.00],
            ],
            dtype=np.float32,
        )
        result = cluster_conformations(rmsd, cutoff_nm=0.15)
        assert result.n_clusters == 3
        # First (largest) cluster should contain the 3 tight frames.
        first = {i for i in range(6) if result.labels[i] == 0}
        assert first == {0, 1, 2}
        # Second cluster: the tight pair.
        second = {i for i in range(6) if result.labels[i] == 1}
        assert second == {3, 4}
        # Third: the singleton.
        third = {i for i in range(6) if result.labels[i] == 2}
        assert third == {5}

    def test_single_frame(self) -> None:
        """A 1x1 matrix must produce a single cluster with one medoid."""
        result = cluster_conformations(np.zeros((1, 1), dtype=np.float32), cutoff_nm=0.1)
        assert result.n_clusters == 1
        assert result.labels[0] == 0
        assert result.medoid_frames.tolist() == [0]

    def test_all_frames_within_cutoff(self) -> None:
        """If every pair is within the cutoff, all frames collapse to one cluster."""
        n = 8
        rmsd = np.zeros((n, n), dtype=np.float32)
        result = cluster_conformations(rmsd, cutoff_nm=0.1)
        assert result.n_clusters == 1
        assert np.all(result.labels == 0)

    def test_all_frames_isolated(self) -> None:
        """If every pair exceeds the cutoff, each frame becomes its own cluster."""
        n = 5
        rmsd = np.full((n, n), 10.0, dtype=np.float32)
        np.fill_diagonal(rmsd, 0.0)
        result = cluster_conformations(rmsd, cutoff_nm=0.1)
        assert result.n_clusters == n
        assert len(np.unique(result.labels)) == n


# ---------------------------------------------------------------------------
# Wrapper dtype / memory tests: verify compute_rmsd_matrix does no redundant copy
# ---------------------------------------------------------------------------


class TestComputeRmsdMatrixNoRedundantCopy:
    """``compute_rmsd_matrix`` must not duplicate the backend's buffer.

    The fix for the 120k-frame OOM relies on two guarantees:

    1. GPU backends return their native ``float32`` (not a ``.astype(np.float64)``).
    2. The wrapper uses ``astype(resolved, copy=False)`` so when the
       backend's dtype already matches the user-resolved dtype, the
       returned buffer is the same object as the backend output.

    These tests drive a stub backend so they do not depend on GPU
    availability.
    """

    @pytest.fixture()
    def tiny_traj(self, backbone_trajectory: md.Trajectory) -> md.Trajectory:
        return backbone_trajectory

    def test_torch_backend_returns_float32(self, tiny_traj: md.Trajectory) -> None:
        """``rmsd_torch`` now returns float32 natively (no float64 upcast)."""
        pytest.importorskip("torch")
        result = compute_rmsd_matrix(tiny_traj, atom_selection="all", backend="torch")
        assert result.rmsd_matrix_nm.dtype == np.float32

    def test_mdtraj_backend_returns_float32(self, tiny_traj: md.Trajectory) -> None:
        """``rmsd_mdtraj`` now allocates float32 (not float64)."""
        result = compute_rmsd_matrix(tiny_traj, atom_selection="all", backend="mdtraj")
        assert result.rmsd_matrix_nm.dtype == np.float32

    def test_wrapper_does_not_copy_when_dtype_matches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_traj: md.Trajectory,
    ) -> None:
        """When backend dtype == resolved dtype, the wrapper's output shares memory.

        Monkeypatch the ``mdtraj`` backend with a stub that returns a
        tagged float32 buffer, then check ``np.shares_memory`` between
        the stub's buffer and the wrapper's output.  If the fix
        regresses (e.g. someone re-adds ``.astype(resolved)`` without
        ``copy=False``), this test catches it immediately.
        """
        from mdpp.analysis._backends._rmsd_matrix import rmsd_matrix_backends

        n = tiny_traj.n_frames
        sentinel = np.zeros((n, n), dtype=np.float32)
        sentinel[0, 0] = 3.1415  # marker so we can prove identity

        def stub_backend(
            traj: md.Trajectory,
            atom_indices: NDArray[np.int_],  # noqa: ARG001
        ) -> NDArray[np.floating]:
            assert traj is tiny_traj
            return sentinel

        monkeypatch.setitem(rmsd_matrix_backends._backends, "mdtraj", stub_backend)

        result = compute_rmsd_matrix(
            tiny_traj, atom_selection="all", backend="mdtraj", dtype=np.float32
        )
        assert result.rmsd_matrix_nm.dtype == np.float32
        assert np.shares_memory(result.rmsd_matrix_nm, sentinel), (
            "wrapper allocated a second buffer even though dtypes match"
        )
        assert result.rmsd_matrix_nm[0, 0] == np.float32(3.1415)

    def test_wrapper_casts_when_dtype_differs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_traj: md.Trajectory,
    ) -> None:
        """When the user asks for float64 and the backend gives float32, a cast happens.

        This is the one case where a copy is unavoidable -- confirm
        the result is numerically correct and has the requested dtype.
        """
        from mdpp.analysis._backends._rmsd_matrix import rmsd_matrix_backends

        n = tiny_traj.n_frames
        sentinel = np.full((n, n), 0.125, dtype=np.float32)

        def stub_backend(
            traj: md.Trajectory,  # noqa: ARG001
            atom_indices: NDArray[np.int_],  # noqa: ARG001
        ) -> NDArray[np.floating]:
            return sentinel

        monkeypatch.setitem(rmsd_matrix_backends._backends, "mdtraj", stub_backend)

        result = compute_rmsd_matrix(
            tiny_traj, atom_selection="all", backend="mdtraj", dtype=np.float64
        )
        assert result.rmsd_matrix_nm.dtype == np.float64
        assert not np.shares_memory(result.rmsd_matrix_nm, sentinel)
        np.testing.assert_allclose(result.rmsd_matrix_nm, 0.125)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


def _is_gpu_oom(exc: BaseException) -> bool:
    """Return True if *exc* looks like a GPU out-of-memory error."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "outofmemory" in name or "out of memory" in msg


def _run_rmsd_benchmark(traj: md.Trajectory) -> None:
    """Run all available RMSD matrix backends and print a comparison table.

    Each backend is warmed up (JIT/CUDA context/XLA compilation) with a
    3-frame slice before the timed run on the full trajectory.  The
    mdtraj result is used as the correctness reference (atol=5e-5 nm)
    since mdtraj is the default backend for every analysis function.

    GPU backends that run out of memory on shared/contended GPUs are
    skipped with a printed note instead of failing the test.  Each
    kernel releases its own framework cache via the ``@clean_*_cache``
    decorators, so no pre-run defragmentation is needed.
    """
    ref = compute_rmsd_matrix(traj, atom_selection="all", backend="mdtraj")
    # Symmetrise mdtraj result (md.rmsd loop is not numerically symmetric).
    ref_mat = (ref.rmsd_matrix_nm + ref.rmsd_matrix_nm.T) / 2.0

    backends: list[tuple[str, bool]] = [
        ("mdtraj", True),
        ("numba", True),
        ("cupy", has_cupy),
        ("torch", has_torch),
        ("jax", has_jax),
    ]

    timings: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name, available in backends:
        if not available:
            continue
        try:
            # Warmup
            compute_rmsd_matrix(traj[:3], atom_selection="all", backend=name)  # type: ignore[arg-type]
            t0 = time.perf_counter()
            result = compute_rmsd_matrix(traj, atom_selection="all", backend=name)  # type: ignore[arg-type]
            timings[name] = time.perf_counter() - t0
            if name != "mdtraj":
                np.testing.assert_allclose(result.rmsd_matrix_nm, ref_mat, atol=5e-5)
        except Exception as exc:
            if _is_gpu_oom(exc):
                skipped[name] = "GPU OOM"
                continue
            raise

    n = traj.n_frames
    n_pairs = n * (n - 1) // 2
    print(f"\n  RMSD matrix benchmark: {n} frames, {len(ref.atom_indices)} atoms ({n_pairs} pairs)")
    print(f"  {'Backend':<10s} {'Time (s)':>10s} {'vs mdtraj':>10s}")
    print(f"  {'-' * 32}")
    t_mdtraj = timings.get("mdtraj", 1.0)
    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        speedup = t_mdtraj / t
        print(f"  {name:<10s} {t:>10.4f} {speedup:>9.1f}x")
    for name, reason in skipped.items():
        print(f"  {name:<10s} {'--':>10s} (skipped: {reason})")


def _make_alanine_traj(n_frames: int, n_residues: int) -> md.Trajectory:
    """Synthetic alanine trajectory with N, CA, C backbone atoms per residue."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for res_idx in range(1, n_residues + 1):
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
    return md.Trajectory(xyz=xyz, topology=topology)


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_residues"),
    [
        pytest.param(100, 50, id="fast-100f-150a"),
        pytest.param(200, 50, id="fast-200f-150a"),
    ],
)
def test_benchmark_rmsd_backends_fast(n_frames: int, n_residues: int) -> None:
    """Fast benchmark -- all available RMSD matrix backends on small trajectories.

    Completes in seconds on a modern machine.  Verifies every backend
    matches the mdtraj reference (atol=5e-5 nm) as a side effect.

    Run only fast benchmarks:    ``pytest -m "benchmark and not slow"``
    Run all benchmarks:           ``pytest -m benchmark``
    Skip benchmarks:              ``pytest -m "not benchmark"``
    """
    traj = _make_alanine_traj(n_frames, n_residues)
    _run_rmsd_benchmark(traj)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_residues"),
    [
        pytest.param(500, 50, id="slow-500f-150a"),
        pytest.param(1000, 100, id="slow-1000f-300a"),
    ],
)
def test_benchmark_rmsd_backends_slow(n_frames: int, n_residues: int) -> None:
    """Slow benchmark -- RMSD matrix backends on larger trajectories.

    The matrix is O(n_frames^2) so the 1000-frame case computes 500k
    pairs and takes tens of seconds on the single-threaded mdtraj loop.
    Marked ``slow`` so it is deselected by ``-m "not slow"`` in fast CI.

    Run only slow benchmarks:  ``pytest -m "benchmark and slow"``
    """
    traj = _make_alanine_traj(n_frames, n_residues)
    _run_rmsd_benchmark(traj)


# ---------------------------------------------------------------------------
# GROMOS clustering benchmarks
# ---------------------------------------------------------------------------


def _make_synthetic_rmsd_matrix(
    n_frames: int,
    n_clusters: int,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic RMSD matrix with a known cluster structure.

    Places each cluster at a distinct point on a 1-D axis, spacing
    2.0 nm apart; every cluster gets at least ``n_frames //
    n_clusters`` members via a shuffled round-robin assignment.
    Intra-cluster RMSD is pure Gaussian noise at ~0.01 nm, so every
    member stays well within a 0.1 nm cutoff while inter-cluster
    RMSD is always >= 2.0 nm -- GROMOS recovers exactly
    ``n_clusters`` groups at any cutoff in (0.05, 2.0).
    """
    rng = np.random.RandomState(seed)
    # Round-robin assignment so every cluster has >= one member.
    assignments = np.arange(n_frames) % n_clusters
    rng.shuffle(assignments)
    # 1-D positions: cluster k is centred at 2.0 * k.
    centres = np.arange(n_clusters, dtype=np.float32) * 2.0
    positions = centres[assignments] + rng.randn(n_frames).astype(np.float32) * 0.01
    # |a - b| in 1-D is a valid metric and the resulting matrix is
    # symmetric with zero diagonal.
    rmsd = np.abs(positions[:, None] - positions[None, :])
    return rmsd.astype(np.float32)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_clusters"),
    [
        pytest.param(1000, 8, id="fast-1000f-8c"),
        pytest.param(2000, 20, id="fast-2000f-20c"),
    ],
)
def test_benchmark_cluster_conformations_fast(
    n_frames: int,
    n_clusters: int,
) -> None:
    """Fast clustering benchmark -- exercise the numba GROMOS kernel.

    Builds a synthetic RMSD matrix with a known cluster structure and
    measures wall time.  The kernel is warmed up with a small pre-pass
    so the reported time excludes numba JIT compilation.
    """
    rmsd = _make_synthetic_rmsd_matrix(n_frames, n_clusters, seed=1)

    warmup = _make_synthetic_rmsd_matrix(32, 4, seed=99)
    cluster_conformations(warmup, cutoff_nm=0.1)

    t0 = time.perf_counter()
    result = cluster_conformations(rmsd, cutoff_nm=0.1)
    elapsed = time.perf_counter() - t0

    print(
        f"\n  GROMOS clustering: n={n_frames} cutoff=0.1 "
        f"({result.n_clusters} clusters)  wall={elapsed:.4f} s"
    )
    # Sanity: we should recover the planted cluster count within a
    # small tolerance (the synthetic centres may occasionally merge or
    # split by one or two clusters at this cutoff).
    assert abs(result.n_clusters - n_clusters) <= max(2, n_clusters // 4)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_clusters"),
    [
        pytest.param(5000, 50, id="slow-5000f-50c"),
        pytest.param(10000, 100, id="slow-10000f-100c"),
    ],
)
def test_benchmark_cluster_conformations_slow(
    n_frames: int,
    n_clusters: int,
) -> None:
    """Slow clustering benchmark -- larger matrices that stress the loop.

    The numba GROMOS loop is O(n^2) for the initial neighbour count
    and then O(|members| * n) per cluster.  For n=10k with 100
    clusters on a modern CPU this should run in a few seconds;
    regressions (e.g. someone re-introducing the O(k*n^2) full
    recompute) would push this to minutes.
    """
    rmsd = _make_synthetic_rmsd_matrix(n_frames, n_clusters, seed=2)

    warmup = _make_synthetic_rmsd_matrix(32, 4, seed=99)
    cluster_conformations(warmup, cutoff_nm=0.1)

    t0 = time.perf_counter()
    result = cluster_conformations(rmsd, cutoff_nm=0.1)
    elapsed = time.perf_counter() - t0

    print(
        f"\n  GROMOS clustering: n={n_frames} cutoff=0.1 "
        f"({result.n_clusters} clusters)  wall={elapsed:.4f} s"
    )
    assert abs(result.n_clusters - n_clusters) <= max(5, n_clusters // 4)
