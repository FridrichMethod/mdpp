"""Tests for featurize_ca_distances and distance backends."""

from __future__ import annotations

from collections.abc import Callable

import mdtraj as md
import numpy as np
import pytest

from mdpp.analysis._backends import has_cupy, has_jax, has_torch
from mdpp.analysis._backends._distances import (
    distances_cupy,
    distances_jax,
    distances_numba,
    distances_torch,
)
from mdpp.analysis.decomposition import DistanceFeatures, featurize_ca_distances
from mdpp.analysis.distance import _pairwise_distances_mdtraj

requires_cupy = pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
requires_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not has_jax, reason="JAX not installed")

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
# Shared test data for kernel tests
# ---------------------------------------------------------------------------

_XYZ_1NM = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
_XYZ_3D = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]], dtype=np.float32)
_XYZ_SELF = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
_XYZ_MULTI = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    ],
    dtype=np.float32,
)
_PAIR_01 = np.array([[0, 1]], dtype=np.int_)
_PAIR_00 = np.array([[0, 0]], dtype=np.int_)


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
            featurize_ca_distances(ca_trajectory, backend="invalid")  # type: ignore[arg-type]

    def test_residue_ids_preserved(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory)
        traj = ca_trajectory.atom_slice(result.atom_indices)
        res_ids = [r.resSeq for r in traj.topology.residues]
        assert res_ids == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Backend equivalence (CPU baselines)
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
# Low-level kernel tests -- Numba
# ---------------------------------------------------------------------------


class TestNumbaKernel:
    """Direct tests on distances_numba."""

    def test_known_distance(self) -> None:
        """Two atoms 1 nm apart along x-axis."""
        result = distances_numba(_XYZ_1NM, _PAIR_01)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_self_distance_is_zero(self) -> None:
        result = distances_numba(_XYZ_SELF, _PAIR_00)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_3d_distance(self) -> None:
        result = distances_numba(_XYZ_3D, _PAIR_01)
        assert result[0, 0] == pytest.approx(np.sqrt(14.0), abs=1e-5)

    def test_multi_frame(self) -> None:
        result = distances_numba(_XYZ_MULTI, _PAIR_01)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(2.0, abs=1e-6)

    def test_output_dtype_float64(self) -> None:
        xyz = np.zeros((2, 2, 3), dtype=np.float32)
        result = distances_numba(xyz, _PAIR_01)
        assert result.dtype == np.float64

    def test_out_of_range_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[0, 5]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_numba(xyz, pairs)

    def test_negative_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[-1, 1]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_numba(xyz, pairs)

    def test_empty_pairs(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.empty((0, 2), dtype=np.int_)
        result = distances_numba(xyz, pairs)
        assert result.shape == (2, 0)


# ---------------------------------------------------------------------------
# Low-level kernel tests -- mdtraj
# ---------------------------------------------------------------------------


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
# Low-level kernel tests -- CuPy
# ---------------------------------------------------------------------------


@requires_cupy
class TestCupyKernel:
    """Direct tests on distances_cupy."""

    def test_known_distance(self) -> None:
        result = distances_cupy(_XYZ_1NM, _PAIR_01)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_self_distance_is_zero(self) -> None:
        result = distances_cupy(_XYZ_SELF, _PAIR_00)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_3d_distance(self) -> None:
        result = distances_cupy(_XYZ_3D, _PAIR_01)
        assert result[0, 0] == pytest.approx(np.sqrt(14.0), abs=1e-5)

    def test_multi_frame(self) -> None:
        result = distances_cupy(_XYZ_MULTI, _PAIR_01)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(2.0, abs=1e-6)

    def test_output_dtype_float64(self) -> None:
        xyz = np.zeros((2, 2, 3), dtype=np.float32)
        result = distances_cupy(xyz, _PAIR_01)
        assert result.dtype == np.float64

    def test_out_of_range_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[0, 5]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_cupy(xyz, pairs)

    def test_negative_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[-1, 1]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_cupy(xyz, pairs)

    def test_empty_pairs(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.empty((0, 2), dtype=np.int_)
        result = distances_cupy(xyz, pairs)
        assert result.shape == (2, 0)


# ---------------------------------------------------------------------------
# Low-level kernel tests -- PyTorch
# ---------------------------------------------------------------------------


@requires_torch
class TestTorchKernel:
    """Direct tests on distances_torch."""

    def test_known_distance(self) -> None:
        result = distances_torch(_XYZ_1NM, _PAIR_01)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_self_distance_is_zero(self) -> None:
        result = distances_torch(_XYZ_SELF, _PAIR_00)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_3d_distance(self) -> None:
        result = distances_torch(_XYZ_3D, _PAIR_01)
        assert result[0, 0] == pytest.approx(np.sqrt(14.0), abs=1e-5)

    def test_multi_frame(self) -> None:
        result = distances_torch(_XYZ_MULTI, _PAIR_01)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(2.0, abs=1e-6)

    def test_output_dtype_float64(self) -> None:
        xyz = np.zeros((2, 2, 3), dtype=np.float32)
        result = distances_torch(xyz, _PAIR_01)
        assert result.dtype == np.float64

    def test_out_of_range_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[0, 5]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_torch(xyz, pairs)

    def test_negative_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[-1, 1]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_torch(xyz, pairs)

    def test_empty_pairs(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.empty((0, 2), dtype=np.int_)
        result = distances_torch(xyz, pairs)
        assert result.shape == (2, 0)


# ---------------------------------------------------------------------------
# Low-level kernel tests -- JAX
# ---------------------------------------------------------------------------


@requires_jax
class TestJaxKernel:
    """Direct tests on distances_jax."""

    def test_known_distance(self) -> None:
        result = distances_jax(_XYZ_1NM, _PAIR_01)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_self_distance_is_zero(self) -> None:
        result = distances_jax(_XYZ_SELF, _PAIR_00)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_3d_distance(self) -> None:
        result = distances_jax(_XYZ_3D, _PAIR_01)
        assert result[0, 0] == pytest.approx(np.sqrt(14.0), abs=1e-5)

    def test_multi_frame(self) -> None:
        result = distances_jax(_XYZ_MULTI, _PAIR_01)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert result[1, 0] == pytest.approx(2.0, abs=1e-6)

    def test_output_dtype_float64(self) -> None:
        xyz = np.zeros((2, 2, 3), dtype=np.float32)
        result = distances_jax(xyz, _PAIR_01)
        assert result.dtype == np.float64

    def test_out_of_range_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[0, 5]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_jax(xyz, pairs)

    def test_negative_pair_raises(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.array([[-1, 1]], dtype=np.int_)
        with pytest.raises(ValueError, match="atom_pairs must contain indices"):
            distances_jax(xyz, pairs)

    def test_empty_pairs(self) -> None:
        xyz = np.zeros((2, 3, 3), dtype=np.float32)
        pairs = np.empty((0, 2), dtype=np.int_)
        result = distances_jax(xyz, pairs)
        assert result.shape == (2, 0)


# ---------------------------------------------------------------------------
# Cross-backend numerical equivalence
# ---------------------------------------------------------------------------


class TestAllBackendsEquivalence:
    """All backends must produce numerically identical results."""

    def _reference(self, traj: md.Trajectory) -> np.ndarray:
        return featurize_ca_distances(traj, backend="numba").values

    @requires_cupy
    def test_cupy_matches_numba(self, ca_trajectory: md.Trajectory) -> None:
        ref = self._reference(ca_trajectory)
        result = featurize_ca_distances(ca_trajectory, backend="cupy")
        np.testing.assert_allclose(result.values, ref, atol=1e-5)

    @requires_torch
    def test_torch_matches_numba(self, ca_trajectory: md.Trajectory) -> None:
        ref = self._reference(ca_trajectory)
        result = featurize_ca_distances(ca_trajectory, backend="torch")
        np.testing.assert_allclose(result.values, ref, atol=1e-5)

    @requires_jax
    def test_jax_matches_numba(self, ca_trajectory: md.Trajectory) -> None:
        ref = self._reference(ca_trajectory)
        result = featurize_ca_distances(ca_trajectory, backend="jax")
        np.testing.assert_allclose(result.values, ref, atol=1e-5)

    @requires_cupy
    def test_cupy_matches_numba_large(self, benchmark_traj: md.Trajectory) -> None:
        ref = self._reference(benchmark_traj)
        result = featurize_ca_distances(benchmark_traj, backend="cupy")
        np.testing.assert_allclose(result.values, ref, atol=1e-5)

    @requires_torch
    def test_torch_matches_numba_large(self, benchmark_traj: md.Trajectory) -> None:
        ref = self._reference(benchmark_traj)
        result = featurize_ca_distances(benchmark_traj, backend="torch")
        np.testing.assert_allclose(result.values, ref, atol=1e-5)

    @requires_jax
    def test_jax_matches_numba_large(self, benchmark_traj: md.Trajectory) -> None:
        ref = self._reference(benchmark_traj)
        result = featurize_ca_distances(benchmark_traj, backend="jax")
        np.testing.assert_allclose(result.values, ref, atol=1e-5)

    @requires_cupy
    def test_cupy_dtype_float32(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, backend="cupy")
        assert result.values.dtype == np.float32

    @requires_torch
    def test_torch_dtype_float32(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, backend="torch")
        assert result.values.dtype == np.float32

    @requires_jax
    def test_jax_dtype_float32(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, backend="jax")
        assert result.values.dtype == np.float32

    @requires_cupy
    def test_cupy_dtype_float64(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, backend="cupy", dtype=np.float64)
        assert result.values.dtype == np.float64

    @requires_torch
    def test_torch_dtype_float64(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, backend="torch", dtype=np.float64)
        assert result.values.dtype == np.float64

    @requires_jax
    def test_jax_dtype_float64(self, ca_trajectory: md.Trajectory) -> None:
        result = featurize_ca_distances(ca_trajectory, backend="jax", dtype=np.float64)
        assert result.values.dtype == np.float64


# ---------------------------------------------------------------------------
# Benchmark -- parametrized multi-scale
# ---------------------------------------------------------------------------

# Available kernel functions keyed by backend name.
_KERNELS: dict[str, tuple[bool, Callable[..., np.ndarray]]] = {}


def _build_kernel_map() -> dict[str, tuple[bool, Callable[..., np.ndarray]]]:
    """Build {name: (available, kernel_fn)} mapping, evaluated once."""
    if _KERNELS:
        return _KERNELS

    def _run_numba(xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return distances_numba(xyz, pairs)

    def _run_cupy(xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return distances_cupy(xyz, pairs)

    def _run_torch(xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return distances_torch(xyz, pairs)

    def _run_jax(xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return distances_jax(xyz, pairs)

    _KERNELS["numba"] = (True, _run_numba)
    _KERNELS["cupy"] = (has_cupy, _run_cupy)
    _KERNELS["torch"] = (has_torch, _run_torch)
    _KERNELS["jax"] = (has_jax, _run_jax)
    return _KERNELS


def _run_benchmark(n_frames: int, n_atoms: int) -> None:
    """Run all available backends on a synthetic trajectory and print results."""
    import time

    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32) * 0.1
    pairs = np.array(
        [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)],
        dtype=np.int_,
    )
    n_pairs = pairs.shape[0]
    warmup_xyz = xyz[:2]
    warmup_pairs = pairs[:10]

    # -- mdtraj reference (needs a Trajectory object) --
    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(n_atoms):
        res = topology.add_residue("ALA", chain, resSeq=i + 1)
        topology.add_atom("CA", md.element.carbon, res)
    traj = md.Trajectory(xyz=xyz, topology=topology)

    t0 = time.perf_counter()
    r_mdtraj = _pairwise_distances_mdtraj(traj, pairs, periodic=False)
    t_mdtraj = time.perf_counter() - t0

    timings: dict[str, float] = {"mdtraj": t_mdtraj}
    kernels = _build_kernel_map()

    for name, (available, fn) in kernels.items():
        if not available:
            continue
        # Warm up (JIT / CUDA context / XLA compile)
        fn(warmup_xyz, warmup_pairs)
        t0 = time.perf_counter()
        r = fn(xyz, pairs)
        timings[name] = time.perf_counter() - t0
        np.testing.assert_allclose(r_mdtraj, r, atol=1e-4)

    print(f"\n  Benchmark: {n_frames} frames x {n_atoms} atoms ({n_pairs} pairs)")
    print(f"  {'Backend':<10s} {'Time (s)':>10s} {'vs mdtraj':>10s}")
    print(f"  {'-' * 32}")
    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        speedup = t_mdtraj / t
        print(f"  {name:<10s} {t:>10.4f} {speedup:>9.1f}x")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_atoms"),
    [
        pytest.param(1000, 100, id="small-1K-100"),
        pytest.param(3000, 200, id="medium-3K-200"),
        pytest.param(3000, 400, id="large-3K-400"),
    ],
)
def test_benchmark_backends(n_frames: int, n_atoms: int) -> None:
    """Benchmark all available backends at different trajectory scales.

    Run only benchmarks:  ``pytest -m benchmark``
    Skip benchmarks:      ``pytest -m "not benchmark"``
    """
    _run_benchmark(n_frames, n_atoms)
