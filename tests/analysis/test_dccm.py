"""Tests for compute_dccm and DCCM backends.

Covers two layers:

1. Public API: ``compute_dccm`` returns the same correlation matrix
   regardless of which backend is selected, on the shared
   ``correlated_ca_trajectory`` fixture from ``tests/conftest.py``.
2. Kernel layer: each backend in ``dccm_backends`` returns the same
   covariance matrix (within float32 tolerance) for a moderately
   sized random trajectory, plus a fast benchmark tier that prints
   wall-time and speedup vs the numpy default.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import mdtraj as md
import numpy as np
import pytest
from numpy.typing import NDArray

from mdpp.analysis._backends import has_cupy, has_jax, has_torch
from mdpp.analysis._backends._dccm import (
    dccm_cupy,
    dccm_jax,
    dccm_numba,
    dccm_numpy,
    dccm_torch,
)
from mdpp.analysis.metrics import compute_dccm

requires_cupy = pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
requires_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not has_jax, reason="JAX not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ca_trajectory(n_frames: int, n_atoms: int, seed: int = 7) -> md.Trajectory:
    """Build a CA-only trajectory with random fluctuations around mean positions."""
    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(1, n_atoms + 1):
        residue = topology.add_residue("ALA", chain, resSeq=i)
        topology.add_atom("CA", md.element.carbon, residue)

    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(1, n_atoms, 3)).astype(np.float32)
    xyz = centres + rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32) * 0.05
    return md.Trajectory(xyz=xyz, topology=topology)


@pytest.fixture()
def random_ca_trajectory() -> md.Trajectory:
    """50-frame, 30-CA trajectory used for cross-backend agreement."""
    return _make_ca_trajectory(n_frames=50, n_atoms=30)


# ---------------------------------------------------------------------------
# Public API: backend selection produces equivalent results
# ---------------------------------------------------------------------------


def test_compute_dccm_default_backend_matches_legacy(correlated_ca_trajectory) -> None:
    """Default ``numpy`` backend must keep the published behaviour."""
    result = compute_dccm(correlated_ca_trajectory, atom_selection="name CA")
    assert result.correlation.shape == (3, 3)
    assert np.allclose(np.diag(result.correlation), 1.0)
    assert result.correlation[0, 1] > 0.99
    assert result.correlation[0, 2] < -0.99


def test_compute_dccm_numba_backend_agrees_with_numpy(correlated_ca_trajectory) -> None:
    """Numba kernel must reproduce the numpy default within float32 noise."""
    reference = compute_dccm(correlated_ca_trajectory, atom_selection="name CA")
    numba_result = compute_dccm(correlated_ca_trajectory, atom_selection="name CA", backend="numba")
    np.testing.assert_allclose(numba_result.correlation, reference.correlation, atol=1e-5)


def test_compute_dccm_unknown_backend_raises(correlated_ca_trajectory) -> None:
    """Unknown backend names must surface a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        compute_dccm(
            correlated_ca_trajectory,
            atom_selection="name CA",
            backend="not-a-backend",  # type: ignore[arg-type]
        )


def test_compute_dccm_too_few_frames_raises() -> None:
    """A single-frame trajectory cannot have a covariance."""
    traj = _make_ca_trajectory(n_frames=1, n_atoms=4)
    with pytest.raises(ValueError, match="at least two frames"):
        compute_dccm(traj, atom_selection="name CA")


# ---------------------------------------------------------------------------
# Kernel layer: backends agree on covariance
# ---------------------------------------------------------------------------


def _reference_covariance(positions: NDArray[np.floating]) -> NDArray[np.float64]:
    """Reference covariance computed in float64 with explicit einsum."""
    pos64 = np.asarray(positions, dtype=np.float64)
    fluct = pos64 - pos64.mean(axis=0, keepdims=True)
    return np.einsum("fid,fjd->ij", fluct, fluct) / pos64.shape[0]


def test_dccm_numpy_matches_einsum_reference(random_ca_trajectory) -> None:
    """Numpy GEMM rewrite must match the float64 einsum reference."""
    positions = random_ca_trajectory.xyz
    expected = _reference_covariance(positions)
    actual = dccm_numpy(positions)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_dccm_numba_matches_einsum_reference(random_ca_trajectory) -> None:
    """Numba kernel must match the float64 einsum reference."""
    positions = random_ca_trajectory.xyz
    expected = _reference_covariance(positions)
    actual = dccm_numba(positions)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@requires_torch
def test_dccm_torch_matches_einsum_reference(random_ca_trajectory) -> None:
    """Torch kernel must match the float64 einsum reference."""
    positions = random_ca_trajectory.xyz
    expected = _reference_covariance(positions)
    actual = dccm_torch(positions)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@requires_jax
def test_dccm_jax_matches_einsum_reference(random_ca_trajectory) -> None:
    """JAX kernel must match the float64 einsum reference."""
    positions = random_ca_trajectory.xyz
    expected = _reference_covariance(positions)
    actual = dccm_jax(positions)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@requires_cupy
def test_dccm_cupy_matches_einsum_reference(random_ca_trajectory) -> None:
    """CuPy kernel must match the float64 einsum reference."""
    positions = random_ca_trajectory.xyz
    expected = _reference_covariance(positions)
    actual = dccm_cupy(positions)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Benchmark -- parametrized multi-scale, all available backends
# ---------------------------------------------------------------------------


def _einsum_baseline(positions: NDArray[np.floating]) -> NDArray[np.floating]:
    """Single-threaded numpy einsum -- the original implementation we replaced."""
    fluct = positions - positions.mean(axis=0, keepdims=True)
    return np.einsum("fid,fjd->ij", fluct, fluct) / positions.shape[0]


_DCCM_KERNELS: dict[str, tuple[bool, Callable[[NDArray[np.floating]], NDArray[np.floating]]]] = {}


def _build_dccm_kernel_map() -> dict[
    str, tuple[bool, Callable[[NDArray[np.floating]], NDArray[np.floating]]]
]:
    """Build ``{name: (available, kernel_fn)}`` for every DCCM backend.

    The ``einsum`` baseline is the legacy single-threaded numpy path;
    keeping it in the mix makes the speedup over the historical
    implementation visible side-by-side with the other backends.
    """
    if _DCCM_KERNELS:
        return _DCCM_KERNELS
    _DCCM_KERNELS["einsum"] = (True, _einsum_baseline)
    _DCCM_KERNELS["numpy"] = (True, dccm_numpy)
    _DCCM_KERNELS["numba"] = (True, dccm_numba)
    _DCCM_KERNELS["cupy"] = (has_cupy, dccm_cupy)
    _DCCM_KERNELS["torch"] = (has_torch, dccm_torch)
    _DCCM_KERNELS["jax"] = (has_jax, dccm_jax)
    return _DCCM_KERNELS


def _is_gpu_oom(exc: BaseException) -> bool:
    """Return True if *exc* looks like a GPU out-of-memory error.

    Mirrors the helper in ``test_ca_distances.py`` so a starved
    shared GPU skips the affected backend rather than failing the
    whole benchmark.
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "outofmemory" in name or "out of memory" in msg


def _run_dccm_benchmark(n_frames: int, n_atoms: int) -> None:
    """Run every available DCCM backend on a synthetic trajectory and print results.

    Each kernel runs once as a warmup (JIT compile / CUDA context /
    XLA trace) before the timed run, and the result is checked
    against the einsum reference at ``atol=1e-4`` so the benchmark
    doubles as a numerical regression test.
    """
    rng = np.random.default_rng(42)
    centres = rng.normal(size=(1, n_atoms, 3)).astype(np.float32)
    positions = centres + rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32) * 0.05
    warmup = positions[: max(2, min(8, n_frames))]
    reference = _einsum_baseline(positions)

    timings: dict[str, float] = {}
    skipped: dict[str, str] = {}

    for name, (available, fn) in _build_dccm_kernel_map().items():
        if not available:
            continue
        try:
            fn(warmup)
            start = perf_counter()
            result = fn(positions)
            timings[name] = perf_counter() - start
            np.testing.assert_allclose(result, reference, atol=1e-4)
        except Exception as exc:
            if _is_gpu_oom(exc):
                skipped[name] = "GPU OOM"
                continue
            raise

    baseline = timings.get("einsum") or min(timings.values())
    print(f"\n  Benchmark: {n_frames} frames x {n_atoms} atoms")
    print(f"  {'Backend':<10s} {'Time (s)':>10s} {'vs einsum':>10s}")
    print(f"  {'-' * 32}")
    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        speedup = baseline / t
        print(f"  {name:<10s} {t:>10.4f} {speedup:>9.1f}x")
    for name, reason in skipped.items():
        print(f"  {name:<10s} {'--':>10s} (skipped: {reason})")


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_atoms"),
    [
        pytest.param(2000, 100, id="fast-2K-100"),
        pytest.param(5000, 200, id="fast-5K-200"),
        pytest.param(10000, 200, id="fast-10K-200"),
    ],
)
def test_benchmark_dccm_backends_fast(n_frames: int, n_atoms: int) -> None:
    """Fast cross-backend DCCM benchmark.

    Sizes chosen so the GPU intermediate ``(F, N, 3)`` plus the
    on-device fluctuations stay under ~500 MB and the legacy einsum
    path still completes in seconds.  Verifies every backend matches
    the einsum reference at ``atol=1e-4`` as a side effect.

    Run only fast benchmarks:    ``pytest -m "benchmark and not slow"``
    Run all benchmarks:           ``pytest -m benchmark``
    Skip benchmarks:              ``pytest -m "not benchmark"``
    """
    _run_dccm_benchmark(n_frames, n_atoms)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_atoms"),
    [
        pytest.param(20000, 300, id="slow-20K-300"),
        pytest.param(50000, 500, id="slow-50K-500"),
    ],
)
def test_benchmark_dccm_backends_slow(n_frames: int, n_atoms: int) -> None:
    """Slow cross-backend DCCM benchmark.

    50K frames x 500 atoms gives a 250k-element covariance plus
    a 30 GB-class einsum reduction on the legacy path -- single-core
    einsum takes tens of seconds while the BLAS / GPU backends stay
    in the sub-second regime.  Marked ``slow`` so it is deselected
    by ``-m "not slow"`` in fast CI.

    Run only slow benchmarks:  ``pytest -m "benchmark and slow"``
    """
    _run_dccm_benchmark(n_frames, n_atoms)
