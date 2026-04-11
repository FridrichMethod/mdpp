"""Tests for GPU memory cache cleanup decorators and kernel integration.

These tests verify that:

1. The ``clean_torch_cache`` / ``clean_cupy_cache`` / ``clean_jax_cache``
   decorators run their framework's cache-release API in a ``finally``
   block on both normal return and exceptions.
2. The decorators preserve the wrapped function's signature so mypy
   can still see the exact shape at the call site.
3. The GPU kernels that use the decorators
   (``rmsd_torch``/``rmsd_cupy`` and
   ``distances_torch``/``distances_cupy``) actually release their
   pooled memory back to the driver after ``compute_rmsd_matrix`` /
   ``compute_distances`` returns.

JAX does not expose a first-class API for querying its device memory
pool, so the JAX tests only verify that the decorator runs without
error and preserves the call signature.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from mdpp.analysis._backends import (
    clean_cupy_cache,
    clean_jax_cache,
    clean_torch_cache,
    has_cupy,
    has_jax,
    has_torch,
)
from mdpp.analysis.clustering import compute_rmsd_matrix
from mdpp.analysis.distance import compute_distances

requires_cupy = pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
requires_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not has_jax, reason="JAX not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_traj() -> md.Trajectory:
    """Return a small alanine trajectory large enough to allocate GPU memory."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms: list[md.core.topology.Atom] = []
    for res_idx in range(1, 21):  # 20 residues x 3 atoms = 60
        residue = topology.add_residue("ALA", chain, resSeq=res_idx)
        n = topology.add_atom("N", md.element.nitrogen, residue)
        ca = topology.add_atom("CA", md.element.carbon, residue)
        c = topology.add_atom("C", md.element.carbon, residue)
        atoms.extend([n, ca, c])

    rng = np.random.RandomState(7)
    xyz = rng.randn(50, len(atoms), 3).astype(np.float32) * 0.15
    return md.Trajectory(xyz=xyz, topology=topology)


# ---------------------------------------------------------------------------
# Decorator unit tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestCleanTorchCacheDecorator:
    """Tests for ``clean_torch_cache``."""

    @requires_torch
    def test_releases_cache_on_normal_return(self) -> None:
        """Wrapped function's torch allocations must be released by ``finally``."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        @clean_torch_cache
        def allocate() -> int:
            # Allocate ~40 MiB on GPU
            tensor = torch.zeros(10_000_000, dtype=torch.float32, device="cuda")
            return tensor.numel()

        n = allocate()
        assert n == 10_000_000
        # After the wrapper's ``finally`` runs, memory should be back to baseline.
        assert torch.cuda.memory_allocated() == baseline

    @requires_torch
    def test_releases_cache_on_exception(self) -> None:
        """Wrapped function that raises must still trigger cache release."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        @clean_torch_cache
        def allocate_then_raise() -> None:
            torch.zeros(10_000_000, dtype=torch.float32, device="cuda")
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            allocate_then_raise()
        assert torch.cuda.memory_allocated() == baseline

    @requires_torch
    def test_preserves_signature(self) -> None:
        """functools.wraps should preserve name and docstring."""

        @clean_torch_cache
        def sample(a: int, b: str = "x") -> str:
            """Sample docstring."""
            return f"{a}-{b}"

        assert sample.__name__ == "sample"
        assert sample.__doc__ == "Sample docstring."
        assert sample(1, b="y") == "1-y"

    def test_is_noop_without_torch(self) -> None:
        """On machines without torch, the decorator must pass results through."""

        @clean_torch_cache
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5


@pytest.mark.gpu
class TestCleanCupyCacheDecorator:
    """Tests for ``clean_cupy_cache``."""

    @requires_cupy
    def test_releases_pool_on_normal_return(self) -> None:
        """Wrapped cupy allocations must be returned to the pool by ``finally``."""
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        baseline = pool.used_bytes()

        @clean_cupy_cache
        def allocate() -> int:
            arr = cp.zeros(10_000_000, dtype=cp.float32)
            return int(arr.size)

        n = allocate()
        assert n == 10_000_000
        assert pool.used_bytes() == baseline

    @requires_cupy
    def test_releases_pool_on_exception(self) -> None:
        """Wrapped function that raises must still trigger pool release."""
        import cupy as cp

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        baseline = pool.used_bytes()

        @clean_cupy_cache
        def allocate_then_raise() -> None:
            cp.zeros(10_000_000, dtype=cp.float32)
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            allocate_then_raise()
        assert pool.used_bytes() == baseline

    @requires_cupy
    def test_preserves_signature(self) -> None:
        """functools.wraps should preserve name and docstring."""

        @clean_cupy_cache
        def sample(x: float) -> float:
            """Sample docstring."""
            return x * 2.0

        assert sample.__name__ == "sample"
        assert sample.__doc__ == "Sample docstring."
        assert sample(1.5) == 3.0

    def test_is_noop_without_cupy(self) -> None:
        """On machines without cupy, the decorator must pass results through."""

        @clean_cupy_cache
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5


@pytest.mark.gpu
class TestCleanJaxCacheDecorator:
    """Tests for ``clean_jax_cache``.

    JAX does not expose a device-memory-usage query that matches the
    ``torch.cuda.memory_allocated`` / ``cp.MemoryPool.used_bytes``
    pattern, so these tests only verify that the decorator runs
    without error and preserves the wrapped function.
    """

    @requires_jax
    def test_runs_on_normal_return(self) -> None:
        """Wrapped function must return its value and not raise."""

        @clean_jax_cache
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    @requires_jax
    def test_runs_on_exception(self) -> None:
        """Wrapped function that raises must still run ``finally``."""

        @clean_jax_cache
        def raise_err() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            raise_err()

    @requires_jax
    def test_preserves_signature(self) -> None:
        """functools.wraps should preserve name and docstring."""

        @clean_jax_cache
        def sample(x: int) -> int:
            """Sample docstring."""
            return x + 1

        assert sample.__name__ == "sample"
        assert sample.__doc__ == "Sample docstring."


# ---------------------------------------------------------------------------
# Kernel integration tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestKernelReleasesGpuCache:
    """Verify that GPU-backed kernels leave the cache at baseline on return.

    These tests go one level up from the decorator unit tests: instead
    of exercising the decorator in isolation, they call the public
    ``compute_rmsd_matrix`` / ``compute_distances`` wrappers with a
    GPU backend and verify that ``torch.cuda.memory_allocated()`` /
    ``cp.MemoryPool.used_bytes()`` return to the pre-call baseline.
    """

    @requires_torch
    def test_compute_rmsd_matrix_torch_no_leak(self, small_traj: md.Trajectory) -> None:
        """``compute_rmsd_matrix(backend="torch")`` must release torch cache."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        # Warm up so any one-time CUDA context allocation is already done.
        compute_rmsd_matrix(small_traj[:3], atom_selection="all", backend="torch")
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        result = compute_rmsd_matrix(small_traj, atom_selection="all", backend="torch")
        # Drop the host-side numpy result -- it does not affect GPU memory.
        del result

        assert torch.cuda.memory_allocated() == baseline

    @requires_cupy
    def test_compute_rmsd_matrix_cupy_no_leak(self, small_traj: md.Trajectory) -> None:
        """``compute_rmsd_matrix(backend="cupy")`` must release the cupy pool."""
        import cupy as cp

        # Warm up so any one-time CUDA context allocation is already done.
        compute_rmsd_matrix(small_traj[:3], atom_selection="all", backend="cupy")
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        baseline = pool.used_bytes()

        result = compute_rmsd_matrix(small_traj, atom_selection="all", backend="cupy")
        del result

        assert pool.used_bytes() == baseline

    @requires_torch
    def test_compute_distances_torch_no_leak(self, small_traj: md.Trajectory) -> None:
        """``compute_distances(backend="torch")`` must release torch cache."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        ca_indices = small_traj.topology.select("name CA")
        pairs = np.array(
            [(i, j) for i in ca_indices for j in ca_indices if i < j],
            dtype=np.int_,
        )

        compute_distances(small_traj[:3], atom_pairs=pairs, backend="torch", periodic=False)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        result = compute_distances(small_traj, atom_pairs=pairs, backend="torch", periodic=False)
        del result

        assert torch.cuda.memory_allocated() == baseline

    @requires_cupy
    def test_compute_distances_cupy_no_leak(self, small_traj: md.Trajectory) -> None:
        """``compute_distances(backend="cupy")`` must release the cupy pool."""
        import cupy as cp

        ca_indices = small_traj.topology.select("name CA")
        pairs = np.array(
            [(i, j) for i in ca_indices for j in ca_indices if i < j],
            dtype=np.int_,
        )

        compute_distances(small_traj[:3], atom_pairs=pairs, backend="cupy", periodic=False)
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        baseline = pool.used_bytes()

        result = compute_distances(small_traj, atom_pairs=pairs, backend="cupy", periodic=False)
        del result

        assert pool.used_bytes() == baseline
