"""Tests for the pinned-memory + CUDA-stream streaming path in rmsd_torch.

The torch RMSD backend streams each row block back to the host through
two pinned buffers and a dedicated copy stream so the D2H transfer of
chunk ``i`` runs in parallel with the compute of chunk ``i + 1``.  This
module exercises that pipeline across many chunk configurations that
are rarely or never hit by the normal row-chunk heuristic:

- Single-chunk runs (``row_chunk >= n_frames``).
- ``row_chunk == 1`` (every row becomes its own chunk, maximum
  double-buffer ping-pong stress).
- Uneven final chunk sizes (``n_frames`` not a multiple of
  ``row_chunk``).
- Pinned-buffer allocation failure -- the helper must silently fall
  back to the synchronous pipeline.
- CPU-only execution -- both ``rmsd_torch`` and its ``_run_cpu``
  helper must produce correct results when no CUDA device is present.

Also benchmarks wall-clock time across chunk sizes so regressions in
the double-buffer overlap are visible in benchmark runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import mdtraj as md
import numpy as np
import pytest

from mdpp.analysis._backends import _rmsd_matrix, has_torch
from mdpp.analysis.clustering import compute_rmsd_matrix

requires_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_backbone_traj(
    n_frames: int,
    n_residues: int,
    seed: int = 0,
) -> md.Trajectory:
    """Build a deterministic alanine backbone trajectory.

    Uses ``n_residues`` ALA residues each with N, CA, C backbone atoms
    so the topology is a thin stand-in for a real protein.  Coordinates
    are a fixed base conformer plus small random perturbations -- the
    resulting pairwise RMSDs sit in the 0.01-0.05 nm range, comfortably
    within the float32 QCP accuracy envelope.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    atoms: list = []
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

    rng = np.random.RandomState(seed)
    n_atoms = len(atoms)
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    return md.Trajectory(xyz=base + perturbation, topology=topology)


@pytest.fixture()
def medium_traj() -> md.Trajectory:
    """Return a 64-frame alanine trajectory (small enough for CI, large enough to chunk)."""
    return _make_backbone_traj(n_frames=64, n_residues=10)


def _force_row_chunk(
    monkeypatch: pytest.MonkeyPatch,
    row_chunk: int,
) -> None:
    """Pin ``_rmsd_torch_row_chunk`` to return a fixed value.

    Stressing the streaming pipeline at chunk sizes the heuristic would
    never pick (like 1 or 3) is the whole point of these tests, so we
    override the heuristic directly instead of trying to starve the GPU
    memory query.
    """
    monkeypatch.setattr(
        _rmsd_matrix,
        "_rmsd_torch_row_chunk",
        lambda free_bytes, n_frames: row_chunk,  # noqa: ARG005
    )


# ---------------------------------------------------------------------------
# Row chunk heuristic unit tests (CPU-only, no GPU required)
# ---------------------------------------------------------------------------


@requires_torch
class TestRowChunkHeuristic:
    """Unit tests for ``_rmsd_torch_row_chunk``.

    These tests are pure arithmetic and need no GPU: they verify that
    the chunk-size helper returns a value in ``[1, n_frames]`` across
    representative free-memory inputs so the streaming loop always makes
    forward progress.
    """

    def test_chunk_capped_by_n_frames(self) -> None:
        """When free memory is huge, chunk must still be <= n_frames."""
        chunk = _rmsd_matrix._rmsd_torch_row_chunk(1 << 40, n_frames=32)
        assert 1 <= chunk <= 32

    def test_chunk_minimum_is_one(self) -> None:
        """Even with a tiny budget, chunk must be at least 1 (to make progress)."""
        chunk = _rmsd_matrix._rmsd_torch_row_chunk(0, n_frames=1000)
        assert chunk == 1

    @pytest.mark.parametrize("free_bytes", [1 << 20, 1 << 25, 1 << 30, 1 << 35])
    def test_chunk_scales_with_free_memory(self, free_bytes: int) -> None:
        """Bigger free budgets must not decrease the chunk size."""
        chunk_small = _rmsd_matrix._rmsd_torch_row_chunk(free_bytes, n_frames=10_000)
        chunk_large = _rmsd_matrix._rmsd_torch_row_chunk(free_bytes * 16, n_frames=10_000)
        assert chunk_large >= chunk_small
        assert chunk_small >= 1


# ---------------------------------------------------------------------------
# CPU-path correctness
# ---------------------------------------------------------------------------


@requires_torch
class TestRmsdTorchCpuPath:
    """Exercise ``_rmsd_torch_run_cpu`` directly.

    Calling the helper with a CPU tensor lets us test the
    synchronous-fallback code path even on a machine with no CUDA
    device, so CI without GPU still gets coverage of the branch that
    handles pinned-allocation failure.
    """

    def test_helper_matches_numba_reference(self) -> None:
        """``_rmsd_torch_run_cpu`` must match numba within 5e-5 nm."""
        import torch

        traj = _make_backbone_traj(n_frames=20, n_residues=6)
        ref = compute_rmsd_matrix(traj, atom_selection="all", backend="numba")

        xyz = torch.as_tensor(np.ascontiguousarray(traj.xyz), dtype=torch.float32)
        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        traces = (xyz * xyz).sum(dim=(1, 2))

        result = np.zeros((traj.n_frames, traj.n_frames), dtype=np.float32)
        _rmsd_matrix._rmsd_torch_run_cpu(
            torch, xyz, traces, traj.n_atoms, row_chunk=5, result=result
        )
        np.fill_diagonal(result, 0.0)

        np.testing.assert_allclose(result.astype(np.float64), ref.rmsd_matrix_nm, atol=5e-5)

    @pytest.mark.parametrize("row_chunk", [1, 3, 20, 100])
    def test_helper_chunk_sizes(self, row_chunk: int) -> None:
        """The CPU helper must work for chunk sizes both below and above n_frames."""
        import torch

        traj = _make_backbone_traj(n_frames=20, n_residues=5)
        ref = compute_rmsd_matrix(traj, atom_selection="all", backend="numba")

        xyz = torch.as_tensor(np.ascontiguousarray(traj.xyz), dtype=torch.float32)
        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        traces = (xyz * xyz).sum(dim=(1, 2))

        result = np.zeros((traj.n_frames, traj.n_frames), dtype=np.float32)
        _rmsd_matrix._rmsd_torch_run_cpu(
            torch, xyz, traces, traj.n_atoms, row_chunk=row_chunk, result=result
        )
        np.fill_diagonal(result, 0.0)

        np.testing.assert_allclose(result.astype(np.float64), ref.rmsd_matrix_nm, atol=5e-5)


# ---------------------------------------------------------------------------
# GPU-path correctness under forced chunk sizes
# ---------------------------------------------------------------------------


def _require_cuda() -> None:
    """Skip the test if no CUDA device is available."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.mark.gpu
@requires_torch
class TestRmsdTorchStreamingChunks:
    """Force specific ``row_chunk`` values through the pinned streaming path.

    These tests monkeypatch the ``_rmsd_torch_row_chunk`` heuristic so
    every call to ``rmsd_torch`` uses the row-block size we pick.  The
    combinations cover:

    - **Single-chunk** runs where the loop body runs exactly once
      (no ping-pong, no overlap).
    - **Multi-chunk** runs with ``row_chunk=1`` or a small prime, so
      each iteration fully exercises the double-buffer swap and the
      ``copy_stream.wait_stream`` ordering.
    - **Uneven chunks** where ``n_frames`` is not a multiple of
      ``row_chunk``, ensuring the partial trailing chunk is written to
      the correct slice of ``result``.

    Each parameterisation is checked for:

    1. Agreement with the numba float64 reference within 5e-5 nm.
    2. Exact symmetry (the two halves of every pair are written by
       different chunks, so a bug in the pipeline ordering would show
       up here immediately).
    3. Zero diagonal (``np.fill_diagonal`` runs after the loop).
    """

    @pytest.mark.parametrize(
        "row_chunk",
        [1, 2, 3, 7, 13, 16, 64, 1000],
        ids=lambda v: f"chunk={v}",
    )
    def test_forced_chunk_matches_numba(
        self,
        monkeypatch: pytest.MonkeyPatch,
        medium_traj: md.Trajectory,
        row_chunk: int,
    ) -> None:
        """Forcing ``row_chunk`` to a fixed value must not change the answer."""
        _require_cuda()
        _force_row_chunk(monkeypatch, row_chunk)

        result = compute_rmsd_matrix(medium_traj, atom_selection="all", backend="torch")
        ref = compute_rmsd_matrix(medium_traj, atom_selection="all", backend="numba")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)
        np.testing.assert_allclose(np.diag(result.rmsd_matrix_nm), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.rmsd_matrix_nm, result.rmsd_matrix_nm.T, atol=1e-6)

    @pytest.mark.parametrize(
        ("n_frames", "row_chunk"),
        [
            pytest.param(17, 5, id="17f-chunk5"),
            pytest.param(23, 8, id="23f-chunk8"),
            pytest.param(31, 10, id="31f-chunk10"),
        ],
    )
    def test_partial_final_chunk(
        self,
        monkeypatch: pytest.MonkeyPatch,
        n_frames: int,
        row_chunk: int,
    ) -> None:
        """Uneven ``n_frames % row_chunk != 0`` must land in the correct rows."""
        _require_cuda()
        traj = _make_backbone_traj(n_frames=n_frames, n_residues=6)
        _force_row_chunk(monkeypatch, row_chunk)

        result = compute_rmsd_matrix(traj, atom_selection="all", backend="torch")
        ref = compute_rmsd_matrix(traj, atom_selection="all", backend="numba")
        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)
        # The final partial chunk covers the rows that would otherwise
        # be skipped -- assert the last row is populated (non-zero off
        # the diagonal) so a drop would fail loudly.
        last_row = result.rmsd_matrix_nm[-1]
        assert np.count_nonzero(last_row) >= n_frames - 1

    def test_single_frame_trajectory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A single-frame trajectory must still produce a valid 1x1 matrix."""
        _require_cuda()
        traj = _make_backbone_traj(n_frames=1, n_residues=4)
        _force_row_chunk(monkeypatch, 1)

        result = compute_rmsd_matrix(traj, atom_selection="all", backend="torch")
        assert result.rmsd_matrix_nm.shape == (1, 1)
        assert result.rmsd_matrix_nm[0, 0] == 0.0

    def test_pinned_alloc_failure_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        medium_traj: md.Trajectory,
    ) -> None:
        """If pinning host memory raises, ``_rmsd_torch_run_cpu`` must take over.

        Simulates a host with a too-low ``ulimit -l`` by making
        ``torch.empty(pin_memory=True)`` raise on every call.  The
        non-pinned codepath (``_rmsd_torch_run_cpu``) is spied on to
        confirm the fallback actually executes; the final result is
        still asserted against the numba reference.
        """
        _require_cuda()
        import torch

        original_empty = torch.empty

        def flaky_empty(*args: object, **kwargs: object):
            if kwargs.get("pin_memory", False):
                raise RuntimeError("simulated: cannot pin host memory")
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", flaky_empty)

        called: dict[str, int] = {"run_cpu": 0}
        original_run_cpu = _rmsd_matrix._rmsd_torch_run_cpu

        def spy_run_cpu(*args: object, **kwargs: object) -> None:
            called["run_cpu"] += 1
            return original_run_cpu(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_rmsd_matrix, "_rmsd_torch_run_cpu", spy_run_cpu)

        result = compute_rmsd_matrix(medium_traj, atom_selection="all", backend="torch")
        ref = compute_rmsd_matrix(medium_traj, atom_selection="all", backend="numba")

        assert called["run_cpu"] == 1, "pinned allocation failure should trigger fallback"
        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    def test_streaming_result_matches_non_streaming(
        self,
        monkeypatch: pytest.MonkeyPatch,
        medium_traj: md.Trajectory,
    ) -> None:
        """Multi-chunk and single-chunk runs must produce identical matrices.

        Sanity check that the streaming pipeline's double-buffered row
        writes line up with the non-streamed variant.  Tolerates
        float32 ordering noise at ~1e-6 nm.
        """
        _require_cuda()

        _force_row_chunk(monkeypatch, 1)
        streamed = compute_rmsd_matrix(
            medium_traj, atom_selection="all", backend="torch"
        ).rmsd_matrix_nm

        _force_row_chunk(monkeypatch, medium_traj.n_frames * 10)
        single = compute_rmsd_matrix(
            medium_traj, atom_selection="all", backend="torch"
        ).rmsd_matrix_nm

        np.testing.assert_allclose(streamed, single, atol=1e-5)


# ---------------------------------------------------------------------------
# Simulated-VRAM tests: chunk size reacts to different "GPUs"
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_torch
class TestRmsdTorchSimulatedVram:
    """Simulate GPUs with different free-memory sizes.

    The streaming path queries ``torch.cuda.mem_get_info`` to size its
    row block.  We can't change real free memory from a test, so we
    instead override ``_rmsd_torch_row_chunk`` with a closure that
    reports the chunk size *as if* free memory were a fixed value.
    This mirrors the branch taken on a small (8 GB), mid (24 GB), and
    large (96 GB) GPU.
    """

    @pytest.mark.parametrize(
        ("free_bytes", "label"),
        [
            pytest.param(8 * (1 << 30), "8 GiB GPU", id="8GiB"),
            pytest.param(24 * (1 << 30), "24 GiB GPU", id="24GiB"),
            pytest.param(96 * (1 << 30), "96 GiB GPU", id="96GiB"),
        ],
    )
    def test_different_gpu_sizes_agree(
        self,
        monkeypatch: pytest.MonkeyPatch,
        medium_traj: md.Trajectory,
        free_bytes: int,
        label: str,  # noqa: ARG002
    ) -> None:
        """Chunk size chosen for 8/24/96 GiB GPUs must all match the reference."""
        _require_cuda()

        original = _rmsd_matrix._rmsd_torch_row_chunk

        def fake_chunk(_free_bytes: int, n_frames: int) -> int:
            return original(free_bytes, n_frames)

        monkeypatch.setattr(_rmsd_matrix, "_rmsd_torch_row_chunk", fake_chunk)

        result = compute_rmsd_matrix(medium_traj, atom_selection="all", backend="torch")
        ref = compute_rmsd_matrix(medium_traj, atom_selection="all", backend="numba")
        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)


# ---------------------------------------------------------------------------
# Benchmark: streaming wall time across chunk sizes
# ---------------------------------------------------------------------------


def _torch_sync() -> None:
    """Block until all queued CUDA work has finished (no-op on CPU)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.mark.gpu
@pytest.mark.benchmark
@requires_torch
class TestRmsdTorchStreamingBenchmark:
    """Report wall time for the streamed torch RMSD kernel at different chunk sizes.

    The point of this benchmark is not to assert absolute speed (that
    depends on the GPU), but to make regressions visible in the test
    output.  Each parameterisation reports ``wall time`` for running
    ``rmsd_torch`` on a fixed 600-frame alanine trajectory with the
    chunk size overridden to a specific value.

    A well-pipelined implementation should have nearly flat wall time
    across chunk sizes because the double-buffer + copy-stream design
    overlaps the D2H transfer and the host memcpy with the next
    chunk's compute.  A pipeline regression (for example, ``.cpu()``
    being called synchronously) would show up as wall time scaling
    roughly linearly with ``n_frames / row_chunk``.
    """

    @pytest.mark.parametrize(
        "row_chunk_factor",
        [1.0, 0.5, 0.25, 0.05],
        ids=["chunk=full", "chunk=half", "chunk=quarter", "chunk=5%"],
    )
    def test_streaming_wall_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        row_chunk_factor: float,
    ) -> None:
        """Time ``rmsd_torch`` at the requested fraction of ``n_frames`` per chunk."""
        _require_cuda()

        traj = _make_backbone_traj(n_frames=600, n_residues=50)
        chunk = max(1, round(traj.n_frames * row_chunk_factor))

        _force_row_chunk(monkeypatch, chunk)

        # Warm up CUDA context and kernel compilation.
        compute_rmsd_matrix(traj[:5], atom_selection="all", backend="torch")
        _torch_sync()

        t0 = time.perf_counter()
        result = compute_rmsd_matrix(traj, atom_selection="all", backend="torch")
        _torch_sync()
        elapsed = time.perf_counter() - t0

        n_chunks = (traj.n_frames + chunk - 1) // chunk
        print(
            f"\n  torch streaming: n={traj.n_frames} chunk={chunk} "
            f"({n_chunks} passes)  wall={elapsed:.4f} s"
        )
        assert result.rmsd_matrix_nm.shape == (traj.n_frames, traj.n_frames)


# ---------------------------------------------------------------------------
# Utility: make sure we haven't accidentally locked the interface surface
# ---------------------------------------------------------------------------


@requires_torch
def test_run_cpu_and_run_gpu_helpers_exist() -> None:
    """Both streaming helpers must be importable and callable.

    This guard catches accidental renames that would break tests in
    this file but succeed in the rest of the suite (the public wrapper
    would still work because it's going through the registry).
    """
    assert callable(getattr(_rmsd_matrix, "_rmsd_torch_run_cpu", None))
    assert callable(getattr(_rmsd_matrix, "_rmsd_torch_run_gpu", None))
    # The row-chunk helper is the third piece of the streaming design.
    helper: Callable = _rmsd_matrix._rmsd_torch_row_chunk
    assert callable(helper)
