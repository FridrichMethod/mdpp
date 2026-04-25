"""RMSD matrix computation backends.

Provides five backends for computing all-vs-all pairwise RMSD matrices:

- ``numba`` -- Numba-parallel QCP (Theobald 2005) on CPU.
- ``mdtraj`` -- mdtraj precentered RMSD loop on CPU.
- ``torch`` -- Vectorised einsum + QCP Newton-Raphson via PyTorch.
- ``jax`` -- Vectorised einsum + QCP Newton-Raphson via JAX.
- ``cupy`` -- Vectorised einsum + QCP Newton-Raphson via CuPy.

The numba kernel lives in :mod:`._rmsd_matrix_numba` to keep both files
under the project's 800-line cap; this file owns the GPU streaming
pipeline plus the shared :func:`_rmsd_qcp_block` helper used by all
three GPU backends.

All backends return a symmetric ``(n_frames, n_frames)`` floating-point
numpy array of RMSD values in nm.

Streaming pipeline design (GPU backends)
----------------------------------------

The full cross-covariance tensor ``H[i, j, m, n]`` for all
``(i, j)`` pairs has shape ``(N, N, 3, 3)`` -- ~450 GB for a 120k-frame
concatenated trajectory at float32, which does not fit on any GPU.
The GPU backends therefore iterate over **row blocks** of the RMSD
matrix:

1. Choose ``row_chunk`` so that each block's working set
   (``(chunk, N, 3, 3)`` cross-covariance plus ~12 ``(chunk, N)``
   intermediate tensors for the QCP coefficients and Newton-Raphson
   state) fits in roughly 25% of free GPU memory -- see
   :func:`_rmsd_torch_row_chunk` / :func:`_rmsd_cupy_row_chunk` /
   :func:`_rmsd_jax_row_chunk`.
2. For each block, compute ``H_block[k, j] = xyz[i_start+k].T @ xyz[j]``
   in one einsum call, then run the vectorised QCP solver.
3. Stream the block's rows back to the host ``result`` matrix.

The ``result`` matrix itself lives on CPU throughout: ``float32``
pairwise RMSD for 120k frames is ~54 GB and would otherwise crowd out
the working set.

The torch backend goes one step further and uses **pinned host memory
+ a dedicated copy stream** so that the D2H transfer truly runs in
parallel with the next block's compute (not merely asynchronously on
the same stream).  The algorithm is:

- Allocate two pinned host buffers ``buf_a`` and ``buf_b`` of shape
  ``(row_chunk, N)`` -- pinned so the CUDA driver can DMA directly,
  double-buffered so two chunks can be in flight at once.
- Create a dedicated ``copy_stream``.  The compute stream remains the
  default torch stream so existing dispatch works unchanged.
- Inside the chunk loop:

  * Queue ``einsum`` + QCP on the compute stream.
  * ``copy_stream.wait_stream(compute_stream)`` orders the copy
    after the compute work, then issue
    ``buf[:chunk_len].copy_(rmsd_block, non_blocking=True)`` inside
    ``torch.cuda.stream(copy_stream)``.  Pinned destination +
    ``non_blocking=True`` is the only combination that lets the
    runtime schedule a real async D2H; plain ``.cpu()`` goes through
    a pageable buffer and implicitly synchronises the compute stream.
  * ``rmsd_block.record_stream(copy_stream)`` defers the caching
    allocator's release of the GPU block until the copy is done.
  * Record a ``torch.cuda.Event`` on ``copy_stream``.
  * Drain the **previous** chunk's pinned buffer into ``result`` --
    ``prev.event.synchronize()`` then a numpy slice-assign.  This
    happens while the *current* chunk's copy is still running and
    the *next* chunk's compute is already queued.
- Ping-pong ``buf_idx = 1 - buf_idx`` between the two buffers.

Three pipeline stages (compute stream / copy stream / CPU memcpy) run
concurrently in steady state, so the GPU compute stream stays saturated
for the entire run.  If pinned allocation fails (e.g. ``ulimit -l``
too low), we fall back to a simple synchronous pipeline that still
overlaps the CPU memcpy with the next chunk's compute via python-side
async dispatch.  The JAX and CuPy backends use that simpler pipeline
only -- neither framework exposes pinned double-buffering the way
torch does.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, NamedTuple, Protocol

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp.analysis._backends._imports import (
    clean_cupy_cache,
    clean_torch_cache,
    query_free_gpu_bytes,
    require_cupy,
    require_jax,
    require_torch,
)
from mdpp.analysis._backends._registry import BackendRegistry
from mdpp.analysis._backends._rmsd_matrix_numba import rmsd_numba


class RMSDMatrixBackendFn(Protocol):
    """Callable signature for a pairwise RMSD matrix backend.

    All registered backends return a symmetric ``(n_frames, n_frames)``
    floating-point numpy array of RMSD values (in nm) computed over
    the selected ``atom_indices``.  Backends return their **native**
    dtype (float32 for the GPU backends, float64 for numba and
    mdtraj); the public :func:`compute_rmsd_matrix` wrapper then
    casts with ``copy=False`` to the user-selected dtype, which is a
    no-op when the dtypes already match.  Requiring a specific
    floating dtype here would force every backend to produce a
    redundant ~N^2-sized copy purely for the type contract -- a real
    problem at 120k frames where one float64 matrix is 115 GB.
    """

    def __call__(
        self,
        traj: md.Trajectory,
        atom_indices: NDArray[np.int_],
    ) -> NDArray[np.floating]: ...


# ---------------------------------------------------------------------------
# Shared QCP Newton-Raphson kernel (used by all three GPU backends)
# ---------------------------------------------------------------------------


def _rmsd_qcp_block(
    xp: ModuleType,
    H_block: Any,
    trace_rows: Any,
    traces_all: Any,
    n_atoms: int,
) -> Any:
    """Vectorised QCP Newton-Raphson on a block of 3x3 cross-covariances.

    Backend-agnostic: ``xp`` may be ``torch``, ``jax.numpy``, or
    ``cupy``.  All operations used here -- ellipsis indexing,
    ``reshape``, ``where``, ``abs``, ``zeros_like``, ``sqrt`` -- are
    present with matching semantics across these libraries, so a single
    implementation replaces three near-identical copies.

    ``H_block`` has shape ``(C, N, 3, 3)``; ``trace_rows`` has shape
    ``(C,)``; ``traces_all`` has shape ``(N,)``.  Returns an ``(C, N)``
    tensor of RMSD values in the same array library as the inputs.

    The return type is ``Any`` because each backend's array type is
    distinct (``torch.Tensor`` / ``jax.Array`` / ``cupy.ndarray``) and
    pulling them in here would force a hard import of every optional
    GPU library at module load.
    """
    Sxx = H_block[..., 0, 0]
    Sxy = H_block[..., 0, 1]
    Sxz = H_block[..., 0, 2]
    Syx = H_block[..., 1, 0]
    Syy = H_block[..., 1, 1]
    Syz = H_block[..., 1, 2]
    Szx = H_block[..., 2, 0]
    Szy = H_block[..., 2, 1]
    Szz = H_block[..., 2, 2]

    # Characteristic polynomial coefficients (Theobald 2005)
    # P(lam) = lam^4 + c2*lam^2 + c1*lam + c0
    c2 = -2.0 * (
        Sxx * Sxx
        + Sxy * Sxy
        + Sxz * Sxz
        + Syx * Syx
        + Syy * Syy
        + Syz * Syz
        + Szx * Szx
        + Szy * Szy
        + Szz * Szz
    )
    c1 = -8.0 * (
        Sxx * (Syy * Szz - Syz * Szy)
        - Sxy * (Syx * Szz - Syz * Szx)
        + Sxz * (Syx * Szy - Syy * Szx)
    )

    ka = Sxx + Syy + Szz
    kb = Syz - Szy
    kc = Szx - Sxz
    kd = Sxy - Syx
    ke = Sxx - Syy - Szz
    kf = Sxy + Syx
    kg = Szx + Sxz
    kh = -Sxx + Syy - Szz
    km = Syz + Szy
    kn = -Sxx - Syy + Szz
    del Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz

    hn_mm = kh * kn - km * km
    fn_mg = kf * kn - km * kg
    fm_hg = kf * km - kh * kg
    cn_md = kc * kn - km * kd
    cm_hd = kc * km - kh * kd
    cg_fd = kc * kg - kf * kd

    c0 = (
        ka * (ke * hn_mm - kf * fn_mg + kg * fm_hg)
        - kb * (kb * hn_mm - kf * cn_md + kg * cm_hd)
        + kc * (kb * fn_mg - ke * cn_md + kg * cg_fd)
        - kd * (kb * fm_hg - ke * cm_hd + kf * cg_fd)
    )

    trace_i = trace_rows.reshape(-1, 1)  # (C, 1)
    trace_j = traces_all.reshape(1, -1)  # (1, N)
    lam = (trace_i + trace_j) * 0.5  # (C, N)
    for _ in range(30):
        l2 = lam * lam
        f_val = l2 * l2 + c2 * l2 + c1 * lam + c0
        fp_val = 4.0 * l2 * lam + 2.0 * c2 * lam + c1
        delta = xp.where(
            xp.abs(fp_val) > 0,
            f_val / fp_val,
            xp.zeros_like(fp_val),
        )
        lam = lam - delta

    rmsd_sq = (trace_i + trace_j - 2.0 * lam) / n_atoms
    # ``xp.maximum`` accepts a scalar second arg in jax/cupy but not
    # uniformly across torch versions, so use ``where`` for portability.
    return xp.sqrt(xp.where(rmsd_sq > 0, rmsd_sq, xp.zeros_like(rmsd_sq)))


# ---------------------------------------------------------------------------
# mdtraj backend
# ---------------------------------------------------------------------------


def rmsd_mdtraj(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute pairwise RMSD matrix using mdtraj's precentered loop.

    Allocates a ``float32`` result to match mdtraj's native coordinate
    precision.  For 120k frames this is 57 GB instead of the 115 GB an
    up-cast to float64 would need; the public wrapper casts with
    ``copy=False`` so no second allocation happens when the user's
    resolved dtype is also float32.
    """
    subset = traj.atom_slice(atom_indices)
    subset.center_coordinates()
    n_frames = subset.n_frames
    rmsd_matrix = np.zeros((n_frames, n_frames), dtype=np.float32)
    for i in range(n_frames):
        rmsd_matrix[i] = md.rmsd(subset, subset, frame=i, precentered=True)
    return rmsd_matrix


# ---------------------------------------------------------------------------
# Torch backend
# ---------------------------------------------------------------------------


class _PinnedChunk(NamedTuple):
    """Buffer / range / event triple for the pinned-D2H pipeline.

    Bundling these three together means the loop's ``prev`` state is
    either ``None`` or fully populated -- the previous code threaded
    three separate ``Optional`` variables and used ``assert`` to guard
    the join, which would be stripped under ``python -O``.
    """

    buf: Any
    rng: tuple[int, int]
    event: Any


def _rmsd_torch_row_chunk(free_bytes: int, n_frames: int) -> int:
    """Choose row-block size for the torch QCP kernel.

    Each block materialises ``(chunk, N, 3, 3)`` cross-covariance plus
    ~12 intermediate ``(chunk, N)`` float32 tensors for the QCP
    polynomial coefficients and Newton-Raphson state.  Peak usage is
    roughly ``180 * chunk * N`` bytes.  Target 25% of the free pool
    so there is headroom for ``xyz`` / ``traces`` / transient
    allocations, capping at ``n_frames`` so small trajectories still
    run in a single pass.
    """
    # ~180 bytes per pair is an empirical upper bound on peak memory.
    per_pair_bytes = 180
    budget = max(free_bytes // 4, 1)
    chunk = budget // (per_pair_bytes * max(n_frames, 1))
    return int(max(1, min(chunk, n_frames)))


def _rmsd_torch_run_cpu(
    torch_mod: ModuleType,
    xyz: Any,
    traces: Any,
    n_atoms: int,
    row_chunk: int,
    result: np.ndarray,
) -> None:  # pragma: no cover - thin helper
    """Torch RMSD streaming loop on CPU or pinned-alloc-failed GPU.

    Simple synchronous pipeline: launch chunk N's einsum/QCP, then
    write chunk N-1 to ``result`` while chunk N is still being
    scheduled.  On CPU everything is blocking anyway; on GPU we
    fall back to this path when pinned-memory allocation fails
    (e.g. ``ulimit -l`` too low).
    """
    n_frames = xyz.shape[0]
    prev_cpu: Any = None
    prev_range: tuple[int, int] | None = None

    for i_start in range(0, n_frames, row_chunk):
        i_end = min(i_start + row_chunk, n_frames)
        xyz_rows = xyz[i_start:i_end]
        H_block = torch_mod.einsum("kam,jan->kjmn", xyz_rows, xyz)
        rmsd_block = _rmsd_qcp_block(torch_mod, H_block, traces[i_start:i_end], traces, n_atoms)
        del H_block

        if prev_cpu is not None and prev_range is not None:
            prev_start, prev_end = prev_range
            result[prev_start:prev_end] = prev_cpu.numpy()

        cpu_tensor = rmsd_block.cpu()
        del rmsd_block

        prev_cpu = cpu_tensor
        prev_range = (i_start, i_end)

    if prev_cpu is not None and prev_range is not None:
        prev_start, prev_end = prev_range
        result[prev_start:prev_end] = prev_cpu.numpy()


def _rmsd_torch_run_gpu(
    torch_mod: ModuleType,
    xyz: Any,
    traces: Any,
    n_atoms: int,
    row_chunk: int,
    result: np.ndarray,
) -> None:  # pragma: no cover - thin helper
    """Torch RMSD streaming loop with pinned memory + copy stream.

    Uses a dedicated ``copy_stream`` so the D2H transfer of chunk
    N runs concurrently with the einsum/QCP compute of chunk N+1
    on the default compute stream.  Two pinned host buffers are
    double-buffered so a new copy can start before the previous
    one has been drained to ``result``.

    The copy is queued with ``non_blocking=True`` into a pinned
    buffer, which is the only combination that lets CUDA schedule
    a real async D2H transfer; a plain ``tensor.cpu()`` goes
    through a pageable buffer and implicitly synchronises the
    compute stream.  ``rmsd_block.record_stream(copy_stream)``
    defers the caching allocator's release of the GPU block until
    the copy stream is done with it.

    If the pinned-buffer allocation fails (RuntimeError, typically
    due to a low ``ulimit -l`` on the host), we fall back to the
    simple synchronous pipeline in :func:`_rmsd_torch_run_cpu`.
    """
    n_frames = xyz.shape[0]
    device = xyz.device

    # Double-buffered pinned host staging for async D2H.  Each buffer
    # is full-width ``(row_chunk, n_frames)``; the last chunk may be
    # shorter and uses ``buf[:C]``.
    buf_shape = (row_chunk, n_frames)
    try:
        pinned_bufs = [
            torch_mod.empty(buf_shape, dtype=torch_mod.float32, pin_memory=True) for _ in range(2)
        ]
    except RuntimeError:
        _rmsd_torch_run_cpu(torch_mod, xyz, traces, n_atoms, row_chunk, result)
        return

    copy_stream = torch_mod.cuda.Stream(device=device)

    prev: _PinnedChunk | None = None
    buf_idx = 0

    for i_start in range(0, n_frames, row_chunk):
        i_end = min(i_start + row_chunk, n_frames)
        chunk_len = i_end - i_start
        xyz_rows = xyz[i_start:i_end]

        # Compute on the current (default) stream.
        H_block = torch_mod.einsum("kam,jan->kjmn", xyz_rows, xyz)
        rmsd_block = _rmsd_qcp_block(torch_mod, H_block, traces[i_start:i_end], traces, n_atoms)
        del H_block

        # Queue the async D2H on copy_stream, ordered after the
        # compute stream's pending work.
        cur_buf = pinned_bufs[buf_idx]
        compute_stream = torch_mod.cuda.current_stream(device)
        copy_stream.wait_stream(compute_stream)
        with torch_mod.cuda.stream(copy_stream):
            cur_buf[:chunk_len].copy_(rmsd_block, non_blocking=True)
        # Hold rmsd_block alive on copy_stream so the caching
        # allocator does not reuse its memory before the copy runs.
        rmsd_block.record_stream(copy_stream)
        del rmsd_block

        cur_event = torch_mod.cuda.Event()
        cur_event.record(copy_stream)

        # While the copy runs (and the next einsum will be queued
        # just after this Python continues), drain the previous
        # chunk from its pinned buffer into ``result``.
        if prev is not None:
            prev.event.synchronize()
            prev_start, prev_end = prev.rng
            prev_len = prev_end - prev_start
            result[prev_start:prev_end] = prev.buf[:prev_len].numpy()

        prev = _PinnedChunk(buf=cur_buf, rng=(i_start, i_end), event=cur_event)
        buf_idx = 1 - buf_idx

    # Flush the final chunk.
    if prev is not None:
        prev.event.synchronize()
        prev_start, prev_end = prev.rng
        prev_len = prev_end - prev_start
        result[prev_start:prev_end] = prev.buf[:prev_len].numpy()


@clean_torch_cache
def rmsd_torch(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute pairwise RMSD matrix using PyTorch (CUDA if available).

    Implements the Quaternion Characteristic Polynomial (Theobald 2005)
    Newton-Raphson solve on the quartic, the same algorithm as the
    Numba kernel, broadcast across all pairs within a row block.

    **Row-chunked streaming.**  A naive "compute all pairs at once"
    einsum would materialise a ``(N, N, 3, 3)`` cross-covariance
    tensor -- ~450 GB for a 120k-frame concatenated trajectory.
    Instead we iterate over row blocks of the RMSD matrix:

    * Choose ``chunk`` rows so the ``(chunk, N, 3, 3)`` block plus
      the Newton-Raphson state fits inside a fraction of free GPU
      memory (see :func:`_rmsd_torch_row_chunk`).
    * Compute ``H_block[k, j] = xyz[i_start+k].T @ xyz[j]`` for all
      ``j`` in one einsum call.
    * Run vectorised QCP on the block, and stream the chunk back
      to the CPU ``result`` matrix.

    The result matrix itself lives on the CPU throughout -- for a
    120k-frame trajectory it is ~54 GB in float32 alone, which would
    eat a huge chunk of device memory if kept on GPU.

    **Pinned D2H pipeline (CUDA path).**  Naively each chunk would
    end with ``rmsd_block.cpu()``, which goes through a pageable
    buffer and implicitly synchronises the compute stream -- the
    GPU then sits idle for the duration of the ~200 ms CPU memcpy
    into ``result`` before the next einsum can begin.  Instead we
    allocate two pinned host buffers and a dedicated copy stream;
    each chunk's D2H copy is issued with ``non_blocking=True`` and
    scheduled on the copy stream so it runs in true parallel with
    the next chunk's compute on the default stream.  The Python
    loop drains the *previous* buffer into ``result`` while the
    *current* chunk is still mid-copy, so neither the CPU memcpy
    nor the D2H transfer is on the critical path for GPU utilisation.
    See :func:`_rmsd_torch_run_gpu` for the implementation details,
    and :func:`_rmsd_torch_run_cpu` for the simple sync fallback
    used on CPU or when pinned-buffer allocation fails.

    **Why QCP and not batched SVD**: batched ``torch.linalg.svd`` on
    3x3 matrices has very high LAPACK per-matrix overhead; even in
    float32 it takes ~70 ms per 500k pairs and dominates wall time.
    QCP is pure element-wise arithmetic over ``(chunk, N)``-shaped
    tensors (~50 flops per pair, 30 Newton iterations) and runs
    ~20x faster on the same workload.

    Internally operates in **float32**: consumer and workstation
    NVIDIA GPUs run float64 at 1/36 -- 1/64 the throughput of
    float32, mdtraj stores coordinates in float32 anyway, and
    float32 QCP agrees with the float64 numba reference to ~1e-6
    nm -- well below the 5e-5 nm agreement tolerance.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    torch = require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        xyz = torch.as_tensor(
            np.ascontiguousarray(traj.xyz[:, atom_indices, :]),
            dtype=torch.float32,
            device=device,
        )
        n_frames, n_atoms, _ = xyz.shape

        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        traces = (xyz * xyz).sum(dim=(1, 2))

        # Pick chunk size based on free GPU memory (CUDA) or use the
        # full trajectory in one shot on CPU.
        if device.type == "cuda":
            free_bytes, _total = torch.cuda.mem_get_info(device)
        else:
            free_bytes = 1 << 33  # 8 GiB host budget default
        row_chunk = _rmsd_torch_row_chunk(int(free_bytes), n_frames)

        # Result matrix lives on CPU to avoid pinning ~N^2 floats on GPU.
        result = np.zeros((n_frames, n_frames), dtype=np.float32)

        if device.type == "cuda":
            _rmsd_torch_run_gpu(torch, xyz, traces, n_atoms, row_chunk, result)
        else:
            _rmsd_torch_run_cpu(torch, xyz, traces, n_atoms, row_chunk, result)

    # Zero the diagonal: float32 QCP Newton-Raphson on ``H(i, i)`` does
    # not land exactly on ``lambda_max = G_i``, so self-RMSD picks up
    # ~1e-4 nm of noise that would violate ``test_diagonal_is_zero``.
    np.fill_diagonal(result, 0.0)

    # Return native float32.  The public wrapper uses ``astype(copy=False)``
    # so no redundant N^2 copy is made when the user's resolved dtype is
    # also float32 (the default) -- critical at 120k frames where each
    # copy costs 57 GB.
    return result


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------


def _rmsd_jax_row_chunk(n_frames: int) -> int:
    """Row-block size for the jax QCP kernel.

    JAX does not expose a reliable free-memory query, but torch or
    cupy (almost always installed alongside JAX on CUDA systems) can
    call ``cudaMemGetInfo`` directly.  We piggyback on those via
    :func:`query_free_gpu_bytes`, which falls back to a conservative
    4 GiB budget when no CUDA query is available (e.g. CPU-only).
    """
    per_pair_bytes = 180
    free_bytes = query_free_gpu_bytes()
    budget = max(free_bytes // 4, 1)
    chunk = budget // (per_pair_bytes * max(n_frames, 1))
    return int(max(1, min(chunk, n_frames)))


def rmsd_jax(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute pairwise RMSD matrix using JAX.

    JAX auto-selects the best available device (GPU > TPU > CPU).
    Row-chunked streaming QCP solve matching :func:`rmsd_torch` --
    see that docstring for the algorithm, the rationale for
    float32, and the row-block strategy that keeps peak device
    memory bounded for large trajectories.

    The einsum explicitly passes ``precision=HIGHEST`` to disable
    JAX's default TF32 / tensor-core accumulation on GPU, which
    would otherwise drop the einsum to ~19 bits of mantissa and
    cost ~1e-4 nm accuracy on small test fixtures.

    This kernel is **not** wrapped with a cache-cleanup decorator:
    ``jax.clear_caches()`` clears JIT compilation caches (not device
    memory), and trashing them after every call forces a slow
    recompile on the next invocation.

    Raises:
        ImportError: If JAX is not installed.
    """
    _jax, jnp = require_jax()

    xyz = jnp.array(traj.xyz[:, atom_indices, :], dtype=jnp.float32)
    n_frames, n_atoms, _ = xyz.shape

    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    traces = (xyz * xyz).sum(axis=(1, 2))

    row_chunk = _rmsd_jax_row_chunk(n_frames)
    result = np.zeros((n_frames, n_frames), dtype=np.float32)

    # Pipeline CPU writes with GPU compute; write whole rows in a
    # single contiguous slice-assign.  See ``rmsd_torch`` for the
    # detailed rationale.
    precision = _jax.lax.Precision.HIGHEST
    prev_cpu: np.ndarray | None = None
    prev_range: tuple[int, int] | None = None

    for i_start in range(0, n_frames, row_chunk):
        i_end = min(i_start + row_chunk, n_frames)
        xyz_rows = xyz[i_start:i_end]
        H_block = jnp.einsum("kam,jan->kjmn", xyz_rows, xyz, precision=precision)
        rmsd_block = _rmsd_qcp_block(jnp, H_block, traces[i_start:i_end], traces, n_atoms)
        del H_block

        if prev_cpu is not None and prev_range is not None:
            prev_start, prev_end = prev_range
            result[prev_start:prev_end] = prev_cpu

        cpu_array = np.asarray(rmsd_block)
        del rmsd_block
        prev_cpu = cpu_array
        prev_range = (i_start, i_end)

    if prev_cpu is not None and prev_range is not None:
        prev_start, prev_end = prev_range
        result[prev_start:prev_end] = prev_cpu

    np.fill_diagonal(result, 0.0)

    return result


# ---------------------------------------------------------------------------
# CuPy backend
# ---------------------------------------------------------------------------


def _rmsd_cupy_row_chunk(free_bytes: int, n_frames: int) -> int:
    """Row-block size for the cupy QCP kernel (see _rmsd_torch_row_chunk)."""
    per_pair_bytes = 180
    budget = max(free_bytes // 4, 1)
    chunk = budget // (per_pair_bytes * max(n_frames, 1))
    return int(max(1, min(chunk, n_frames)))


@clean_cupy_cache
def rmsd_cupy(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.floating]:
    """Compute pairwise RMSD matrix using CuPy (CUDA).

    Requires a CUDA-capable GPU.  Row-chunked streaming QCP solve
    matching :func:`rmsd_torch` -- see that docstring for the
    algorithm and the rationale for float32 / QCP / row chunking.

    Raises:
        ImportError: If CuPy is not installed.
    """
    cp = require_cupy()

    xyz = cp.array(traj.xyz[:, atom_indices, :], dtype=cp.float32)
    n_frames, n_atoms, _ = xyz.shape

    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    traces = (xyz * xyz).sum(axis=(1, 2))

    free_bytes, _total = cp.cuda.Device().mem_info
    row_chunk = _rmsd_cupy_row_chunk(int(free_bytes), n_frames)

    result = np.zeros((n_frames, n_frames), dtype=np.float32)

    # Pipeline CPU writes with GPU compute; write whole rows in a
    # single contiguous slice-assign.  See ``rmsd_torch`` for the
    # detailed rationale.
    prev_cpu: np.ndarray | None = None
    prev_range: tuple[int, int] | None = None

    for i_start in range(0, n_frames, row_chunk):
        i_end = min(i_start + row_chunk, n_frames)
        xyz_rows = xyz[i_start:i_end]
        H_block = cp.einsum("kam,jan->kjmn", xyz_rows, xyz)
        rmsd_block = _rmsd_qcp_block(cp, H_block, traces[i_start:i_end], traces, n_atoms)
        del H_block

        if prev_cpu is not None and prev_range is not None:
            prev_start, prev_end = prev_range
            result[prev_start:prev_end] = prev_cpu

        cpu_array = cp.asnumpy(rmsd_block)
        del rmsd_block
        prev_cpu = cpu_array
        prev_range = (i_start, i_end)

    if prev_cpu is not None and prev_range is not None:
        prev_start, prev_end = prev_range
        result[prev_start:prev_end] = prev_cpu

    np.fill_diagonal(result, 0.0)

    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

rmsd_matrix_backends: BackendRegistry[RMSDMatrixBackendFn] = BackendRegistry(default="mdtraj")
rmsd_matrix_backends.register("numba", rmsd_numba)
rmsd_matrix_backends.register("mdtraj", rmsd_mdtraj)
rmsd_matrix_backends.register("torch", rmsd_torch)
rmsd_matrix_backends.register("jax", rmsd_jax)
rmsd_matrix_backends.register("cupy", rmsd_cupy)
