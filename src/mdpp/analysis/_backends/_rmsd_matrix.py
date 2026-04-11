"""RMSD matrix computation backends.

Provides five backends for computing all-vs-all pairwise RMSD matrices:

- ``numba`` -- Numba-parallel QCP (Theobald 2005) on CPU.
- ``mdtraj`` -- mdtraj precentered RMSD loop on CPU.
- ``torch`` -- Vectorised einsum + batched SVD via PyTorch.
- ``jax`` -- Vectorised einsum + batched SVD via JAX.
- ``cupy`` -- Vectorised einsum + batched SVD via CuPy.

All backends return a symmetric ``(n_frames, n_frames)`` float64 numpy
array of RMSD values in nm.
"""

from __future__ import annotations

from typing import Protocol

import mdtraj as md
import numpy as np
from numba import njit, prange
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


class RMSDMatrixBackendFn(Protocol):
    """Callable signature for a pairwise RMSD matrix backend.

    All registered backends return a symmetric ``(n_frames, n_frames)``
    float64 numpy array of RMSD values (in nm) computed over the
    selected ``atom_indices``.
    """

    def __call__(
        self,
        traj: md.Trajectory,
        atom_indices: NDArray[np.int_],
    ) -> NDArray[np.float64]: ...


# ---------------------------------------------------------------------------
# Numba-parallel pairwise RMSD (QCP / Theobald 2005)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _center_and_traces(
    xyz: NDArray[np.float64],
) -> NDArray[np.float64]:  # pragma: no cover - JIT
    """Center each frame in-place and return per-frame sum-of-squares."""
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]
    traces = np.empty(n_frames, dtype=np.float64)
    for f in range(n_frames):
        cx = cy = cz = 0.0
        for i in range(n_atoms):
            cx += xyz[f, i, 0]
            cy += xyz[f, i, 1]
            cz += xyz[f, i, 2]
        cx /= n_atoms
        cy /= n_atoms
        cz /= n_atoms
        t = 0.0
        for i in range(n_atoms):
            xyz[f, i, 0] -= cx
            xyz[f, i, 1] -= cy
            xyz[f, i, 2] -= cz
            t += xyz[f, i, 0] ** 2 + xyz[f, i, 1] ** 2 + xyz[f, i, 2] ** 2
        traces[f] = t
    return traces


@njit(parallel=True, cache=True)
def _pairwise_rmsd(
    xyz: NDArray[np.float64],
    traces: NDArray[np.float64],
    pair_i: NDArray[np.int64],
    pair_j: NDArray[np.int64],
) -> NDArray[np.float64]:  # pragma: no cover - JIT
    """Compute symmetric pairwise RMSD matrix with QCP superposition.

    Uses the Quaternion Characteristic Polynomial method (Theobald 2005)
    to find the optimal rotational RMSD for each pair.  The largest
    eigenvalue of the 4x4 key matrix is found via Newton-Raphson on
    the characteristic polynomial -- pure scalar arithmetic with no
    LAPACK calls.

    The kernel iterates over a **flat** list of upper-triangle pair
    indices (``pair_i[p]``, ``pair_j[p]``) rather than the original
    nested ``prange(n_frames)`` / ``range(i+1, n_frames)``.  A nested
    loop is statically load-imbalanced -- thread 0 gets ``i=0``
    (``n-1`` pairs) while the last thread gets ``i=n-1`` (0 pairs) --
    and caps CPU utilisation at 60-80%.  A single ``prange`` over the
    flat pair list gives every thread an equal slab of work, pushing
    utilisation close to 100%.
    """
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]
    n_pairs = pair_i.shape[0]
    result = np.zeros((n_frames, n_frames))
    for p in prange(n_pairs):
        i = pair_i[p]
        j = pair_j[p]
        # Cross-covariance matrix elements
        Sxx = Sxy = Sxz = 0.0
        Syx = Syy = Syz = 0.0
        Szx = Szy = Szz = 0.0
        for k in range(n_atoms):
            x1 = xyz[i, k, 0]
            y1 = xyz[i, k, 1]
            z1 = xyz[i, k, 2]
            x2 = xyz[j, k, 0]
            y2 = xyz[j, k, 1]
            z2 = xyz[j, k, 2]
            Sxx += x1 * x2
            Sxy += x1 * y2
            Sxz += x1 * z2
            Syx += y1 * x2
            Syy += y1 * y2
            Syz += y1 * z2
            Szx += z1 * x2
            Szy += z1 * y2
            Szz += z1 * z2

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

        # det(K) via cofactor expansion of the 4x4 key matrix
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

        # Newton-Raphson for the largest eigenvalue
        lam = (traces[i] + traces[j]) * 0.5
        for _ in range(50):
            l2 = lam * lam
            f_val = l2 * l2 + c2 * l2 + c1 * lam + c0
            fp_val = 4.0 * l2 * lam + 2.0 * c2 * lam + c1
            if fp_val == 0.0:
                break
            delta = f_val / fp_val
            lam -= delta
            if abs(delta) < 1e-11 * abs(lam):
                break

        rmsd_sq = (traces[i] + traces[j] - 2.0 * lam) / n_atoms
        val = np.sqrt(max(0.0, rmsd_sq))
        result[i, j] = val
        result[j, i] = val
    return result


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def rmsd_numba(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using the Numba QCP kernel.

    Pre-centers each frame, then dispatches all ``n*(n-1)/2`` upper
    triangle pair indices to a single ``prange`` so every thread gets
    an equal share of the work (the nested-loop form would leave
    high-index threads idle early).
    """
    xyz = np.ascontiguousarray(traj.xyz[:, atom_indices, :], dtype=np.float64)
    traces = _center_and_traces(xyz)
    n_frames = xyz.shape[0]
    pair_i, pair_j = np.triu_indices(n_frames, k=1)
    return _pairwise_rmsd(
        xyz,
        traces,
        np.ascontiguousarray(pair_i, dtype=np.int64),
        np.ascontiguousarray(pair_j, dtype=np.int64),
    )


def rmsd_mdtraj(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using mdtraj's precentered loop."""
    subset = traj.atom_slice(atom_indices)
    subset.center_coordinates()
    n_frames = subset.n_frames
    rmsd_matrix = np.zeros((n_frames, n_frames), dtype=np.float64)
    for i in range(n_frames):
        rmsd_matrix[i] = md.rmsd(subset, subset, frame=i, precentered=True)
    return rmsd_matrix


def _rmsd_torch_row_chunk(torch_mod, free_bytes: int, n_frames: int) -> int:
    """Choose row-block size for the torch QCP kernel.

    Each block materialises ``(chunk, N, 3, 3)`` cross-covariance plus
    ~12 intermediate ``(chunk, N)`` float32 tensors for the QCP
    polynomial coefficients and Newton-Raphson state.  Peak usage is
    roughly ``180 * chunk * N`` bytes.  Target 25% of the free pool
    so there is headroom for ``xyz`` / ``traces`` / transient
    allocations, capping at ``n_frames`` so small trajectories still
    run in a single pass.
    """
    del torch_mod  # linked via type sig; kept for future expansion
    # ~180 bytes per pair is an empirical upper bound on peak memory.
    per_pair_bytes = 180
    budget = max(free_bytes // 4, 1)
    chunk = budget // (per_pair_bytes * max(n_frames, 1))
    return int(max(1, min(chunk, n_frames)))


def _rmsd_qcp_torch(
    torch_mod, H_block, trace_rows, traces_all, n_atoms: int
):  # pragma: no cover - thin helper
    """Vectorised QCP Newton-Raphson on a block of 3x3 cross-covariances.

    ``H_block`` has shape ``(C, N, 3, 3)``.  ``trace_rows`` has shape
    ``(C,)`` (traces for the current row-block); ``traces_all`` has
    shape ``(N,)``.  Returns an ``(C, N)`` tensor of RMSD values.
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

    trace_i = trace_rows.unsqueeze(1)  # (C, 1)
    trace_j = traces_all.unsqueeze(0)  # (1, N)
    lam = (trace_i + trace_j) * 0.5  # (C, N)
    for _ in range(30):
        l2 = lam * lam
        f_val = l2 * l2 + c2 * l2 + c1 * lam + c0
        fp_val = 4.0 * l2 * lam + 2.0 * c2 * lam + c1
        delta = torch_mod.where(
            fp_val.abs() > 0,
            f_val / fp_val,
            torch_mod.zeros_like(fp_val),
        )
        lam = lam - delta

    rmsd_sq = (trace_i + trace_j - 2.0 * lam) / n_atoms
    return torch_mod.sqrt(torch_mod.clamp(rmsd_sq, min=0.0))


@clean_torch_cache
def rmsd_torch(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
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
    * Run vectorised QCP on the block, mask the lower triangle, and
      transfer the chunk back to the CPU result matrix.

    The result matrix itself lives on the CPU throughout -- for a
    120k-frame trajectory it is ~54 GB in float32 alone, which would
    eat a huge chunk of device memory if kept on GPU.

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
        row_chunk = _rmsd_torch_row_chunk(torch, int(free_bytes), n_frames)

        # Result matrix lives on CPU to avoid pinning ~N^2 floats on GPU.
        result = np.zeros((n_frames, n_frames), dtype=np.float32)

        for i_start in range(0, n_frames - 1, row_chunk):
            i_end = min(i_start + row_chunk, n_frames - 1)
            xyz_rows = xyz[i_start:i_end]  # (C, M, 3) view
            # H_block[k, j, m, n] = sum_a xyz_rows[k, a, m] * xyz[j, a, n]
            H_block = torch.einsum("kam,jan->kjmn", xyz_rows, xyz)
            rmsd_block = _rmsd_qcp_torch(torch, H_block, traces[i_start:i_end], traces, n_atoms)
            del H_block

            rmsd_block_cpu = rmsd_block.cpu().numpy()
            del rmsd_block
            for k in range(i_end - i_start):
                i = i_start + k
                row = rmsd_block_cpu[k, i + 1 :]
                result[i, i + 1 :] = row
                result[i + 1 :, i] = row

    return result.astype(np.float64)


def _rmsd_qcp_jax(
    jnp, H_block, trace_rows, traces_all, n_atoms: int
):  # pragma: no cover - thin helper
    """Vectorised QCP Newton-Raphson on a block of 3x3 cross-covariances (JAX)."""
    Sxx = H_block[..., 0, 0]
    Sxy = H_block[..., 0, 1]
    Sxz = H_block[..., 0, 2]
    Syx = H_block[..., 1, 0]
    Syy = H_block[..., 1, 1]
    Syz = H_block[..., 1, 2]
    Szx = H_block[..., 2, 0]
    Szy = H_block[..., 2, 1]
    Szz = H_block[..., 2, 2]

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

    trace_i = trace_rows.reshape(-1, 1)
    trace_j = traces_all.reshape(1, -1)
    lam = (trace_i + trace_j) * 0.5
    for _ in range(30):
        l2 = lam * lam
        f_val = l2 * l2 + c2 * l2 + c1 * lam + c0
        fp_val = 4.0 * l2 * lam + 2.0 * c2 * lam + c1
        delta = jnp.where(
            jnp.abs(fp_val) > 0,
            f_val / fp_val,
            jnp.zeros_like(fp_val),
        )
        lam = lam - delta

    rmsd_sq = (trace_i + trace_j - 2.0 * lam) / n_atoms
    return jnp.sqrt(jnp.maximum(0.0, rmsd_sq))


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
) -> NDArray[np.float64]:
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

    precision = _jax.lax.Precision.HIGHEST
    for i_start in range(0, n_frames - 1, row_chunk):
        i_end = min(i_start + row_chunk, n_frames - 1)
        xyz_rows = xyz[i_start:i_end]
        H_block = jnp.einsum("kam,jan->kjmn", xyz_rows, xyz, precision=precision)
        rmsd_block = _rmsd_qcp_jax(jnp, H_block, traces[i_start:i_end], traces, n_atoms)
        del H_block

        rmsd_block_cpu = np.asarray(rmsd_block)
        del rmsd_block
        for k in range(i_end - i_start):
            i = i_start + k
            row = rmsd_block_cpu[k, i + 1 :]
            result[i, i + 1 :] = row
            result[i + 1 :, i] = row

    return result.astype(np.float64)


def _rmsd_qcp_cupy(
    cp, H_block, trace_rows, traces_all, n_atoms: int
):  # pragma: no cover - thin helper
    """Vectorised QCP Newton-Raphson on a block of 3x3 cross-covariances.

    Same algorithm as :func:`_rmsd_qcp_torch` but expressed with CuPy
    ops.  ``H_block`` has shape ``(C, N, 3, 3)``; ``trace_rows`` has
    shape ``(C,)``; ``traces_all`` has shape ``(N,)``.
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
    lam = (trace_i + trace_j) * 0.5
    for _ in range(30):
        l2 = lam * lam
        f_val = l2 * l2 + c2 * l2 + c1 * lam + c0
        fp_val = 4.0 * l2 * lam + 2.0 * c2 * lam + c1
        delta = cp.where(
            cp.abs(fp_val) > 0,
            f_val / fp_val,
            cp.zeros_like(fp_val),
        )
        lam = lam - delta

    rmsd_sq = (trace_i + trace_j - 2.0 * lam) / n_atoms
    return cp.sqrt(cp.maximum(0.0, rmsd_sq))


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
) -> NDArray[np.float64]:
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

    for i_start in range(0, n_frames - 1, row_chunk):
        i_end = min(i_start + row_chunk, n_frames - 1)
        xyz_rows = xyz[i_start:i_end]
        H_block = cp.einsum("kam,jan->kjmn", xyz_rows, xyz)
        rmsd_block = _rmsd_qcp_cupy(cp, H_block, traces[i_start:i_end], traces, n_atoms)
        del H_block

        rmsd_block_cpu = cp.asnumpy(rmsd_block)
        del rmsd_block
        for k in range(i_end - i_start):
            i = i_start + k
            row = rmsd_block_cpu[k, i + 1 :]
            result[i, i + 1 :] = row
            result[i + 1 :, i] = row

    return result.astype(np.float64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

rmsd_matrix_backends: BackendRegistry[RMSDMatrixBackendFn] = BackendRegistry(default="mdtraj")
rmsd_matrix_backends.register("numba", rmsd_numba)
rmsd_matrix_backends.register("mdtraj", rmsd_mdtraj)
rmsd_matrix_backends.register("torch", rmsd_torch)
rmsd_matrix_backends.register("jax", rmsd_jax)
rmsd_matrix_backends.register("cupy", rmsd_cupy)
