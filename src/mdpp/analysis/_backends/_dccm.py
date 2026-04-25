"""DCCM (dynamic cross-correlation) covariance backends.

Provides five backends for computing the per-atom covariance
``cov[i, j] = <(r_i - <r_i>) . (r_j - <r_j>)>`` averaged over frames:

- ``numpy`` -- BLAS GEMM via reshape + matmul.
- ``numba`` -- Numba-parallel CPU kernel.
- ``cupy`` -- CuPy matmul on GPU.
- ``torch`` -- PyTorch matmul on GPU/CPU.
- ``jax`` -- JAX/XLA matmul on GPU/CPU.

Why a backend system at all
---------------------------

The original implementation of :func:`mdpp.analysis.compute_dccm`
called ``np.einsum("fid,fjd->ij", fluct, fluct)``.  NumPy's einsum
falls back to a single-threaded contraction loop for this 4-index
pattern -- it does **not** dispatch to BLAS even when ``optimize=True``
is set, because the underlying ``tensordot`` requires reshaping the
operands first.  The result is one core saturated regardless of
``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``, with the contraction
becoming the bottleneck for any non-trivial trajectory.

The ``numpy`` backend here flattens the fluctuations to a 2D
``(N, F*3)`` matrix and uses ``A @ A.T``; matmul on 2D arrays
dispatches to BLAS GEMM, which is multi-threaded.  On a 16-core host
this gives roughly an order-of-magnitude speedup at F=100k, N=500
with no extra dependencies and is therefore the default backend.

The numba and GPU backends layer on top for users who want either
pure-Python parallelism (numba) or GPU acceleration (torch/jax/cupy).

Backend signature
-----------------

All backends accept a single ``(F, N, 3)`` float positions array and
return an ``(N, N)`` covariance matrix in the backend's **native**
dtype (float32 for numpy / GPU backends, float64 for numba).  Mean
subtraction is performed inside each backend so that GPU kernels can
keep the entire pipeline on-device.  The public :func:`compute_dccm`
wrapper casts with ``copy=False`` and derives the final correlation
matrix from the covariance.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from mdpp.analysis._backends._imports import (
    clean_cupy_cache,
    clean_torch_cache,
    require_cupy,
    require_jax,
    require_torch,
)
from mdpp.analysis._backends._registry import BackendRegistry


class DCCMBackendFn(Protocol):
    """Callable signature for a DCCM covariance backend.

    Backends accept atom positions of shape ``(F, N, 3)`` already
    sliced to the atom subset of interest, and return an ``(N, N)``
    floating-point covariance matrix in the backend's native dtype.
    Centering (mean subtraction) is performed inside the backend so
    that GPU kernels keep the full pipeline on-device.
    """

    def __call__(
        self,
        positions_nm: NDArray[np.floating],
    ) -> NDArray[np.floating]: ...


# ---------------------------------------------------------------------------
# numpy backend (multi-threaded via BLAS GEMM)
# ---------------------------------------------------------------------------


def dccm_numpy(positions_nm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute DCCM covariance using BLAS GEMM via reshape + matmul.

    Reshapes the fluctuations from ``(F, N, 3)`` to ``(N, F*3)`` and
    computes ``A @ A.T``, which dispatches to a multi-threaded BLAS
    GEMM kernel.  ``np.einsum("fid,fjd->ij", ...)`` -- the original
    formulation -- saturates a single core regardless of
    ``OPENBLAS_NUM_THREADS``, so this rewrite gives roughly linear
    speedup with cores at no API cost.

    Args:
        positions_nm: Atom positions of shape ``(F, N, 3)`` in nm.

    Returns:
        Covariance matrix of shape ``(N, N)`` in the input dtype
        (typically float32 for mdtraj coordinates).  The public
        :func:`compute_dccm` wrapper casts with ``copy=False`` so no
        redundant ``(N, N)`` copy is made when dtypes match.
    """
    n_frames, n_atoms, _ = positions_nm.shape
    fluct = positions_nm - positions_nm.mean(axis=0, keepdims=True)
    # ``ascontiguousarray`` realises the transposed view as a contiguous
    # row-major buffer once, so the BLAS call sees a clean stride
    # pattern instead of paying a hidden copy inside matmul.
    matrix = np.ascontiguousarray(fluct.transpose(1, 0, 2).reshape(n_atoms, n_frames * 3))
    return (matrix @ matrix.T) / float(n_frames)


# ---------------------------------------------------------------------------
# numba backend (parallel CPU)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _dccm_numba_kernel(
    positions: NDArray[np.float32],
) -> NDArray[np.floating]:  # pragma: no cover - JIT-compiled
    """Two-pass numba kernel: per-atom mean, then symmetric covariance.

    The outer atom loops use ``prange`` so each thread owns a disjoint
    slice of output rows -- no cross-thread reduction is required.
    Inner reductions promote to C ``double`` via ``float()`` so the
    result has float64 precision; the wrapper casts to the user's
    resolved dtype with ``copy=False`` afterward.
    """
    n_frames, n_atoms, _ = positions.shape

    # Pass 1: per-atom mean.
    means = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in prange(n_atoms):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for f in range(n_frames):
            sx += float(positions[f, i, 0])
            sy += float(positions[f, i, 1])
            sz += float(positions[f, i, 2])
        means[i, 0] = sx / n_frames
        means[i, 1] = sy / n_frames
        means[i, 2] = sz / n_frames

    # Pass 2: symmetric covariance.
    cov = np.empty((n_atoms, n_atoms), dtype=np.float64)
    for i in prange(n_atoms):
        mxi = means[i, 0]
        myi = means[i, 1]
        mzi = means[i, 2]
        for j in range(i, n_atoms):
            mxj = means[j, 0]
            myj = means[j, 1]
            mzj = means[j, 2]
            s = 0.0
            for f in range(n_frames):
                dxi = float(positions[f, i, 0]) - mxi
                dyi = float(positions[f, i, 1]) - myi
                dzi = float(positions[f, i, 2]) - mzi
                dxj = float(positions[f, j, 0]) - mxj
                dyj = float(positions[f, j, 1]) - myj
                dzj = float(positions[f, j, 2]) - mzj
                s += dxi * dxj + dyi * dyj + dzi * dzj
            value = s / n_frames
            cov[i, j] = value
            cov[j, i] = value
    return cov


def dccm_numba(positions_nm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute DCCM covariance using a Numba-parallel CPU kernel.

    Parallelises the outer atom loop with ``prange`` so each thread
    owns a disjoint band of output rows.  Returns float64 (numba's
    natural dtype via ``float()`` promotion); the public wrapper
    casts to the user's resolved dtype with ``copy=False``.

    Args:
        positions_nm: Atom positions of shape ``(F, N, 3)`` in nm.

    Returns:
        Covariance matrix of shape ``(N, N)`` in **float64**.
    """
    return _dccm_numba_kernel(np.ascontiguousarray(positions_nm, dtype=np.float32))


# ---------------------------------------------------------------------------
# CuPy backend (GPU)
# ---------------------------------------------------------------------------


@clean_cupy_cache
def dccm_cupy(positions_nm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute DCCM covariance on GPU using CuPy.

    Mean subtraction and the covariance GEMM both run on-device; only
    the final ``(N, N)`` matrix is transferred back to host memory.

    Args:
        positions_nm: Atom positions of shape ``(F, N, 3)`` in nm.

    Returns:
        Covariance matrix of shape ``(N, N)`` in **float32** (cupy
        inherits the float32 of mdtraj coordinates).

    Raises:
        ImportError: If CuPy is not installed.
    """
    cp = require_cupy()
    n_frames, n_atoms, _ = positions_nm.shape
    pos_gpu = cp.asarray(positions_nm)
    fluct = pos_gpu - pos_gpu.mean(axis=0, keepdims=True)
    matrix = cp.ascontiguousarray(fluct.transpose(1, 0, 2).reshape(n_atoms, n_frames * 3))
    cov = (matrix @ matrix.T) / float(n_frames)
    return cp.asnumpy(cov)


# ---------------------------------------------------------------------------
# Torch backend (GPU / CPU)
# ---------------------------------------------------------------------------


@clean_torch_cache
def dccm_torch(positions_nm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute DCCM covariance using PyTorch (CUDA if available).

    Uses CUDA when visible, otherwise falls back to CPU torch (still
    multi-threaded via ATen).  Wrapped in ``torch.inference_mode``
    to skip autograd bookkeeping; the GEMM dispatches to cuBLAS
    (GPU) or MKL/OpenBLAS (CPU).

    Args:
        positions_nm: Atom positions of shape ``(F, N, 3)`` in nm.

    Returns:
        Covariance matrix of shape ``(N, N)`` in **float32**.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_frames, n_atoms, _ = positions_nm.shape
    with torch.inference_mode():
        pos = torch.as_tensor(np.ascontiguousarray(positions_nm), device=device)
        fluct = pos - pos.mean(dim=0, keepdim=True)
        matrix = fluct.permute(1, 0, 2).reshape(n_atoms, n_frames * 3).contiguous()
        cov = (matrix @ matrix.T) / float(n_frames)
    return cov.cpu().numpy()


# ---------------------------------------------------------------------------
# JAX backend (GPU / TPU / CPU)
# ---------------------------------------------------------------------------


def dccm_jax(positions_nm: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute DCCM covariance using JAX (auto GPU/TPU/CPU).

    The matmul explicitly passes ``precision=HIGHEST`` to disable JAX's
    default TF32 / tensor-core accumulation on GPU, which would
    otherwise drop the GEMM to ~19 bits of mantissa and pick up
    visible error on small fixtures.  Deliberately not wrapped with a
    cache-cleanup decorator -- ``jax.clear_caches()`` clears JIT
    compilation caches (not device memory) and trashing them after
    every call forces a multi-second recompile on the next invocation.

    Args:
        positions_nm: Atom positions of shape ``(F, N, 3)`` in nm.

    Returns:
        Covariance matrix of shape ``(N, N)`` in **float32**.

    Raises:
        ImportError: If JAX is not installed.
    """
    jax_mod, jnp = require_jax()
    n_frames, n_atoms, _ = positions_nm.shape
    pos = jnp.asarray(positions_nm)
    fluct = pos - pos.mean(axis=0, keepdims=True)
    matrix = jnp.transpose(fluct, (1, 0, 2)).reshape(n_atoms, n_frames * 3)
    cov = jnp.matmul(matrix, matrix.T, precision=jax_mod.lax.Precision.HIGHEST) / float(n_frames)
    return np.asarray(cov)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

dccm_backends: BackendRegistry[DCCMBackendFn] = BackendRegistry(default="numpy")
dccm_backends.register("numpy", dccm_numpy)
dccm_backends.register("numba", dccm_numba)
dccm_backends.register("cupy", dccm_cupy)
dccm_backends.register("torch", dccm_torch)
dccm_backends.register("jax", dccm_jax)
