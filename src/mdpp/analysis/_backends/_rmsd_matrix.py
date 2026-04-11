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

from mdpp.analysis._backends._imports import require_cupy, require_jax, require_torch
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
) -> NDArray[np.float64]:  # pragma: no cover - JIT
    """Compute symmetric pairwise RMSD matrix with QCP superposition.

    Uses the Quaternion Characteristic Polynomial method (Theobald 2005)
    to find the optimal rotational RMSD for each pair.  The largest
    eigenvalue of the 4x4 key matrix is found via Newton-Raphson on
    the characteristic polynomial -- pure scalar arithmetic with no
    LAPACK calls, so the ``prange`` outer loop scales across all cores.
    """
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]
    result = np.zeros((n_frames, n_frames))
    for i in prange(n_frames):
        for j in range(i + 1, n_frames):
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
    """Compute pairwise RMSD matrix using the Numba QCP kernel."""
    xyz = np.ascontiguousarray(traj.xyz[:, atom_indices, :], dtype=np.float64)
    traces = _center_and_traces(xyz)
    return _pairwise_rmsd(xyz, traces)


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


def rmsd_torch(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using PyTorch (CUDA if available).

    Uses vectorised ``einsum`` for all pairwise cross-covariance
    matrices followed by batched ``torch.linalg.svd`` on the 3x3
    matrices.  Falls back to CPU when no CUDA device is found.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    torch = require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        xyz = torch.as_tensor(
            np.ascontiguousarray(traj.xyz[:, atom_indices, :]),
            dtype=torch.float64,
            device=device,
        )
        n_frames, n_atoms, _ = xyz.shape

        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        traces = (xyz * xyz).sum(dim=(1, 2))

        H_all = torch.einsum("iak,jal->ijkl", xyz, xyz)

        ii, jj = torch.triu_indices(n_frames, n_frames, offset=1, device=device)
        H_pairs = H_all[ii, jj]

        U, S, Vh = torch.linalg.svd(H_pairs)
        d = torch.det(U) * torch.det(Vh)
        S[:, 2] = torch.where(d < 0, -S[:, 2], S[:, 2])

        rmsd_sq = (traces[ii] + traces[jj] - 2.0 * S.sum(dim=1)) / n_atoms
        rmsd_vals = torch.sqrt(torch.clamp(rmsd_sq, min=0.0))

        result = torch.zeros(n_frames, n_frames, dtype=torch.float64, device=device)
        result[ii, jj] = rmsd_vals
        result[jj, ii] = rmsd_vals

    return result.cpu().numpy()


def rmsd_jax(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using JAX.

    JAX auto-selects the best available device (GPU > TPU > CPU).
    Uses ``jnp.einsum`` for cross-covariance and ``jnp.linalg.svd``
    for batched superposition.

    Raises:
        ImportError: If JAX is not installed.
    """
    _jax, jnp = require_jax()

    xyz = jnp.array(traj.xyz[:, atom_indices, :], dtype=jnp.float64)
    n_frames, n_atoms, _ = xyz.shape

    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    traces = (xyz * xyz).sum(axis=(1, 2))

    H_all = jnp.einsum("iak,jal->ijkl", xyz, xyz)

    ii, jj = jnp.triu_indices(n_frames, k=1)
    H_pairs = H_all[ii, jj]

    U, S, Vh = jnp.linalg.svd(H_pairs)
    d = jnp.linalg.det(U) * jnp.linalg.det(Vh)
    S = S.at[:, 2].set(jnp.where(d < 0, -S[:, 2], S[:, 2]))

    rmsd_sq = (traces[ii] + traces[jj] - 2.0 * S.sum(axis=1)) / n_atoms
    rmsd_vals = jnp.sqrt(jnp.maximum(0.0, rmsd_sq))

    result = jnp.zeros((n_frames, n_frames), dtype=jnp.float64)
    result = result.at[ii, jj].set(rmsd_vals)
    result = result.at[jj, ii].set(rmsd_vals)

    return np.asarray(result)


def rmsd_cupy(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using CuPy (CUDA).

    Requires a CUDA-capable GPU.  Uses ``cupy.einsum`` for
    cross-covariance and ``cupy.linalg.svd`` for batched
    superposition.

    Raises:
        ImportError: If CuPy is not installed.
    """
    cp = require_cupy()

    xyz = cp.array(traj.xyz[:, atom_indices, :], dtype=cp.float64)
    n_frames, n_atoms, _ = xyz.shape

    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    traces = (xyz * xyz).sum(axis=(1, 2))

    H_all = cp.einsum("iak,jal->ijkl", xyz, xyz)

    ii, jj = (
        cp.array(np.triu_indices(n_frames, k=1)[0]),
        cp.array(np.triu_indices(n_frames, k=1)[1]),
    )
    H_pairs = H_all[ii, jj]

    U, S, Vh = cp.linalg.svd(H_pairs)
    d = cp.linalg.det(U) * cp.linalg.det(Vh)
    S[:, 2] = cp.where(d < 0, -S[:, 2], S[:, 2])

    rmsd_sq = (traces[ii] + traces[jj] - 2.0 * S.sum(axis=1)) / n_atoms
    rmsd_vals = cp.sqrt(cp.maximum(0.0, rmsd_sq))

    result = cp.zeros((n_frames, n_frames), dtype=cp.float64)
    result[ii, jj] = rmsd_vals
    result[jj, ii] = rmsd_vals

    return cp.asnumpy(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

rmsd_matrix_backends: BackendRegistry[RMSDMatrixBackendFn] = BackendRegistry(default="mdtraj")
rmsd_matrix_backends.register("numba", rmsd_numba)
rmsd_matrix_backends.register("mdtraj", rmsd_mdtraj)
rmsd_matrix_backends.register("torch", rmsd_torch)
rmsd_matrix_backends.register("jax", rmsd_jax)
rmsd_matrix_backends.register("cupy", rmsd_cupy)
