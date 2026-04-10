"""Conformational clustering from RMSD matrices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.core.trajectory import select_atom_indices

type _BackendFn = Callable[[md.Trajectory, NDArray[np.int_]], NDArray[np.float64]]

# ---------------------------------------------------------------------------
# Numba-parallel pairwise RMSD (QCP / Theobald 2005)
#
# Each pair is independently centered, then the optimal rotational RMSD is
# found via the Quaternion Characteristic Polynomial method.  The outer
# loop over pairs is parallelised with ``prange`` so all cores are used.
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
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RMSDMatrixResult:
    """Pairwise RMSD matrix between trajectory frames."""

    rmsd_matrix_nm: NDArray[np.floating]
    atom_indices: NDArray[np.int_]

    @property
    def rmsd_matrix_angstrom(self) -> NDArray[np.floating]:
        """Return the RMSD matrix in Angstrom."""
        return self.rmsd_matrix_nm * 10.0


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Conformational clustering output."""

    labels: NDArray[np.int_]
    n_clusters: int
    medoid_frames: NDArray[np.int_]


type RMSDBackend = str


def _rmsd_matrix_numba(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using the Numba QCP kernel."""
    xyz = np.ascontiguousarray(traj.xyz[:, atom_indices, :], dtype=np.float64)
    traces = _center_and_traces(xyz)
    return _pairwise_rmsd(xyz, traces)


def _rmsd_matrix_mdtraj(
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


# ---------------------------------------------------------------------------
# Vectorised GPU backends (einsum + batched SVD)
#
# All three share the same algorithm:
#   1. Center each frame, compute per-frame traces.
#   2. Compute all NxN cross-covariance 3x3 matrices via einsum.
#   3. Extract upper-triangle pairs and run batched SVD.
#   4. Handle reflections, compute RMSD from singular values + traces.
# ---------------------------------------------------------------------------


def _rmsd_matrix_torch(
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
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for backend='torch'. Install with: pip install torch"
        ) from None

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


def _rmsd_matrix_jax(
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
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for backend='jax'. Install with: pip install jax[cuda12]"
        ) from None

    jax.config.update("jax_enable_x64", True)

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


def _rmsd_matrix_cupy(
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
    try:
        import cupy as cp
    except ImportError:
        raise ImportError(
            "CuPy is required for backend='cupy'. Install with: pip install cupy-cuda12x"
        ) from None

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


_BACKENDS: dict[str, _BackendFn] = {
    "numba": _rmsd_matrix_numba,
    "mdtraj": _rmsd_matrix_mdtraj,
    "torch": _rmsd_matrix_torch,
    "jax": _rmsd_matrix_jax,
    "cupy": _rmsd_matrix_cupy,
}


def compute_rmsd_matrix(
    traj: md.Trajectory,
    *,
    atom_selection: str = "backbone",
    backend: RMSDBackend = "numba",
    dtype: DtypeArg = None,
) -> RMSDMatrixResult:
    """Compute an all-vs-all RMSD matrix between trajectory frames.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used for RMSD calculation.
        backend: Computation backend.

            - ``"numba"`` (default) -- Numba-parallel QCP kernel (CPU).
            - ``"mdtraj"`` -- mdtraj precentered RMSD loop (CPU).
            - ``"torch"`` -- PyTorch einsum + batched SVD (CUDA/CPU).
            - ``"jax"`` -- JAX einsum + batched SVD (GPU/TPU/CPU).
            - ``"cupy"`` -- CuPy einsum + batched SVD (CUDA).

        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        RMSDMatrixResult with a symmetric ``(n_frames, n_frames)`` matrix.

    Raises:
        ValueError: If an unsupported backend is specified.
        ImportError: If the requested backend package is not installed.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Unsupported backend: {backend!r}. Choose from {sorted(_BACKENDS)}.")
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    compute_fn = _BACKENDS[backend]
    rmsd_matrix = compute_fn(traj, atom_indices)

    return RMSDMatrixResult(
        rmsd_matrix_nm=rmsd_matrix.astype(resolved),
        atom_indices=atom_indices,
    )


def cluster_conformations(
    rmsd_matrix: NDArray[np.floating],
    *,
    method: str = "gromos",
    cutoff_nm: float = 0.15,
) -> ClusteringResult:
    """Cluster trajectory frames from a pairwise RMSD matrix.

    Args:
        rmsd_matrix: Symmetric pairwise RMSD matrix of shape ``(n, n)``
            in nm.
        method: Clustering method. ``"gromos"`` uses the GROMOS algorithm
            (largest-cluster-first greedy assignment).
        cutoff_nm: RMSD cutoff in nm for the GROMOS algorithm.

    Returns:
        ClusteringResult with per-frame labels and medoid frame indices.

    Raises:
        ValueError: If an unsupported method is specified.
    """
    if method != "gromos":
        raise ValueError(f"Unsupported clustering method: {method!r}. Use 'gromos'.")

    n_frames = rmsd_matrix.shape[0]
    labels = np.full(n_frames, -1, dtype=np.int_)
    assigned = np.zeros(n_frames, dtype=bool)
    cluster_id = 0
    medoids: list[int] = []

    while not np.all(assigned):
        neighbor_counts = np.zeros(n_frames, dtype=np.int_)
        for i in range(n_frames):
            if assigned[i]:
                continue
            neighbor_counts[i] = np.sum((rmsd_matrix[i] <= cutoff_nm) & (~assigned))

        center = int(np.argmax(neighbor_counts))
        members = (~assigned) & (rmsd_matrix[center] <= cutoff_nm)
        labels[members] = cluster_id
        assigned[members] = True
        medoids.append(center)
        cluster_id += 1

    return ClusteringResult(
        labels=labels,
        n_clusters=cluster_id,
        medoid_frames=np.array(medoids, dtype=np.int_),
    )
