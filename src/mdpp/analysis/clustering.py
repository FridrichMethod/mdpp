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


_BACKENDS: dict[str, _BackendFn] = {
    "numba": _rmsd_matrix_numba,
    "mdtraj": _rmsd_matrix_mdtraj,
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
        backend: Computation backend. ``"numba"`` (default) uses a
            Numba-parallel QCP kernel; ``"mdtraj"`` uses mdtraj's
            precentered RMSD loop.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        RMSDMatrixResult with a symmetric ``(n_frames, n_frames)`` matrix.

    Raises:
        ValueError: If an unsupported backend is specified.
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
