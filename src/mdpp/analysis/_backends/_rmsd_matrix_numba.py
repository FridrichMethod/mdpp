"""Numba-parallel pairwise RMSD kernels (QCP / Theobald 2005).

Lives in its own module so the GPU streaming code in
:mod:`mdpp.analysis._backends._rmsd_matrix` can stay under the project's
800-line file cap without giving up the dedicated comments on the QCP
algorithm and dtype handling here.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(cache=True)
def _center_and_traces(
    xyz: NDArray[np.floating],
) -> NDArray[np.floating]:  # pragma: no cover - JIT
    """Center each frame in-place and return per-frame sum-of-squares.

    ``traces`` is allocated in float64 so the QCP Newton-Raphson
    subtraction ``G_a + G_b - 2*lambda`` preserves the few extra
    significant bits that float32 would lose when ``lambda`` is
    close to ``(G_a + G_b) / 2``.  This buffer is ``O(n_frames)`` so
    the fp64 cost is negligible even at 120k frames (1 MB).
    """
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
    xyz: NDArray[np.floating],
    traces: NDArray[np.floating],
    pair_i: NDArray[np.int64],
    pair_j: NDArray[np.int64],
) -> NDArray[np.floating]:  # pragma: no cover - JIT
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

    **Dtype policy.**  The accumulators (``Sxx`` etc.) and the QCP
    Newton-Raphson state are all ``float64`` scalars (numba's
    ``0.0`` literal maps to a C ``double``), so the quartic solve
    preserves full double precision regardless of the input dtype.
    Only the final store ``result[i, j] = val`` truncates to
    ``float32``, which halves the O(N^2) output-matrix footprint
    (58 GB saved at n=120k) while keeping the QCP precision that
    the float64 accumulation provides.  The ``traces`` buffer is
    also float64 for the same reason -- see
    :func:`_center_and_traces`.
    """
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]
    n_pairs = pair_i.shape[0]
    result = np.zeros((n_frames, n_frames), dtype=np.float32)
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


def rmsd_numba(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.floating]:
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
