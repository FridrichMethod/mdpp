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


@clean_torch_cache
def rmsd_torch(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using PyTorch (CUDA if available).

    Uses vectorised ``einsum`` for all pairwise cross-covariance
    matrices followed by the Quaternion Characteristic Polynomial
    (Theobald 2005) Newton-Raphson solve on the quartic -- same
    algorithm as the Numba kernel, but broadcast across all pairs
    in ``(n_pairs,)``-shaped tensors.  Falls back to CPU when no
    CUDA device is found.

    **Why QCP and not batched SVD**: batched ``torch.linalg.svd``
    on 3x3 matrices has high LAPACK per-matrix overhead; for
    500k pairs it takes ~70 ms even in float32 and dominates wall
    time.  QCP is pure element-wise arithmetic on flat tensors
    (~50 flops per pair, fixed 30 Newton-Raphson iterations) and
    runs ~20x faster on the same workload.

    Internally operates in **float32** because consumer and
    workstation NVIDIA GPUs run float64 at 1/36 -- 1/64 the
    throughput of float32.  mdtraj stores coordinates in float32
    anyway, and float32 QCP agrees with the float64 numba reference
    to ~1e-6 nm -- well below the 5e-5 nm agreement tolerance.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    torch = require_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        xyz = torch.as_tensor(
            np.ascontiguousarray(traj.xyz[:, atom_indices, :]),
            dtype=torch.float32,
            device=device,
        )
        n_frames, n_atoms, _ = xyz.shape

        xyz = xyz - xyz.mean(dim=1, keepdim=True)
        traces = (xyz * xyz).sum(dim=(1, 2))

        # Cross-covariance ``H[i, j, m, n] = sum_a xyz[i, a, m] * xyz[j, a, n]``
        H_all = torch.einsum("iam,jan->ijmn", xyz, xyz)
        ii, jj = torch.triu_indices(n_frames, n_frames, offset=1, device=device)

        Sxx = H_all[ii, jj, 0, 0]
        Sxy = H_all[ii, jj, 0, 1]
        Sxz = H_all[ii, jj, 0, 2]
        Syx = H_all[ii, jj, 1, 0]
        Syy = H_all[ii, jj, 1, 1]
        Syz = H_all[ii, jj, 1, 2]
        Szx = H_all[ii, jj, 2, 0]
        Szy = H_all[ii, jj, 2, 1]
        Szz = H_all[ii, jj, 2, 2]
        del H_all

        # Quartic characteristic polynomial coefficients.
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

        # 4x4 key matrix elements.
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

        # Newton-Raphson for the largest eigenvalue.  Initial guess
        # ``(G_a + G_b) / 2`` is an upper bound on lambda_max.
        lam = (traces[ii] + traces[jj]) * 0.5
        for _ in range(30):
            l2 = lam * lam
            f_val = l2 * l2 + c2 * l2 + c1 * lam + c0
            fp_val = 4.0 * l2 * lam + 2.0 * c2 * lam + c1
            # Avoid div-by-zero if Newton-Raphson stalls on a flat derivative.
            delta = torch.where(
                fp_val.abs() > 0,
                f_val / fp_val,
                torch.zeros_like(fp_val),
            )
            lam = lam - delta

        rmsd_sq = (traces[ii] + traces[jj] - 2.0 * lam) / n_atoms
        rmsd_vals = torch.sqrt(torch.clamp(rmsd_sq, min=0.0))

        result = torch.zeros(n_frames, n_frames, dtype=torch.float32, device=device)
        result[ii, jj] = rmsd_vals
        result[jj, ii] = rmsd_vals

    return result.cpu().numpy().astype(np.float64)


def rmsd_jax(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using JAX.

    JAX auto-selects the best available device (GPU > TPU > CPU).
    Uses ``jnp.einsum`` for cross-covariance and a vectorised
    QCP Newton-Raphson solve for the rotational superposition --
    same algorithm as :func:`rmsd_torch` and :func:`rmsd_cupy`.

    Internally operates in **float32** for GPU throughput, with
    ``precision=HIGHEST`` on the einsum to disable TF32 accumulation
    on NVIDIA GPUs.  This kernel deliberately does **not** use the
    ``clean_jax_cache`` decorator -- JAX's ``clear_caches()`` clears
    JIT compilation caches (not device memory), and trashing the
    compilation cache after every call forces a 1+ second recompile
    on the next call.  JAX does not expose a public API for returning
    pooled device memory to the driver anyway.

    Raises:
        ImportError: If JAX is not installed.
    """
    _jax, jnp = require_jax()

    xyz = jnp.array(traj.xyz[:, atom_indices, :], dtype=jnp.float32)
    n_frames, n_atoms, _ = xyz.shape

    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    traces = (xyz * xyz).sum(axis=(1, 2))

    # ``precision=HIGHEST`` disables JAX's default TF32 / tensor-core
    # accumulation on GPU, which would otherwise drop the einsum
    # down to ~19 bits of mantissa and cost ~1e-4 nm accuracy on
    # small test fixtures.
    H_all = jnp.einsum(
        "iam,jan->ijmn",
        xyz,
        xyz,
        precision=_jax.lax.Precision.HIGHEST,
    )
    ii, jj = jnp.triu_indices(n_frames, k=1)

    Sxx = H_all[ii, jj, 0, 0]
    Sxy = H_all[ii, jj, 0, 1]
    Sxz = H_all[ii, jj, 0, 2]
    Syx = H_all[ii, jj, 1, 0]
    Syy = H_all[ii, jj, 1, 1]
    Syz = H_all[ii, jj, 1, 2]
    Szx = H_all[ii, jj, 2, 0]
    Szy = H_all[ii, jj, 2, 1]
    Szz = H_all[ii, jj, 2, 2]
    del H_all

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

    lam = (traces[ii] + traces[jj]) * 0.5
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

    rmsd_sq = (traces[ii] + traces[jj] - 2.0 * lam) / n_atoms
    rmsd_vals = jnp.sqrt(jnp.maximum(0.0, rmsd_sq))

    result = jnp.zeros((n_frames, n_frames), dtype=jnp.float32)
    result = result.at[ii, jj].set(rmsd_vals)
    result = result.at[jj, ii].set(rmsd_vals)

    return np.asarray(result).astype(np.float64)


@clean_cupy_cache
def rmsd_cupy(
    traj: md.Trajectory,
    atom_indices: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute pairwise RMSD matrix using CuPy (CUDA).

    Requires a CUDA-capable GPU.  Uses ``cupy.einsum`` for
    cross-covariance and a vectorised QCP Newton-Raphson solve
    for the rotational superposition -- same algorithm as
    :func:`rmsd_torch`.

    Internally operates in **float32** for GPU performance (see
    :func:`rmsd_torch` for rationale).  The result is cast to
    float64 on the way out to preserve the public API contract.

    Raises:
        ImportError: If CuPy is not installed.
    """
    cp = require_cupy()

    xyz = cp.array(traj.xyz[:, atom_indices, :], dtype=cp.float32)
    n_frames, n_atoms, _ = xyz.shape

    xyz = xyz - xyz.mean(axis=1, keepdims=True)
    traces = (xyz * xyz).sum(axis=(1, 2))

    H_all = cp.einsum("iam,jan->ijmn", xyz, xyz)
    ii_host, jj_host = np.triu_indices(n_frames, k=1)
    ii = cp.asarray(ii_host)
    jj = cp.asarray(jj_host)

    Sxx = H_all[ii, jj, 0, 0]
    Sxy = H_all[ii, jj, 0, 1]
    Sxz = H_all[ii, jj, 0, 2]
    Syx = H_all[ii, jj, 1, 0]
    Syy = H_all[ii, jj, 1, 1]
    Syz = H_all[ii, jj, 1, 2]
    Szx = H_all[ii, jj, 2, 0]
    Szy = H_all[ii, jj, 2, 1]
    Szz = H_all[ii, jj, 2, 2]
    del H_all

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

    lam = (traces[ii] + traces[jj]) * 0.5
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

    rmsd_sq = (traces[ii] + traces[jj] - 2.0 * lam) / n_atoms
    rmsd_vals = cp.sqrt(cp.maximum(0.0, rmsd_sq))

    result = cp.zeros((n_frames, n_frames), dtype=cp.float32)
    result[ii, jj] = rmsd_vals
    result[jj, ii] = rmsd_vals

    return cp.asnumpy(result).astype(np.float64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

rmsd_matrix_backends: BackendRegistry[RMSDMatrixBackendFn] = BackendRegistry(default="mdtraj")
rmsd_matrix_backends.register("numba", rmsd_numba)
rmsd_matrix_backends.register("mdtraj", rmsd_mdtraj)
rmsd_matrix_backends.register("torch", rmsd_torch)
rmsd_matrix_backends.register("jax", rmsd_jax)
rmsd_matrix_backends.register("cupy", rmsd_cupy)
