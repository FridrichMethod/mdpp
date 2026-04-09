"""Core structure and dynamics metrics computed from trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
from mdpp.core.trajectory import (
    residue_ids_from_indices,
    select_atom_indices,
    trajectory_time_ps,
)


@dataclass(frozen=True, slots=True)
class RMSDResult:
    """RMSD time series."""

    time_ps: NDArray[np.floating]
    rmsd_nm: NDArray[np.floating]
    atom_indices: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def rmsd_angstrom(self) -> NDArray[np.floating]:
        """Return RMSD values in Angstrom."""
        return self.rmsd_nm * 10.0


@dataclass(frozen=True, slots=True)
class RMSFResult:
    """Per-atom RMSF values."""

    rmsf_nm: NDArray[np.floating]
    atom_indices: NDArray[np.int_]
    residue_ids: NDArray[np.int_] | None

    @property
    def rmsf_angstrom(self) -> NDArray[np.floating]:
        """Return RMSF values in Angstrom."""
        return self.rmsf_nm * 10.0


@dataclass(frozen=True, slots=True)
class DeltaRMSFResult:
    """Per-residue RMSF difference between two systems (B minus A).

    Averaging is done in MSF (mean-square fluctuation) space: per-residue
    RMSF^2 values are averaged across replicas, then the square root is
    taken.  The delta is computed on the resulting average RMSF values.

    The SEM on each system's average RMSF is propagated through the sqrt
    transform, then the two independent SEMs are combined in quadrature
    to give the SEM on the delta.
    """

    delta_rmsf_nm: NDArray[np.floating]
    residue_ids: NDArray[np.int_] | None
    sem_nm: NDArray[np.floating] | None

    @property
    def delta_rmsf_angstrom(self) -> NDArray[np.floating]:
        """Return delta-RMSF values in Angstrom."""
        return self.delta_rmsf_nm * 10.0

    @property
    def sem_angstrom(self) -> NDArray[np.floating] | None:
        """Return SEM on the delta-RMSF in Angstrom."""
        if self.sem_nm is None:
            return None
        return self.sem_nm * 10.0


@dataclass(frozen=True, slots=True)
class DCCMResult:
    """Dynamic cross-correlation matrix."""

    correlation: NDArray[np.floating]
    atom_indices: NDArray[np.int_]
    residue_ids: NDArray[np.int_] | None


@dataclass(frozen=True, slots=True)
class SASAResult:
    """Solvent accessible surface area."""

    time_ps: NDArray[np.floating]
    values_nm2: NDArray[np.floating]
    atom_indices: NDArray[np.int_]
    mode: str
    residue_ids: NDArray[np.int_] | None

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def total_nm2(self) -> NDArray[np.floating]:
        """Return summed SASA for each frame."""
        return np.sum(self.values_nm2, axis=1)


@dataclass(frozen=True, slots=True)
class RadiusOfGyrationResult:
    """Radius of gyration time series."""

    time_ps: NDArray[np.floating]
    radius_gyration_nm: NDArray[np.floating]
    atom_indices: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def radius_gyration_angstrom(self) -> NDArray[np.floating]:
        """Return radius of gyration values in Angstrom."""
        return self.radius_gyration_nm * 10.0


def compute_rmsd(
    traj: md.Trajectory,
    *,
    atom_selection: str = "backbone",
    reference_frame: int = 0,
    timestep_ps: float | None = None,
    dtype: type[np.floating] | None = None,
) -> RMSDResult:
    """Compute RMSD over time.

    The trajectory should be aligned before calling this function
    (see :func:`~mdpp.core.trajectory.align_trajectory`).

    Args:
        traj: Input trajectory (pre-aligned).
        atom_selection: Atoms used in RMSD calculation.
        reference_frame: Reference frame index for RMSD.
        timestep_ps: Optional time step in ps to override trajectory time.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        RMSDResult containing time and RMSD.
    """
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    rmsd_nm = np.asarray(
        md.rmsd(
            traj,
            traj,
            frame=reference_frame,
            atom_indices=atom_indices,
            precentered=False,
        ),
        dtype=resolved,
    )
    return RMSDResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps, dtype=resolved),
        rmsd_nm=rmsd_nm,
        atom_indices=atom_indices,
    )


def compute_rmsf(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    dtype: type[np.floating] | None = None,
) -> RMSFResult:
    """Compute per-atom RMSF from positional fluctuations.

    The trajectory should be aligned before calling this function
    (see :func:`~mdpp.core.trajectory.align_trajectory`).

    Args:
        traj: Input trajectory (pre-aligned).
        atom_selection: Atoms included in RMSF calculation.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        RMSFResult with atom and residue mapping.
    """
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    # Upcast to float64 for sum-of-squares accumulation stability
    positions_nm = traj.xyz[:, atom_indices, :].astype(np.float64)
    mean_positions_nm = np.mean(positions_nm, axis=0)
    squared_displacements = np.sum((positions_nm - mean_positions_nm) ** 2, axis=2)
    rmsf_nm = np.sqrt(np.mean(squared_displacements, axis=0))
    residue_ids = residue_ids_from_indices(traj.topology, atom_indices)
    return RMSFResult(
        rmsf_nm=np.asarray(rmsf_nm, dtype=resolved),
        atom_indices=atom_indices,
        residue_ids=residue_ids,
    )


def compute_dccm(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    dtype: type[np.floating] | None = None,
) -> DCCMResult:
    """Compute dynamic cross-correlation matrix (DCCM).

    The trajectory should be aligned before calling this function
    (see :func:`~mdpp.core.trajectory.align_trajectory`).

    Covariance is always computed in float64 for numerical stability;
    the final correlation matrix is cast to the resolved *dtype*.

    Args:
        traj: Input trajectory (pre-aligned).
        atom_selection: Atoms used in DCCM.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        DCCMResult with correlation matrix and residue IDs.
    """
    resolved = resolve_dtype(dtype)

    if traj.n_frames < 2:
        raise ValueError("DCCM requires at least two frames.")

    atom_indices = select_atom_indices(traj.topology, atom_selection)
    # Upcast to float64 before covariance: einsum on float32 loses precision.
    positions_nm = traj.xyz[:, atom_indices, :].astype(np.float64)
    mean_positions_nm = np.mean(positions_nm, axis=0)
    fluctuation_nm = positions_nm - mean_positions_nm

    covariance = np.einsum("fid,fjd->ij", fluctuation_nm, fluctuation_nm) / float(traj.n_frames)
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    normalization = np.outer(standard_deviation, standard_deviation)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / normalization
    correlation = np.asarray(correlation, dtype=np.float64)
    correlation[~np.isfinite(correlation)] = 0.0
    np.fill_diagonal(correlation, 1.0)

    # Cast to output dtype after all float64 arithmetic.
    correlation = np.asarray(correlation, dtype=resolved)

    residue_ids = residue_ids_from_indices(traj.topology, atom_indices)
    return DCCMResult(
        correlation=correlation,
        atom_indices=atom_indices,
        residue_ids=residue_ids,
    )


def compute_sasa(
    traj: md.Trajectory,
    *,
    atom_selection: str | None = "protein",
    mode: str = "residue",
    probe_radius: float = 0.14,
    n_sphere_points: int = 960,
    timestep_ps: float | None = None,
    dtype: type[np.floating] | None = None,
) -> SASAResult:
    """Compute solvent-accessible surface area via Shrake-Rupley.

    Args:
        traj: Input trajectory.
        atom_selection: Optional atom selection before SASA.
        mode: Either ``"atom"`` or ``"residue"``.
        probe_radius: Probe radius in nm.
        n_sphere_points: Number of sphere points per atom.
        timestep_ps: Optional timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        SASAResult containing frame-resolved values.
    """
    resolved = resolve_dtype(dtype)

    if mode not in {"atom", "residue"}:
        raise ValueError("mode must be either 'atom' or 'residue'.")

    if atom_selection is None:
        atom_indices = np.arange(traj.n_atoms, dtype=np.int_)
    else:
        atom_indices = select_atom_indices(traj.topology, atom_selection)

    sliced = traj.atom_slice(atom_indices)
    values_nm2 = np.asarray(
        md.shrake_rupley(
            sliced,
            mode=mode,
            probe_radius=probe_radius,
            n_sphere_points=n_sphere_points,
        ),
        dtype=resolved,
    )
    residue_ids = None
    if mode == "residue":
        residue_ids = np.array(
            [residue.resSeq for residue in sliced.topology.residues],
            dtype=np.int_,
        )

    return SASAResult(
        time_ps=trajectory_time_ps(sliced, timestep_ps=timestep_ps, dtype=resolved),
        values_nm2=values_nm2,
        atom_indices=atom_indices,
        mode=mode,
        residue_ids=residue_ids,
    )


def compute_radius_of_gyration(
    traj: md.Trajectory,
    *,
    atom_selection: str = "protein",
    timestep_ps: float | None = None,
    dtype: type[np.floating] | None = None,
) -> RadiusOfGyrationResult:
    """Compute radius of gyration over time.

    Args:
        traj: Input trajectory.
        atom_selection: Atom selection used to compute radius of gyration.
        timestep_ps: Optional timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        RadiusOfGyrationResult with per-frame values.
    """
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    sliced = traj.atom_slice(atom_indices)
    rg_nm = np.asarray(md.compute_rg(sliced), dtype=resolved)
    return RadiusOfGyrationResult(
        time_ps=trajectory_time_ps(sliced, timestep_ps=timestep_ps, dtype=resolved),
        radius_gyration_nm=rg_nm,
        atom_indices=atom_indices,
    )


def _average_rmsf_with_sem(
    results: list[RMSFResult],
    *,
    dtype: type[np.floating] | np.dtype[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """Average RMSF across replicas in MSF space and propagate SEM.

    Computation is performed in float64 for numerical stability (MSF
    averaging through sqrt); the outputs are cast to *dtype*.

    Args:
        results: RMSF results from each replica.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        (avg_rmsf_nm, sem_rmsf_nm).  SEM is ``None`` when fewer than 2
        replicas are provided.

    The SEM on MSF is propagated through the sqrt transform:
    ``sem_rmsf = sem_msf / (2 * avg_rmsf)``.
    """
    resolved = resolve_dtype(dtype)

    # Compute in float64 for numerical stability.
    msf_stack = np.stack([r.rmsf_nm.astype(np.float64) ** 2 for r in results])
    avg_msf = np.mean(msf_stack, axis=0)
    avg_rmsf = np.sqrt(avg_msf).astype(resolved)

    if len(results) < 2:
        return avg_rmsf, None

    n_replicas = len(results)
    sem_msf = np.std(msf_stack, axis=0, ddof=1) / np.sqrt(n_replicas)
    avg_rmsf_f64 = np.sqrt(avg_msf)
    sem_rmsf = np.where(avg_rmsf_f64 > 0, sem_msf / (2.0 * avg_rmsf_f64), 0.0).astype(resolved)
    return avg_rmsf, sem_rmsf


def _validate_rmsf_replicas(results: list[RMSFResult], name: str) -> None:
    """Validate that a list of RMSF results is non-empty and consistent."""
    if not results:
        raise ValueError(f"{name} must not be empty.")
    sizes = {r.rmsf_nm.size for r in results}
    if len(sizes) > 1:
        raise ValueError(f"{name} replicas have inconsistent sizes: {sizes}.")


def compute_delta_rmsf(
    results_a: list[RMSFResult],
    results_b: list[RMSFResult],
    *,
    indices_a: NDArray[np.int_] | None = None,
    indices_b: NDArray[np.int_] | None = None,
    residue_ids: NDArray[np.int_] | None = None,
    dtype: type[np.floating] | None = None,
) -> DeltaRMSFResult:
    """Compute per-residue RMSF difference between two systems.

    The RMSF for each system is first averaged across replicas in MSF space
    (``sqrt(mean(RMSF^2))``), then the delta is taken as B minus A.
    Positive values indicate that system B is more flexible.

    The SEM on each system's average RMSF is propagated through the sqrt
    transform (``sem_rmsf = sem_msf / (2 * avg_rmsf)``), then the two
    independent SEMs are combined in quadrature to give the SEM on the
    delta.  At least 2 replicas per system are required for SEM; otherwise
    ``DeltaRMSFResult.sem_nm`` is ``None``.

    For systems with **identical residue counts**, ``indices_a`` and
    ``indices_b`` may be omitted and the comparison is element-wise.

    For systems with **different sequences**, supply aligned index arrays
    so that ``indices_a[i]`` and ``indices_b[i]`` point to the same
    structural position in each system.  The caller is responsible for
    generating these mappings (e.g. from a multiple sequence alignment).

    Args:
        results_a: RMSF results for system A (one per replica).
        results_b: RMSF results for system B (one per replica).
        indices_a: Optional 0-based residue indices into system A's RMSF
            array at aligned positions.  Must have the same length as
            ``indices_b``.
        indices_b: Optional 0-based residue indices into system B's RMSF
            array at aligned positions.
        residue_ids: Optional residue IDs for the x-axis of the resulting
            delta-RMSF (e.g. a reference sequence numbering).  When
            ``None`` and indices are not provided, residue IDs are taken
            from ``results_a[0]``.  When ``None`` and indices *are*
            provided, residue IDs are taken from ``results_a[0]`` at the
            positions given by ``indices_a``.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        DeltaRMSFResult with the per-residue difference and SEM.

    Raises:
        ValueError: If input lists are empty, replicas within a system
            have inconsistent lengths, index arrays differ in length, or
            unindexed systems have different residue counts.
    """
    resolved = resolve_dtype(dtype)

    _validate_rmsf_replicas(results_a, "results_a")
    _validate_rmsf_replicas(results_b, "results_b")

    avg_a, sem_a = _average_rmsf_with_sem(results_a, dtype=resolved)
    avg_b, sem_b = _average_rmsf_with_sem(results_b, dtype=resolved)

    if indices_a is not None and indices_b is not None:
        if indices_a.shape[0] != indices_b.shape[0]:
            raise ValueError(
                f"indices_a and indices_b must have the same length, "
                f"got {indices_a.shape[0]} and {indices_b.shape[0]}."
            )
        avg_a = avg_a[indices_a]
        avg_b = avg_b[indices_b]
        if sem_a is not None:
            sem_a = sem_a[indices_a]
        if sem_b is not None:
            sem_b = sem_b[indices_b]

        if residue_ids is None:
            ref = results_a[0]
            if ref.residue_ids is not None:
                residue_ids = ref.residue_ids[indices_a]
    elif indices_a is not None or indices_b is not None:
        raise ValueError("indices_a and indices_b must both be provided or both be None.")
    else:
        if avg_a.shape[0] != avg_b.shape[0]:
            raise ValueError(
                f"Systems have different residue counts ({avg_a.shape[0]} vs "
                f"{avg_b.shape[0]}). Provide indices_a and indices_b to map "
                f"aligned positions."
            )
        if residue_ids is None:
            residue_ids = results_a[0].residue_ids

    delta = avg_b - avg_a

    # Combine SEMs in quadrature (independent systems)
    sem: NDArray[np.floating] | None = None
    if sem_a is not None and sem_b is not None:
        sem = np.sqrt(sem_a.astype(np.float64) ** 2 + sem_b.astype(np.float64) ** 2).astype(
            resolved
        )

    return DeltaRMSFResult(
        delta_rmsf_nm=delta,
        residue_ids=residue_ids,
        sem_nm=sem,
    )
