"""Core structure and dynamics metrics computed from trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp.core.trajectory import (
    align_trajectory,
    residue_ids_from_indices,
    select_atom_indices,
    trajectory_time_ps,
)


@dataclass(frozen=True, slots=True)
class RMSDResult:
    """RMSD time series."""

    time_ps: NDArray[np.float64]
    rmsd_nm: NDArray[np.float64]
    atom_indices: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.float64]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def rmsd_angstrom(self) -> NDArray[np.float64]:
        """Return RMSD values in Angstrom."""
        return self.rmsd_nm * 10.0


@dataclass(frozen=True, slots=True)
class RMSFResult:
    """Per-atom RMSF values."""

    rmsf_nm: NDArray[np.float64]
    atom_indices: NDArray[np.int_]
    residue_ids: NDArray[np.int_] | None

    @property
    def rmsf_angstrom(self) -> NDArray[np.float64]:
        """Return RMSF values in Angstrom."""
        return self.rmsf_nm * 10.0


@dataclass(frozen=True, slots=True)
class DeltaRMSFResult:
    """Per-residue RMSF difference between two systems (B minus A).

    Averaging is done in MSF (mean-square fluctuation) space: per-residue
    RMSF^2 values are averaged across replicas, then the square root is
    taken.  The delta is computed on the resulting average RMSF values.
    """

    delta_rmsf_nm: NDArray[np.float64]
    residue_ids: NDArray[np.int_] | None

    @property
    def delta_rmsf_angstrom(self) -> NDArray[np.float64]:
        """Return delta-RMSF values in Angstrom."""
        return self.delta_rmsf_nm * 10.0


@dataclass(frozen=True, slots=True)
class DCCMResult:
    """Dynamic cross-correlation matrix."""

    correlation: NDArray[np.float64]
    atom_indices: NDArray[np.int_]
    residue_ids: NDArray[np.int_] | None


@dataclass(frozen=True, slots=True)
class SASAResult:
    """Solvent accessible surface area."""

    time_ps: NDArray[np.float64]
    values_nm2: NDArray[np.float64]
    atom_indices: NDArray[np.int_]
    mode: str
    residue_ids: NDArray[np.int_] | None

    @property
    def time_ns(self) -> NDArray[np.float64]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def total_nm2(self) -> NDArray[np.float64]:
        """Return summed SASA for each frame."""
        return np.sum(self.values_nm2, axis=1)


@dataclass(frozen=True, slots=True)
class RadiusOfGyrationResult:
    """Radius of gyration time series."""

    time_ps: NDArray[np.float64]
    radius_gyration_nm: NDArray[np.float64]
    atom_indices: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.float64]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0

    @property
    def radius_gyration_angstrom(self) -> NDArray[np.float64]:
        """Return radius of gyration values in Angstrom."""
        return self.radius_gyration_nm * 10.0


def compute_rmsd(
    traj: md.Trajectory,
    *,
    atom_selection: str = "backbone",
    reference_frame: int = 0,
    align: bool = True,
    align_selection: str | None = None,
    timestep_ps: float | None = None,
) -> RMSDResult:
    """Compute RMSD over time.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used in RMSD calculation.
        reference_frame: Reference frame index.
        align: Whether to align the trajectory before RMSD.
        align_selection: Optional atom selection for alignment.
        timestep_ps: Optional time step in ps to override trajectory time.

    Returns:
        RMSDResult containing time and RMSD.
    """
    working = (
        align_trajectory(
            traj,
            atom_selection=align_selection or atom_selection,
            reference_frame=reference_frame,
            inplace=False,
        )
        if align
        else traj
    )
    atom_indices = select_atom_indices(working.topology, atom_selection)
    rmsd_nm = np.asarray(
        md.rmsd(
            working,
            working,
            frame=reference_frame,
            atom_indices=atom_indices,
            precentered=False,
        ),
        dtype=np.float64,
    )
    return RMSDResult(
        time_ps=trajectory_time_ps(working, timestep_ps=timestep_ps),
        rmsd_nm=rmsd_nm,
        atom_indices=atom_indices,
    )


def compute_rmsf(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    align: bool = True,
    align_selection: str | None = None,
    reference_frame: int = 0,
) -> RMSFResult:
    """Compute per-atom RMSF from positional fluctuations.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms included in RMSF calculation.
        align: Whether to align trajectory before RMSF.
        align_selection: Optional atom selection used for alignment.
        reference_frame: Reference frame index for alignment.

    Returns:
        RMSFResult with atom and residue mapping.
    """
    working = (
        align_trajectory(
            traj,
            atom_selection=align_selection or atom_selection,
            reference_frame=reference_frame,
            inplace=False,
        )
        if align
        else traj
    )
    atom_indices = select_atom_indices(working.topology, atom_selection)
    positions_nm = working.xyz[:, atom_indices, :]
    mean_positions_nm = np.mean(positions_nm, axis=0)
    squared_displacements = np.sum((positions_nm - mean_positions_nm) ** 2, axis=2)
    rmsf_nm = np.sqrt(np.mean(squared_displacements, axis=0, dtype=np.float64))
    residue_ids = residue_ids_from_indices(working.topology, atom_indices)
    return RMSFResult(
        rmsf_nm=np.asarray(rmsf_nm, dtype=np.float64),
        atom_indices=atom_indices,
        residue_ids=residue_ids,
    )


def compute_dccm(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    align: bool = True,
    align_selection: str | None = None,
    reference_frame: int = 0,
) -> DCCMResult:
    """Compute dynamic cross-correlation matrix (DCCM).

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used in DCCM.
        align: Whether to align trajectory first.
        align_selection: Optional atom selection used for alignment.
        reference_frame: Reference frame index for alignment.

    Returns:
        DCCMResult with correlation matrix and residue IDs.
    """
    if traj.n_frames < 2:
        raise ValueError("DCCM requires at least two frames.")

    working = (
        align_trajectory(
            traj,
            atom_selection=align_selection or atom_selection,
            reference_frame=reference_frame,
            inplace=False,
        )
        if align
        else traj
    )
    atom_indices = select_atom_indices(working.topology, atom_selection)
    positions_nm = working.xyz[:, atom_indices, :]
    mean_positions_nm = np.mean(positions_nm, axis=0)
    fluctuation_nm = positions_nm - mean_positions_nm

    covariance = np.einsum("fid,fjd->ij", fluctuation_nm, fluctuation_nm) / float(working.n_frames)
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    normalization = np.outer(standard_deviation, standard_deviation)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / normalization
    correlation = np.asarray(correlation, dtype=np.float64)
    correlation[~np.isfinite(correlation)] = 0.0
    np.fill_diagonal(correlation, 1.0)

    residue_ids = residue_ids_from_indices(working.topology, atom_indices)
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
) -> SASAResult:
    """Compute solvent-accessible surface area via Shrake-Rupley.

    Args:
        traj: Input trajectory.
        atom_selection: Optional atom selection before SASA.
        mode: Either ``"atom"`` or ``"residue"``.
        probe_radius: Probe radius in nm.
        n_sphere_points: Number of sphere points per atom.
        timestep_ps: Optional timestep override in ps.

    Returns:
        SASAResult containing frame-resolved values.
    """
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
        dtype=np.float64,
    )
    residue_ids = None
    if mode == "residue":
        residue_ids = np.array(
            [residue.resSeq for residue in sliced.topology.residues],
            dtype=np.int_,
        )

    return SASAResult(
        time_ps=trajectory_time_ps(sliced, timestep_ps=timestep_ps),
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
) -> RadiusOfGyrationResult:
    """Compute radius of gyration over time.

    Args:
        traj: Input trajectory.
        atom_selection: Atom selection used to compute radius of gyration.
        timestep_ps: Optional timestep override in ps.

    Returns:
        RadiusOfGyrationResult with per-frame values.
    """
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    sliced = traj.atom_slice(atom_indices)
    rg_nm = np.asarray(md.compute_rg(sliced), dtype=np.float64)
    return RadiusOfGyrationResult(
        time_ps=trajectory_time_ps(sliced, timestep_ps=timestep_ps),
        radius_gyration_nm=rg_nm,
        atom_indices=atom_indices,
    )


def _average_rmsf_nm(results: list[RMSFResult]) -> NDArray[np.float64]:
    """Average RMSF values across replicas in MSF space.

    Returns the per-residue average RMSF in nm, computed as
    ``sqrt(mean(RMSF^2))``.
    """
    msf_stack = np.stack([r.rmsf_nm**2 for r in results])
    return np.sqrt(np.mean(msf_stack, axis=0)).astype(np.float64)


def compute_delta_rmsf(
    results_a: list[RMSFResult],
    results_b: list[RMSFResult],
    *,
    indices_a: NDArray[np.int_] | None = None,
    indices_b: NDArray[np.int_] | None = None,
    residue_ids: NDArray[np.int_] | None = None,
) -> DeltaRMSFResult:
    """Compute per-residue RMSF difference between two systems.

    The RMSF for each system is first averaged across replicas in MSF space
    (``sqrt(mean(RMSF^2))``), then the delta is taken as B minus A.
    Positive values indicate that system B is more flexible.

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

    Returns:
        DeltaRMSFResult with the per-residue difference.

    Raises:
        ValueError: If input lists are empty, replicas within a system
            have inconsistent lengths, index arrays differ in length, or
            unindexed systems have different residue counts.
    """
    if not results_a:
        raise ValueError("results_a must not be empty.")
    if not results_b:
        raise ValueError("results_b must not be empty.")

    sizes_a = {r.rmsf_nm.size for r in results_a}
    if len(sizes_a) > 1:
        raise ValueError(f"results_a replicas have inconsistent sizes: {sizes_a}.")
    sizes_b = {r.rmsf_nm.size for r in results_b}
    if len(sizes_b) > 1:
        raise ValueError(f"results_b replicas have inconsistent sizes: {sizes_b}.")

    avg_a = _average_rmsf_nm(results_a)
    avg_b = _average_rmsf_nm(results_b)

    if indices_a is not None and indices_b is not None:
        if indices_a.shape[0] != indices_b.shape[0]:
            raise ValueError(
                f"indices_a and indices_b must have the same length, "
                f"got {indices_a.shape[0]} and {indices_b.shape[0]}."
            )
        avg_a = avg_a[indices_a]
        avg_b = avg_b[indices_b]

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
    return DeltaRMSFResult(
        delta_rmsf_nm=delta,
        residue_ids=residue_ids,
    )
