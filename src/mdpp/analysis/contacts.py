"""Contact analysis for molecular dynamics trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.core.trajectory import trajectory_time_ps


@dataclass(frozen=True, slots=True)
class ContactResult:
    """Per-frame inter-residue contact distances."""

    time_ps: NDArray[np.floating]
    distances_nm: NDArray[np.floating]
    residue_pairs: NDArray[np.int_]

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0


@dataclass(frozen=True, slots=True)
class NativeContactResult:
    """Fraction of native contacts (Q) over time."""

    time_ps: NDArray[np.floating]
    fraction: NDArray[np.floating]
    native_pairs: NDArray[np.int_]
    cutoff_nm: float

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0


def compute_contacts(
    traj: md.Trajectory,
    *,
    contacts: str | NDArray[np.int_] = "all",
    scheme: str = "closest-heavy",
    periodic: bool = True,
    timestep_ps: float | None = None,
    dtype: DtypeArg = None,
) -> ContactResult:
    """Compute inter-residue contact distances over time.

    Args:
        traj: Input trajectory.
        contacts: Residue pairs to monitor. ``"all"`` computes all pairs;
            otherwise an ``(n_pairs, 2)`` integer array of residue index pairs.
        scheme: Contact scheme passed to ``mdtraj.compute_contacts``.
            One of ``"closest-heavy"``, ``"closest"``, ``"ca"``, or ``"sidechain-heavy"``.
        periodic: Whether to apply periodic boundary conditions.
        timestep_ps: Optional frame timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        ContactResult with per-frame distances and residue pair indices.
    """
    resolved = resolve_dtype(dtype)
    distances, pairs = md.compute_contacts(
        traj,
        contacts=contacts,
        scheme=scheme,
        periodic=periodic,
    )
    return ContactResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps, dtype=resolved),
        distances_nm=np.asarray(distances, dtype=resolved),
        residue_pairs=np.asarray(pairs, dtype=np.int_),
    )


def compute_contact_frequency(
    traj: md.Trajectory,
    *,
    cutoff_nm: float = 0.45,
    scheme: str = "closest-heavy",
    periodic: bool = True,
    dtype: DtypeArg = None,
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """Compute the fraction of frames each residue pair is in contact.

    Args:
        traj: Input trajectory.
        cutoff_nm: Distance threshold in nm below which a contact is counted.
        scheme: Contact scheme passed to ``mdtraj.compute_contacts``.
        periodic: Whether to apply periodic boundary conditions.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        A tuple of ``(frequency, residue_pairs)`` where ``frequency`` has
        shape ``(n_pairs,)`` with values in ``[0, 1]`` and ``residue_pairs``
        has shape ``(n_pairs, 2)``.
    """
    resolved = resolve_dtype(dtype)
    distances, pairs = md.compute_contacts(
        traj,
        contacts="all",
        scheme=scheme,
        periodic=periodic,
    )
    frequency = np.mean(distances < cutoff_nm, axis=0)
    return np.asarray(frequency, dtype=resolved), np.asarray(pairs, dtype=np.int_)


def compute_native_contacts(
    traj: md.Trajectory,
    *,
    reference_frame: int = 0,
    cutoff_nm: float = 0.45,
    scheme: str = "closest-heavy",
    periodic: bool = True,
    timestep_ps: float | None = None,
    dtype: DtypeArg = None,
) -> NativeContactResult:
    """Compute the fraction of native contacts Q(t) over time.

    Native contacts are residue pairs that are within ``cutoff_nm`` in
    the reference frame. Q(t) is the fraction of those pairs that remain
    in contact at each frame.

    Args:
        traj: Input trajectory.
        reference_frame: Frame index defining native contacts.
        cutoff_nm: Distance threshold in nm for a contact.
        scheme: Contact scheme for ``mdtraj.compute_contacts``.
        periodic: Whether to apply periodic boundary conditions.
        timestep_ps: Optional frame timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        NativeContactResult with per-frame Q values.

    Raises:
        ValueError: If ``reference_frame`` is out of range or no native
            contacts are found.
    """
    if not 0 <= reference_frame < traj.n_frames:
        raise ValueError(
            f"reference_frame must be in [0, {traj.n_frames - 1}], got {reference_frame}."
        )

    distances, pairs = md.compute_contacts(
        traj,
        contacts="all",
        scheme=scheme,
        periodic=periodic,
    )
    ref_distances = distances[reference_frame]
    native_mask = ref_distances < cutoff_nm
    if not np.any(native_mask):
        raise ValueError(
            f"No native contacts found at frame {reference_frame} with cutoff {cutoff_nm} nm."
        )

    native_pairs = pairs[native_mask]
    native_distances = distances[:, native_mask]
    resolved = resolve_dtype(dtype)
    fraction = np.mean(native_distances < cutoff_nm, axis=1)

    return NativeContactResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps, dtype=resolved),
        fraction=np.asarray(fraction, dtype=resolved),
        native_pairs=np.asarray(native_pairs, dtype=np.int_),
        cutoff_nm=float(cutoff_nm),
    )
