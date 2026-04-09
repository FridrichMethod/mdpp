"""Hydrogen-bond analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np
from numpy.typing import NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.core.trajectory import trajectory_time_ps


@dataclass(frozen=True, slots=True)
class HBondResult:
    """Hydrogen-bond detection results."""

    time_ps: NDArray[np.floating]
    triplets: NDArray[np.int_]
    presence: NDArray[np.bool_]
    count_per_frame: NDArray[np.int_]
    occupancy: NDArray[np.floating]
    method: str
    distance_cutoff_nm: float
    angle_cutoff_deg: float

    @property
    def time_ns(self) -> NDArray[np.floating]:
        """Return frame times in nanoseconds."""
        return self.time_ps / 1000.0


def format_hbond_triplets(
    topology: md.Topology,
    triplets: NDArray[np.int_],
) -> list[str]:
    """Format donor-hydrogen-acceptor triplets into readable labels.

    Args:
        topology: Trajectory topology.
        triplets: Integer array with shape ``(n_hbonds, 3)``.

    Returns:
        List of labels such as ``"ALA1:N-H ... GLU10:OE1"``.
    """
    labels: list[str] = []
    for donor_index, hydrogen_index, acceptor_index in triplets:
        donor_atom = topology.atom(int(donor_index))
        hydrogen_atom = topology.atom(int(hydrogen_index))
        acceptor_atom = topology.atom(int(acceptor_index))

        donor_label = f"{donor_atom.residue.name}{donor_atom.residue.resSeq}:{donor_atom.name}"
        acceptor_label = (
            f"{acceptor_atom.residue.name}{acceptor_atom.residue.resSeq}:{acceptor_atom.name}"
        )
        labels.append(f"{donor_label}-{hydrogen_atom.name} ... {acceptor_label}")
    return labels


def _presence_from_geometry(
    traj: md.Trajectory,
    triplets: NDArray[np.int_],
    *,
    periodic: bool,
    distance_cutoff_nm: float,
    angle_cutoff_deg: float,
) -> NDArray[np.bool_]:
    """Compute per-frame hydrogen-bond presence from geometric criteria."""
    if triplets.size == 0:
        return np.zeros((traj.n_frames, 0), dtype=np.bool_)

    ha_pairs = triplets[:, [1, 2]]
    distances_nm = md.compute_distances(traj, ha_pairs, periodic=periodic)
    angles_rad = md.compute_angles(traj, triplets, periodic=periodic)
    angle_cutoff_rad = np.deg2rad(angle_cutoff_deg)
    return (distances_nm <= distance_cutoff_nm) & (angles_rad >= angle_cutoff_rad)


def _triplets_from_wernet_nilsson(
    traj: md.Trajectory,
    *,
    exclude_water: bool,
    periodic: bool,
) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    """Build unique hydrogen-bond triplets and a per-frame presence matrix."""
    frame_triplets = md.wernet_nilsson(
        traj,
        exclude_water=exclude_water,
        periodic=periodic,
    )

    unique_triplets = sorted({
        tuple(int(atom_index) for atom_index in triplet)
        for frame_triplet in frame_triplets
        for triplet in frame_triplet
    })
    if not unique_triplets:
        empty_triplets = np.empty((0, 3), dtype=np.int_)
        empty_presence = np.zeros((traj.n_frames, 0), dtype=np.bool_)
        return empty_triplets, empty_presence

    triplets = np.asarray(unique_triplets, dtype=np.int_)
    presence = np.zeros((traj.n_frames, triplets.shape[0]), dtype=np.bool_)
    triplet_to_index = {
        tuple(int(atom_index) for atom_index in triplet): index
        for index, triplet in enumerate(triplets)
    }

    for frame_index, frame_triplet in enumerate(frame_triplets):
        for triplet in frame_triplet:
            key = tuple(int(atom_index) for atom_index in triplet)
            presence[frame_index, triplet_to_index[key]] = True
    return triplets, presence


def compute_hbonds(
    traj: md.Trajectory,
    *,
    method: str = "baker_hubbard",
    exclude_water: bool = True,
    periodic: bool = True,
    sidechain_only: bool = False,
    freq: float = 0.1,
    distance_cutoff_nm: float = 0.25,
    angle_cutoff_deg: float = 120.0,
    timestep_ps: float | None = None,
    dtype: DtypeArg = None,
) -> HBondResult:
    """Compute hydrogen bonds and per-frame counts.

    Args:
        traj: Input trajectory.
        method: Hydrogen bond method: ``"baker_hubbard"`` or ``"wernet_nilsson"``.
        exclude_water: Whether to ignore water-mediated hydrogen bonds.
        periodic: Whether to apply periodic boundary conditions.
        sidechain_only: For ``"baker_hubbard"``, restrict to sidechain interactions.
        freq: For ``"baker_hubbard"``, minimum occupancy fraction for returned bonds.
        distance_cutoff_nm: H...A distance cutoff used for presence matrix.
        angle_cutoff_deg: D-H...A angle cutoff used for presence matrix.
        timestep_ps: Optional frame timestep override in ps.
        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        HBondResult containing detected bonds, occupancy, and per-frame counts.
    """
    if method not in {"baker_hubbard", "wernet_nilsson"}:
        raise ValueError("method must be either 'baker_hubbard' or 'wernet_nilsson'.")
    if not 0.0 <= freq <= 1.0:
        raise ValueError("freq must be in [0, 1].")
    if distance_cutoff_nm <= 0.0:
        raise ValueError("distance_cutoff_nm must be positive.")
    if not 0.0 < angle_cutoff_deg <= 180.0:
        raise ValueError("angle_cutoff_deg must be in (0, 180].")

    if method == "baker_hubbard":
        triplets = np.asarray(
            md.baker_hubbard(
                traj,
                freq=freq,
                exclude_water=exclude_water,
                periodic=periodic,
                sidechain_only=sidechain_only,
            ),
            dtype=np.int_,
        )
        presence = _presence_from_geometry(
            traj,
            triplets,
            periodic=periodic,
            distance_cutoff_nm=distance_cutoff_nm,
            angle_cutoff_deg=angle_cutoff_deg,
        )
    else:
        triplets, presence = _triplets_from_wernet_nilsson(
            traj,
            exclude_water=exclude_water,
            periodic=periodic,
        )

    resolved = resolve_dtype(dtype)
    count_per_frame = np.sum(presence, axis=1, dtype=np.int_)
    occupancy = (
        np.mean(presence, axis=0) if presence.shape[1] > 0 else np.empty((0,), dtype=resolved)
    )
    return HBondResult(
        time_ps=trajectory_time_ps(traj, timestep_ps=timestep_ps, dtype=resolved),
        triplets=triplets,
        presence=presence,
        count_per_frame=np.asarray(count_per_frame, dtype=np.int_),
        occupancy=np.asarray(occupancy, dtype=resolved),
        method=method,
        distance_cutoff_nm=distance_cutoff_nm,
        angle_cutoff_deg=angle_cutoff_deg,
    )
