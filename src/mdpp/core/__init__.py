"""Core trajectory I/O, selection helpers, and file parsers."""

from mdpp.core.parsers import read_edr, read_xvg
from mdpp.core.trajectory import (
    align_trajectory,
    load_trajectories,
    load_trajectory,
    residue_ids_from_indices,
    select_atom_indices,
    trajectory_time_ps,
)

__all__ = [
    "align_trajectory",
    "load_trajectories",
    "load_trajectory",
    "read_edr",
    "read_xvg",
    "residue_ids_from_indices",
    "select_atom_indices",
    "trajectory_time_ps",
]
