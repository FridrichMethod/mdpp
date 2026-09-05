"""Core trajectory I/O, selection helpers, and file parsers."""

from mdpp.core.fepp import (
    extract_edge_mappings,
    extract_fep_results,
    fep_mapping_fingerprint,
    read_fep_edges,
)
from mdpp.core.fepp_results import (
    Edge,
    ParsedEdgeFile,
    read_edge_file,
)
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
    "Edge",
    "ParsedEdgeFile",
    "align_trajectory",
    "extract_edge_mappings",
    "extract_fep_results",
    "fep_mapping_fingerprint",
    "load_trajectories",
    "load_trajectory",
    "read_edge_file",
    "read_edr",
    "read_fep_edges",
    "read_xvg",
    "residue_ids_from_indices",
    "select_atom_indices",
    "trajectory_time_ps",
]
