"""Analysis utilities for molecular dynamics trajectories."""

from mdpp.analysis.decomposition import (
    PCAResult,
    TICAResult,
    TorsionFeatures,
    compute_pca,
    compute_tica,
    featurize_backbone_torsions,
)
from mdpp.analysis.fes import FES2DResult, compute_fes_2d, compute_fes_from_projection
from mdpp.analysis.hbond import HBondResult, compute_hbonds, format_hbond_triplets
from mdpp.analysis.metrics import (
    DCCMResult,
    RadiusOfGyrationResult,
    RMSDResult,
    RMSFResult,
    SASAResult,
    compute_dccm,
    compute_radius_of_gyration,
    compute_rmsd,
    compute_rmsf,
    compute_sasa,
)
from mdpp.analysis.trajectory import (
    align_trajectory,
    load_trajectories,
    load_trajectory,
    residue_ids_from_indices,
    select_atom_indices,
    trajectory_time_ps,
)

__all__ = [
    "DCCMResult",
    "FES2DResult",
    "HBondResult",
    "PCAResult",
    "RMSDResult",
    "RMSFResult",
    "RadiusOfGyrationResult",
    "SASAResult",
    "TICAResult",
    "TorsionFeatures",
    "align_trajectory",
    "compute_dccm",
    "compute_fes_2d",
    "compute_fes_from_projection",
    "compute_hbonds",
    "compute_pca",
    "compute_radius_of_gyration",
    "compute_rmsd",
    "compute_rmsf",
    "compute_sasa",
    "compute_tica",
    "featurize_backbone_torsions",
    "format_hbond_triplets",
    "load_trajectories",
    "load_trajectory",
    "residue_ids_from_indices",
    "select_atom_indices",
    "trajectory_time_ps",
]
