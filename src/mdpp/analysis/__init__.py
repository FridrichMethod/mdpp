"""Analysis subpackage for molecular dynamics trajectories."""

from mdpp.analysis.clustering import (
    DBSCAN,
    HDBSCAN,
    Gromos,
    Hierarchical,
    KMeans,
    MiniBatchKMeans,
    RegularSpace,
    compute_rmsd_matrix,
)
from mdpp.analysis.contacts import (
    compute_contact_frequency,
    compute_contacts,
    compute_native_contacts,
)
from mdpp.analysis.decomposition import (
    compute_pca,
    compute_tica,
    featurize_backbone_torsions,
    featurize_ca_distances,
    project_pca,
)
from mdpp.analysis.distance import compute_distances, compute_minimum_distance
from mdpp.analysis.dssp import compute_dssp
from mdpp.analysis.fes import compute_fes_2d, compute_fes_from_projection
from mdpp.analysis.hbond import compute_hbonds, format_hbond_triplets
from mdpp.analysis.metrics import (
    compute_dccm,
    compute_delta_rmsf,
    compute_radius_of_gyration,
    compute_rmsd,
    compute_rmsf,
    compute_sasa,
)

__all__ = [
    "DBSCAN",
    "HDBSCAN",
    "Gromos",
    "Hierarchical",
    "KMeans",
    "MiniBatchKMeans",
    "RegularSpace",
    "compute_contact_frequency",
    "compute_contacts",
    "compute_dccm",
    "compute_delta_rmsf",
    "compute_distances",
    "compute_dssp",
    "compute_fes_2d",
    "compute_fes_from_projection",
    "compute_hbonds",
    "compute_minimum_distance",
    "compute_native_contacts",
    "compute_pca",
    "compute_radius_of_gyration",
    "compute_rmsd",
    "compute_rmsd_matrix",
    "compute_rmsf",
    "compute_sasa",
    "compute_tica",
    "featurize_backbone_torsions",
    "featurize_ca_distances",
    "format_hbond_triplets",
    "project_pca",
]
