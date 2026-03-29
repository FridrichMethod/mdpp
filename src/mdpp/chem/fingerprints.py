"""Molecular fingerprint generation and clustering utilities."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from tqdm.auto import trange

from mdpp.chem.similarity import (
    _VALID_CLUSTERING_METRICS,
    CLUSTERING_SIM_METRICS,
    PARALLEL_SIM_KERNELS,
    FingerPrint,
    calc_bulk_sim,
    calc_similarities,
)

FP_GENERATORS: dict[str, Callable[[Chem.rdchem.Mol], FingerPrint]] = {
    "morgan": AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint,
    "ecfp2": AllChem.GetMorganGenerator(radius=1, fpSize=1024).GetFingerprint,
    "ecfp4": AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint,
    "ecfp6": AllChem.GetMorganGenerator(radius=3, fpSize=1024).GetFingerprint,
    "maccs": Chem.rdMolDescriptors.GetMACCSKeysFingerprint,
    "rdkit": AllChem.GetRDKitFPGenerator().GetFingerprint,
    "atom_pair": AllChem.GetAtomPairGenerator().GetFingerprint,
    "topological_torsion": AllChem.GetTopologicalTorsionGenerator().GetFingerprint,
}

_VALID_FP_TYPES = ", ".join(f"'{k}'" for k in FP_GENERATORS)


@dataclass(frozen=True, slots=True)
class FingerprintClusteringResult:
    """Fingerprint-based Butina clustering output.

    Attributes:
        clusters: Cluster memberships sorted by size (largest first).
            Each tuple contains molecule indices belonging to that cluster.
        n_clusters: Total number of clusters.
    """

    clusters: tuple[tuple[int, ...], ...]
    n_clusters: int


def gen_fp(mol: Chem.rdchem.Mol, *, fp_type: str = "morgan") -> FingerPrint:
    """Generate a molecular fingerprint.

    Args:
        mol: An RDKit molecule.
        fp_type: Fingerprint type. One of 'morgan', 'ecfp2', 'ecfp4', 'ecfp6',
            'maccs', 'rdkit', 'atom_pair', 'topological_torsion'.

    Returns:
        A fingerprint bit vector.

    Raises:
        ValueError: If *fp_type* is not recognised.
    """
    key = fp_type.lower()
    if key not in FP_GENERATORS:
        raise ValueError(f"fp_type should be one of {_VALID_FP_TYPES}. Got '{fp_type}'.")
    return FP_GENERATORS[key](mol)


def _validate_clustering_metric(similarity_metric: str) -> None:
    """Raise ``ValueError`` if the metric is unsuitable for clustering."""
    if similarity_metric not in CLUSTERING_SIM_METRICS:
        raise ValueError(
            f"similarity_metric should be one of {_VALID_CLUSTERING_METRICS} for clustering. "
            f"'russel' is excluded because its self-similarity is not 1."
        )


def _butina_cluster(
    distances: np.ndarray,
    n: int,
    cutoff: float,
) -> FingerprintClusteringResult:
    """Run Butina clustering on a condensed distance array."""
    clusters = list(Butina.ClusterData(distances, n, 1 - cutoff, isDistData=True))
    clusters.sort(key=len, reverse=True)
    return FingerprintClusteringResult(
        clusters=tuple(clusters),
        n_clusters=len(clusters),
    )


def cluster_fps(
    fps: Sequence[FingerPrint],
    *,
    cutoff: float = 0.6,
    similarity_metric: str = "tanimoto",
) -> FingerprintClusteringResult:
    """Cluster fingerprints using RDKit bulk similarity and the Butina algorithm.

    Distances are computed as ``1 - similarity``.  Only metrics where
    ``sim(A, A) == 1`` (i.e. self-distance is zero) are valid for clustering.
    ``'russel'`` (Russell-Rao) is excluded because its self-similarity equals
    ``popcount / n_bits``, which is generally less than 1.

    Args:
        fps: Fingerprint bit vectors.
        cutoff: Similarity cutoff for clustering.
        similarity_metric: Similarity metric name.  Must be one of the metrics
            listed in ``CLUSTERING_SIM_METRICS``.

    Returns:
        Clustering result with sorted cluster tuples and cluster count.

    Raises:
        ValueError: If *similarity_metric* is not in ``CLUSTERING_SIM_METRICS``.
    """
    _validate_clustering_metric(similarity_metric)

    n = len(fps)
    if n <= 1:
        clusters = tuple((i,) for i in range(n))
        return FingerprintClusteringResult(clusters=clusters, n_clusters=n)

    logging.info("Calculating similarities...")
    similarities = np.concatenate([
        calc_bulk_sim(fps[i], fps[:i], similarity_metric=similarity_metric) for i in trange(1, n)
    ])

    logging.info("Clustering...")
    return _butina_cluster(1 - similarities, n, cutoff)


def cluster_fps_parallel(
    fps: np.ndarray,
    *,
    cutoff: float = 0.6,
    similarity_metric: str = "tanimoto",
) -> FingerprintClusteringResult:
    """Cluster fingerprints using Numba-parallel similarity and the Butina algorithm.

    Distances are computed as ``1 - similarity``.  Only metrics where
    ``sim(A, A) == 1`` (i.e. self-distance is zero) are valid for clustering.
    ``'russel'`` (Russell-Rao) is excluded because its self-similarity equals
    ``popcount / n_bits``, which is generally less than 1.

    Args:
        fps: 2D numpy array of shape ``(n_mols, n_bits)`` with binary fingerprints.
        cutoff: Similarity cutoff for clustering.
        similarity_metric: Similarity metric name.  Must be one of the metrics
            listed in ``CLUSTERING_SIM_METRICS``.

    Returns:
        Clustering result with sorted cluster tuples and cluster count.

    Raises:
        ValueError: If *similarity_metric* is not in ``CLUSTERING_SIM_METRICS``
            or *fps* is not a 2D array.
    """
    _validate_clustering_metric(similarity_metric)

    if fps.ndim != 2:
        raise ValueError("fps should be a 2D numpy.ndarray with shape (n_mols, n_bits).")

    n = len(fps)
    if n <= 1:
        clusters = tuple((i,) for i in range(n))
        return FingerprintClusteringResult(clusters=clusters, n_clusters=n)

    logging.info("Calculating similarities...")
    similarities = calc_similarities(fps, PARALLEL_SIM_KERNELS[similarity_metric])

    logging.info("Clustering...")
    return _butina_cluster(1 - similarities, n, cutoff)
