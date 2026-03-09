"""Dimensionality reduction and feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mdtraj as md
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

from mdpp.analysis.io import select_atom_indices


@dataclass(frozen=True, slots=True)
class TorsionFeatures:
    """Backbone torsion features."""

    values: NDArray[np.float64]
    labels: list[str]


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Principal component analysis outputs."""

    projections: NDArray[np.float64]
    components: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    model: PCA


@dataclass(frozen=True, slots=True)
class TICAResult:
    """Time-lagged independent component analysis outputs."""

    projections: NDArray[np.float64]
    lagtime: int
    model: Any


def _as_feature_matrix(features: ArrayLike) -> NDArray[np.float64]:
    """Validate and coerce a feature matrix."""
    feature_matrix = np.asarray(features, dtype=np.float64)
    if feature_matrix.ndim != 2:
        raise ValueError("features must be a 2D array with shape (n_samples, n_features).")
    if feature_matrix.shape[0] < 2:
        raise ValueError("features must contain at least two samples.")
    if feature_matrix.shape[1] < 1:
        raise ValueError("features must contain at least one feature.")
    return feature_matrix


def featurize_backbone_torsions(
    traj: md.Trajectory,
    *,
    atom_selection: str | None = "protein",
    periodic: bool = True,
) -> TorsionFeatures:
    """Featurize backbone phi/psi torsions.

    Args:
        traj: Input trajectory.
        atom_selection: Optional atom selection before featurization.
        periodic: If True, return sin/cos embedding for periodic torsions.

    Returns:
        TorsionFeatures with values and labels.

    Raises:
        ValueError: If no phi/psi torsions are available.
    """
    if atom_selection is None:
        sliced = traj
    else:
        atom_indices = select_atom_indices(traj.topology, atom_selection)
        sliced = traj.atom_slice(atom_indices)

    _, phi = md.compute_phi(sliced)
    _, psi = md.compute_psi(sliced)

    blocks: list[NDArray[np.float64]] = []
    labels: list[str] = []

    phi_count = int(phi.shape[1]) if phi.ndim == 2 else 0
    psi_count = int(psi.shape[1]) if psi.ndim == 2 else 0

    if periodic:
        if phi_count > 0:
            blocks.extend([np.cos(phi), np.sin(phi)])
            labels.extend([f"cos(phi_{index})" for index in range(phi_count)])
            labels.extend([f"sin(phi_{index})" for index in range(phi_count)])
        if psi_count > 0:
            blocks.extend([np.cos(psi), np.sin(psi)])
            labels.extend([f"cos(psi_{index})" for index in range(psi_count)])
            labels.extend([f"sin(psi_{index})" for index in range(psi_count)])
    else:
        if phi_count > 0:
            blocks.append(phi)
            labels.extend([f"phi_{index}" for index in range(phi_count)])
        if psi_count > 0:
            blocks.append(psi)
            labels.extend([f"psi_{index}" for index in range(psi_count)])

    if not blocks:
        raise ValueError("No phi/psi torsions were found for the selected atoms.")

    values = np.hstack(blocks).astype(np.float64, copy=False)
    return TorsionFeatures(values=values, labels=labels)


def compute_pca(
    features: ArrayLike,
    *,
    n_components: int = 2,
    standardize: bool = True,
) -> PCAResult:
    """Compute PCA projection from feature vectors.

    Args:
        features: Input feature matrix ``(n_samples, n_features)``.
        n_components: Number of principal components.
        standardize: Whether to z-score features before PCA.

    Returns:
        PCAResult containing projections and explained variance ratio.
    """
    feature_matrix = _as_feature_matrix(features)
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if n_components > feature_matrix.shape[1]:
        raise ValueError("n_components cannot exceed the feature dimension.")

    feature_mean = np.mean(feature_matrix, axis=0)
    if standardize:
        feature_scale = np.std(feature_matrix, axis=0)
        feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
        transformed = (feature_matrix - feature_mean) / feature_scale
    else:
        feature_scale = np.ones(feature_matrix.shape[1], dtype=np.float64)
        transformed = feature_matrix - feature_mean

    model = PCA(n_components=n_components)
    projections = np.asarray(model.fit_transform(transformed), dtype=np.float64)
    return PCAResult(
        projections=projections,
        components=np.asarray(model.components_, dtype=np.float64),
        explained_variance_ratio=np.asarray(model.explained_variance_ratio_, dtype=np.float64),
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_scale=np.asarray(feature_scale, dtype=np.float64),
        model=model,
    )


def compute_tica(
    features: ArrayLike,
    *,
    lagtime: int,
    n_components: int = 2,
) -> TICAResult:
    """Compute TICA projection from feature vectors.

    Args:
        features: Input feature matrix ``(n_samples, n_features)``.
        lagtime: Lag time in frames.
        n_components: Number of independent components.

    Returns:
        TICAResult containing projected coordinates and fitted model.
    """
    feature_matrix = _as_feature_matrix(features)
    if lagtime < 1:
        raise ValueError("lagtime must be >= 1.")
    if lagtime >= feature_matrix.shape[0]:
        raise ValueError("lagtime must be smaller than the number of samples.")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    from deeptime.decomposition import TICA

    estimator = TICA(lagtime=lagtime, dim=n_components)
    model = estimator.fit(feature_matrix).fetch_model()
    projections = np.asarray(model.transform(feature_matrix), dtype=np.float64)
    return TICAResult(
        projections=projections,
        lagtime=lagtime,
        model=model,
    )
