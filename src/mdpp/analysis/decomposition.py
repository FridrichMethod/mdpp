"""Dimensionality reduction and feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mdtraj as md
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA

from mdpp._dtype import resolve_dtype
from mdpp.core.trajectory import select_atom_indices


@dataclass(frozen=True, slots=True)
class DistanceFeatures:
    """Pairwise distance features (e.g. CA-CA distances)."""

    values: NDArray[np.floating]
    pairs: NDArray[np.int_]
    atom_indices: NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class TorsionFeatures:
    """Backbone torsion features."""

    values: NDArray[np.floating]
    labels: list[str]


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Principal component analysis outputs."""

    projections: NDArray[np.floating]
    components: NDArray[np.floating]
    explained_variance_ratio: NDArray[np.floating]
    feature_mean: NDArray[np.floating]
    feature_scale: NDArray[np.floating]
    model: PCA


@dataclass(frozen=True, slots=True)
class TICAResult:
    """Time-lagged independent component analysis outputs."""

    projections: NDArray[np.floating]
    lagtime: int
    model: Any


def _as_feature_matrix(
    features: ArrayLike,
) -> NDArray[np.float64]:
    """Validate and coerce a feature matrix.

    Always converts to float64 because downstream estimators (sklearn PCA,
    deeptime TICA) require float64 internally.

    Args:
        features: Input feature array.
    """
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
    dtype: type[np.floating] | None = None,
) -> TorsionFeatures:
    """Featurize backbone phi/psi torsions.

    Args:
        traj: Input trajectory.
        atom_selection: Optional atom selection before featurization.
        periodic: If True, return sin/cos embedding for periodic torsions.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        TorsionFeatures with values and labels.

    Raises:
        ValueError: If no phi/psi torsions are available.
    """
    resolved = resolve_dtype(dtype)

    if atom_selection is None:
        sliced = traj
    else:
        atom_indices = select_atom_indices(traj.topology, atom_selection)
        sliced = traj.atom_slice(atom_indices)

    _, phi = md.compute_phi(sliced)
    _, psi = md.compute_psi(sliced)

    blocks: list[NDArray[np.floating]] = []
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

    values = np.hstack(blocks).astype(resolved, copy=False)
    return TorsionFeatures(values=values, labels=labels)


type DistanceBackend = str


def _pairwise_distances_numba(
    xyz: NDArray[np.float32],
    pairs: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute non-periodic pairwise distances using a Numba-parallel kernel.

    Parallelises the frame loop using ``prange``, giving ~5x speedup over
    mdtraj's single-threaded C/SSE kernel on multi-core machines.

    **Limitations compared to mdtraj:**

    - No periodic boundary condition (minimum image convention) support.
      Use ``backend="mdtraj"`` if the trajectory has unit-cell information
      and PBC distances are required.
    - Returns float64 (mdtraj returns float32).

    Args:
        xyz: Coordinates of shape ``(n_frames, n_atoms, 3)``, typically
            ``traj.xyz`` in nm.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.

    Returns:
        Distances of shape ``(n_frames, n_pairs)`` in the same unit as
        *xyz* (nm for mdtraj trajectories).

    Raises:
        ValueError: If any pair index is out of range.
    """
    n_atoms = xyz.shape[1]
    if pairs.size > 0 and (np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError(
            f"atom_pairs must contain indices in [0, {n_atoms}), "
            f"got range [{int(pairs.min())}, {int(pairs.max())}]."
        )

    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _kernel(
        xyz: NDArray[np.float32], pairs: NDArray[np.int_]
    ) -> NDArray[np.float64]:  # pragma: no cover - JIT-compiled
        n_frames = xyz.shape[0]
        n_pairs = pairs.shape[0]
        out = np.empty((n_frames, n_pairs), dtype=np.float64)
        for f in prange(n_frames):
            for k in range(n_pairs):
                i = pairs[k, 0]
                j = pairs[k, 1]
                dx = float(xyz[f, i, 0]) - float(xyz[f, j, 0])
                dy = float(xyz[f, i, 1]) - float(xyz[f, j, 1])
                dz = float(xyz[f, i, 2]) - float(xyz[f, j, 2])
                out[f, k] = np.sqrt(dx * dx + dy * dy + dz * dz)
        return out

    return _kernel(xyz, pairs)


def _pairwise_distances_mdtraj(
    traj: md.Trajectory,
    pairs: NDArray[np.int_],
    *,
    periodic: bool,
    dtype: type[np.floating] | np.dtype[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute pairwise distances using mdtraj's optimised C/SSE kernel.

    Supports periodic boundary conditions via minimum image convention
    when the trajectory contains unit-cell information.

    Args:
        traj: Atom-sliced trajectory.
        pairs: 0-based atom-index pairs of shape ``(n_pairs, 2)``.
        periodic: Whether to apply minimum image convention.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        Distances of shape ``(n_frames, n_pairs)``.
    """
    resolved = resolve_dtype(dtype)
    return np.asarray(
        md.compute_distances(traj, pairs, periodic=periodic),
        dtype=resolved,
    )


def featurize_ca_distances(
    traj: md.Trajectory,
    *,
    atom_selection: str = "name CA",
    backend: DistanceBackend = "numba",
    periodic: bool = False,
    dtype: type[np.floating] | None = None,
) -> DistanceFeatures:
    """Featurize all pairwise distances between selected atoms.

    Computes the ``N*(N-1)/2`` pairwise distances for the selected atoms
    at each frame, producing a feature matrix suitable for PCA or TICA.

    Two backends are available:

    ``"numba"`` (default)
        Numba-parallel kernel that distributes frames across all CPU
        cores.  ~5x faster than mdtraj for non-periodic systems on
        multi-core machines.  **Does not support periodic boundary
        conditions** -- the *periodic* parameter is ignored.

    ``"mdtraj"``
        mdtraj's optimised C/SSE ``compute_distances``.  Supports
        periodic boundary conditions via minimum image convention when
        the trajectory contains unit-cell information.  Single-threaded.

    Args:
        traj: Input trajectory.
        atom_selection: MDTraj selection string for the atoms to include.
            Defaults to ``"name CA"`` for alpha-carbon distances.
        backend: Distance computation backend. One of ``"numba"``
            (default, fast, non-periodic) or ``"mdtraj"`` (PBC-capable).
        periodic: Whether to apply minimum image convention. Only
            effective with ``backend="mdtraj"``.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        DistanceFeatures with values, atom pairs, and atom indices.

    Raises:
        ValueError: If the selection matches fewer than 2 atoms, or an
            unknown backend is requested.
    """
    resolved = resolve_dtype(dtype)

    if backend not in ("numba", "mdtraj"):
        raise ValueError(f"Unknown backend {backend!r}. Use 'numba' or 'mdtraj'.")

    atom_indices = select_atom_indices(traj.topology, atom_selection)
    if atom_indices.size < 2:
        raise ValueError(
            f"At least 2 atoms are required for pairwise distances, "
            f"got {atom_indices.size} from selection {atom_selection!r}."
        )

    n_atoms = atom_indices.size
    pairs = np.array(
        [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)],
        dtype=np.int_,
    )
    sliced = traj.atom_slice(atom_indices)

    if backend == "numba":
        # Numba JIT kernel always outputs float64; cast to resolved dtype.
        values = np.asarray(_pairwise_distances_numba(sliced.xyz, pairs), dtype=resolved)
    else:
        values = _pairwise_distances_mdtraj(sliced, pairs, periodic=periodic, dtype=resolved)

    return DistanceFeatures(values=values, pairs=pairs, atom_indices=atom_indices)


def compute_pca(
    features: ArrayLike,
    *,
    n_components: int = 2,
    standardize: bool = True,
    dtype: type[np.floating] | None = None,
) -> PCAResult:
    """Compute PCA projection from feature vectors.

    Sklearn PCA requires float64 internally; the *dtype* parameter
    controls the dtype of the **output** arrays only.

    Args:
        features: Input feature matrix ``(n_samples, n_features)``.
        n_components: Number of principal components.
        standardize: Whether to z-score features before PCA.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        PCAResult containing projections and explained variance ratio.
    """
    resolved = resolve_dtype(dtype)
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
    projections = np.asarray(model.fit_transform(transformed), dtype=resolved)
    return PCAResult(
        projections=projections,
        components=np.asarray(model.components_, dtype=resolved),
        explained_variance_ratio=np.asarray(model.explained_variance_ratio_, dtype=resolved),
        feature_mean=np.asarray(feature_mean, dtype=resolved),
        feature_scale=np.asarray(feature_scale, dtype=resolved),
        model=model,
    )


def project_pca(
    features: ArrayLike,
    *,
    fitted: PCAResult,
    dtype: type[np.floating] | None = None,
) -> PCAResult:
    """Project new features using a previously fitted PCA.

    The features are standardized using the mean and scale from the
    fitted PCA, then transformed using its model.  This is the correct
    way to project a second dataset (e.g. a different system) onto the
    same principal component axes for direct comparison.

    Sklearn PCA requires float64 internally; the *dtype* parameter
    controls the dtype of the **output** arrays only.

    Args:
        features: Input feature matrix ``(n_samples, n_features)``.
            Must have the same number of features as the fitted PCA.
        fitted: PCAResult from a previous ``compute_pca`` call whose
            principal component axes will be used.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        PCAResult with projections onto the fitted PCA axes.  The
        ``components``, ``explained_variance_ratio``, ``feature_mean``,
        ``feature_scale``, and ``model`` are shared from ``fitted``.

    Raises:
        ValueError: If the feature dimension does not match the fitted PCA.
    """
    resolved = resolve_dtype(dtype)
    feature_matrix = _as_feature_matrix(features)
    expected_dim = fitted.feature_mean.shape[0]
    if feature_matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: got {feature_matrix.shape[1]}, "
            f"expected {expected_dim} (from fitted PCA)."
        )

    # Standardize in float64 for numerical stability, then cast output.
    feature_mean_f64 = np.asarray(fitted.feature_mean, dtype=np.float64)
    feature_scale_f64 = np.asarray(fitted.feature_scale, dtype=np.float64)
    transformed = (feature_matrix - feature_mean_f64) / feature_scale_f64
    projections = np.asarray(fitted.model.transform(transformed), dtype=resolved)
    return PCAResult(
        projections=projections,
        components=fitted.components,
        explained_variance_ratio=fitted.explained_variance_ratio,
        feature_mean=fitted.feature_mean,
        feature_scale=fitted.feature_scale,
        model=fitted.model,
    )


def compute_tica(
    features: ArrayLike,
    *,
    lagtime: int,
    n_components: int = 2,
    dtype: type[np.floating] | None = None,
) -> TICAResult:
    """Compute TICA projection from feature vectors.

    Deeptime requires float64 internally; the *dtype* parameter
    controls the dtype of the **output** arrays only.

    Args:
        features: Input feature matrix ``(n_samples, n_features)``.
        lagtime: Lag time in frames.
        n_components: Number of independent components.
        dtype: Output float dtype. If ``None``, uses the package default
            (see :func:`mdpp.set_default_dtype`).

    Returns:
        TICAResult containing projected coordinates and fitted model.
    """
    resolved = resolve_dtype(dtype)
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
    projections = np.asarray(model.transform(feature_matrix), dtype=resolved)
    return TICAResult(
        projections=projections,
        lagtime=lagtime,
        model=model,
    )
