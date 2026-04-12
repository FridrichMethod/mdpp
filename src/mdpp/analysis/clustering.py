"""Conformational clustering from RMSD matrices.

Each clustering algorithm is a frozen dataclass configured at
construction time and invoked as a callable::

    result = Gromos(cutoff_nm=0.2)(rmsd_matrix)
    result = DBSCAN(eps=0.15, min_samples=5)(rmsd_matrix)
    result = KMeans(n_clusters=10)(pca.projections)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mdtraj as md
import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from mdpp._dtype import resolve_dtype
from mdpp._types import DtypeArg
from mdpp.analysis._backends import RMSDBackend
from mdpp.analysis._backends._rmsd_matrix import rmsd_matrix_backends
from mdpp.core.trajectory import select_atom_indices

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RMSDMatrixResult:
    """Pairwise RMSD matrix between trajectory frames."""

    rmsd_matrix_nm: NDArray[np.floating]
    atom_indices: NDArray[np.int_]

    @property
    def rmsd_matrix_angstrom(self) -> NDArray[np.floating]:
        """Return the RMSD matrix in Angstrom.

        Note:
            Each access allocates a new ``(n_frames, n_frames)`` array.
            Cache the result in a local variable if you need it more
            than once -- at 120k frames this is ~54 GB per call.
        """
        return self.rmsd_matrix_nm * 10.0


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Conformational clustering output."""

    labels: NDArray[np.int_]
    n_clusters: int
    medoid_frames: NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class FeatureClusteringResult:
    """Clustering result from feature-vector-based methods."""

    labels: NDArray[np.int_]
    n_clusters: int
    cluster_centers: NDArray[np.floating]
    medoid_frames: NDArray[np.int_]
    inertia: float


# ---------------------------------------------------------------------------
# Public API: RMSD matrix computation
# ---------------------------------------------------------------------------


def compute_rmsd_matrix(
    traj: md.Trajectory,
    *,
    atom_selection: str = "backbone",
    backend: RMSDBackend = "mdtraj",
    dtype: DtypeArg = None,
) -> RMSDMatrixResult:
    """Compute an all-vs-all RMSD matrix between trajectory frames.

    Args:
        traj: Input trajectory.
        atom_selection: Atoms used for RMSD calculation.
        backend: Computation backend.  Defaults to ``"mdtraj"`` for
            API consistency with other analysis functions; switch to a
            faster backend explicitly when performance matters.

            - ``"mdtraj"`` (default) -- mdtraj precentered RMSD loop (CPU).
            - ``"numba"`` -- Numba-parallel QCP kernel (CPU, 50-200x faster).
            - ``"torch"`` -- PyTorch einsum + QCP (CUDA/CPU, float32).
            - ``"jax"`` -- JAX einsum + QCP (GPU/TPU/CPU, float32).
            - ``"cupy"`` -- CuPy einsum + QCP (CUDA, float32).

        dtype: Output float dtype. If ``None``, uses the package default.

    Returns:
        RMSDMatrixResult with a symmetric ``(n_frames, n_frames)`` matrix.

    Raises:
        ValueError: If an unsupported backend is specified.
        ImportError: If the requested backend package is not installed.

    Memory note:
        Every backend returns its native ``float32`` output matrix
        (the numba kernel uses float64 accumulators internally but
        stores float32 in the result buffer; GPU kernels compute in
        float32 end-to-end).  This wrapper casts with ``copy=False``
        so when the resolved dtype is float32 (the package default)
        there is **no second copy** of the ``(n_frames, n_frames)``
        matrix.  For a 120k-frame trajectory this saves ~115 GB of
        peak RAM versus the old "cast to float64 for the Protocol
        contract, then cast back" path.  Passing ``dtype=np.float64``
        still forces a one-time upcast.
    """
    resolved = resolve_dtype(dtype)
    atom_indices = select_atom_indices(traj.topology, atom_selection)
    compute_fn = rmsd_matrix_backends.get(backend)
    rmsd_matrix = compute_fn(traj, atom_indices)

    # Force exact symmetry: some backends (mdtraj, torch) produce tiny
    # asymmetry (~1e-7 nm) that causes sklearn HDBSCAN to reject the
    # matrix.  The Numba kernel averages (i,j) and (j,i) in-place with
    # zero extra allocation.
    rmsd_matrix = np.ascontiguousarray(rmsd_matrix)
    _symmetrize_inplace(rmsd_matrix)

    # ``copy=False`` is critical at large N: when the backend's native
    # dtype already matches ``resolved`` we reuse the same buffer
    # instead of allocating a second ~N^2 matrix.
    return RMSDMatrixResult(
        rmsd_matrix_nm=rmsd_matrix.astype(resolved, copy=False),
        atom_indices=atom_indices,
    )


# ---------------------------------------------------------------------------
# Numba JIT kernels
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _symmetrize_inplace(
    matrix: NDArray[np.floating],
) -> None:  # pragma: no cover - JIT-compiled
    """Average upper and lower triangles in place (zero allocation).

    RMSD is a true metric so the matrix must be symmetric, but some
    backends (mdtraj, torch) produce tiny asymmetry (~1e-7 nm) due to
    floating-point ordering in the superposition loop.  This kernel
    forces exact symmetry so downstream consumers (e.g. sklearn HDBSCAN)
    that check ``allclose(X, X.T)`` do not reject the matrix.

    Parallelised over rows; each thread owns a disjoint set of (i, j)
    pairs so writes are race-free.
    """
    n = matrix.shape[0]
    for i in prange(n):
        for j in range(i + 1, n):
            avg = (matrix[i, j] + matrix[j, i]) / 2
            matrix[i, j] = avg
            matrix[j, i] = avg


@njit(parallel=True, cache=True)
def _gromos_initial_counts(
    rmsd_matrix: NDArray[np.floating],
    cutoff_nm: float,
) -> NDArray[np.int64]:  # pragma: no cover - JIT-compiled
    """Count each row's neighbors within ``cutoff_nm`` (inclusive).

    Parallel over rows.  Each row index is owned by exactly one thread
    so the write to ``counts[i]`` is race-free.
    """
    n = rmsd_matrix.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        c = 0
        for j in range(n):
            if rmsd_matrix[i, j] <= cutoff_nm:
                c += 1
        counts[i] = c
    return counts


@njit(cache=True)
def _gromos_loop(
    rmsd_matrix: NDArray[np.floating],
    counts: NDArray[np.int64],
    cutoff_nm: float,
) -> tuple[NDArray[np.int64], int, NDArray[np.int64]]:  # pragma: no cover - JIT-compiled
    """Run the incremental GROMOS greedy assignment loop.

    On entry ``counts[i]`` is the number of rows within ``cutoff_nm``
    of row ``i`` (including ``i`` itself).  The loop repeatedly picks
    the unassigned row with the largest count as the next cluster
    centre, assigns every unassigned neighbour of that centre, and
    **incrementally** decrements ``counts`` so that ``counts[i]`` is
    always the number of *still-unassigned* neighbours of ``i``.

    The incremental update is the key speed win: for each newly
    assigned member ``m``, we scan its row once (the matrix is
    symmetric, so its column is its row) and decrement ``counts`` at
    every neighbour.  This is ``O(|members| * n)`` per cluster
    instead of ``O(n^2)``.  For 120k frames with ~100 clusters the
    whole loop runs in ~10 seconds on a modern CPU, versus ~100
    minutes for the original fully-recomputing Python implementation.

    When multiple unassigned frames have the same neighbour count,
    the one with the smallest frame index is chosen as the cluster
    centre (deterministic tie-breaking).

    Warning:
        ``counts`` is **mutated in place** -- the caller must not
        reuse the array after this function returns.

    Returns:
        ``(labels, n_clusters, medoids)`` as three numpy arrays.
    """
    n = rmsd_matrix.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=np.bool_)
    medoids_buf = np.empty(n, dtype=np.int64)
    members_buf = np.empty(n, dtype=np.int64)
    cluster_id = 0

    while True:
        best = -1
        best_count = -1
        for i in range(n):
            if not assigned[i] and counts[i] > best_count:
                best_count = counts[i]
                best = i
        if best < 0:
            break

        n_members = 0
        for j in range(n):
            if not assigned[j] and rmsd_matrix[best, j] <= cutoff_nm:
                members_buf[n_members] = j
                n_members += 1
                labels[j] = cluster_id
                assigned[j] = True

        for k in range(n_members):
            m = members_buf[k]
            for i in range(n):
                if rmsd_matrix[m, i] <= cutoff_nm:
                    counts[i] -= 1

        medoids_buf[cluster_id] = best
        cluster_id += 1

    return labels, cluster_id, medoids_buf[:cluster_id].copy()


@njit(cache=True)
def _dbscan_label(
    rmsd_matrix: NDArray[np.floating],
    counts: NDArray[np.int64],
    eps: float,
    min_samples: int,
) -> tuple[NDArray[np.int64], int]:  # pragma: no cover - JIT-compiled
    """Assign DBSCAN cluster labels via BFS from core points.

    Uses only ``O(n)`` auxiliary memory (labels, is_core, stack) and
    reads the RMSD matrix in place -- no copies.  At 120k frames this
    avoids the ~57 GB of internal copies that sklearn DBSCAN makes.

    A point ``i`` is a *core* point when ``counts[i] >= min_samples``
    (neighbour count includes self, matching sklearn convention).  BFS
    expands from each unvisited core point: neighbours within ``eps``
    are assigned to the cluster and, if they are also core, pushed onto
    the stack for further expansion.  Non-core neighbours become border
    points (assigned but not expanded).  Points never reached stay as
    label -1 (noise).

    When multiple core points could seed a cluster, the one with the
    smallest frame index seeds first (deterministic tie-breaking).

    Returns:
        ``(labels, n_clusters)`` -- labels array with -1 for noise,
        and the number of non-noise clusters found.
    """
    n = rmsd_matrix.shape[0]
    labels = np.full(n, -1, dtype=np.int64)

    is_core = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if counts[i] >= min_samples:
            is_core[i] = True

    stack = np.empty(n, dtype=np.int64)
    cluster_id = 0

    for seed in range(n):
        if labels[seed] != -1 or not is_core[seed]:
            continue

        labels[seed] = cluster_id
        stack_top = 1
        stack[0] = seed

        while stack_top > 0:
            stack_top -= 1
            p = stack[stack_top]

            for q in range(n):
                if rmsd_matrix[p, q] <= eps and labels[q] == -1:
                    labels[q] = cluster_id
                    if is_core[q]:
                        stack[stack_top] = q
                        stack_top += 1

        cluster_id += 1

    return labels, cluster_id


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_rmsd_matrix(rmsd_matrix: NDArray[np.floating]) -> None:
    """Validate that *rmsd_matrix* is a finite, square, 2-D array."""
    if rmsd_matrix.ndim != 2 or rmsd_matrix.shape[0] != rmsd_matrix.shape[1]:
        raise ValueError(f"rmsd_matrix must be a square 2-D array, got shape {rmsd_matrix.shape}")
    if rmsd_matrix.size > 0 and not np.isfinite(rmsd_matrix).all():
        raise ValueError("rmsd_matrix contains NaN or Inf values")


def _compute_medoids(
    rmsd_matrix: NDArray[np.floating],
    labels: NDArray[np.int_],
    n_clusters: int,
) -> NDArray[np.int_]:
    """Compute medoid frame index per cluster from the RMSD matrix."""
    medoids = np.empty(n_clusters, dtype=np.int_)
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        sub_matrix = rmsd_matrix[np.ix_(members, members)]
        medoids[k] = members[np.argmin(sub_matrix.sum(axis=1))]
    return medoids


def _make_clustering_result(
    rmsd_matrix: NDArray[np.floating],
    labels: NDArray[np.int_],
    n_clusters: int,
    medoids: NDArray[np.int_] | None = None,
) -> ClusteringResult:
    """Build a :class:`ClusteringResult`, computing medoids if needed."""
    if medoids is None:
        if n_clusters > 0:
            medoids = _compute_medoids(rmsd_matrix, labels, n_clusters)
        else:
            medoids = np.array([], dtype=np.int_)
    return ClusteringResult(
        labels=labels.astype(np.int_, copy=False),
        n_clusters=int(n_clusters),
        medoid_frames=medoids.astype(np.int_, copy=False),
    )


def _make_feature_result(
    feature_matrix: NDArray[np.floating],
    labels: NDArray[np.int_],
    centers: NDArray[np.floating],
    n_clusters: int,
    inertia: float,
    resolved: np.dtype[np.floating],
) -> FeatureClusteringResult:
    """Build a :class:`FeatureClusteringResult` with medoid computation."""
    labels = np.asarray(labels)
    centers = np.asarray(centers)
    medoids = np.empty(n_clusters, dtype=np.int_)
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        dists = np.linalg.norm(feature_matrix[members] - centers[k], axis=1)
        medoids[k] = members[np.argmin(dists)]
    return FeatureClusteringResult(
        labels=labels.astype(np.int_, copy=False),
        n_clusters=int(n_clusters),
        cluster_centers=centers.astype(resolved, copy=False),
        medoid_frames=medoids,
        inertia=float(inertia),
    )


# ---------------------------------------------------------------------------
# Distance-matrix clustering methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gromos:
    """GROMOS clustering (Daura et al. 1999).

    Greedy largest-cluster-first assignment via Numba-JIT kernels.
    O(n) auxiliary memory -- no copies of the RMSD matrix.

    Args:
        cutoff_nm: Neighbour cutoff in nm.

    Example::

        result = Gromos(cutoff_nm=0.2)(rmsd_matrix)
    """

    cutoff_nm: float = 0.15

    def __call__(self, rmsd_matrix: NDArray[np.floating]) -> ClusteringResult:
        """Cluster *rmsd_matrix* and return a :class:`ClusteringResult`."""
        _validate_rmsd_matrix(rmsd_matrix)
        if self.cutoff_nm <= 0.0:
            raise ValueError(f"cutoff_nm must be positive, got {self.cutoff_nm!r}")
        rmsd_matrix = np.ascontiguousarray(rmsd_matrix)
        counts = _gromos_initial_counts(rmsd_matrix, self.cutoff_nm)
        labels, n_cls, medoids = _gromos_loop(rmsd_matrix, counts, self.cutoff_nm)
        return _make_clustering_result(rmsd_matrix, labels, n_cls, medoids)


@dataclass(frozen=True, slots=True)
class Hierarchical:
    """Agglomerative hierarchical clustering (scipy).

    Uses ``distance_threshold`` by default. Set ``n_clusters`` to use a
    fixed cluster count instead (overrides ``distance_threshold``).

    Note:
        Scipy builds an ``O(n^2)`` float64 condensed distance matrix
        internally.  At 120k frames this is ~57 GB.

    Args:
        linkage_method: ``"average"``, ``"complete"``, or ``"single"``.
            ``"ward"`` is not valid for RMSD matrices.
        distance_threshold: Distance cutoff in nm.
        n_clusters: Fixed cluster count (overrides *distance_threshold*).

    Example::

        result = Hierarchical(linkage_method="average", distance_threshold=0.2)(rmsd_matrix)
    """

    linkage_method: str = "average"
    distance_threshold: float = 0.15
    n_clusters: int | None = None

    def __call__(self, rmsd_matrix: NDArray[np.floating]) -> ClusteringResult:
        """Cluster *rmsd_matrix* and return a :class:`ClusteringResult`."""
        _validate_rmsd_matrix(rmsd_matrix)
        if self.n_clusters is None and self.distance_threshold <= 0.0:
            raise ValueError(
                f"distance_threshold must be positive, got {self.distance_threshold!r}"
            )

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(rmsd_matrix, checks=False)
        z = linkage(condensed, method=self.linkage_method)
        if self.n_clusters is not None:
            raw_labels = fcluster(z, t=self.n_clusters, criterion="maxclust")
        else:
            raw_labels = fcluster(z, t=self.distance_threshold, criterion="distance")
        labels = (raw_labels - 1).astype(np.int_)
        n_cls = int(labels.max()) + 1
        return _make_clustering_result(rmsd_matrix, labels, n_cls)


@dataclass(frozen=True, slots=True)
class DBSCAN:
    """DBSCAN density-based clustering.

    Two backends:

    - ``"numba"`` (default) -- custom Numba-JIT kernel.  Reuses the
      parallel neighbour-count kernel from GROMOS and a sequential BFS
      for label assignment.  O(n) auxiliary memory, no copies.
    - ``"sklearn"`` -- official scikit-learn ``DBSCAN`` with
      ``metric="precomputed"``.

    Noise frames receive label -1.

    Args:
        eps: Neighbourhood radius in nm.
        min_samples: Minimum neighbours (including self) for a core point.
        backend: ``"numba"`` or ``"sklearn"``.

    Example::

        result = DBSCAN(eps=0.15, min_samples=5)(rmsd_matrix)
        result = DBSCAN(eps=0.15, backend="sklearn")(rmsd_matrix)
    """

    eps: float = 0.15
    min_samples: int = 5
    backend: Literal["numba", "sklearn"] = "numba"

    def __call__(self, rmsd_matrix: NDArray[np.floating]) -> ClusteringResult:
        """Cluster *rmsd_matrix* and return a :class:`ClusteringResult`."""
        _validate_rmsd_matrix(rmsd_matrix)
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps!r}")

        if self.backend == "numba":
            rmsd_matrix = np.ascontiguousarray(rmsd_matrix)
            counts = _gromos_initial_counts(rmsd_matrix, self.eps)
            labels, n_cls = _dbscan_label(rmsd_matrix, counts, self.eps, self.min_samples)
            labels = labels.astype(np.int_)
        elif self.backend == "sklearn":
            from sklearn.cluster import DBSCAN as _SklearnDBSCAN

            db = _SklearnDBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric="precomputed",
            )
            labels = db.fit_predict(rmsd_matrix).astype(np.int_)
            n_cls = int(labels.max()) + 1 if labels.max() >= 0 else 0
        else:
            raise ValueError(f"Unknown DBSCAN backend: {self.backend!r}. Use 'numba' or 'sklearn'.")

        return _make_clustering_result(rmsd_matrix, labels, n_cls)


@dataclass(frozen=True, slots=True)
class HDBSCAN:
    """HDBSCAN hierarchical density-based clustering (sklearn >= 1.3).

    Noise frames receive label -1.

    Args:
        min_cluster_size: Minimum number of frames in a cluster.
        min_samples: Number of neighbours for core-point estimation.

    Example::

        result = HDBSCAN(min_cluster_size=50, min_samples=5)(rmsd_matrix)
    """

    min_cluster_size: int = 5
    min_samples: int = 5

    def __call__(self, rmsd_matrix: NDArray[np.floating]) -> ClusteringResult:
        """Cluster *rmsd_matrix* and return a :class:`ClusteringResult`."""
        _validate_rmsd_matrix(rmsd_matrix)
        from sklearn.cluster import HDBSCAN as _SklearnHDBSCAN

        hdb = _SklearnHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
        )
        labels = hdb.fit_predict(rmsd_matrix).astype(np.int_)
        n_cls = int(labels.max()) + 1 if labels.max() >= 0 else 0
        return _make_clustering_result(rmsd_matrix, labels, n_cls)


# ---------------------------------------------------------------------------
# Feature-vector clustering methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KMeans:
    """K-Means clustering (scikit-learn).

    Args:
        n_clusters: Number of clusters.
        dtype: Output float dtype for *cluster_centers*.

    Example::

        result = KMeans(n_clusters=10)(pca.projections)
    """

    n_clusters: int = 10
    dtype: DtypeArg = None

    def __call__(self, features: ArrayLike) -> FeatureClusteringResult:
        """Cluster *features* and return a :class:`FeatureClusteringResult`."""
        from sklearn.cluster import KMeans as _SklearnKMeans

        from mdpp.analysis.decomposition import _as_feature_matrix

        resolved = resolve_dtype(self.dtype)
        feature_matrix = _as_feature_matrix(features)
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters!r}")

        km = _SklearnKMeans(n_clusters=self.n_clusters, n_init="auto", random_state=42)
        labels = km.fit_predict(feature_matrix)
        centers = np.asarray(km.cluster_centers_)
        inertia = float(km.inertia_)  # type: ignore[arg-type]  # set after fit
        return _make_feature_result(
            feature_matrix, labels, centers, self.n_clusters, inertia, resolved
        )


@dataclass(frozen=True, slots=True)
class MiniBatchKMeans:
    """Mini-Batch K-Means clustering (scikit-learn).

    Args:
        n_clusters: Number of clusters.
        batch_size: Mini-batch size.
        dtype: Output float dtype for *cluster_centers*.

    Example::

        result = MiniBatchKMeans(n_clusters=10, batch_size=1024)(pca.projections)
    """

    n_clusters: int = 10
    batch_size: int = 1024
    dtype: DtypeArg = None

    def __call__(self, features: ArrayLike) -> FeatureClusteringResult:
        """Cluster *features* and return a :class:`FeatureClusteringResult`."""
        from sklearn.cluster import MiniBatchKMeans as _SklearnMBK

        from mdpp.analysis.decomposition import _as_feature_matrix

        resolved = resolve_dtype(self.dtype)
        feature_matrix = _as_feature_matrix(features)
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size!r}")

        mbk = _SklearnMBK(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            n_init="auto",
            random_state=42,
        )
        labels = mbk.fit_predict(feature_matrix)
        centers = np.asarray(mbk.cluster_centers_)
        inertia = float(mbk.inertia_)  # type: ignore[arg-type]  # set after fit
        return _make_feature_result(
            feature_matrix, labels, centers, self.n_clusters, inertia, resolved
        )


@dataclass(frozen=True, slots=True)
class RegularSpace:
    """Regular-space clustering (deeptime).

    The number of clusters is determined by ``dmin``, not specified
    upfront.

    Args:
        dmin: Minimum distance between cluster centres.
        dtype: Output float dtype for *cluster_centers*.

    Example::

        result = RegularSpace(dmin=0.5)(pca.projections)
    """

    dmin: float = 0.5
    dtype: DtypeArg = None

    def __call__(self, features: ArrayLike) -> FeatureClusteringResult:
        """Cluster *features* and return a :class:`FeatureClusteringResult`."""
        from mdpp.analysis.decomposition import _as_feature_matrix

        resolved = resolve_dtype(self.dtype)
        feature_matrix = _as_feature_matrix(features)
        if self.dmin <= 0.0:
            raise ValueError(f"dmin must be positive, got {self.dmin!r}")

        from deeptime.clustering import RegularSpace as _DeeptimeRegSpace

        estimator = _DeeptimeRegSpace(dmin=self.dmin, max_centers=10000)
        model = estimator.fit(feature_matrix).fetch_model()
        centers = np.asarray(model.cluster_centers)
        labels = np.asarray(model.transform(feature_matrix))
        n_cls = len(centers)

        dists = np.linalg.norm(feature_matrix[:, None, :] - centers[None, :, :], axis=2)
        inertia = float(np.sum(np.min(dists, axis=1) ** 2))
        return _make_feature_result(feature_matrix, labels, centers, n_cls, inertia, resolved)
