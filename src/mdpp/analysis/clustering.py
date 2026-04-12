"""Conformational clustering from RMSD matrices."""

from __future__ import annotations

from dataclasses import dataclass

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
# Public API
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

    # ``copy=False`` is critical at large N: when the backend's native
    # dtype already matches ``resolved`` we reuse the same buffer
    # instead of allocating a second ~N^2 matrix.
    return RMSDMatrixResult(
        rmsd_matrix_nm=rmsd_matrix.astype(resolved, copy=False),
        atom_indices=atom_indices,
    )


@njit(parallel=True, cache=True)
def _gromos_initial_counts(
    rmsd_matrix: NDArray[np.floating],
    cutoff_nm: float,
) -> NDArray[np.int64]:  # pragma: no cover - JIT-compiled
    """Count each row's neighbors within ``cutoff_nm`` (inclusive).

    Parallel over rows.  This runs once at the start of the GROMOS
    loop and dominates wall time for small cluster counts.
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
    # Pre-allocate medoid storage -- at most ``n`` clusters.
    medoids_buf = np.empty(n, dtype=np.int64)
    cluster_id = 0

    # Working buffer for newly-assigned member indices in the current
    # cluster -- sized for the worst case (first cluster might absorb
    # the entire trajectory).
    members_buf = np.empty(n, dtype=np.int64)

    while True:
        # Pick the unassigned row with the largest neighbour count.
        best = -1
        best_count = -1
        for i in range(n):
            if not assigned[i] and counts[i] > best_count:
                best_count = counts[i]
                best = i
        if best < 0:
            break

        # Collect this cluster's members: every unassigned row within
        # cutoff of ``best`` (includes ``best`` itself).
        n_members = 0
        for j in range(n):
            if not assigned[j] and rmsd_matrix[best, j] <= cutoff_nm:
                members_buf[n_members] = j
                n_members += 1
                labels[j] = cluster_id
                assigned[j] = True

        # Incrementally refresh ``counts``: for every newly assigned
        # member ``m``, each of ``m``'s neighbours loses one
        # still-unassigned neighbour.  Walk row ``m`` (cache-friendly
        # contiguous access) and decrement in place.
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

    Args:
        rmsd_matrix: Symmetric ``(n, n)`` pairwise RMSD matrix.
        counts: Pre-computed neighbour counts from
            ``_gromos_initial_counts(rmsd_matrix, eps)``.
        eps: Neighbourhood radius (same as ``cutoff_nm``).
        min_samples: Minimum neighbours to qualify as a core point.

    Returns:
        ``(labels, n_clusters)`` -- labels array with -1 for noise,
        and the number of non-noise clusters found.
    """
    n = rmsd_matrix.shape[0]
    labels = np.full(n, -1, dtype=np.int64)

    # Core-point mask.
    is_core = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if counts[i] >= min_samples:
            is_core[i] = True

    # Stack buffer for BFS -- worst case all points in one component.
    stack = np.empty(n, dtype=np.int64)
    cluster_id = 0

    for seed in range(n):
        if labels[seed] != -1 or not is_core[seed]:
            continue

        # Seed a new cluster from this core point.
        labels[seed] = cluster_id
        stack_top = 1
        stack[0] = seed

        while stack_top > 0:
            stack_top -= 1
            p = stack[stack_top]

            # Scan row p for unassigned neighbours within eps.
            for q in range(n):
                if rmsd_matrix[p, q] <= eps and labels[q] == -1:
                    labels[q] = cluster_id
                    if is_core[q]:
                        stack[stack_top] = q
                        stack_top += 1

        cluster_id += 1

    return labels, cluster_id


def _compute_medoids(
    rmsd_matrix: NDArray[np.floating],
    labels: NDArray[np.int_],
    n_clusters: int,
) -> NDArray[np.int_]:
    """Compute the medoid frame index for each cluster.

    The medoid is the frame with the minimum sum of within-cluster
    distances.  Noise frames (label == -1) are ignored.
    """
    medoids = np.empty(n_clusters, dtype=np.int_)
    for k in range(n_clusters):
        members = np.where(labels == k)[0]
        sub_matrix = rmsd_matrix[np.ix_(members, members)]
        medoids[k] = members[np.argmin(sub_matrix.sum(axis=1))]
    return medoids


def _cluster_hierarchical(
    rmsd_matrix: NDArray[np.floating],
    *,
    cutoff_nm: float,
    linkage_method: str,
    n_clusters: int | None,
) -> tuple[NDArray[np.int_], int]:
    """Hierarchical agglomerative clustering via scipy.

    Raises:
        MemoryError: When the condensed distance matrix would exceed
            8 GB.  Subsample the RMSD matrix first (e.g.
            ``rmsd_matrix[::stride, ::stride]``).
    """
    n = rmsd_matrix.shape[0]
    condensed_bytes = n * (n - 1) // 2 * 8  # float64 condensed vector
    if condensed_bytes > 8 * 1024**3:
        gb = condensed_bytes / 1024**3
        raise MemoryError(
            f"Hierarchical clustering on {n} frames needs ~{gb:.0f} GB for the "
            f"condensed distance matrix (float64). Subsample first, e.g. "
            f"rmsd_matrix[::10, ::10]."
        )

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(rmsd_matrix, checks=False)
    z = linkage(condensed, method=linkage_method)
    if n_clusters is not None:
        raw_labels = fcluster(z, t=n_clusters, criterion="maxclust")
    else:
        raw_labels = fcluster(z, t=cutoff_nm, criterion="distance")
    # fcluster labels start at 1; shift to 0-based
    labels = raw_labels - 1
    n = int(labels.max()) + 1
    return labels.astype(np.int_), n


def _cluster_dbscan(
    rmsd_matrix: NDArray[np.floating],
    *,
    cutoff_nm: float,
    min_samples: int,
) -> tuple[NDArray[np.int_], int]:
    """DBSCAN clustering via Numba JIT kernels.

    Reuses ``_gromos_initial_counts`` for the parallel neighbour count
    and ``_dbscan_label`` for the sequential BFS assignment.  Total
    auxiliary memory is ``O(n)`` -- no copies of the RMSD matrix.
    """
    counts = _gromos_initial_counts(rmsd_matrix, cutoff_nm)
    labels, n_cls = _dbscan_label(rmsd_matrix, counts, cutoff_nm, min_samples)
    return labels.astype(np.int_), int(n_cls)


def _cluster_hdbscan(
    rmsd_matrix: NDArray[np.floating],
    *,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[NDArray[np.int_], int]:
    """HDBSCAN clustering via scikit-learn (>= 1.3)."""
    from sklearn.cluster import HDBSCAN

    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = hdb.fit_predict(rmsd_matrix).astype(np.int_)
    n = int(labels.max()) + 1 if labels.max() >= 0 else 0
    return labels, n


def cluster_conformations(
    rmsd_matrix: NDArray[np.floating],
    *,
    method: str = "gromos",
    cutoff_nm: float = 0.15,
    # --- hierarchical ---
    linkage_method: str = "average",
    n_clusters: int | None = None,
    # --- DBSCAN / HDBSCAN ---
    min_samples: int = 5,
    # --- HDBSCAN only ---
    min_cluster_size: int = 5,
) -> ClusteringResult:
    """Cluster trajectory frames from a pairwise RMSD matrix.

    Args:
        rmsd_matrix: Symmetric pairwise RMSD matrix of shape ``(n, n)``
            in nm.  Accepts any floating dtype -- float32 is preferred
            at large ``n`` because the matrix itself is already the
            biggest allocation in the pipeline (57 GB at n=120k).
        method: Clustering method.

            - ``"gromos"`` -- GROMOS algorithm (largest-cluster-first
              greedy assignment).  Uses: ``cutoff_nm``.
            - ``"hierarchical"`` -- scipy agglomerative clustering.
              Uses: ``cutoff_nm`` (as distance threshold) or
              ``n_clusters``, ``linkage_method``.
            - ``"dbscan"`` -- Numba-JIT DBSCAN (O(n) aux memory, no
              copies).  Uses: ``cutoff_nm`` (as eps), ``min_samples``.
              Noise frames get label -1.
            - ``"hdbscan"`` -- scikit-learn HDBSCAN (>= 1.3).  Uses:
              ``min_cluster_size``, ``min_samples``.  Noise frames get
              label -1.

        cutoff_nm: RMSD cutoff in nm.  Interpretation depends on the
            method (GROMOS cutoff, hierarchical distance threshold,
            or DBSCAN eps).
        linkage_method: Linkage criterion for hierarchical clustering
            (e.g. ``"average"``, ``"complete"``, ``"single"``).
        n_clusters: If set, hierarchical clustering uses a fixed
            cluster count instead of ``cutoff_nm``.
        min_samples: Minimum number of samples in a neighbourhood for
            DBSCAN / HDBSCAN.
        min_cluster_size: Minimum cluster size for HDBSCAN.

    Returns:
        ClusteringResult with per-frame labels and medoid frame indices.

    Raises:
        ValueError: If an unsupported method is specified or invalid
            parameters are given.

    Memory note:
        ``"gromos"`` and ``"dbscan"`` use Numba-JIT kernels with only
        ``O(n)`` auxiliary buffers -- no copies of the RMSD matrix.
        ``"hierarchical"`` requires an ``O(n^2)`` float64 condensed
        matrix via scipy and will raise ``MemoryError`` when this
        exceeds 8 GB (~45k frames).  ``"hdbscan"`` delegates to
        sklearn which may create internal copies at large ``n``.
    """
    if rmsd_matrix.ndim != 2 or rmsd_matrix.shape[0] != rmsd_matrix.shape[1]:
        raise ValueError(f"rmsd_matrix must be a square 2-D array, got shape {rmsd_matrix.shape}")
    if rmsd_matrix.size > 0 and not np.isfinite(rmsd_matrix).all():
        raise ValueError("rmsd_matrix contains NaN or Inf values")

    if method == "gromos":
        if cutoff_nm <= 0.0:
            raise ValueError(f"cutoff_nm must be positive, got {cutoff_nm!r}")
        # ``np.ascontiguousarray`` is a no-op for already-contiguous
        # inputs and avoids a numba strided-access slow path for views.
        rmsd_matrix = np.ascontiguousarray(rmsd_matrix)
        counts = _gromos_initial_counts(rmsd_matrix, cutoff_nm)
        labels, n_cls, medoids = _gromos_loop(rmsd_matrix, counts, cutoff_nm)
    elif method == "hierarchical":
        if n_clusters is None and cutoff_nm <= 0.0:
            raise ValueError(f"cutoff_nm must be positive, got {cutoff_nm!r}")
        labels, n_cls = _cluster_hierarchical(
            rmsd_matrix,
            cutoff_nm=cutoff_nm,
            linkage_method=linkage_method,
            n_clusters=n_clusters,
        )
        medoids = _compute_medoids(rmsd_matrix, labels, n_cls)
    elif method == "dbscan":
        if cutoff_nm <= 0.0:
            raise ValueError(f"cutoff_nm must be positive, got {cutoff_nm!r}")
        rmsd_matrix = np.ascontiguousarray(rmsd_matrix)
        labels, n_cls = _cluster_dbscan(
            rmsd_matrix,
            cutoff_nm=cutoff_nm,
            min_samples=min_samples,
        )
        medoids = (
            _compute_medoids(rmsd_matrix, labels, n_cls)
            if n_cls > 0
            else np.array([], dtype=np.int_)
        )
    elif method == "hdbscan":
        labels, n_cls = _cluster_hdbscan(
            rmsd_matrix,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        medoids = (
            _compute_medoids(rmsd_matrix, labels, n_cls)
            if n_cls > 0
            else np.array([], dtype=np.int_)
        )
    else:
        supported = ("gromos", "hierarchical", "dbscan", "hdbscan")
        raise ValueError(
            f"Unsupported clustering method: {method!r}. Supported methods: {supported}"
        )

    return ClusteringResult(
        labels=labels.astype(np.int_, copy=False),
        n_clusters=int(n_cls),
        medoid_frames=medoids.astype(np.int_, copy=False),
    )


def cluster_features(
    features: ArrayLike,
    *,
    method: str = "kmeans",
    n_clusters: int = 10,
    batch_size: int = 1024,
    dmin: float = 0.5,
    dtype: DtypeArg = None,
) -> FeatureClusteringResult:
    """Cluster frames from a feature matrix (e.g. PCA/TICA projections).

    Args:
        features: Feature matrix of shape ``(n_frames, n_features)``,
            e.g. from ``compute_pca(...).projections`` or
            ``compute_tica(...).projections``.
        method: Clustering algorithm.

            - ``"kmeans"`` -- standard k-means (scikit-learn).
              Uses: ``n_clusters``.
            - ``"minibatch"`` -- mini-batch k-means (scikit-learn).
              Uses: ``n_clusters``, ``batch_size``.
            - ``"regspace"`` -- regular-space clustering (deeptime).
              Uses: ``dmin``. ``n_clusters`` is ignored.

        n_clusters: Number of clusters for k-means / mini-batch.
        batch_size: Batch size for mini-batch k-means.
        dmin: Minimum center-to-center distance for regular-space.
        dtype: Output float dtype.

    Returns:
        FeatureClusteringResult with labels, centers, medoids, and
        inertia.

    Raises:
        ValueError: If an unsupported method or invalid parameters are
            given.
    """
    from mdpp._dtype import resolve_dtype
    from mdpp.analysis.decomposition import _as_feature_matrix

    resolved = resolve_dtype(dtype)
    feature_matrix = _as_feature_matrix(features)

    if method == "kmeans":
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters!r}")
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        labels = km.fit_predict(feature_matrix)
        centers = np.asarray(km.cluster_centers_)
        inertia = float(km.inertia_)  # type: ignore[arg-type]  # set after fit
        n_cls = n_clusters
    elif method == "minibatch":
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        from sklearn.cluster import MiniBatchKMeans

        mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init="auto",
            random_state=42,
        )
        labels = mbk.fit_predict(feature_matrix)
        centers = np.asarray(mbk.cluster_centers_)
        inertia = float(mbk.inertia_)  # type: ignore[arg-type]  # set after fit
        n_cls = n_clusters
    elif method == "regspace":
        if dmin <= 0.0:
            raise ValueError(f"dmin must be positive, got {dmin!r}")
        from deeptime.clustering import RegularSpace

        estimator = RegularSpace(dmin=dmin, max_centers=10000)
        model = estimator.fit(feature_matrix).fetch_model()
        centers = np.asarray(model.cluster_centers)
        labels = model.transform(feature_matrix)
        n_cls = len(centers)
        # Compute inertia manually: sum of squared distances to nearest
        # cluster center.
        dists = np.linalg.norm(
            feature_matrix[:, None, :] - centers[None, :, :],
            axis=2,
        )
        inertia = float(np.sum(np.min(dists, axis=1) ** 2))
    else:
        supported = ("kmeans", "minibatch", "regspace")
        raise ValueError(
            f"Unsupported feature clustering method: {method!r}. Supported methods: {supported}"
        )

    labels = np.asarray(labels)
    centers = np.asarray(centers)

    # Compute medoid frames: for each cluster, the member frame closest
    # to the cluster centroid in feature space.
    medoids = np.empty(n_cls, dtype=np.int_)
    for k in range(n_cls):
        members = np.where(labels == k)[0]
        dists_to_center = np.linalg.norm(feature_matrix[members] - centers[k], axis=1)
        medoids[k] = members[np.argmin(dists_to_center)]

    return FeatureClusteringResult(
        labels=labels.astype(np.int_, copy=False),
        n_clusters=int(n_cls),
        cluster_centers=np.asarray(centers, dtype=resolved),
        medoid_frames=medoids,
        inertia=float(inertia),
    )
