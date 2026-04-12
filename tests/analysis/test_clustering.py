"""Tests for conformational clustering (RMSD matrix + class-based API)."""

from __future__ import annotations

import time

import mdtraj as md
import numpy as np
import pytest
from numpy.typing import NDArray

from mdpp.analysis._backends import has_cupy, has_jax, has_torch
from mdpp.analysis.clustering import (
    DBSCAN,
    HDBSCAN,
    ClusteringResult,
    FeatureClusteringResult,
    Gromos,
    Hierarchical,
    KMeans,
    MiniBatchKMeans,
    RegularSpace,
    RMSDMatrixResult,
    compute_rmsd_matrix,
)

requires_cupy = pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
requires_torch = pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
requires_jax = pytest.mark.skipif(not has_jax, reason="JAX not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backbone_trajectory() -> md.Trajectory:
    """Return a small trajectory with backbone-like atoms and known geometry.

    5 ALA residues x 3 backbone atoms (N, CA, C) = 15 atoms, 30 frames.
    Coordinates are a shared base structure plus small random perturbations
    (0.02 nm), giving pairwise RMSDs in the 0.01-0.06 nm range.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for res_idx in range(1, 6):
        residue = topology.add_residue("ALA", chain, resSeq=res_idx)
        n = topology.add_atom("N", md.element.nitrogen, residue)
        ca = topology.add_atom("CA", md.element.carbon, residue)
        c = topology.add_atom("C", md.element.carbon, residue)
        atoms.extend([n, ca, c])
        topology.add_bond(n, ca)
        topology.add_bond(ca, c)
        if res_idx > 1:
            topology.add_bond(atoms[-6], c)

    rng = np.random.RandomState(42)
    n_frames = 30
    n_atoms = len(atoms)
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    xyz = base + perturbation
    time_ps = np.arange(n_frames, dtype=np.float64) * 10.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


@pytest.fixture()
def clustered_features() -> NDArray[np.floating]:
    """Feature matrix with 3 well-separated clusters of 20 points each."""
    rng = np.random.RandomState(42)
    centers = np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float32)
    points = [c + rng.randn(20, 2).astype(np.float32) * 0.3 for c in centers]
    return np.vstack(points)


# ---------------------------------------------------------------------------
# RMSD matrix tests
# ---------------------------------------------------------------------------


class TestComputeRmsdMatrix:
    """Tests for ``compute_rmsd_matrix``."""

    def test_result_shape_and_type(self, backbone_trajectory: md.Trajectory) -> None:
        """Result should have the correct shape and be a frozen dataclass."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        n = backbone_trajectory.n_frames

        assert isinstance(result, RMSDMatrixResult)
        assert result.rmsd_matrix_nm.shape == (n, n)
        assert result.rmsd_matrix_angstrom.shape == (n, n)

    def test_diagonal_is_zero(self, backbone_trajectory: md.Trajectory) -> None:
        """Self-RMSD (diagonal) should be zero."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        np.testing.assert_allclose(np.diag(result.rmsd_matrix_nm), 0.0, atol=1e-6)

    def test_matrix_is_symmetric(self, backbone_trajectory: md.Trajectory) -> None:
        """Pairwise RMSD matrix should be symmetric."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        np.testing.assert_allclose(result.rmsd_matrix_nm, result.rmsd_matrix_nm.T, atol=1e-6)

    def test_values_are_nonnegative(self, backbone_trajectory: md.Trajectory) -> None:
        """All RMSD values should be >= 0."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        assert np.all(result.rmsd_matrix_nm >= 0.0)

    def test_angstrom_conversion(self, backbone_trajectory: md.Trajectory) -> None:
        """Angstrom property should be 10x the nm values."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        np.testing.assert_allclose(result.rmsd_matrix_angstrom, result.rmsd_matrix_nm * 10.0)

    def test_invalid_backend_raises(self, backbone_trajectory: md.Trajectory) -> None:
        """An unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="bogus")  # type: ignore[arg-type]

    def test_numba_and_mdtraj_agree(self, backbone_trajectory: md.Trajectory) -> None:
        """Numba QCP and mdtraj backends should produce the same matrix."""
        result_numba = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", backend="numba"
        )
        result_mdtraj = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", backend="mdtraj"
        )
        mdtraj_sym = (result_mdtraj.rmsd_matrix_nm + result_mdtraj.rmsd_matrix_nm.T) / 2.0

        np.testing.assert_allclose(result_numba.rmsd_matrix_nm, mdtraj_sym, atol=5e-5)

    def test_backends_agree_on_atom_selection(self, backbone_trajectory: md.Trajectory) -> None:
        """Both backends should respect atom_selection and agree on the result."""
        result_numba = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="name CA", backend="numba"
        )
        result_mdtraj = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="name CA", backend="mdtraj"
        )
        mdtraj_sym = (result_mdtraj.rmsd_matrix_nm + result_mdtraj.rmsd_matrix_nm.T) / 2.0

        assert len(result_numba.atom_indices) == 5  # 5 residues, 1 CA each
        np.testing.assert_array_equal(result_numba.atom_indices, result_mdtraj.atom_indices)
        np.testing.assert_allclose(result_numba.rmsd_matrix_nm, mdtraj_sym, atol=5e-5)


# ---------------------------------------------------------------------------
# GPU / optional backend agreement tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGpuBackendAgreement:
    """Verify that torch, jax, and cupy backends agree with numba.

    Each test computes the pairwise RMSD matrix with the GPU backend and
    the numba reference, then asserts element-wise agreement within
    ``atol=5e-5 nm`` (~0.0005 Angstrom).  The numba result is itself
    validated against mdtraj in ``TestComputeRmsdMatrix``.

    Tests are skipped when the corresponding package is not installed
    or when the ``gpu`` marker is deselected (``-m "not gpu"``).
    """

    @requires_torch
    def test_torch_agrees_with_numba(self, backbone_trajectory: md.Trajectory) -> None:
        """PyTorch backend should match numba QCP within 5e-5 nm."""
        ref = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="numba")
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="torch")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    @requires_jax
    def test_jax_agrees_with_numba(self, backbone_trajectory: md.Trajectory) -> None:
        """JAX backend should match numba QCP within 5e-5 nm."""
        ref = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="numba")
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="jax")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    @requires_cupy
    def test_cupy_agrees_with_numba(self, backbone_trajectory: md.Trajectory) -> None:
        """CuPy backend should match numba QCP within 5e-5 nm."""
        ref = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="numba")
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all", backend="cupy")

        np.testing.assert_allclose(result.rmsd_matrix_nm, ref.rmsd_matrix_nm, atol=5e-5)

    @requires_torch
    @requires_jax
    @requires_cupy
    def test_all_gpu_backends_agree(self, backbone_trajectory: md.Trajectory) -> None:
        """All three GPU backends should agree with each other within 5e-5 nm."""
        results = {
            name: compute_rmsd_matrix(
                backbone_trajectory,
                atom_selection="all",
                backend=name,  # type: ignore[arg-type]
            ).rmsd_matrix_nm
            for name in ("torch", "jax", "cupy")
        }
        for a_name, a_mat in results.items():
            for b_name, b_mat in results.items():
                if a_name >= b_name:
                    continue
                np.testing.assert_allclose(
                    a_mat,
                    b_mat,
                    atol=5e-5,
                    err_msg=f"{a_name} vs {b_name}",
                )


# ---------------------------------------------------------------------------
# Hand-crafted 6x6 matrix reused by several test classes
# ---------------------------------------------------------------------------

_SMALL_RMSD = np.array(
    [
        [0.00, 0.05, 0.07, 0.80, 0.82, 1.50],
        [0.05, 0.00, 0.06, 0.81, 0.83, 1.51],
        [0.07, 0.06, 0.00, 0.79, 0.80, 1.49],
        [0.80, 0.81, 0.79, 0.00, 0.10, 1.00],
        [0.82, 0.83, 0.80, 0.10, 0.00, 0.99],
        [1.50, 1.51, 1.49, 1.00, 0.99, 0.00],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Gromos clustering tests
# ---------------------------------------------------------------------------


class TestGromos:
    """Tests for ``Gromos`` clustering."""

    def test_all_frames_assigned(self, backbone_trajectory: md.Trajectory) -> None:
        """Every frame should receive a cluster label >= 0."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = Gromos(cutoff_nm=0.5)(result.rmsd_matrix_nm)

        assert isinstance(clustering, ClusteringResult)
        assert clustering.labels.shape == (backbone_trajectory.n_frames,)
        assert np.all(clustering.labels >= 0)

    def test_cluster_count_matches_labels(self, backbone_trajectory: md.Trajectory) -> None:
        """n_clusters should equal the number of unique labels."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = Gromos(cutoff_nm=0.5)(result.rmsd_matrix_nm)

        assert clustering.n_clusters == len(np.unique(clustering.labels))

    def test_medoid_count_matches_clusters(self, backbone_trajectory: md.Trajectory) -> None:
        """There should be one medoid per cluster."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = Gromos(cutoff_nm=0.5)(result.rmsd_matrix_nm)

        assert len(clustering.medoid_frames) == clustering.n_clusters

    def test_tight_cutoff_gives_more_clusters(self, backbone_trajectory: md.Trajectory) -> None:
        """A tighter cutoff should produce at least as many clusters."""
        result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        loose = Gromos(cutoff_nm=1.0)(result.rmsd_matrix_nm)
        tight = Gromos(cutoff_nm=0.01)(result.rmsd_matrix_nm)

        assert tight.n_clusters >= loose.n_clusters

    def test_float32_matrix_matches_float64(self, backbone_trajectory: md.Trajectory) -> None:
        """Clustering must accept float32 matrices and give the same result as float64."""
        result_f32 = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", dtype=np.float32
        )
        result_f64 = compute_rmsd_matrix(
            backbone_trajectory, atom_selection="all", dtype=np.float64
        )
        assert result_f32.rmsd_matrix_nm.dtype == np.float32
        assert result_f64.rmsd_matrix_nm.dtype == np.float64

        clust_f32 = Gromos(cutoff_nm=0.2)(result_f32.rmsd_matrix_nm)
        clust_f64 = Gromos(cutoff_nm=0.2)(result_f64.rmsd_matrix_nm)

        # Labels may not be identical if two centres tie, but the number
        # of clusters should match at a safe cutoff above the matrix noise.
        assert clust_f32.n_clusters == clust_f64.n_clusters
        # Every member of cluster k in f32 should share the same label in f64
        # (modulo the label id -- use a canonical remapping).
        mapping: dict[int, int] = {}
        for a, b in zip(clust_f32.labels, clust_f64.labels, strict=True):
            mapping.setdefault(int(a), int(b))
            assert mapping[int(a)] == int(b)

    def test_known_greedy_result_small(self) -> None:
        """Hand-checked GROMOS result on a 6x6 matrix.

        Frames 0-2 are within cutoff of each other; frames 3-4 form a
        second tight pair; frame 5 is an isolated singleton.  GROMOS
        picks the largest cluster first (3 members), then the pair, then
        the singleton.
        """
        result = Gromos(cutoff_nm=0.15)(_SMALL_RMSD)
        assert result.n_clusters == 3
        # First (largest) cluster should contain the 3 tight frames.
        first = {i for i in range(6) if result.labels[i] == 0}
        assert first == {0, 1, 2}
        # Second cluster: the tight pair.
        second = {i for i in range(6) if result.labels[i] == 1}
        assert second == {3, 4}
        # Third: the singleton.
        third = {i for i in range(6) if result.labels[i] == 2}
        assert third == {5}

    def test_single_frame(self) -> None:
        """A 1x1 matrix must produce a single cluster with one medoid."""
        result = Gromos(cutoff_nm=0.1)(np.zeros((1, 1), dtype=np.float32))
        assert result.n_clusters == 1
        assert result.labels[0] == 0
        assert result.medoid_frames.tolist() == [0]

    def test_all_frames_within_cutoff(self) -> None:
        """If every pair is within the cutoff, all frames collapse to one cluster."""
        n = 8
        rmsd = np.zeros((n, n), dtype=np.float32)
        result = Gromos(cutoff_nm=0.1)(rmsd)
        assert result.n_clusters == 1
        assert np.all(result.labels == 0)

    def test_all_frames_isolated(self) -> None:
        """If every pair exceeds the cutoff, each frame becomes its own cluster."""
        n = 5
        rmsd = np.full((n, n), 10.0, dtype=np.float32)
        np.fill_diagonal(rmsd, 0.0)
        result = Gromos(cutoff_nm=0.1)(rmsd)
        assert result.n_clusters == n
        assert len(np.unique(result.labels)) == n

    def test_negative_cutoff_raises(self) -> None:
        """A negative cutoff must raise ValueError, not hang."""
        rmsd = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="cutoff_nm must be positive"):
            Gromos(cutoff_nm=-0.1)(rmsd)

    def test_zero_cutoff_raises(self) -> None:
        """A zero cutoff must raise ValueError."""
        rmsd = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="cutoff_nm must be positive"):
            Gromos(cutoff_nm=0.0)(rmsd)

    def test_non_square_matrix_raises(self) -> None:
        """A non-square matrix must raise ValueError."""
        rmsd = np.zeros((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="square 2-D array"):
            Gromos(cutoff_nm=0.1)(rmsd)

    def test_1d_array_raises(self) -> None:
        """A 1-D array must raise ValueError."""
        rmsd = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="square 2-D array"):
            Gromos(cutoff_nm=0.1)(rmsd)

    def test_nan_in_matrix_raises(self) -> None:
        """NaN values in the RMSD matrix must raise ValueError."""
        rmsd = np.zeros((4, 4), dtype=np.float32)
        rmsd[1, 2] = np.nan
        rmsd[2, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            Gromos(cutoff_nm=0.1)(rmsd)

    def test_inf_in_matrix_raises(self) -> None:
        """Inf values in the RMSD matrix must raise ValueError."""
        rmsd = np.zeros((4, 4), dtype=np.float32)
        rmsd[0, 3] = np.inf
        rmsd[3, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            Gromos(cutoff_nm=0.1)(rmsd)

    def test_empty_matrix(self) -> None:
        """A 0x0 matrix should produce zero clusters."""
        rmsd = np.zeros((0, 0), dtype=np.float32)
        result = Gromos(cutoff_nm=0.1)(rmsd)
        assert result.n_clusters == 0
        assert len(result.labels) == 0
        assert len(result.medoid_frames) == 0


# ---------------------------------------------------------------------------
# Hierarchical clustering tests
# ---------------------------------------------------------------------------


class TestHierarchical:
    """Tests for ``Hierarchical`` clustering."""

    def test_hierarchical_with_cutoff(self) -> None:
        """Cutoff-based hierarchical clustering should find 3 groups in the 6x6 matrix."""
        result = Hierarchical(distance_threshold=0.15, linkage_method="average")(_SMALL_RMSD)
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3
        assert result.labels.shape == (6,)
        assert len(result.medoid_frames) == 3

    def test_hierarchical_with_n_clusters(self) -> None:
        """Fixed n_clusters=2 should produce exactly 2 clusters."""
        result = Hierarchical(n_clusters=2)(_SMALL_RMSD)
        assert result.n_clusters == 2
        assert len(np.unique(result.labels)) == 2

    def test_hierarchical_all_frames_assigned(self, backbone_trajectory: md.Trajectory) -> None:
        """All frames should have labels >= 0 (no noise) for hierarchical clustering."""
        rmsd_result = compute_rmsd_matrix(backbone_trajectory, atom_selection="all")
        clustering = Hierarchical(distance_threshold=0.5)(rmsd_result.rmsd_matrix_nm)
        assert np.all(clustering.labels >= 0)
        assert clustering.n_clusters == len(np.unique(clustering.labels))

    @pytest.mark.parametrize("linkage_method", ["average", "complete", "single"])
    def test_hierarchical_linkage_methods(self, linkage_method: str) -> None:
        """Each linkage method should produce valid results on the 6x6 matrix."""
        result = Hierarchical(distance_threshold=0.15, linkage_method=linkage_method)(_SMALL_RMSD)
        assert result.n_clusters >= 1
        assert result.labels.shape == (6,)
        assert np.all(result.labels >= 0)
        assert len(result.medoid_frames) == result.n_clusters

    def test_hierarchical_medoids_are_valid(self) -> None:
        """Medoid frames should be valid frame indices within their clusters."""
        result = Hierarchical(distance_threshold=0.15)(_SMALL_RMSD)
        for k in range(result.n_clusters):
            members = np.where(result.labels == k)[0]
            assert result.medoid_frames[k] in members

    def test_negative_distance_threshold_raises(self) -> None:
        """A negative distance_threshold must raise ValueError."""
        rmsd = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            Hierarchical(distance_threshold=-0.1)(rmsd)


# ---------------------------------------------------------------------------
# DBSCAN clustering tests
# ---------------------------------------------------------------------------


class TestDBSCAN:
    """Tests for ``DBSCAN`` clustering."""

    def test_dbscan_basic(self) -> None:
        """DBSCAN should find the 3 groups in the 6x6 matrix."""
        result = DBSCAN(eps=0.15, min_samples=1)(_SMALL_RMSD)
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3

    def test_dbscan_noise_labels(self) -> None:
        """An isolated frame should receive the noise label -1."""
        # Frame 5 is far from everything; with min_samples=2 it cannot
        # form its own cluster and should be labeled as noise.
        result = DBSCAN(eps=0.15, min_samples=2)(_SMALL_RMSD)
        assert result.labels[5] == -1

    def test_dbscan_n_clusters_excludes_noise(self) -> None:
        """n_clusters should count only non-noise clusters."""
        result = DBSCAN(eps=0.15, min_samples=2)(_SMALL_RMSD)
        unique_non_noise = set(result.labels) - {-1}
        assert result.n_clusters == len(unique_non_noise)

    def test_dbscan_medoids_only_for_clusters(self) -> None:
        """medoid_frames length should equal n_clusters (no noise medoid)."""
        result = DBSCAN(eps=0.15, min_samples=2)(_SMALL_RMSD)
        assert len(result.medoid_frames) == result.n_clusters

    def test_dbscan_empty_result(self) -> None:
        """Very small eps + large min_samples: all frames may be noise."""
        result = DBSCAN(eps=0.001, min_samples=10)(_SMALL_RMSD)
        assert result.n_clusters == 0
        assert len(result.medoid_frames) == 0
        assert np.all(result.labels == -1)

    def test_numba_dbscan_matches_sklearn(self) -> None:
        """Numba DBSCAN kernel should produce the same clusters as sklearn.

        Labels may differ in numbering (both are deterministic but seed
        from different starting points), so we compare via a canonical
        partition: two frames share a label iff they are in the same
        cluster, and noise frames (-1) match exactly.
        """
        from sklearn.cluster import DBSCAN as SklearnDBSCAN

        eps, min_samples = 0.15, 2
        # sklearn reference
        sk = SklearnDBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        sk_labels = sk.fit_predict(_SMALL_RMSD)
        # Numba (via DBSCAN class)
        result = DBSCAN(eps=eps, min_samples=min_samples)(_SMALL_RMSD)
        nb_labels = result.labels

        # Noise frames must match exactly.
        np.testing.assert_array_equal(sk_labels == -1, nb_labels == -1)

        # Non-noise: same partition (co-membership).
        valid = sk_labels >= 0
        if valid.any():
            for i in range(len(sk_labels)):
                for j in range(i + 1, len(sk_labels)):
                    if sk_labels[i] >= 0 and sk_labels[j] >= 0:
                        assert (sk_labels[i] == sk_labels[j]) == (nb_labels[i] == nb_labels[j]), (
                            f"Partition mismatch at frames ({i}, {j})"
                        )

    def test_numba_dbscan_matches_sklearn_random(self) -> None:
        """Numba DBSCAN matches sklearn on a larger random matrix."""
        rng = np.random.RandomState(123)
        n = 50
        coords = rng.randn(n, 3).astype(np.float32)
        # Euclidean distance matrix as a proxy for RMSD
        diff = coords[:, None, :] - coords[None, :, :]
        rmsd = np.sqrt((diff**2).sum(axis=2))

        from sklearn.cluster import DBSCAN as SklearnDBSCAN

        eps, min_samples = 1.5, 3
        sk_labels = SklearnDBSCAN(
            eps=eps, min_samples=min_samples, metric="precomputed"
        ).fit_predict(rmsd)
        nb_result = DBSCAN(eps=eps, min_samples=min_samples)(rmsd)

        # Noise agreement
        np.testing.assert_array_equal(sk_labels == -1, nb_result.labels == -1)
        # Same number of clusters
        sk_n = int(sk_labels.max()) + 1 if sk_labels.max() >= 0 else 0
        assert nb_result.n_clusters == sk_n

    def test_dbscan_sklearn_backend(self) -> None:
        """DBSCAN with sklearn backend should find 3 clusters in the 6x6 matrix."""
        result = DBSCAN(eps=0.15, min_samples=1, backend="sklearn")(_SMALL_RMSD)
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters == 3

    def test_dbscan_invalid_backend_raises(self) -> None:
        """An invalid DBSCAN backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown DBSCAN backend"):
            DBSCAN(eps=0.15, backend="invalid")(_SMALL_RMSD)  # type: ignore[arg-type]

    def test_negative_eps_raises(self) -> None:
        """A negative eps must raise ValueError."""
        with pytest.raises(ValueError, match="eps must be positive"):
            DBSCAN(eps=-0.1)(_SMALL_RMSD)

    def test_non_square_matrix_raises(self) -> None:
        """A non-square matrix must raise ValueError."""
        with pytest.raises(ValueError, match="square 2-D array"):
            DBSCAN(eps=0.15)(np.zeros((3, 5)))

    def test_nan_matrix_raises(self) -> None:
        """NaN values in the RMSD matrix must raise ValueError."""
        nan_matrix = np.zeros((4, 4), dtype=np.float32)
        nan_matrix[1, 2] = np.nan
        nan_matrix[2, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            DBSCAN(eps=0.15)(nan_matrix)


# ---------------------------------------------------------------------------
# HDBSCAN clustering tests
# ---------------------------------------------------------------------------


class TestHDBSCAN:
    """Tests for ``HDBSCAN`` clustering."""

    def test_hdbscan_basic(self) -> None:
        """HDBSCAN should find clusters in the 6x6 matrix."""
        result = HDBSCAN(min_cluster_size=2, min_samples=1)(_SMALL_RMSD)
        assert isinstance(result, ClusteringResult)
        assert result.n_clusters >= 1

    def test_hdbscan_noise_handling(self) -> None:
        """HDBSCAN may produce noise labels (-1) and they should be handled."""
        result = HDBSCAN(min_cluster_size=2, min_samples=1)(_SMALL_RMSD)
        # n_clusters should exclude noise even if noise is present
        unique_non_noise = set(result.labels) - {-1}
        assert result.n_clusters == len(unique_non_noise)
        assert len(result.medoid_frames) == result.n_clusters

    def test_hdbscan_result_type(self) -> None:
        """Result should be ClusteringResult with correct field types."""
        result = HDBSCAN(min_cluster_size=2, min_samples=1)(_SMALL_RMSD)
        assert isinstance(result, ClusteringResult)
        assert isinstance(result.labels, np.ndarray)
        assert isinstance(result.n_clusters, int)
        assert isinstance(result.medoid_frames, np.ndarray)
        assert result.labels.shape == (6,)


# ---------------------------------------------------------------------------
# Feature-vector clustering tests: KMeans
# ---------------------------------------------------------------------------


class TestKMeans:
    """Tests for ``KMeans`` clustering."""

    def test_kmeans_basic(self, clustered_features: NDArray[np.floating]) -> None:
        """KMeans with n_clusters=3 should recover 3 clusters."""
        result = KMeans(n_clusters=3)(clustered_features)
        assert isinstance(result, FeatureClusteringResult)
        assert result.n_clusters == 3

    def test_kmeans_labels_shape(self, clustered_features: NDArray[np.floating]) -> None:
        """Labels should have shape (60,) with values in {0, 1, 2}."""
        result = KMeans(n_clusters=3)(clustered_features)
        assert result.labels.shape == (60,)
        assert set(result.labels.tolist()).issubset({0, 1, 2})

    def test_kmeans_cluster_centers_shape(self, clustered_features: NDArray[np.floating]) -> None:
        """Cluster centers should have shape (3, 2)."""
        result = KMeans(n_clusters=3)(clustered_features)
        assert result.cluster_centers.shape == (3, 2)

    def test_kmeans_medoid_frames_valid(self, clustered_features: NDArray[np.floating]) -> None:
        """Each medoid_frames[k] should be a member of cluster k."""
        result = KMeans(n_clusters=3)(clustered_features)
        for k in range(result.n_clusters):
            members = np.where(result.labels == k)[0]
            assert result.medoid_frames[k] in members

    def test_kmeans_inertia_positive(self, clustered_features: NDArray[np.floating]) -> None:
        """Inertia should be a positive float."""
        result = KMeans(n_clusters=3)(clustered_features)
        assert result.inertia > 0.0

    def test_feature_dtype(self, clustered_features: NDArray[np.floating]) -> None:
        """The dtype parameter should be respected for cluster_centers."""
        result = KMeans(n_clusters=3, dtype=np.float64)(clustered_features)
        assert result.cluster_centers.dtype == np.float64

    def test_invalid_n_clusters_raises(self, clustered_features: NDArray[np.floating]) -> None:
        """n_clusters=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            KMeans(n_clusters=0)(clustered_features)


# ---------------------------------------------------------------------------
# Feature-vector clustering tests: MiniBatchKMeans
# ---------------------------------------------------------------------------


class TestMiniBatchKMeans:
    """Tests for ``MiniBatchKMeans`` clustering."""

    def test_minibatch_agrees_with_kmeans(self, clustered_features: NDArray[np.floating]) -> None:
        """MiniBatch KMeans should produce the same number of clusters as KMeans."""
        result_kmeans = KMeans(n_clusters=3)(clustered_features)
        result_mb = MiniBatchKMeans(n_clusters=3)(clustered_features)
        assert result_mb.n_clusters == result_kmeans.n_clusters

    def test_minibatch_batch_size(self, clustered_features: NDArray[np.floating]) -> None:
        """The batch_size parameter should be accepted without error."""
        result = MiniBatchKMeans(n_clusters=3, batch_size=16)(clustered_features)
        assert isinstance(result, FeatureClusteringResult)
        assert result.n_clusters == 3

    def test_invalid_n_clusters_raises(self, clustered_features: NDArray[np.floating]) -> None:
        """n_clusters=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            MiniBatchKMeans(n_clusters=0)(clustered_features)

    def test_invalid_batch_size_raises(self, clustered_features: NDArray[np.floating]) -> None:
        """batch_size=0 must raise ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            MiniBatchKMeans(batch_size=0)(clustered_features)


# ---------------------------------------------------------------------------
# Feature-vector clustering tests: RegularSpace
# ---------------------------------------------------------------------------


class TestRegularSpace:
    """Tests for ``RegularSpace`` clustering."""

    def test_regspace_basic(self, clustered_features: NDArray[np.floating]) -> None:
        """Regspace with dmin=2.0 should find approximately 3 clusters."""
        result = RegularSpace(dmin=2.0)(clustered_features)
        assert isinstance(result, FeatureClusteringResult)
        assert result.n_clusters >= 2
        assert result.cluster_centers.shape[1] == 2
        assert len(result.medoid_frames) == result.n_clusters

    def test_regspace_n_clusters_determined_by_dmin(
        self, clustered_features: NDArray[np.floating]
    ) -> None:
        """Smaller dmin should produce more clusters."""
        result_large = RegularSpace(dmin=3.0)(clustered_features)
        result_small = RegularSpace(dmin=0.5)(clustered_features)
        assert result_small.n_clusters >= result_large.n_clusters

    def test_invalid_dmin_raises(self, clustered_features: NDArray[np.floating]) -> None:
        """A negative dmin must raise ValueError."""
        with pytest.raises(ValueError, match="dmin must be positive"):
            RegularSpace(dmin=-1.0)(clustered_features)


# ---------------------------------------------------------------------------
# Wrapper dtype / memory tests: verify compute_rmsd_matrix does no redundant copy
# ---------------------------------------------------------------------------


class TestComputeRmsdMatrixNoRedundantCopy:
    """``compute_rmsd_matrix`` must not duplicate the backend's buffer.

    The fix for the 120k-frame OOM relies on two guarantees:

    1. GPU backends return their native ``float32`` (not a ``.astype(np.float64)``).
    2. The wrapper uses ``astype(resolved, copy=False)`` so when the
       backend's dtype already matches the user-resolved dtype, the
       returned buffer is the same object as the backend output.

    These tests drive a stub backend so they do not depend on GPU
    availability.
    """

    @pytest.fixture()
    def tiny_traj(self, backbone_trajectory: md.Trajectory) -> md.Trajectory:
        return backbone_trajectory

    def test_torch_backend_returns_float32(self, tiny_traj: md.Trajectory) -> None:
        """``rmsd_torch`` now returns float32 natively (no float64 upcast)."""
        pytest.importorskip("torch")
        result = compute_rmsd_matrix(tiny_traj, atom_selection="all", backend="torch")
        assert result.rmsd_matrix_nm.dtype == np.float32

    def test_mdtraj_backend_returns_float32(self, tiny_traj: md.Trajectory) -> None:
        """``rmsd_mdtraj`` now allocates float32 (not float64)."""
        result = compute_rmsd_matrix(tiny_traj, atom_selection="all", backend="mdtraj")
        assert result.rmsd_matrix_nm.dtype == np.float32

    def test_numba_backend_returns_float32(self, tiny_traj: md.Trajectory) -> None:
        """``rmsd_numba`` now stores float32 in the result buffer.

        The QCP accumulators and Newton-Raphson state are still
        float64 inside the JIT kernel (numba's ``0.0`` literal is a C
        ``double``), so precision is preserved; only the final
        ``result[i, j] = val`` store truncates to float32.  At
        n=120k this halves the output-matrix footprint from 115 GB
        to 57 GB.
        """
        result = compute_rmsd_matrix(tiny_traj, atom_selection="all", backend="numba")
        assert result.rmsd_matrix_nm.dtype == np.float32

    def test_wrapper_does_not_copy_when_dtype_matches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_traj: md.Trajectory,
    ) -> None:
        """When backend dtype == resolved dtype, the wrapper's output shares memory.

        Monkeypatch the ``mdtraj`` backend with a stub that returns a
        tagged float32 buffer, then check ``np.shares_memory`` between
        the stub's buffer and the wrapper's output.  If the fix
        regresses (e.g. someone re-adds ``.astype(resolved)`` without
        ``copy=False``), this test catches it immediately.
        """
        from mdpp.analysis._backends._rmsd_matrix import rmsd_matrix_backends

        n = tiny_traj.n_frames
        sentinel = np.zeros((n, n), dtype=np.float32)
        sentinel[0, 0] = 3.1415  # marker so we can prove identity

        def stub_backend(
            traj: md.Trajectory,
            atom_indices: NDArray[np.int_],  # noqa: ARG001
        ) -> NDArray[np.floating]:
            assert traj is tiny_traj
            return sentinel

        monkeypatch.setitem(rmsd_matrix_backends._backends, "mdtraj", stub_backend)

        result = compute_rmsd_matrix(
            tiny_traj, atom_selection="all", backend="mdtraj", dtype=np.float32
        )
        assert result.rmsd_matrix_nm.dtype == np.float32
        assert np.shares_memory(result.rmsd_matrix_nm, sentinel), (
            "wrapper allocated a second buffer even though dtypes match"
        )
        assert result.rmsd_matrix_nm[0, 0] == np.float32(3.1415)

    def test_wrapper_casts_when_dtype_differs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_traj: md.Trajectory,
    ) -> None:
        """When the user asks for float64 and the backend gives float32, a cast happens.

        This is the one case where a copy is unavoidable -- confirm
        the result is numerically correct and has the requested dtype.
        """
        from mdpp.analysis._backends._rmsd_matrix import rmsd_matrix_backends

        n = tiny_traj.n_frames
        sentinel = np.full((n, n), 0.125, dtype=np.float32)

        def stub_backend(
            traj: md.Trajectory,  # noqa: ARG001
            atom_indices: NDArray[np.int_],  # noqa: ARG001
        ) -> NDArray[np.floating]:
            return sentinel

        monkeypatch.setitem(rmsd_matrix_backends._backends, "mdtraj", stub_backend)

        result = compute_rmsd_matrix(
            tiny_traj, atom_selection="all", backend="mdtraj", dtype=np.float64
        )
        assert result.rmsd_matrix_nm.dtype == np.float64
        assert not np.shares_memory(result.rmsd_matrix_nm, sentinel)
        np.testing.assert_allclose(result.rmsd_matrix_nm, 0.125)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


def _is_gpu_oom(exc: BaseException) -> bool:
    """Return True if *exc* looks like a GPU out-of-memory error."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "outofmemory" in name or "out of memory" in msg


def _run_rmsd_benchmark(traj: md.Trajectory) -> None:
    """Run all available RMSD matrix backends and print a comparison table.

    Each backend is warmed up (JIT/CUDA context/XLA compilation) with a
    3-frame slice before the timed run on the full trajectory.  The
    mdtraj result is used as the correctness reference (atol=5e-5 nm)
    since mdtraj is the default backend for every analysis function.

    GPU backends that run out of memory on shared/contended GPUs are
    skipped with a printed note instead of failing the test.  Each
    kernel releases its own framework cache via the ``@clean_*_cache``
    decorators, so no pre-run defragmentation is needed.
    """
    ref = compute_rmsd_matrix(traj, atom_selection="all", backend="mdtraj")
    # Symmetrise mdtraj result (md.rmsd loop is not numerically symmetric).
    ref_mat = (ref.rmsd_matrix_nm + ref.rmsd_matrix_nm.T) / 2.0

    backends: list[tuple[str, bool]] = [
        ("mdtraj", True),
        ("numba", True),
        ("cupy", has_cupy),
        ("torch", has_torch),
        ("jax", has_jax),
    ]

    timings: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name, available in backends:
        if not available:
            continue
        try:
            # Warmup
            compute_rmsd_matrix(traj[:3], atom_selection="all", backend=name)  # type: ignore[arg-type]
            t0 = time.perf_counter()
            result = compute_rmsd_matrix(traj, atom_selection="all", backend=name)  # type: ignore[arg-type]
            timings[name] = time.perf_counter() - t0
            if name != "mdtraj":
                np.testing.assert_allclose(result.rmsd_matrix_nm, ref_mat, atol=5e-5)
        except Exception as exc:
            if _is_gpu_oom(exc):
                skipped[name] = "GPU OOM"
                continue
            raise

    n = traj.n_frames
    n_pairs = n * (n - 1) // 2
    print(f"\n  RMSD matrix benchmark: {n} frames, {len(ref.atom_indices)} atoms ({n_pairs} pairs)")
    print(f"  {'Backend':<10s} {'Time (s)':>10s} {'vs mdtraj':>10s}")
    print(f"  {'-' * 32}")
    t_mdtraj = timings.get("mdtraj", 1.0)
    for name, t in sorted(timings.items(), key=lambda x: x[1]):
        speedup = t_mdtraj / t
        print(f"  {name:<10s} {t:>10.4f} {speedup:>9.1f}x")
    for name, reason in skipped.items():
        print(f"  {name:<10s} {'--':>10s} (skipped: {reason})")


def _make_alanine_traj(n_frames: int, n_residues: int) -> md.Trajectory:
    """Synthetic alanine trajectory with N, CA, C backbone atoms per residue."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for res_idx in range(1, n_residues + 1):
        residue = topology.add_residue("ALA", chain, resSeq=res_idx)
        n_atom = topology.add_atom("N", md.element.nitrogen, residue)
        ca = topology.add_atom("CA", md.element.carbon, residue)
        c = topology.add_atom("C", md.element.carbon, residue)
        atoms.extend([n_atom, ca, c])
        topology.add_bond(n_atom, ca)
        topology.add_bond(ca, c)
        if res_idx > 1:
            topology.add_bond(atoms[-6], c)

    rng = np.random.RandomState(99)
    n_atoms = len(atoms)
    base = rng.randn(1, n_atoms, 3).astype(np.float32) * 0.15
    perturbation = rng.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.02
    xyz = base + perturbation
    return md.Trajectory(xyz=xyz, topology=topology)


@pytest.mark.gpu
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_residues"),
    [
        pytest.param(100, 50, id="fast-100f-150a"),
        pytest.param(200, 50, id="fast-200f-150a"),
    ],
)
def test_benchmark_rmsd_backends_fast(n_frames: int, n_residues: int) -> None:
    """Fast benchmark -- all available RMSD matrix backends on small trajectories.

    Completes in seconds on a modern machine.  Verifies every backend
    matches the mdtraj reference (atol=5e-5 nm) as a side effect.

    Run only fast benchmarks:    ``pytest -m "benchmark and not slow"``
    Run all benchmarks:           ``pytest -m benchmark``
    Skip benchmarks:              ``pytest -m "not benchmark"``
    """
    traj = _make_alanine_traj(n_frames, n_residues)
    _run_rmsd_benchmark(traj)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_residues"),
    [
        pytest.param(500, 50, id="slow-500f-150a"),
        pytest.param(1000, 100, id="slow-1000f-300a"),
    ],
)
def test_benchmark_rmsd_backends_slow(n_frames: int, n_residues: int) -> None:
    """Slow benchmark -- RMSD matrix backends on larger trajectories.

    The matrix is O(n_frames^2) so the 1000-frame case computes 500k
    pairs and takes tens of seconds on the single-threaded mdtraj loop.
    Marked ``slow`` so it is deselected by ``-m "not slow"`` in fast CI.

    Run only slow benchmarks:  ``pytest -m "benchmark and slow"``
    """
    traj = _make_alanine_traj(n_frames, n_residues)
    _run_rmsd_benchmark(traj)


# ---------------------------------------------------------------------------
# GROMOS clustering benchmarks
# ---------------------------------------------------------------------------


def _make_synthetic_rmsd_matrix(
    n_frames: int,
    n_clusters: int,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic RMSD matrix with a known cluster structure.

    Places each cluster at a distinct point on a 1-D axis, spacing
    2.0 nm apart; every cluster gets at least ``n_frames //
    n_clusters`` members via a shuffled round-robin assignment.
    Intra-cluster RMSD is pure Gaussian noise at ~0.01 nm, so every
    member stays well within a 0.1 nm cutoff while inter-cluster
    RMSD is always >= 2.0 nm -- GROMOS recovers exactly
    ``n_clusters`` groups at any cutoff in (0.05, 2.0).
    """
    rng = np.random.RandomState(seed)
    # Round-robin assignment so every cluster has >= one member.
    assignments = np.arange(n_frames) % n_clusters
    rng.shuffle(assignments)
    # 1-D positions: cluster k is centred at 2.0 * k.
    centres = np.arange(n_clusters, dtype=np.float32) * 2.0
    positions = centres[assignments] + rng.randn(n_frames).astype(np.float32) * 0.01
    # |a - b| in 1-D is a valid metric and the resulting matrix is
    # symmetric with zero diagonal.
    rmsd = np.abs(positions[:, None] - positions[None, :])
    return rmsd.astype(np.float32)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_clusters"),
    [
        pytest.param(1000, 8, id="fast-1000f-8c"),
        pytest.param(2000, 20, id="fast-2000f-20c"),
    ],
)
def test_benchmark_cluster_conformations_fast(
    n_frames: int,
    n_clusters: int,
) -> None:
    """Fast clustering benchmark -- exercise the numba GROMOS kernel.

    Builds a synthetic RMSD matrix with a known cluster structure and
    measures wall time.  The kernel is warmed up with a small pre-pass
    so the reported time excludes numba JIT compilation.
    """
    rmsd = _make_synthetic_rmsd_matrix(n_frames, n_clusters, seed=1)

    warmup = _make_synthetic_rmsd_matrix(32, 4, seed=99)
    Gromos(cutoff_nm=0.1)(warmup)

    t0 = time.perf_counter()
    result = Gromos(cutoff_nm=0.1)(rmsd)
    elapsed = time.perf_counter() - t0

    print(
        f"\n  GROMOS clustering: n={n_frames} cutoff=0.1 "
        f"({result.n_clusters} clusters)  wall={elapsed:.4f} s"
    )
    # Sanity: we should recover the planted cluster count within a
    # small tolerance (the synthetic centres may occasionally merge or
    # split by one or two clusters at this cutoff).
    assert abs(result.n_clusters - n_clusters) <= max(2, n_clusters // 4)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_clusters"),
    [
        pytest.param(5000, 50, id="slow-5000f-50c"),
        pytest.param(10000, 100, id="slow-10000f-100c"),
    ],
)
def test_benchmark_cluster_conformations_slow(
    n_frames: int,
    n_clusters: int,
) -> None:
    """Slow clustering benchmark -- larger matrices that stress the loop.

    The numba GROMOS loop is O(n^2) for the initial neighbour count
    and then O(|members| * n) per cluster.  For n=10k with 100
    clusters on a modern CPU this should run in a few seconds;
    regressions (e.g. someone re-introducing the O(k*n^2) full
    recompute) would push this to minutes.
    """
    rmsd = _make_synthetic_rmsd_matrix(n_frames, n_clusters, seed=2)

    warmup = _make_synthetic_rmsd_matrix(32, 4, seed=99)
    Gromos(cutoff_nm=0.1)(warmup)

    t0 = time.perf_counter()
    result = Gromos(cutoff_nm=0.1)(rmsd)
    elapsed = time.perf_counter() - t0

    print(
        f"\n  GROMOS clustering: n={n_frames} cutoff=0.1 "
        f"({result.n_clusters} clusters)  wall={elapsed:.4f} s"
    )


# ---------------------------------------------------------------------------
# DBSCAN backend benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_frames", "n_clusters"),
    [
        pytest.param(1000, 8, id="fast-1000f-8c"),
        pytest.param(2000, 20, id="fast-2000f-20c"),
    ],
)
def test_benchmark_dbscan_backends_fast(n_frames: int, n_clusters: int) -> None:
    """DBSCAN backend benchmark: numba vs sklearn."""
    rmsd = _make_synthetic_rmsd_matrix(n_frames, n_clusters, seed=3)

    # Warmup
    warmup = _make_synthetic_rmsd_matrix(32, 4, seed=99)
    DBSCAN(eps=0.1)(warmup)
    DBSCAN(eps=0.1, backend="sklearn")(warmup)

    timings: dict[str, float] = {}
    for backend in ("numba", "sklearn"):
        t0 = time.perf_counter()
        result = DBSCAN(eps=0.1, min_samples=3, backend=backend)(rmsd)  # noqa: F841
        timings[backend] = time.perf_counter() - t0

    print(
        f"\n  DBSCAN benchmark: n={n_frames}"
        f"  numba={timings['numba']:.4f}s  sklearn={timings['sklearn']:.4f}s"
        f"  speedup={timings['sklearn'] / timings['numba']:.1f}x"
    )
