"""Tests for clustering plot functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle

from mdpp.analysis.clustering import (
    ClusteringResult,
    FeatureClusteringResult,
    KMeans,
    MiniBatchKMeans,
)
from mdpp.analysis.decomposition import PCAResult, compute_pca
from mdpp.plots.clustering import plot_cluster_populations, plot_feature_clustering


@pytest.fixture()
def pca_result() -> PCAResult:
    rng = np.random.default_rng(42)
    return compute_pca(rng.normal(size=(200, 20)), n_components=5)


@pytest.fixture()
def kmeans_result(pca_result: PCAResult) -> FeatureClusteringResult:
    return KMeans(n_clusters=3)(pca_result.projections)


@pytest.fixture()
def clustering_result() -> ClusteringResult:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 5, size=200).astype(np.int_)
    medoids = np.array([0, 40, 80, 120, 160], dtype=np.int_)
    return ClusteringResult(labels=labels, n_clusters=5, medoid_frames=medoids)


@pytest.fixture()
def clustering_result_with_noise() -> ClusteringResult:
    rng = np.random.default_rng(7)
    labels = rng.integers(-1, 4, size=200).astype(np.int_)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    medoids = np.arange(n_clusters, dtype=np.int_)
    return ClusteringResult(labels=labels, n_clusters=n_clusters, medoid_frames=medoids)


# ---------------------------------------------------------------------------
# plot_feature_clustering
# ---------------------------------------------------------------------------


class TestPlotFeatureClustering:
    def test_returns_axes(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        fig, ax = plt.subplots()
        ret = plot_feature_clustering(kmeans_result, pca_result, ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_creates_own_axis(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        ax = plot_feature_clustering(kmeans_result, pca_result)
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"
        fig = ax.get_figure()
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_pca_labels(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        fig, ax = plt.subplots()
        plot_feature_clustering(kmeans_result, pca_result, ax=ax)
        assert ax.get_xlabel() == "PC1"
        assert ax.get_ylabel() == "PC2"
        plt.close(fig)

    def test_pca_custom_indices(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        fig, ax = plt.subplots()
        plot_feature_clustering(kmeans_result, pca_result, x_index=1, y_index=2, ax=ax)
        assert ax.get_xlabel() == "PC2"
        assert ax.get_ylabel() == "PC3"
        plt.close(fig)

    def test_raw_array(self, kmeans_result: FeatureClusteringResult) -> None:
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(200, 3)).astype(np.float32)
        fig, ax = plt.subplots()
        plot_feature_clustering(kmeans_result, raw, ax=ax)
        assert ax.get_xlabel() == "Component 1"
        assert ax.get_ylabel() == "Component 2"
        plt.close(fig)

    def test_scatter_and_centers(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        fig, ax = plt.subplots()
        plot_feature_clustering(kmeans_result, pca_result, ax=ax)
        # Two scatter collections: data points + cluster centers
        collections = ax.collections
        assert len(collections) == 2
        plt.close(fig)

    def test_no_centers(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        fig, ax = plt.subplots()
        plot_feature_clustering(kmeans_result, pca_result, show_centers=False, ax=ax)
        # Only data points, no center markers
        assert len(ax.collections) == 1
        plt.close(fig)

    def test_center_count_matches_clusters(
        self, kmeans_result: FeatureClusteringResult, pca_result: PCAResult
    ) -> None:
        fig, ax = plt.subplots()
        plot_feature_clustering(kmeans_result, pca_result, ax=ax)
        # Second collection is the centers
        center_offsets = np.asarray(ax.collections[1].get_offsets())
        assert len(center_offsets) == kmeans_result.n_clusters
        plt.close(fig)

    def test_minibatch_result(self, pca_result: PCAResult) -> None:
        mb = MiniBatchKMeans(n_clusters=4)(pca_result.projections)
        fig, ax = plt.subplots()
        ret = plot_feature_clustering(mb, pca_result, ax=ax)
        assert ret is ax
        assert len(ax.collections) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_cluster_populations
# ---------------------------------------------------------------------------


class TestPlotClusterPopulations:
    def test_returns_axes(self, clustering_result: ClusteringResult) -> None:
        fig, ax = plt.subplots()
        ret = plot_cluster_populations(clustering_result, ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_creates_own_axis(self, clustering_result: ClusteringResult) -> None:
        ax = plot_cluster_populations(clustering_result)
        assert ax.get_xlabel() == "Cluster (ranked)"
        assert ax.get_ylabel() == "Frames"
        fig = ax.get_figure()
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_bar_count(self, clustering_result: ClusteringResult) -> None:
        fig, ax = plt.subplots()
        plot_cluster_populations(clustering_result, ax=ax)
        # 5 clusters -> 5 bars
        assert len(ax.patches) == 5
        plt.close(fig)

    def test_top_k_limits_bars(self, clustering_result: ClusteringResult) -> None:
        fig, ax = plt.subplots()
        plot_cluster_populations(clustering_result, top_k=3, ax=ax)
        assert len(ax.patches) == 3
        plt.close(fig)

    def test_bars_sorted_descending(self, clustering_result: ClusteringResult) -> None:
        fig, ax = plt.subplots()
        plot_cluster_populations(clustering_result, ax=ax)
        heights = [cast(Rectangle, p).get_height() for p in ax.patches]
        assert heights == sorted(heights, reverse=True)
        plt.close(fig)

    def test_with_noise_labels(self, clustering_result_with_noise: ClusteringResult) -> None:
        fig, ax = plt.subplots()
        plot_cluster_populations(clustering_result_with_noise, ax=ax)
        # Noise frames (label -1) should be excluded
        assert len(ax.patches) == clustering_result_with_noise.n_clusters
        plt.close(fig)

    def test_custom_color(self, clustering_result: ClusteringResult) -> None:
        fig, ax = plt.subplots()
        plot_cluster_populations(clustering_result, color="red", ax=ax)
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_feature_clustering_result(self, kmeans_result: FeatureClusteringResult) -> None:
        fig, ax = plt.subplots()
        ret = plot_cluster_populations(kmeans_result, ax=ax)
        assert ret is ax
        assert len(ax.patches) == kmeans_result.n_clusters
        plt.close(fig)

    def test_empty_clusters(self) -> None:
        labels = np.full(50, -1, dtype=np.int_)
        result = ClusteringResult(
            labels=labels, n_clusters=0, medoid_frames=np.array([], dtype=np.int_)
        )
        fig, ax = plt.subplots()
        ret = plot_cluster_populations(result, ax=ax)
        assert ret is ax
        assert len(ax.patches) == 0
        plt.close(fig)
