"""Tests for mdpp.chem.fingerprints."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import DataStructs

from mdpp.chem.fingerprints import (
    FP_GENERATORS,
    FingerprintClusteringResult,
    cluster_fps,
    cluster_fps_parallel,
    gen_fp,
)
from mdpp.chem.similarity import CLUSTERING_SIM_METRICS


def _fps_to_numpy(fps: list) -> np.ndarray:
    """Convert RDKit fingerprints to a 2D binary numpy array."""
    n_bits = fps[0].GetNumBits()
    arr = np.zeros((len(fps), n_bits), dtype=np.int8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


class TestGenFp:
    def test_morgan_returns_bitvect(self, ethanol):
        fp = gen_fp(ethanol, fp_type="morgan")
        assert isinstance(fp, DataStructs.cDataStructs.ExplicitBitVect)

    def test_morgan_bit_length(self, ethanol):
        fp = gen_fp(ethanol, fp_type="morgan")
        assert fp.GetNumBits() == 1024

    @pytest.mark.parametrize("fp_type", list(FP_GENERATORS.keys()))
    def test_all_fp_types(self, ethanol, fp_type):
        fp = gen_fp(ethanol, fp_type=fp_type)
        assert isinstance(fp, DataStructs.cDataStructs.ExplicitBitVect)

    def test_case_insensitive(self, ethanol):
        fp_lower = gen_fp(ethanol, fp_type="morgan")
        fp_upper = gen_fp(ethanol, fp_type="MORGAN")
        assert fp_lower == fp_upper

    def test_invalid_fp_type(self, ethanol):
        with pytest.raises(ValueError, match="fp_type should be one of"):
            gen_fp(ethanol, fp_type="invalid")

    def test_different_types_differ(self, aspirin):
        fp_morgan = gen_fp(aspirin, fp_type="morgan")
        fp_rdkit = gen_fp(aspirin, fp_type="rdkit")
        assert fp_morgan != fp_rdkit


class TestClusterFps:
    def test_identical_molecules_single_cluster(self, ethanol):
        fp = gen_fp(ethanol)
        result = cluster_fps([fp, fp, fp], cutoff=0.5)
        assert result.n_clusters == 1
        assert len(result.clusters[0]) == 3

    def test_returns_dataclass(self, ethanol, aspirin):
        fps = [gen_fp(ethanol), gen_fp(aspirin)]
        result = cluster_fps(fps, cutoff=0.5)
        assert isinstance(result, FingerprintClusteringResult)
        assert isinstance(result.clusters, tuple)
        assert isinstance(result.n_clusters, int)

    def test_clusters_sorted_by_size(self, ethanol, aspirin, benzene):
        fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        result = cluster_fps(fps, cutoff=0.1)
        sizes = [len(c) for c in result.clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_all_molecules_assigned(self, ethanol, aspirin, benzene):
        fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        result = cluster_fps(fps, cutoff=0.5)
        all_indices = sorted(idx for cluster in result.clusters for idx in cluster)
        assert all_indices == [0, 1, 2]

    def test_single_fingerprint(self, ethanol):
        fp = gen_fp(ethanol)
        result = cluster_fps([fp], cutoff=0.5)
        assert result.n_clusters == 1
        assert result.clusters == ((0,),)

    def test_invalid_metric_rejected(self, ethanol):
        fps = [gen_fp(ethanol), gen_fp(ethanol)]
        with pytest.raises(ValueError, match="similarity_metric should be one of"):
            cluster_fps(fps, similarity_metric="russel")

    @pytest.mark.parametrize(
        "metric",
        sorted(CLUSTERING_SIM_METRICS & {"tanimoto", "dice", "cosine"}),
    )
    def test_clustering_with_different_metrics(self, ethanol, aspirin, benzene, metric):
        fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        result = cluster_fps(fps, cutoff=0.5, similarity_metric=metric)
        all_indices = sorted(idx for cluster in result.clusters for idx in cluster)
        assert all_indices == [0, 1, 2]


class TestClusterFpsParallel:
    def test_identical_molecules_single_cluster(self, ethanol):
        fp = gen_fp(ethanol)
        np_fps = _fps_to_numpy([fp, fp, fp])
        result = cluster_fps_parallel(np_fps, cutoff=0.5)
        assert result.n_clusters == 1
        assert len(result.clusters[0]) == 3

    def test_returns_dataclass(self, ethanol, aspirin):
        fps = [gen_fp(ethanol), gen_fp(aspirin)]
        np_fps = _fps_to_numpy(fps)
        result = cluster_fps_parallel(np_fps, cutoff=0.5)
        assert isinstance(result, FingerprintClusteringResult)

    def test_all_molecules_assigned(self, ethanol, aspirin, benzene):
        fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        np_fps = _fps_to_numpy(fps)
        result = cluster_fps_parallel(np_fps, cutoff=0.5)
        all_indices = sorted(idx for cluster in result.clusters for idx in cluster)
        assert all_indices == [0, 1, 2]

    def test_single_fingerprint(self, ethanol):
        fp = gen_fp(ethanol)
        np_fps = _fps_to_numpy([fp])
        result = cluster_fps_parallel(np_fps, cutoff=0.5)
        assert result.n_clusters == 1
        assert result.clusters == ((0,),)

    def test_rejects_1d_array(self, ethanol):
        fp = gen_fp(ethanol)
        np_fps = _fps_to_numpy([fp])
        with pytest.raises(ValueError, match=r"2D numpy\.ndarray"):
            cluster_fps_parallel(np_fps[0], cutoff=0.5)

    def test_invalid_metric_rejected(self, ethanol):
        fp = gen_fp(ethanol)
        np_fps = _fps_to_numpy([fp, fp])
        with pytest.raises(ValueError, match="similarity_metric should be one of"):
            cluster_fps_parallel(np_fps, similarity_metric="russel")


class TestClusterFpsParallelMatchesSerial:
    """The parallel and serial clustering paths must produce identical clusters."""

    @pytest.mark.parametrize(
        "metric",
        sorted((CLUSTERING_SIM_METRICS & set(FP_GENERATORS)) | {"tanimoto", "dice", "cosine"}),
    )
    def test_same_clusters(self, ethanol, aspirin, benzene, metric):
        rdkit_fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        np_fps = _fps_to_numpy(rdkit_fps)

        serial = cluster_fps(rdkit_fps, cutoff=0.5, similarity_metric=metric)
        parallel = cluster_fps_parallel(np_fps, cutoff=0.5, similarity_metric=metric)

        assert serial.n_clusters == parallel.n_clusters
        assert serial.clusters == parallel.clusters

    def test_same_clusters_tight_cutoff(self, ethanol, aspirin, benzene):
        rdkit_fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        np_fps = _fps_to_numpy(rdkit_fps)

        serial = cluster_fps(rdkit_fps, cutoff=0.9)
        parallel = cluster_fps_parallel(np_fps, cutoff=0.9)

        assert serial.clusters == parallel.clusters

    def test_same_clusters_loose_cutoff(self, ethanol, aspirin, benzene):
        rdkit_fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        np_fps = _fps_to_numpy(rdkit_fps)

        serial = cluster_fps(rdkit_fps, cutoff=0.01)
        parallel = cluster_fps_parallel(np_fps, cutoff=0.01)

        assert serial.clusters == parallel.clusters
