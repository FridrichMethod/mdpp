"""Tests for mdpp.chem.similarity."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import DataStructs

from mdpp.chem.fingerprints import gen_fp
from mdpp.chem.similarity import (
    BULK_SIM_FUNCS,
    CLUSTERING_SIM_METRICS,
    PARALLEL_SIM_KERNELS,
    SIM_FUNCS,
    calc_bulk_sim,
    calc_sim,
    calc_similarities,
)


def _fps_to_numpy(fps: list) -> np.ndarray:
    """Convert RDKit fingerprints to a 2D binary numpy array."""
    n_bits = fps[0].GetNumBits()
    arr = np.zeros((len(fps), n_bits), dtype=np.int8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


class TestCalcSim:
    def test_identical_molecules(self, ethanol):
        fp = gen_fp(ethanol)
        assert calc_sim(fp, fp) == pytest.approx(1.0)

    def test_different_molecules(self, ethanol, aspirin):
        fp1 = gen_fp(ethanol)
        fp2 = gen_fp(aspirin)
        sim = calc_sim(fp1, fp2)
        assert 0.0 < sim < 1.0

    @pytest.mark.parametrize("metric", list(SIM_FUNCS.keys()))
    def test_all_metrics(self, ethanol, aspirin, metric):
        fp1 = gen_fp(ethanol)
        fp2 = gen_fp(aspirin)
        sim = calc_sim(fp1, fp2, similarity_metric=metric)
        assert -1.0 <= sim <= 1.0

    def test_symmetry(self, ethanol, aspirin):
        fp1 = gen_fp(ethanol)
        fp2 = gen_fp(aspirin)
        assert calc_sim(fp1, fp2) == pytest.approx(calc_sim(fp2, fp1))

    def test_invalid_metric(self, ethanol):
        fp = gen_fp(ethanol)
        with pytest.raises(ValueError, match="similarity_metric should be one of"):
            calc_sim(fp, fp, similarity_metric="invalid")


class TestCalcBulkSim:
    def test_length_matches_targets(self, ethanol, aspirin, benzene):
        fp_query = gen_fp(ethanol)
        fps_target = [gen_fp(aspirin), gen_fp(benzene)]
        result = calc_bulk_sim(fp_query, fps_target)
        assert len(result) == 2

    def test_consistent_with_pairwise(self, ethanol, aspirin):
        fp1 = gen_fp(ethanol)
        fp2 = gen_fp(aspirin)
        bulk = calc_bulk_sim(fp1, [fp2])
        pairwise = calc_sim(fp1, fp2)
        assert bulk[0] == pytest.approx(pairwise)

    def test_all_bulk_metrics(self, ethanol, aspirin):
        fp1 = gen_fp(ethanol)
        fp2 = gen_fp(aspirin)
        for metric in BULK_SIM_FUNCS:
            result = calc_bulk_sim(fp1, [fp2], similarity_metric=metric)
            assert len(result) == 1

    def test_invalid_metric(self, ethanol):
        fp = gen_fp(ethanol)
        with pytest.raises(ValueError, match="similarity_metric should be one of"):
            calc_bulk_sim(fp, [fp], similarity_metric="invalid")


class TestCalcSimilarities:
    @pytest.fixture()
    def mol_fps(self, ethanol, aspirin, benzene):
        rdkit_fps = [gen_fp(ethanol), gen_fp(aspirin), gen_fp(benzene)]
        return rdkit_fps, _fps_to_numpy(rdkit_fps)

    def test_output_length(self, mol_fps):
        _, np_fps = mol_fps
        result = calc_similarities(np_fps, PARALLEL_SIM_KERNELS["tanimoto"])
        n = len(np_fps)
        assert len(result) == n * (n - 1) // 2

    def test_output_dtype(self, mol_fps):
        _, np_fps = mol_fps
        result = calc_similarities(np_fps, PARALLEL_SIM_KERNELS["tanimoto"])
        assert result.dtype == np.float32

    @pytest.mark.parametrize("metric", list(PARALLEL_SIM_KERNELS.keys()))
    def test_matches_rdkit(self, mol_fps, metric):
        """Numba kernels must agree with RDKit bulk similarity for every metric."""
        rdkit_fps, np_fps = mol_fps
        parallel_sims = calc_similarities(np_fps, PARALLEL_SIM_KERNELS[metric])

        rdkit_sims = np.concatenate([
            calc_bulk_sim(rdkit_fps[i], rdkit_fps[:i], similarity_metric=metric)
            for i in range(1, len(rdkit_fps))
        ])
        np.testing.assert_allclose(parallel_sims, rdkit_sims, atol=1e-6)

    def test_self_similarity_is_one(self, ethanol):
        fp = gen_fp(ethanol)
        np_fps = _fps_to_numpy([fp, fp])
        result = calc_similarities(np_fps, PARALLEL_SIM_KERNELS["tanimoto"])
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_two_molecules(self, ethanol, aspirin):
        fp1 = gen_fp(ethanol)
        fp2 = gen_fp(aspirin)
        rdkit_sim = calc_sim(fp1, fp2)
        np_fps = _fps_to_numpy([fp1, fp2])
        parallel_sims = calc_similarities(np_fps, PARALLEL_SIM_KERNELS["tanimoto"])
        assert parallel_sims[0] == pytest.approx(rdkit_sim, abs=1e-6)


class TestConstants:
    def test_parallel_and_bulk_share_metrics(self):
        parallel = set(PARALLEL_SIM_KERNELS)
        bulk = set(BULK_SIM_FUNCS) - {"russel"}
        assert parallel == bulk

    def test_russel_excluded_from_clustering(self):
        assert "russel" not in CLUSTERING_SIM_METRICS

    def test_clustering_metrics_subset_of_bulk(self):
        assert frozenset(BULK_SIM_FUNCS) >= CLUSTERING_SIM_METRICS

    def test_clustering_metrics_nonempty(self):
        assert len(CLUSTERING_SIM_METRICS) > 0
