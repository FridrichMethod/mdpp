"""Tests for PCA QC plotting functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdpp.analysis.decomposition import PCAResult, compute_pca
from mdpp.plots.scatter import plot_pca_cumulative_variance, plot_pca_scree


@pytest.fixture()
def pca_result_a() -> PCAResult:
    rng = np.random.default_rng(42)
    return compute_pca(rng.normal(size=(100, 20)), n_components=10)


@pytest.fixture()
def pca_result_b() -> PCAResult:
    rng = np.random.default_rng(99)
    return compute_pca(rng.normal(size=(100, 20)), n_components=10)


class TestPlotPcaScree:
    def test_returns_axes(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        ret = plot_pca_scree(pca_result_a, ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_creates_own_axis(self, pca_result_a: PCAResult) -> None:
        ax = plot_pca_scree(pca_result_a)
        assert ax.get_ylabel() == "Variance Explained (%)"
        fig = ax.get_figure()
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_single_result(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_scree(pca_result_a, ax=ax)
        # 10 bars
        patches = ax.patches
        assert len(patches) == 10
        plt.close(fig)

    def test_two_results(self, pca_result_a: PCAResult, pca_result_b: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_scree(pca_result_a, pca_result_b, labels=["A", "B"], ax=ax)
        # 10 bars each = 20 total
        assert len(ax.patches) == 20
        plt.close(fig)

    def test_labels_in_legend(self, pca_result_a: PCAResult, pca_result_b: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_scree(pca_result_a, pca_result_b, labels=["sys A", "sys B"], ax=ax)
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "sys A" in texts
        assert "sys B" in texts
        plt.close(fig)

    def test_custom_colors(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_scree(pca_result_a, colors=["red"], ax=ax)
        assert len(ax.patches) == 10
        plt.close(fig)

    def test_no_labels_no_legend(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_scree(pca_result_a, ax=ax)
        assert ax.get_legend() is None
        plt.close(fig)


class TestPlotPcaCumulativeVariance:
    def test_returns_axes(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        ret = plot_pca_cumulative_variance(pca_result_a, ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_creates_own_axis(self, pca_result_a: PCAResult) -> None:
        ax = plot_pca_cumulative_variance(pca_result_a)
        assert ax.get_ylabel() == "Cumulative Variance (%)"
        fig = ax.get_figure()
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_single_result(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_cumulative_variance(pca_result_a, ax=ax)
        # 1 data line + 2 threshold lines
        assert len(ax.get_lines()) == 3
        plt.close(fig)

    def test_two_results(self, pca_result_a: PCAResult, pca_result_b: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_cumulative_variance(pca_result_a, pca_result_b, labels=["A", "B"], ax=ax)
        # 2 data lines + 2 threshold lines
        assert len(ax.get_lines()) == 4
        plt.close(fig)

    def test_custom_thresholds(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_cumulative_variance(pca_result_a, thresholds=[50, 75, 95], ax=ax)
        # 1 data line + 3 threshold lines
        assert len(ax.get_lines()) == 4
        plt.close(fig)

    def test_no_thresholds(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_cumulative_variance(pca_result_a, thresholds=[], ax=ax)
        # 1 data line only
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_ylim_zero_to_hundred(self, pca_result_a: PCAResult) -> None:
        fig, ax = plt.subplots()
        plot_pca_cumulative_variance(pca_result_a, ax=ax)
        ylim = ax.get_ylim()
        assert ylim[0] == pytest.approx(0.0)
        assert ylim[1] == pytest.approx(100.0)
        plt.close(fig)

    def test_monotonically_increasing(self, pca_result_a: PCAResult) -> None:
        """Cumulative variance line should be non-decreasing."""
        fig, ax = plt.subplots()
        plot_pca_cumulative_variance(pca_result_a, thresholds=[], ax=ax)
        ydata = ax.get_lines()[0].get_ydata()
        assert np.all(np.diff(ydata) >= -1e-10)
        plt.close(fig)
