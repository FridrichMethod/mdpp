"""Tests for plot_delta_rmsf."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from mdpp.analysis.metrics import DeltaRMSFResult
from mdpp.plots.timeseries import plot_delta_rmsf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    delta: list[float],
    residue_ids: list[int] | None = None,
    sem: list[float] | None = None,
) -> DeltaRMSFResult:
    return DeltaRMSFResult(
        delta_rmsf_nm=np.array(delta, dtype=np.float64),
        residue_ids=np.array(residue_ids, dtype=np.int_) if residue_ids else None,
        sem_nm=np.array(sem, dtype=np.float64) if sem else None,
    )


# ---------------------------------------------------------------------------
# Tests -- basic plot structure
# ---------------------------------------------------------------------------


class TestPlotDeltaRMSFBasic:
    def test_returns_axes(self) -> None:
        result = _make_result([0.01, -0.02, 0.03], residue_ids=[1, 2, 3])
        fig, ax = plt.subplots()
        ret = plot_delta_rmsf(result, ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_creates_own_axis(self) -> None:
        result = _make_result([0.01, -0.02])
        ax = plot_delta_rmsf(result)
        assert ax is not None
        assert ax.get_ylabel() == r"$\Delta$RMSF (Å)"
        assert ax.get_xlabel() == "Residue ID"
        fig = ax.get_figure()
        assert fig is not None
        plt.close(fig)  # type: ignore[arg-type]

    def test_axis_labels(self) -> None:
        result = _make_result([0.01, -0.02], residue_ids=[10, 20])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        assert ax.get_ylabel() == r"$\Delta$RMSF (Å)"
        assert ax.get_xlabel() == "Residue ID"
        plt.close(fig)

    def test_draws_line(self) -> None:
        result = _make_result([0.01, -0.02, 0.03])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        # Should have the ΔRMSF line + the axhline
        assert len(ax.get_lines()) >= 2
        plt.close(fig)

    def test_zero_line_present(self) -> None:
        result = _make_result([0.01, -0.02])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        lines = ax.get_lines()
        assert any(np.allclose(line.get_ydata(), 0.0) for line in lines)
        plt.close(fig)

    def test_no_residue_ids_uses_one_based_index(self) -> None:
        result = _make_result([0.01, -0.02, 0.03])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        xlim = ax.get_xlim()
        assert xlim[0] <= 1.0
        assert xlim[1] >= 3.0
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests -- no SEM (single replica)
# ---------------------------------------------------------------------------


class TestPlotDeltaRMSFNoSEM:
    def test_no_sem_no_fill(self) -> None:
        """Without SEM data, no fill_between collections should be drawn."""
        result = _make_result([0.01, -0.02, 0.03])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_no_sem_still_has_line(self) -> None:
        result = _make_result([0.01, -0.02])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        # ΔRMSF line + axhline
        assert len(ax.get_lines()) >= 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests -- with SEM band
# ---------------------------------------------------------------------------


class TestPlotDeltaRMSFWithSEM:
    def test_sem_band_drawn(self) -> None:
        result = _make_result([0.01, -0.02], sem=[0.005, 0.003])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        # Two fill_between collections (positive + negative regions)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_default_labels_in_legend(self) -> None:
        result = _make_result([0.01, -0.02], sem=[0.005, 0.003])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "system B more flexible" in legend_texts
        assert "system A more flexible" in legend_texts
        plt.close(fig)

    def test_custom_labels_in_legend(self) -> None:
        result = _make_result([0.01, -0.02], sem=[0.005, 0.003])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax, labels=("BirA", "TurboID"))
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "TurboID more flexible" in legend_texts
        assert "BirA more flexible" in legend_texts
        plt.close(fig)

    def test_custom_colors(self) -> None:
        result = _make_result([0.01, -0.02], sem=[0.005, 0.003])
        fig, ax = plt.subplots()
        plot_delta_rmsf(
            result,
            ax=ax,
            positive_color="orange",
            negative_color="purple",
        )
        collections = ax.collections
        assert len(collections) >= 2
        pos_fc = tuple(collections[0].get_facecolor()[0])  # type: ignore[arg-type]
        neg_fc = tuple(collections[1].get_facecolor()[0])  # type: ignore[arg-type]
        assert pos_fc == pytest.approx(to_rgba("orange", alpha=0.3))
        assert neg_fc == pytest.approx(to_rgba("purple", alpha=0.3))
        plt.close(fig)

    def test_custom_sem_alpha(self) -> None:
        result = _make_result([0.01, -0.02], sem=[0.005, 0.003])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax, sem_alpha=0.6)
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if len(fc) > 0:
                rgba = tuple(fc[0])  # type: ignore[arg-type]
                assert rgba[3] == pytest.approx(0.6)
        plt.close(fig)

    def test_all_positive_with_sem(self) -> None:
        result = _make_result([0.01, 0.02, 0.03], sem=[0.005, 0.005, 0.005])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_all_negative_with_sem(self) -> None:
        result = _make_result([-0.01, -0.02, -0.03], sem=[0.005, 0.005, 0.005])
        fig, ax = plt.subplots()
        plot_delta_rmsf(result, ax=ax)
        assert len(ax.collections) >= 1
        plt.close(fig)
