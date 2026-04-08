"""Tests for the ``color`` parameter on timeseries plot functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from mdpp.analysis.contacts import NativeContactResult
from mdpp.analysis.hbond import HBondResult
from mdpp.analysis.metrics import (
    RadiusOfGyrationResult,
    RMSDResult,
    RMSFResult,
    SASAResult,
)
from mdpp.plots.timeseries import (
    plot_hbond_counts,
    plot_native_contacts,
    plot_radius_of_gyration,
    plot_rmsd,
    plot_rmsf,
    plot_rmsf_average,
    plot_sasa,
)

# ---------------------------------------------------------------------------
# Fixtures -- lightweight synthetic result objects
# ---------------------------------------------------------------------------

_TIME_PS = np.array([0.0, 10.0, 20.0], dtype=np.float64)
_ATOM_IDX = np.array([0, 1], dtype=np.int_)
_RESIDUE_IDS = np.array([1, 2], dtype=np.int_)


@pytest.fixture()
def rmsd_result() -> RMSDResult:
    return RMSDResult(
        time_ps=_TIME_PS,
        rmsd_nm=np.array([0.1, 0.2, 0.15], dtype=np.float64),
        atom_indices=_ATOM_IDX,
    )


@pytest.fixture()
def rmsf_result() -> RMSFResult:
    return RMSFResult(
        rmsf_nm=np.array([0.05, 0.10], dtype=np.float64),
        atom_indices=_ATOM_IDX,
        residue_ids=_RESIDUE_IDS,
    )


@pytest.fixture()
def sasa_result() -> SASAResult:
    return SASAResult(
        time_ps=_TIME_PS,
        values_nm2=np.array([[0.5, 0.3], [0.6, 0.4], [0.55, 0.35]], dtype=np.float64),
        atom_indices=_ATOM_IDX,
        mode="residue",
        residue_ids=_RESIDUE_IDS,
    )


@pytest.fixture()
def hbond_result() -> HBondResult:
    return HBondResult(
        time_ps=_TIME_PS,
        triplets=np.array([[0, 1, 2]], dtype=np.int_),
        presence=np.array([[True, False, True]]),
        count_per_frame=np.array([1, 0, 1], dtype=np.int_),
        occupancy=np.array([0.67], dtype=np.float64),
        method="baker_hubbard",
        distance_cutoff_nm=0.35,
        angle_cutoff_deg=120.0,
    )


@pytest.fixture()
def rg_result() -> RadiusOfGyrationResult:
    return RadiusOfGyrationResult(
        time_ps=_TIME_PS,
        radius_gyration_nm=np.array([1.0, 1.1, 1.05], dtype=np.float64),
        atom_indices=_ATOM_IDX,
    )


@pytest.fixture()
def native_contact_result() -> NativeContactResult:
    return NativeContactResult(
        time_ps=_TIME_PS,
        fraction=np.array([1.0, 0.8, 0.9], dtype=np.float64),
        native_pairs=np.array([[0, 1]], dtype=np.int_),
        cutoff_nm=0.45,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_line_color(ax: plt.Axes, index: int = 0) -> tuple:
    """Return the RGBA color of the *index*-th line on *ax*."""
    return to_rgba(ax.get_lines()[index].get_color())


# ---------------------------------------------------------------------------
# Tests -- color=None (default) uses property cycle
# ---------------------------------------------------------------------------


class TestColorDefaultNone:
    """When ``color`` is not specified, the property cycle picks the color."""

    def test_plot_rmsd_default(self, rmsd_result: RMSDResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsd(rmsd_result, ax=ax)
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_rmsf_default(self, rmsf_result: RMSFResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsf(rmsf_result, ax=ax)
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_sasa_default(self, sasa_result: SASAResult) -> None:
        fig, ax = plt.subplots()
        plot_sasa(sasa_result, ax=ax, aggregate="sum")
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_hbond_counts_default(self, hbond_result: HBondResult) -> None:
        fig, ax = plt.subplots()
        plot_hbond_counts(hbond_result, ax=ax)
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_rg_default(self, rg_result: RadiusOfGyrationResult) -> None:
        fig, ax = plt.subplots()
        plot_radius_of_gyration(rg_result, ax=ax)
        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_plot_native_contacts_default(self, native_contact_result: NativeContactResult) -> None:
        fig, ax = plt.subplots()
        plot_native_contacts(native_contact_result, ax=ax)
        assert len(ax.get_lines()) == 1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests -- explicit color is applied to the line
# ---------------------------------------------------------------------------


class TestColorExplicit:
    """When ``color`` is specified, the line must use that exact color."""

    def test_plot_rmsd_red(self, rmsd_result: RMSDResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsd(rmsd_result, ax=ax, color="red")
        assert _get_line_color(ax) == to_rgba("red")
        plt.close(fig)

    def test_plot_rmsd_with_moving_average(self, rmsd_result: RMSDResult) -> None:
        """Both raw trace and MA overlay should share the specified color."""
        fig, ax = plt.subplots()
        plot_rmsd(rmsd_result, ax=ax, color="green", moving_average=2)
        lines = ax.get_lines()
        assert len(lines) == 2
        assert _get_line_color(ax, 0) == to_rgba("green")
        assert _get_line_color(ax, 1) == to_rgba("green")
        plt.close(fig)

    def test_plot_rmsf_blue(self, rmsf_result: RMSFResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsf(rmsf_result, ax=ax, color="blue")
        assert _get_line_color(ax) == to_rgba("blue")
        plt.close(fig)

    def test_plot_rmsf_average_color(self, rmsf_result: RMSFResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsf_average([rmsf_result, rmsf_result], ax=ax, color="crimson")
        assert _get_line_color(ax) == to_rgba("crimson")
        plt.close(fig)

    def test_plot_sasa_sum(self, sasa_result: SASAResult) -> None:
        fig, ax = plt.subplots()
        plot_sasa(sasa_result, ax=ax, aggregate="sum", color="orange")
        assert _get_line_color(ax) == to_rgba("orange")
        plt.close(fig)

    def test_plot_sasa_mean(self, sasa_result: SASAResult) -> None:
        fig, ax = plt.subplots()
        plot_sasa(sasa_result, ax=ax, aggregate="mean", color="purple")
        assert _get_line_color(ax) == to_rgba("purple")
        plt.close(fig)

    def test_plot_sasa_none_ignores_color(self, sasa_result: SASAResult) -> None:
        """``aggregate='none'`` draws per-atom traces; color param is ignored."""
        fig, ax = plt.subplots()
        plot_sasa(sasa_result, ax=ax, aggregate="none", color="red")
        # Multiple lines drawn, none forced to red
        assert len(ax.get_lines()) == sasa_result.values_nm2.shape[1]
        for line in ax.get_lines():
            assert to_rgba(line.get_color()) != to_rgba("red")
        plt.close(fig)

    def test_plot_hbond_counts_color(self, hbond_result: HBondResult) -> None:
        fig, ax = plt.subplots()
        plot_hbond_counts(hbond_result, ax=ax, color="teal")
        assert _get_line_color(ax) == to_rgba("teal")
        plt.close(fig)

    def test_plot_rg_color(self, rg_result: RadiusOfGyrationResult) -> None:
        fig, ax = plt.subplots()
        plot_radius_of_gyration(rg_result, ax=ax, color="navy")
        assert _get_line_color(ax) == to_rgba("navy")
        plt.close(fig)

    def test_plot_native_contacts_color(self, native_contact_result: NativeContactResult) -> None:
        fig, ax = plt.subplots()
        plot_native_contacts(native_contact_result, ax=ax, color="coral")
        assert _get_line_color(ax) == to_rgba("coral")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests -- hex color strings
# ---------------------------------------------------------------------------


class TestColorHex:
    """Hex color strings should work identically to named colors."""

    def test_plot_rmsd_hex(self, rmsd_result: RMSDResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsd(rmsd_result, ax=ax, color="#FF5733")
        assert _get_line_color(ax) == to_rgba("#FF5733")
        plt.close(fig)

    def test_plot_rmsf_hex(self, rmsf_result: RMSFResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsf(rmsf_result, ax=ax, color="#2ECC71")
        assert _get_line_color(ax) == to_rgba("#2ECC71")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests -- multiple calls on the same axis with different colors
# ---------------------------------------------------------------------------


class TestColorOverlay:
    """Overlaying multiple traces on a single axis with distinct colors."""

    def test_rmsd_two_colors(self, rmsd_result: RMSDResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsd(rmsd_result, ax=ax, color="red", label="r1")
        plot_rmsd(rmsd_result, ax=ax, color="blue", label="r2")
        assert _get_line_color(ax, 0) == to_rgba("red")
        assert _get_line_color(ax, 1) == to_rgba("blue")
        plt.close(fig)

    def test_rmsf_two_colors(self, rmsf_result: RMSFResult) -> None:
        fig, ax = plt.subplots()
        plot_rmsf(rmsf_result, ax=ax, color="green", label="r1")
        plot_rmsf(rmsf_result, ax=ax, color="orange", label="r2")
        assert _get_line_color(ax, 0) == to_rgba("green")
        assert _get_line_color(ax, 1) == to_rgba("orange")
        plt.close(fig)

    def test_hbond_counts_two_colors(self, hbond_result: HBondResult) -> None:
        fig, ax = plt.subplots()
        plot_hbond_counts(hbond_result, ax=ax, color="purple", label="r1")
        plot_hbond_counts(hbond_result, ax=ax, color="cyan", label="r2")
        assert _get_line_color(ax, 0) == to_rgba("purple")
        assert _get_line_color(ax, 1) == to_rgba("cyan")
        plt.close(fig)
