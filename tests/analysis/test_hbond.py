"""Tests for hydrogen-bond analysis."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis.hbond import compute_hbonds, format_hbond_triplets


def test_compute_hbonds_baker_hubbard_counts(hbond_trajectory) -> None:
    """Baker-Hubbard should detect on/off hydrogen-bond frames."""
    result = compute_hbonds(
        hbond_trajectory,
        method="baker_hubbard",
        freq=0.0,
        periodic=False,
    )

    assert result.triplets.shape == (1, 3)
    assert np.array_equal(result.count_per_frame, np.array([1, 0, 1], dtype=np.int_))
    assert result.occupancy.shape == (1,)
    assert result.occupancy[0] == pytest.approx(2.0 / 3.0)
    assert result.time_ns[-1] == pytest.approx(0.04)


def test_format_hbond_triplets_returns_readable_labels(hbond_trajectory) -> None:
    """Triplet formatting should include residue and atom names."""
    result = compute_hbonds(
        hbond_trajectory,
        method="baker_hubbard",
        freq=0.0,
        periodic=False,
    )
    labels = format_hbond_triplets(hbond_trajectory.topology, result.triplets)

    assert len(labels) == 1
    assert "DON1:N-H" in labels[0]
    assert "ACC2:O" in labels[0]


@pytest.mark.parametrize(
    ("distance_nm", "angle_deg", "cutoff_nm", "cutoff_deg"),
    [(0.28, 180.0, 0.30, 120.0), (0.20, 110.0, 0.25, 100.0)],
)
def test_custom_hbond_cutoffs_include_relaxed_candidates(
    hbond_trajectory, distance_nm, angle_deg, cutoff_nm, cutoff_deg
) -> None:
    """Relaxed criteria must be applied before the candidate/occupancy filter."""
    theta = np.deg2rad(angle_deg)
    hbond_trajectory.xyz[:, 2, :] = [
        0.1 - distance_nm * np.cos(theta),
        distance_nm * np.sin(theta),
        0.0,
    ]
    result = compute_hbonds(
        hbond_trajectory,
        periodic=False,
        freq=0.5,
        distance_cutoff_nm=cutoff_nm,
        angle_cutoff_deg=cutoff_deg,
    )
    assert result.triplets.shape == (1, 3)
    np.testing.assert_array_equal(result.count_per_frame, [1, 1, 1])
    np.testing.assert_allclose(result.occupancy, [1.0])


def test_custom_hbond_cutoff_filters_by_actual_occupancy(hbond_trajectory) -> None:
    """Tightened criteria must not retain bonds below the requested frequency."""
    hbond_trajectory.xyz[:, 2, 0] = [0.25, 0.25, 0.20]
    result = compute_hbonds(
        hbond_trajectory,
        periodic=False,
        freq=0.5,
        distance_cutoff_nm=0.12,
    )
    assert result.triplets.shape == (0, 3)
    np.testing.assert_array_equal(result.count_per_frame, [0, 0, 0])
