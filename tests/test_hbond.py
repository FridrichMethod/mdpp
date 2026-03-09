"""Tests for hydrogen-bond analysis."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis import compute_hbonds, format_hbond_triplets


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
