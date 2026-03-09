"""Tests for free-energy surface computation."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis import compute_fes_2d, compute_fes_from_projection


def test_compute_fes_2d_sets_minimum_to_zero() -> None:
    """FES should be shifted so the minimum finite value is zero."""
    rng = np.random.default_rng(2)
    x_values = rng.normal(loc=0.0, scale=1.0, size=8000)
    y_values = 0.5 * x_values + rng.normal(loc=0.0, scale=0.4, size=x_values.shape[0])

    result = compute_fes_2d(
        x_values,
        y_values,
        bins=40,
        temperature_k=310.15,
        min_probability=1e-8,
        mask_unsampled=True,
    )

    assert result.free_energy_kj_mol.shape == (40, 40)
    assert np.nanmin(result.free_energy_kj_mol) == pytest.approx(0.0, abs=1e-12)
    assert np.nanmax(result.free_energy_kj_mol) > 0.0
    assert np.any(result.observed_mask)


def test_compute_fes_from_projection_selects_components() -> None:
    """FES-from-projection should use selected component columns."""
    rng = np.random.default_rng(3)
    projection = rng.normal(size=(5000, 3))
    result = compute_fes_from_projection(
        projection,
        x_index=1,
        y_index=2,
        bins=(25, 30),
        min_probability=1e-8,
    )

    assert result.free_energy_kj_mol.shape == (25, 30)
    assert result.probability_density.shape == (25, 30)
