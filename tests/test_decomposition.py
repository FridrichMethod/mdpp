"""Tests for decomposition and projection utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis import compute_pca, compute_tica


def test_compute_pca_identifies_dominant_axis() -> None:
    """PCA should place most variance on the first component for anisotropic data."""
    rng = np.random.default_rng(0)
    x_axis = np.linspace(-5.0, 5.0, 500)
    y_axis = 0.02 * rng.normal(size=x_axis.shape[0])
    features = np.column_stack([x_axis, y_axis])

    result = compute_pca(features, n_components=2, standardize=False)

    assert result.projections.shape == (features.shape[0], 2)
    assert result.components.shape == (2, 2)
    assert result.explained_variance_ratio[0] > 0.999


def test_compute_tica_returns_valid_projection() -> None:
    """TICA should produce finite projections with the requested dimensionality."""
    pytest.importorskip("deeptime")

    rng = np.random.default_rng(1)
    n_samples = 400
    slow = np.zeros(n_samples, dtype=np.float64)
    for index in range(1, n_samples):
        slow[index] = 0.95 * slow[index - 1] + 0.1 * rng.normal()
    fast = rng.normal(scale=0.3, size=n_samples)
    features = np.column_stack([slow, fast])

    result = compute_tica(features, lagtime=5, n_components=2)

    assert result.projections.shape == (n_samples, 2)
    assert np.all(np.isfinite(result.projections))
