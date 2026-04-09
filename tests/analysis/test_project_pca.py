"""Tests for project_pca."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis.decomposition import PCAResult, compute_pca, project_pca


@pytest.fixture()
def reference_pca() -> tuple[np.ndarray, np.ndarray, PCAResult]:
    """Two feature matrices (same dimension) and a PCA fitted on the first."""
    rng = np.random.default_rng(42)
    features_a = rng.normal(size=(100, 10))
    features_b = rng.normal(loc=1.0, size=(80, 10))
    fitted = compute_pca(features_a, n_components=2, dtype=np.float64)
    return features_a, features_b, fitted


class TestProjectPCA:
    def test_returns_pca_result(
        self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]
    ) -> None:
        _, features_b, fitted = reference_pca
        result = project_pca(features_b, fitted=fitted)
        assert isinstance(result, PCAResult)

    def test_projection_shape(
        self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]
    ) -> None:
        _, features_b, fitted = reference_pca
        result = project_pca(features_b, fitted=fitted)
        assert result.projections.shape == (80, 2)

    def test_shares_model(self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]) -> None:
        """Projected result should share model, components, etc. from the fitted PCA."""
        _, features_b, fitted = reference_pca
        result = project_pca(features_b, fitted=fitted)
        assert result.model is fitted.model
        np.testing.assert_array_equal(result.components, fitted.components)
        np.testing.assert_array_equal(
            result.explained_variance_ratio, fitted.explained_variance_ratio
        )
        np.testing.assert_array_equal(result.feature_mean, fitted.feature_mean)
        np.testing.assert_array_equal(result.feature_scale, fitted.feature_scale)

    def test_self_projection_matches_fit(
        self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]
    ) -> None:
        """Projecting the same data used for fitting should reproduce the original projections."""
        features_a, _, fitted = reference_pca
        result = project_pca(features_a, fitted=fitted, dtype=np.float64)
        np.testing.assert_allclose(result.projections, fitted.projections, atol=1e-10)

    def test_different_data_gives_different_projections(
        self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]
    ) -> None:
        """Projecting different data should give different projections."""
        _, features_b, fitted = reference_pca
        result = project_pca(features_b, fitted=fitted)
        # Different means -> projections should differ from fitted
        assert not np.allclose(result.projections[:10], fitted.projections[:10])

    def test_dimension_mismatch_raises(
        self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]
    ) -> None:
        """Feature dimension mismatch should raise ValueError."""
        _, _, fitted = reference_pca
        wrong_dim = np.random.default_rng(0).normal(size=(50, 5))  # 5 != 10
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            project_pca(wrong_dim, fitted=fitted)

    def test_single_sample(self, reference_pca: tuple[np.ndarray, np.ndarray, PCAResult]) -> None:
        """Projecting a single sample should work (2 required by _as_feature_matrix)."""
        _, _, fitted = reference_pca
        # _as_feature_matrix requires >= 2 samples
        two_samples = np.random.default_rng(0).normal(size=(2, 10))
        result = project_pca(two_samples, fitted=fitted)
        assert result.projections.shape == (2, 2)

    def test_without_standardize(self) -> None:
        """project_pca should work correctly with non-standardized PCA."""
        rng = np.random.default_rng(99)
        features_a = rng.normal(size=(50, 5))
        features_b = rng.normal(size=(30, 5))
        fitted = compute_pca(features_a, n_components=2, standardize=False, dtype=np.float64)
        result = project_pca(features_b, fitted=fitted, dtype=np.float64)
        assert result.projections.shape == (30, 2)
        # Self-projection should still match
        self_proj = project_pca(features_a, fitted=fitted, dtype=np.float64)
        np.testing.assert_allclose(self_proj.projections, fitted.projections, atol=1e-10)
