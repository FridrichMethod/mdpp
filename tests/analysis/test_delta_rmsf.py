"""Tests for compute_delta_rmsf."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis.metrics import (
    DeltaRMSFResult,
    RMSFResult,
    compute_delta_rmsf,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rmsf(
    rmsf_nm: list[float],
    residue_ids: list[int] | None = None,
) -> RMSFResult:
    arr = np.array(rmsf_nm, dtype=np.float64)
    return RMSFResult(
        rmsf_nm=arr,
        atom_indices=np.arange(arr.size, dtype=np.int_),
        residue_ids=np.array(residue_ids, dtype=np.int_) if residue_ids is not None else None,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestDeltaRMSFResult:
    def test_unit_conversion(self) -> None:
        result = DeltaRMSFResult(
            delta_rmsf_nm=np.array([0.1, -0.2], dtype=np.float64),
            residue_ids=np.array([1, 2], dtype=np.int_),
            sem_nm=np.array([0.01, 0.02], dtype=np.float64),
        )
        np.testing.assert_allclose(result.delta_rmsf_angstrom, [1.0, -2.0])
        assert result.sem_angstrom is not None
        np.testing.assert_allclose(result.sem_angstrom, [0.1, 0.2])

    def test_sem_none(self) -> None:
        result = DeltaRMSFResult(
            delta_rmsf_nm=np.array([0.1], dtype=np.float64),
            residue_ids=None,
            sem_nm=None,
        )
        assert result.sem_angstrom is None

    def test_frozen(self) -> None:
        result = DeltaRMSFResult(
            delta_rmsf_nm=np.array([0.1], dtype=np.float64),
            residue_ids=None,
            sem_nm=None,
        )
        with pytest.raises(AttributeError):
            result.delta_rmsf_nm = np.array([0.5])  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Identical-length systems (no index mapping)
# ---------------------------------------------------------------------------


class TestComputeDeltaRMSFIdentical:
    def test_basic_delta(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3], residue_ids=[1, 2, 3])
        b = _make_rmsf([0.2, 0.2, 0.1], residue_ids=[1, 2, 3])
        result = compute_delta_rmsf([a], [b])

        assert isinstance(result, DeltaRMSFResult)
        # delta = b - a
        expected = np.array([0.2 - 0.1, 0.2 - 0.2, 0.1 - 0.3])
        np.testing.assert_allclose(result.delta_rmsf_nm, expected)

    def test_residue_ids_from_a(self) -> None:
        a = _make_rmsf([0.1, 0.2], residue_ids=[10, 20])
        b = _make_rmsf([0.1, 0.2], residue_ids=[30, 40])
        result = compute_delta_rmsf([a], [b])
        np.testing.assert_array_equal(result.residue_ids, [10, 20])

    def test_no_residue_ids(self) -> None:
        a = _make_rmsf([0.1, 0.2])
        b = _make_rmsf([0.3, 0.4])
        result = compute_delta_rmsf([a], [b])
        assert result.residue_ids is None

    def test_explicit_residue_ids_override(self) -> None:
        a = _make_rmsf([0.1, 0.2], residue_ids=[1, 2])
        b = _make_rmsf([0.3, 0.4], residue_ids=[1, 2])
        custom = np.array([100, 200], dtype=np.int_)
        result = compute_delta_rmsf([a], [b], residue_ids=custom)
        np.testing.assert_array_equal(result.residue_ids, [100, 200])

    def test_multi_replica_averaging(self) -> None:
        """Averaging should happen in MSF space (mean of RMSF^2, then sqrt)."""
        a1 = _make_rmsf([0.1, 0.3])
        a2 = _make_rmsf([0.3, 0.1])
        b = _make_rmsf([0.2, 0.2])
        result = compute_delta_rmsf([a1, a2], [b])

        avg_a = np.sqrt(np.mean([np.array([0.1, 0.3]) ** 2, np.array([0.3, 0.1]) ** 2], axis=0))
        expected = np.array([0.2, 0.2]) - avg_a
        np.testing.assert_allclose(result.delta_rmsf_nm, expected)

    def test_zero_delta_identical_systems(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3])
        result = compute_delta_rmsf([a], [a])
        np.testing.assert_allclose(result.delta_rmsf_nm, 0.0, atol=1e-15)

    def test_sem_none_with_single_replica(self) -> None:
        a = _make_rmsf([0.1, 0.2])
        b = _make_rmsf([0.3, 0.4])
        result = compute_delta_rmsf([a], [b])
        assert result.sem_nm is None

    def test_sem_none_when_only_one_system_has_replicas(self) -> None:
        a1 = _make_rmsf([0.1, 0.2])
        a2 = _make_rmsf([0.3, 0.2])
        b = _make_rmsf([0.3, 0.4])
        result = compute_delta_rmsf([a1, a2], [b])
        assert result.sem_nm is None

    def test_sem_computed_with_multi_replicas(self) -> None:
        a1 = _make_rmsf([0.1, 0.3])
        a2 = _make_rmsf([0.3, 0.1])
        b1 = _make_rmsf([0.2, 0.4])
        b2 = _make_rmsf([0.4, 0.2])
        result = compute_delta_rmsf([a1, a2], [b1, b2])
        assert result.sem_nm is not None
        assert result.sem_nm.shape == result.delta_rmsf_nm.shape
        # SEM should be positive everywhere
        assert np.all(result.sem_nm >= 0)

    def test_sem_with_indices(self) -> None:
        a1 = _make_rmsf([0.1, 0.2, 0.3])
        a2 = _make_rmsf([0.15, 0.25, 0.35])
        b1 = _make_rmsf([0.4, 0.5])
        b2 = _make_rmsf([0.45, 0.55])
        idx_a = np.array([0, 2], dtype=np.int_)
        idx_b = np.array([0, 1], dtype=np.int_)
        result = compute_delta_rmsf([a1, a2], [b1, b2], indices_a=idx_a, indices_b=idx_b)
        assert result.sem_nm is not None
        assert result.sem_nm.shape == (2,)


# ---------------------------------------------------------------------------
# Different-length systems (index mapping)
# ---------------------------------------------------------------------------


class TestComputeDeltaRMSFWithIndices:
    def test_basic_with_indices(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3, 0.4], residue_ids=[1, 2, 3, 4])
        b = _make_rmsf([0.5, 0.6, 0.7], residue_ids=[10, 20, 30])
        idx_a = np.array([0, 2, 3], dtype=np.int_)
        idx_b = np.array([0, 1, 2], dtype=np.int_)
        result = compute_delta_rmsf([a], [b], indices_a=idx_a, indices_b=idx_b)

        expected = np.array([0.5 - 0.1, 0.6 - 0.3, 0.7 - 0.4])
        np.testing.assert_allclose(result.delta_rmsf_nm, expected)

    def test_residue_ids_from_a_at_indices(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3], residue_ids=[10, 20, 30])
        b = _make_rmsf([0.4, 0.5], residue_ids=[1, 2])
        idx_a = np.array([0, 2], dtype=np.int_)
        idx_b = np.array([0, 1], dtype=np.int_)
        result = compute_delta_rmsf([a], [b], indices_a=idx_a, indices_b=idx_b)
        np.testing.assert_array_equal(result.residue_ids, [10, 30])

    def test_explicit_residue_ids_with_indices(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3])
        b = _make_rmsf([0.4, 0.5])
        idx_a = np.array([0, 2], dtype=np.int_)
        idx_b = np.array([0, 1], dtype=np.int_)
        custom = np.array([100, 300], dtype=np.int_)
        result = compute_delta_rmsf([a], [b], indices_a=idx_a, indices_b=idx_b, residue_ids=custom)
        np.testing.assert_array_equal(result.residue_ids, [100, 300])

    def test_no_residue_ids_on_source(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3])
        b = _make_rmsf([0.4, 0.5])
        idx_a = np.array([0, 2], dtype=np.int_)
        idx_b = np.array([0, 1], dtype=np.int_)
        result = compute_delta_rmsf([a], [b], indices_a=idx_a, indices_b=idx_b)
        assert result.residue_ids is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestComputeDeltaRMSFErrors:
    def test_empty_results_a(self) -> None:
        b = _make_rmsf([0.1])
        with pytest.raises(ValueError, match="results_a must not be empty"):
            compute_delta_rmsf([], [b])

    def test_empty_results_b(self) -> None:
        a = _make_rmsf([0.1])
        with pytest.raises(ValueError, match="results_b must not be empty"):
            compute_delta_rmsf([a], [])

    def test_inconsistent_replicas_a(self) -> None:
        a1 = _make_rmsf([0.1, 0.2])
        a2 = _make_rmsf([0.1, 0.2, 0.3])
        b = _make_rmsf([0.1, 0.2])
        with pytest.raises(ValueError, match="results_a replicas have inconsistent sizes"):
            compute_delta_rmsf([a1, a2], [b])

    def test_inconsistent_replicas_b(self) -> None:
        a = _make_rmsf([0.1, 0.2])
        b1 = _make_rmsf([0.1, 0.2])
        b2 = _make_rmsf([0.1])
        with pytest.raises(ValueError, match="results_b replicas have inconsistent sizes"):
            compute_delta_rmsf([a], [b1, b2])

    def test_mismatched_lengths_without_indices(self) -> None:
        a = _make_rmsf([0.1, 0.2])
        b = _make_rmsf([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="different residue counts"):
            compute_delta_rmsf([a], [b])

    def test_mismatched_index_lengths(self) -> None:
        a = _make_rmsf([0.1, 0.2, 0.3])
        b = _make_rmsf([0.4, 0.5, 0.6])
        idx_a = np.array([0, 1], dtype=np.int_)
        idx_b = np.array([0, 1, 2], dtype=np.int_)
        with pytest.raises(ValueError, match="same length"):
            compute_delta_rmsf([a], [b], indices_a=idx_a, indices_b=idx_b)

    def test_only_indices_a_provided(self) -> None:
        a = _make_rmsf([0.1, 0.2])
        b = _make_rmsf([0.1, 0.2])
        with pytest.raises(ValueError, match="both be provided or both be None"):
            compute_delta_rmsf([a], [b], indices_a=np.array([0, 1], dtype=np.int_))

    def test_only_indices_b_provided(self) -> None:
        a = _make_rmsf([0.1, 0.2])
        b = _make_rmsf([0.1, 0.2])
        with pytest.raises(ValueError, match="both be provided or both be None"):
            compute_delta_rmsf([a], [b], indices_b=np.array([0, 1], dtype=np.int_))
