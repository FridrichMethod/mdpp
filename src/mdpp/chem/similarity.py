"""Similarity metrics, kernels, and pairwise computation utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numba import njit, prange
from rdkit import DataStructs

type FingerPrint = DataStructs.cDataStructs.ExplicitBitVect


# ---------------------------------------------------------------------------
# Numba-parallel similarity kernels
#
# Each kernel computes similarity from four integer counts derived from two
# binary fingerprints:
#   c      = |A & B|   (intersection / shared on-bits)
#   a      = |A|       (popcount of A)
#   b      = |B|       (popcount of B)
#   n_bits = total bits in the fingerprint
# ---------------------------------------------------------------------------


@njit
def _tanimoto_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    denom = a + b - c
    return c / denom if denom > 0 else 0.0


@njit
def _dice_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    denom = a + b
    return 2.0 * c / denom if denom > 0 else 0.0


@njit
def _cosine_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    prod = a * b
    return c / np.sqrt(float(prod)) if prod > 0 else 0.0


@njit
def _sokal_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    denom = 2 * a + 2 * b - 3 * c
    return c / denom if denom > 0 else 0.0


@njit
def _rogotgoldberg_sim(c: int, a: int, b: int, n_bits: int) -> float:
    d = n_bits - a - b + c
    denom1 = a + b
    denom2 = 2 * n_bits - a - b
    t1 = c / denom1 if denom1 > 0 else 0.0
    t2 = d / denom2 if denom2 > 0 else 0.0
    return t1 + t2


@njit
def _allbit_sim(c: int, a: int, b: int, n_bits: int) -> float:
    d = n_bits - a - b + c
    return (c + d) / n_bits if n_bits > 0 else 0.0


@njit
def _kulczynski_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    if a == 0 or b == 0:
        return 0.0
    return 0.5 * (c / a + c / b)


@njit
def _mcconnaughey_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    denom = a * b
    return (c * (a + b) - denom) / denom if denom > 0 else 0.0


@njit
def _asymmetric_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    denom = min(a, b)
    return c / denom if denom > 0 else 0.0


@njit
def _braunblanquet_sim(c: int, a: int, b: int, _n_bits: int) -> float:
    denom = max(a, b)
    return c / denom if denom > 0 else 0.0


@njit(parallel=True)
def calc_similarities(
    fps: np.ndarray,
    sim_kernel: Callable[[int, int, int, int], float],
) -> np.ndarray:
    """Compute condensed pairwise similarity array using a Numba-parallel kernel.

    Args:
        fps: 2D numpy array of shape ``(n_mols, n_bits)`` with binary fingerprints.
        sim_kernel: A ``@njit`` function ``(c, a, b, n_bits) -> float`` returning
            similarity in [0, 1] (or [-1, 1] for McConnaughey).

    Returns:
        1D condensed similarity array of length ``n*(n-1)/2``, dtype float32.
    """
    n = len(fps)
    n_bits = fps.shape[1]
    n_pairs = n * (n - 1) // 2
    popcounts = fps.sum(axis=1)
    # prange over scalar pairs rather than full vectorization: each iteration
    # only touches two fingerprint rows (~n_bits bytes), keeping the working set
    # in L1/L2 cache.  A vectorized broadcast would materialize O(n_pairs * n_bits)
    # temporaries (~50 GB for 10k molecules), thrashing memory for no speed gain.
    similarities = np.empty(n_pairs, dtype=np.float32)
    for k in prange(n_pairs):
        i = int(np.sqrt(k + 0.125) * np.sqrt(2.0) + 0.5)
        j = k - i * (i - 1) // 2
        c = np.sum(fps[i] & fps[j])
        similarities[k] = sim_kernel(c, popcounts[i], popcounts[j], n_bits)
    return similarities


PARALLEL_SIM_KERNELS: dict[str, Callable[[int, int, int, int], float]] = {
    "tanimoto": _tanimoto_sim,
    "dice": _dice_sim,
    "cosine": _cosine_sim,
    "sokal": _sokal_sim,
    "rogotgoldberg": _rogotgoldberg_sim,
    "allbit": _allbit_sim,
    "kulczynski": _kulczynski_sim,
    "mcconnaughey": _mcconnaughey_sim,
    "asymmetric": _asymmetric_sim,
    "braunblanquet": _braunblanquet_sim,
}


# ---------------------------------------------------------------------------
# RDKit similarity wrappers
# ---------------------------------------------------------------------------

SIM_FUNCS: dict[str, Callable[[FingerPrint, FingerPrint], float]] = {
    name.lower(): func for name, func, _ in DataStructs.similarityFunctions
}

BULK_SIM_FUNCS: dict[str, Callable[[FingerPrint, Sequence[FingerPrint]], list[float]]] = {
    "tanimoto": DataStructs.cDataStructs.BulkTanimotoSimilarity,
    "dice": DataStructs.cDataStructs.BulkDiceSimilarity,
    "cosine": DataStructs.cDataStructs.BulkCosineSimilarity,
    "sokal": DataStructs.cDataStructs.BulkSokalSimilarity,
    "russel": DataStructs.cDataStructs.BulkRusselSimilarity,
    "rogotgoldberg": DataStructs.cDataStructs.BulkRogotGoldbergSimilarity,
    "allbit": DataStructs.cDataStructs.BulkAllBitSimilarity,
    "kulczynski": DataStructs.cDataStructs.BulkKulczynskiSimilarity,
    "mcconnaughey": DataStructs.cDataStructs.BulkMcConnaugheySimilarity,
    "asymmetric": DataStructs.cDataStructs.BulkAsymmetricSimilarity,
    "braunblanquet": DataStructs.cDataStructs.BulkBraunBlanquetSimilarity,
}

_VALID_SIM_METRICS = ", ".join(f"'{k}'" for k in BULK_SIM_FUNCS)

_METRICS_UNSUITABLE_FOR_CLUSTERING = frozenset({"russel"})

CLUSTERING_SIM_METRICS: frozenset[str] = (
    frozenset(PARALLEL_SIM_KERNELS) | frozenset(BULK_SIM_FUNCS)
) - _METRICS_UNSUITABLE_FOR_CLUSTERING
"""Similarity metrics whose ``1 - sim`` transform yields a valid distance for clustering.

Excluded metrics:

* ``'russel'`` (Russell-Rao): self-similarity equals ``popcount / n_bits`` rather than 1,
  so ``1 - sim(A, A) > 0`` for most fingerprints, breaking distance-based clustering.
"""

_VALID_CLUSTERING_METRICS = ", ".join(f"'{k}'" for k in sorted(CLUSTERING_SIM_METRICS))


def calc_sim(
    fp1: FingerPrint,
    fp2: FingerPrint,
    *,
    similarity_metric: str = "tanimoto",
) -> float:
    """Calculate similarity between two fingerprints.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.
        similarity_metric: Similarity metric name (case-insensitive).

    Returns:
        Similarity score.

    Raises:
        ValueError: If *similarity_metric* is not recognised.
    """
    if similarity_metric not in SIM_FUNCS:
        raise ValueError(f"similarity_metric should be one of {_VALID_SIM_METRICS}.")
    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=SIM_FUNCS[similarity_metric])


def calc_bulk_sim(
    fp: FingerPrint,
    fps: Sequence[FingerPrint],
    *,
    similarity_metric: str = "tanimoto",
) -> list[float]:
    """Calculate similarity between one fingerprint and a list of fingerprints.

    Args:
        fp: Query fingerprint.
        fps: Target fingerprints.
        similarity_metric: Similarity metric name (case-insensitive).

    Returns:
        List of similarity scores, one per target fingerprint.

    Raises:
        ValueError: If *similarity_metric* is not recognised.
    """
    if similarity_metric not in BULK_SIM_FUNCS:
        raise ValueError(f"similarity_metric should be one of {_VALID_SIM_METRICS}.")
    return BULK_SIM_FUNCS[similarity_metric](fp, fps)
