"""Free-energy surface computation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

type BinsType = int | tuple[int, int]
type RangeType = tuple[tuple[float, float], tuple[float, float]] | None


@dataclass(frozen=True, slots=True)
class FES2DResult:
    """2D free-energy surface derived from a histogram."""

    free_energy_kj_mol: NDArray[np.float64]
    probability_density: NDArray[np.float64]
    x_edges: NDArray[np.float64]
    y_edges: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    temperature_k: float

    @property
    def x_centers(self) -> NDArray[np.float64]:
        """Return x-axis bin centers."""
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def y_centers(self) -> NDArray[np.float64]:
        """Return y-axis bin centers."""
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])


def compute_fes_2d(
    x_values: ArrayLike,
    y_values: ArrayLike,
    *,
    bins: BinsType = 100,
    value_range: RangeType = None,
    temperature_k: float = 300.0,
    gas_constant_kj_mol_k: float = 0.00831446261815324,
    min_probability: float = 1e-12,
    mask_unsampled: bool = True,
) -> FES2DResult:
    """Compute a 2D free-energy surface from two collective variables.

    Args:
        x_values: Samples for CV1.
        y_values: Samples for CV2.
        bins: Histogram bin count.
        value_range: Optional ``((x_min, x_max), (y_min, y_max))`` range.
        temperature_k: Temperature in Kelvin.
        gas_constant_kj_mol_k: Gas constant in ``kJ/mol/K``.
        min_probability: Lower bound to avoid ``log(0)``.
        mask_unsampled: If True, unsampled bins are set to ``NaN``.

    Returns:
        FES2DResult with free energy shifted so its minimum is 0.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive.")
    if gas_constant_kj_mol_k <= 0.0:
        raise ValueError("gas_constant_kj_mol_k must be positive.")
    if min_probability <= 0.0:
        raise ValueError("min_probability must be positive.")

    x_array = np.ravel(np.asarray(x_values, dtype=np.float64))
    y_array = np.ravel(np.asarray(y_values, dtype=np.float64))
    if x_array.shape != y_array.shape:
        raise ValueError("x_values and y_values must have matching shape.")
    if x_array.size < 2:
        raise ValueError("At least two samples are required to compute a FES.")

    probability_density, x_edges, y_edges = np.histogram2d(
        x_array,
        y_array,
        bins=bins,
        range=value_range,
        density=True,
    )
    probability_density = np.asarray(probability_density, dtype=np.float64)

    observed_mask = probability_density > min_probability
    if mask_unsampled and not np.any(observed_mask):
        raise ValueError("No sampled bins remain. Reduce min_probability or adjust bins.")

    clipped_probability = np.clip(probability_density, min_probability, None)
    free_energy_kj_mol = -gas_constant_kj_mol_k * temperature_k * np.log(clipped_probability)
    if mask_unsampled:
        free_energy_kj_mol = np.where(observed_mask, free_energy_kj_mol, np.nan)

    minimum = np.nanmin(free_energy_kj_mol) if mask_unsampled else np.min(free_energy_kj_mol)
    free_energy_kj_mol = free_energy_kj_mol - float(minimum)

    return FES2DResult(
        free_energy_kj_mol=np.asarray(free_energy_kj_mol, dtype=np.float64),
        probability_density=probability_density,
        x_edges=np.asarray(x_edges, dtype=np.float64),
        y_edges=np.asarray(y_edges, dtype=np.float64),
        observed_mask=np.asarray(observed_mask, dtype=bool),
        temperature_k=float(temperature_k),
    )


def compute_fes_from_projection(
    projection: ArrayLike,
    *,
    x_index: int = 0,
    y_index: int = 1,
    bins: BinsType = 100,
    value_range: RangeType = None,
    temperature_k: float = 300.0,
    gas_constant_kj_mol_k: float = 0.00831446261815324,
    min_probability: float = 1e-12,
    mask_unsampled: bool = True,
) -> FES2DResult:
    """Compute a 2D FES from a projection matrix.

    Args:
        projection: Matrix ``(n_samples, n_components)``.
        x_index: Component index for x-axis.
        y_index: Component index for y-axis.
        bins: Histogram bin count.
        value_range: Optional histogram range.
        temperature_k: Temperature in Kelvin.
        gas_constant_kj_mol_k: Gas constant in ``kJ/mol/K``.
        min_probability: Lower bound to avoid ``log(0)``.
        mask_unsampled: If True, unsampled bins are set to ``NaN``.

    Returns:
        FES2DResult computed from selected projection components.
    """
    projection_array = np.asarray(projection, dtype=np.float64)
    if projection_array.ndim != 2:
        raise ValueError("projection must be a 2D array.")
    if projection_array.shape[1] <= max(x_index, y_index):
        raise ValueError("x_index and y_index must be valid projection component indices.")

    return compute_fes_2d(
        projection_array[:, x_index],
        projection_array[:, y_index],
        bins=bins,
        value_range=value_range,
        temperature_k=temperature_k,
        gas_constant_kj_mol_k=gas_constant_kj_mol_k,
        min_probability=min_probability,
        mask_unsampled=mask_unsampled,
    )
