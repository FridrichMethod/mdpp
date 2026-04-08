"""Physical constants used across mdpp modules."""

from scipy.constants import R as _R_J_MOL_K

GAS_CONSTANT_KJ_MOL_K: float = _R_J_MOL_K / 1000.0
"""Molar gas constant in kJ/mol/K (8.31446... x 10^-3)."""

DEFAULT_TEMPERATURE_K: float = 298.15
"""Standard room temperature in Kelvin (25 degrees C)."""
