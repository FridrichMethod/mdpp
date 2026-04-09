"""Package-wide floating-point dtype configuration.

Default is ``np.float32`` (matches MD simulation precision).
Use ``set_default_dtype(np.float64)`` to switch globally, or pass
``dtype=np.float64`` to individual functions.
"""

from __future__ import annotations

import numpy as np

_VALID_DTYPES = frozenset({np.dtype(np.float32), np.dtype(np.float64)})
_default_dtype = np.dtype(np.float32)


def get_default_dtype() -> np.dtype[np.floating]:
    """Return the current default float dtype."""
    return _default_dtype


def set_default_dtype(dtype: type[np.floating] | np.dtype[np.floating]) -> None:
    """Set the package-wide default float dtype.

    Args:
        dtype: ``np.float32`` or ``np.float64``.

    Raises:
        ValueError: If *dtype* is not float32 or float64.
    """
    global _default_dtype  # noqa: PLW0603
    resolved = np.dtype(dtype)
    if resolved not in _VALID_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {resolved}.")
    _default_dtype = resolved


def resolve_dtype(
    dtype: type[np.floating] | np.dtype[np.floating] | None,
) -> np.dtype[np.floating]:
    """Resolve the effective dtype for a function call.

    Returns *dtype* if explicitly provided, otherwise the global default.

    Args:
        dtype: Explicit dtype override, or ``None`` to use the default.

    Returns:
        The resolved numpy dtype.

    Raises:
        ValueError: If *dtype* is not float32, float64, or None.
    """
    if dtype is None:
        return _default_dtype
    resolved = np.dtype(dtype)
    if resolved not in _VALID_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {resolved}.")
    return resolved
