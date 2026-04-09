"""Package-wide floating-point dtype configuration.

Resolution order (highest to lowest precedence):
1. Explicit ``dtype`` parameter on a function call
2. ``default_dtype()`` context manager scope
3. ``set_default_dtype()`` global setting
4. Package default: ``np.float32``
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

_VALID_DTYPES = frozenset({np.dtype(np.float32), np.dtype(np.float64)})
_DEFAULT = np.dtype(np.float32)
_default_dtype: ContextVar[np.dtype[np.floating]] = ContextVar(
    "mdpp_default_dtype", default=_DEFAULT
)


def get_default_dtype() -> np.dtype[np.floating]:
    """Return the current default float dtype."""
    return _default_dtype.get()


def set_default_dtype(dtype: type[np.floating] | np.dtype[np.floating]) -> None:
    """Set the package-wide default float dtype.

    Args:
        dtype: ``np.float32`` or ``np.float64``.

    Raises:
        ValueError: If *dtype* is not float32 or float64.
    """
    resolved = np.dtype(dtype)
    if resolved not in _VALID_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {resolved}.")
    _default_dtype.set(resolved)


@contextmanager
def default_dtype(
    dtype: type[np.floating] | np.dtype[np.floating],
) -> Generator[None]:
    """Context manager for scoped dtype override.

    Args:
        dtype: ``np.float32`` or ``np.float64``.

    Example::

        with default_dtype(np.float64):
            result = compute_rmsd(traj)  # uses float64
        # reverts to previous default here
    """
    resolved = np.dtype(dtype)
    if resolved not in _VALID_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {resolved}.")
    token = _default_dtype.set(resolved)
    try:
        yield
    finally:
        _default_dtype.reset(token)


def resolve_dtype(
    dtype: type[np.floating] | np.dtype[np.floating] | None,
) -> np.dtype[np.floating]:
    """Resolve the effective dtype for a function call.

    If *dtype* is explicitly provided, validate and return it.
    Otherwise, return the current default from the ContextVar.

    Args:
        dtype: Explicit dtype override, or ``None`` to use the default.

    Returns:
        The resolved numpy dtype.

    Raises:
        ValueError: If *dtype* is not float32, float64, or None.
    """
    if dtype is None:
        return _default_dtype.get()
    resolved = np.dtype(dtype)
    if resolved not in _VALID_DTYPES:
        raise ValueError(f"dtype must be float32 or float64, got {resolved}.")
    return resolved
