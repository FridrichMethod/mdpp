"""Package-wide floating-point dtype configuration.

Default is ``np.float32``, which matches the precision of MD trajectory
coordinates (mdtraj stores ``traj.xyz`` as float32) and is sufficient
for all analysis operations in this package.

Float64 is **not** forced anywhere in the analysis pipeline.  The only
places where float64 appears are:

- **Numba JIT kernel** (``decomposition._pairwise_distances_numba``):
  the compiled kernel outputs float64 due to Numba's ``float()`` cast
  semantics; callers cast the result to the resolved dtype afterward.
- **Deeptime TICA** (``decomposition.compute_tica``): deeptime upcasts
  to float64 internally for covariance estimation -- no explicit cast
  is needed from our side.
- **``np.histogram2d``** (``fes.compute_fes_2d``): returns float64
  probability density regardless of input dtype (edges follow the
  input dtype); the downstream log and energy arithmetic therefore
  runs in float64 naturally.
- **``np.mean`` on boolean arrays** (contacts, h-bonds): NumPy defaults
  to float64 for boolean reductions.

Use ``set_default_dtype(np.float64)`` to switch globally, or pass
``dtype=np.float64`` to individual functions.
"""

from __future__ import annotations

import numpy as np

from mdpp._types import DtypeArg

_VALID_DTYPES = frozenset({np.dtype(np.float32), np.dtype(np.float64)})
_default_dtype = np.dtype(np.float32)


def get_default_dtype() -> np.dtype[np.floating]:
    """Return the current default float dtype."""
    return _default_dtype


def set_default_dtype(
    dtype: type[np.floating] | np.dtype[np.floating],
) -> None:  # not DtypeArg (None disallowed)
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
    dtype: DtypeArg,
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
