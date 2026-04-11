"""Package-wide floating-point dtype configuration.

Default is ``np.float32``, which matches the precision of MD trajectory
coordinates (mdtraj stores ``traj.xyz`` as float32) and is sufficient
for all analysis operations in this package.

Float64 appears in the analysis pipeline only where it is genuinely
necessary or where an external library forces it:

- **Numba JIT kernels** (``_backends/_distances.distances_numba``,
  ``_backends/_rmsd_matrix._pairwise_rmsd``): compiled kernels output
  float64 because Numba's ``float()`` cast maps to C ``double``.
  Numba runs on CPU where float64 is at ~50% of float32 throughput,
  so the cost is negligible and the extra precision is useful for the
  QCP Newton-Raphson subtraction ``G_a + G_b - 2*lambda``.  Callers
  cast the result to the resolved user dtype afterward.
- **GPU backends** (``_backends/_distances`` and
  ``_backends/_rmsd_matrix`` ``torch``/``jax``/``cupy`` variants):
  compute **internally in float32** because consumer and workstation
  NVIDIA GPUs run float64 at 1/36 -- 1/64 the throughput of float32.
  Since 2026-04-11 these backends also **return native float32**
  (the ``RMSDMatrixBackendFn`` / ``DistanceBackendFn`` Protocols were
  widened from ``NDArray[np.float64]`` to ``NDArray[np.floating]``
  so backends can report their natural dtype).  The public
  ``compute_*`` wrappers then cast with ``astype(resolved, copy=False)``
  so when the resolved dtype is also float32 (the package default)
  **no additional copy is made** -- critical for large N where
  every redundant copy of the ``(n_frames, n_frames)`` RMSD matrix
  costs tens of GB (57 GB at n=120k).  Float32 QCP agrees with the
  float64 numba reference to ~1e-6 nm on realistic trajectories.
- **Deeptime TICA** (``decomposition.compute_tica``): deeptime upcasts
  to float64 internally for covariance estimation -- no explicit cast
  is needed from our side.
- **``np.histogram2d``** (``fes.compute_fes_2d``): returns float64
  probability density regardless of input dtype (edges follow the
  input dtype); the downstream log and energy arithmetic therefore
  runs in float64 naturally.
- **``np.mean`` on boolean arrays** (contacts, h-bonds): NumPy defaults
  to float64 for boolean reductions.
- **``jax.config.update("jax_enable_x64", True)``** in
  ``_backends/_imports.require_jax``: enables float64 support in JAX
  so ``jnp.float64`` arrays can round-trip through the JIT.  The
  actual JAX compute still runs in float32 on GPU.

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
