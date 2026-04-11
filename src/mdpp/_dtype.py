"""Package-wide floating-point dtype configuration.

Default is ``np.float32``, which matches the precision of MD trajectory
coordinates (mdtraj stores ``traj.xyz`` as float32) and is sufficient
for all analysis operations in this package.  **Every** compute function
returns float32 by default; users who want float64 must opt in either
globally via :func:`set_default_dtype` or per-call via ``dtype=np.float64``.

Design rules for new compute code
---------------------------------

1. The public function's last keyword argument is
   ``dtype: DtypeArg = None``.
2. Call ``resolved = resolve_dtype(dtype)`` at the top.
3. Pass ``resolved`` through to every downstream buffer allocation and
   cast outputs via ``np.asarray(result, dtype=resolved)`` /
   ``result.astype(resolved, copy=False)`` so same-dtype returns do
   not duplicate memory.
4. **Backend kernels** (numba/torch/jax/cupy) return their native dtype
   (``NDArray[np.floating]``) and should prefer float32 output unless
   external precision is required.  The public wrapper's ``copy=False``
   cast becomes a no-op when the kernel already returns the resolved
   dtype, which is essential at large N where each redundant N^2 copy
   can cost tens of GB.

Where float64 still appears (and why)
-------------------------------------

These are the only places fp64 remains in the compute pipeline; each is
either an O(1)-to-O(n) scalar buffer (not an OOM risk) or forced by an
external library:

- **QCP Newton-Raphson scalars** in ``_backends/_rmsd_matrix._pairwise_rmsd``
  and the ``traces`` buffer in ``_center_and_traces``: accumulators
  (``Sxx`` etc.) and the ``(G_a + G_b - 2*lambda)`` subtraction run in
  double precision because Numba's ``0.0`` literal maps to C
  ``double``.  Only the final ``result[i, j] = val`` store truncates
  to float32 so the O(N^2) output matrix is half the memory of the
  old float64 output (58 GB saved at n=120k).  The ``traces`` buffer
  is O(n_frames) so the fp64 cost is negligible.
- **GPU backends** (``_backends/_distances`` and
  ``_backends/_rmsd_matrix`` ``torch``/``jax``/``cupy`` variants):
  compute internally in float32 because consumer and workstation
  NVIDIA GPUs run float64 at 1/36 -- 1/64 the throughput of float32,
  and return native float32 directly.  Float32 QCP agrees with the
  float64 numba reference to ~1e-6 nm on realistic trajectories.
- **Deeptime TICA** (``decomposition.compute_tica``): deeptime upcasts
  to float64 internally for covariance estimation -- external to us.
  The output is cast back to the resolved dtype by the wrapper.
- **``np.histogram2d``** (``fes.compute_fes_2d``): returns float64
  probability density regardless of input dtype; the downstream log
  and energy arithmetic therefore runs in float64 naturally.  Output
  is O(bins^2), tiny.
- **``np.mean`` on boolean arrays** (contacts, h-bonds): NumPy defaults
  to float64 for boolean reductions.  Output is O(n), tiny.
- **``jax.config.update("jax_enable_x64", True)``** in
  ``_backends/_imports.require_jax``: enables float64 support in JAX
  so ``jnp.float64`` arrays can round-trip through the JIT when the
  user explicitly opts in.  The actual JAX compute still runs in
  float32 on GPU by default.

Opting into float64
-------------------

Use ``set_default_dtype(np.float64)`` to switch globally, or pass
``dtype=np.float64`` to individual functions.  Be aware that float64
doubles the memory of every O(N^2) or O(N*M) intermediate, which will
OOM at trajectory sizes above ~40k frames on a 128 GB host.
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
