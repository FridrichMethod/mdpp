"""Lazy imports and availability flags for optional GPU packages."""

from __future__ import annotations

from types import ModuleType

# ---------------------------------------------------------------------------
# Availability flags (safe at import time, no side effects)
# ---------------------------------------------------------------------------

try:
    import cupy as _cupy  # noqa: F401

    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import torch as _torch  # noqa: F401

    has_torch = True
except ImportError:
    has_torch = False

try:
    import jax as _jax  # noqa: F401

    has_jax = True
except ImportError:
    has_jax = False


# ---------------------------------------------------------------------------
# Lazy import helpers (called inside backend functions at runtime)
# ---------------------------------------------------------------------------


def require_cupy() -> ModuleType:
    """Return the ``cupy`` module or raise with install instructions."""
    try:
        import cupy
    except ImportError:
        raise ImportError(
            "CuPy is required for backend='cupy'. Install with: pip install cupy-cuda12x"
        ) from None
    return cupy


def require_torch() -> ModuleType:
    """Return the ``torch`` module or raise with install instructions."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for backend='torch'. Install with: pip install torch"
        ) from None
    return torch


def require_jax() -> tuple[ModuleType, ModuleType]:
    """Return ``(jax, jax.numpy)`` with float64 enabled.

    Calls ``jax.config.update("jax_enable_x64", True)`` so that
    ``jnp.float64`` arrays are supported on all devices.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for backend='jax'. Install with: pip install jax[cuda12]"
        ) from None
    jax.config.update("jax_enable_x64", True)
    return jax, jnp
