"""Lazy imports and availability flags for optional GPU packages."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
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


# ---------------------------------------------------------------------------
# GPU cache management
# ---------------------------------------------------------------------------


def clean_torch_cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator: release torch's CUDA caching allocator after *func*.

    PyTorch's caching allocator holds GPU blocks in a pool even after
    tensor references are dropped.  Wrap a GPU kernel with this
    decorator so ``torch.cuda.empty_cache()`` runs in a ``finally``
    block -- the cache is returned to the CUDA driver on normal
    return *and* on exceptions.

    The decorator is a no-op when torch is not installed or no CUDA
    device is visible, so it is safe to apply unconditionally to
    torch-backed kernels.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        finally:
            if has_torch:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return wrapper


def clean_cupy_cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator: release cupy's GPU memory pool after *func*.

    CuPy's default memory pool caches GPU allocations for reuse.
    Wrap a GPU kernel with this decorator so
    ``cp.get_default_memory_pool().free_all_blocks()`` runs in a
    ``finally`` block, returning cached blocks to the CUDA driver
    on normal return and on exceptions.

    The decorator is a no-op when cupy is not installed, so it is
    safe to apply unconditionally to cupy-backed kernels.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        finally:
            if has_cupy:
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()

    return wrapper


def clean_jax_cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator: release JAX compilation caches after *func*.

    JAX has no public API for returning pooled device memory to the
    driver, but ``jax.clear_caches()`` (if available) releases
    compiled-function caches which can indirectly free device-side
    state.  Wrap a JAX-backed kernel with this decorator so the call
    runs in a ``finally`` block.

    The decorator is a no-op when jax is not installed or the
    ``clear_caches`` symbol is missing, so it is safe to apply
    unconditionally to jax-backed kernels.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        finally:
            if has_jax:
                import jax

                clear_caches = getattr(jax, "clear_caches", None)
                if clear_caches is not None:
                    clear_caches()

    return wrapper
