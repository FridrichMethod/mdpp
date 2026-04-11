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


def query_free_gpu_bytes(fallback_bytes: int = 1 << 32) -> int:
    """Best-effort query of free GPU memory in bytes.

    Returns the number of bytes currently free on the default CUDA
    device, accounting for allocations from other processes and other
    frameworks on the same card.  Tries torch and cupy in order, since
    each calls the CUDA driver's ``cudaMemGetInfo`` directly.  Falls
    back to *fallback_bytes* (default 4 GiB) when no CUDA query is
    available -- e.g. on CPU-only machines or when neither torch nor
    cupy is installed.

    Useful for backends that need adaptive chunk sizing but whose
    native framework (e.g. JAX) does not expose a reliable free-memory
    query.

    Args:
        fallback_bytes: Value returned when no CUDA query path is
            available.  Defaults to 4 GiB, chosen to fit comfortably
            on even modest GPUs.

    Returns:
        Free GPU memory in bytes, or ``fallback_bytes`` if no query
        path succeeds.
    """
    if has_torch:
        try:
            import torch

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return int(free)
        except Exception:  # pragma: no cover - defensive
            pass
    if has_cupy:
        try:
            import cupy as cp

            free, _total = cp.cuda.Device().mem_info
            return int(free)
        except Exception:  # pragma: no cover - defensive
            pass
    return fallback_bytes


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


# Note: there is deliberately no ``clean_jax_cache`` decorator.
# ``jax.clear_caches()`` clears JIT *compilation* caches (not device
# memory) and trashing the compilation cache after every call forces
# a multi-second recompile on the next call, which is catastrophic
# for repeated kernel invocations.  JAX also has no public API for
# returning pooled device memory to the driver -- XLA manages it
# directly.  JAX-backed kernels therefore rely on XLA's internal
# memory management and do not need a per-call cleanup decorator.
