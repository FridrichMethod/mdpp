"""Computation backends for analysis functions."""

from mdpp.analysis._backends._imports import (
    clean_cupy_cache,
    clean_jax_cache,
    clean_torch_cache,
    has_cupy,
    has_jax,
    has_torch,
)
from mdpp.analysis._backends._registry import (
    BackendRegistry,
    DistanceBackend,
    RMSDBackend,
)

__all__ = [
    "BackendRegistry",
    "DistanceBackend",
    "RMSDBackend",
    "clean_cupy_cache",
    "clean_jax_cache",
    "clean_torch_cache",
    "has_cupy",
    "has_jax",
    "has_torch",
]
