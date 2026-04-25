"""Computation backends for analysis functions."""

from mdpp.analysis._backends._imports import (
    clean_cupy_cache,
    clean_torch_cache,
    has_cupy,
    has_jax,
    has_torch,
    query_free_gpu_bytes,
)
from mdpp.analysis._backends._registry import (
    BackendRegistry,
    DCCMBackend,
    DistanceBackend,
    RMSDBackend,
)

__all__ = [
    "BackendRegistry",
    "DCCMBackend",
    "DistanceBackend",
    "RMSDBackend",
    "clean_cupy_cache",
    "clean_torch_cache",
    "has_cupy",
    "has_jax",
    "has_torch",
    "query_free_gpu_bytes",
]
