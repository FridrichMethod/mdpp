"""Computation backends for analysis functions."""

from mdpp.analysis._backends._imports import has_cupy, has_jax, has_torch
from mdpp.analysis._backends._registry import (
    BackendRegistry,
    DistanceBackend,
    RMSDBackend,
)

__all__ = [
    "BackendRegistry",
    "DistanceBackend",
    "RMSDBackend",
    "has_cupy",
    "has_jax",
    "has_torch",
]
