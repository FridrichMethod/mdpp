"""Computation backends for analysis functions."""

from mdpp.analysis._backends._imports import has_cupy, has_jax, has_torch
from mdpp.analysis._backends._registry import BackendRegistry

__all__ = [
    "BackendRegistry",
    "has_cupy",
    "has_jax",
    "has_torch",
]
