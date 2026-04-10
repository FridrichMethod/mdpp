"""Generic backend registry for dispatching compute functions by name."""

from __future__ import annotations

from typing import Literal

type DistanceBackend = Literal["mdtraj", "numba", "cupy", "torch", "jax"]
"""Valid backend names for pairwise distance computation."""

type RMSDBackend = Literal["mdtraj", "numba", "cupy", "torch", "jax"]
"""Valid backend names for pairwise RMSD matrix computation."""


class BackendRegistry[F]:
    """Map backend name strings to callables of type ``F``.

    Args:
        default: Default backend name returned by :meth:`get` when
            the caller does not specify one.
    """

    def __init__(self, *, default: str) -> None:
        self._backends: dict[str, F] = {}
        self._default = default

    def register(self, name: str, fn: F) -> None:
        """Register a backend function under *name*."""
        self._backends[name] = fn

    def get(self, name: str | None = None) -> F:
        """Return the backend function for *name*.

        Raises:
            ValueError: If *name* is not registered.
        """
        key = name if name is not None else self._default
        if key not in self._backends:
            raise ValueError(f"Unknown backend {key!r}. Choose from {sorted(self._backends)}.")
        return self._backends[key]

    @property
    def names(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._backends)
