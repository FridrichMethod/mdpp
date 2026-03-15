"""Shared plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def get_axis(ax: Axes | None) -> Axes:
    """Return the passed axis or create a new one.

    Args:
        ax: An existing matplotlib axis, or ``None`` to create a new figure.

    Returns:
        The provided axis or a freshly created one.
    """
    if ax is not None:
        return ax
    _, new_axis = plt.subplots()
    return new_axis
