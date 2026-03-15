"""Trajectory manipulation utilities for system preparation."""

from __future__ import annotations

from collections.abc import Sequence

import mdtraj as md
import numpy as np


def merge_trajectories(trajectories: Sequence[md.Trajectory]) -> md.Trajectory:
    """Concatenate multiple trajectories along the time axis.

    All trajectories must share the same topology and number of atoms.

    Args:
        trajectories: Sequence of trajectories to concatenate.

    Returns:
        A single trajectory containing all frames in order.

    Raises:
        ValueError: If fewer than two trajectories are provided or topologies
            do not match.
    """
    if len(trajectories) < 2:
        raise ValueError("At least two trajectories are required for merging.")
    return md.join(trajectories)


def slice_trajectory(
    traj: md.Trajectory,
    *,
    start: int | None = None,
    stop: int | None = None,
    stride: int | None = None,
) -> md.Trajectory:
    """Slice a trajectory by frame range with validation.

    Args:
        traj: Input trajectory.
        start: Starting frame index (inclusive). Defaults to ``0``.
        stop: Stopping frame index (exclusive). Defaults to ``n_frames``.
        stride: Frame stride. Defaults to ``1``.

    Returns:
        A new trajectory with the selected frames.
    """
    return traj[start:stop:stride]


def subsample_trajectory(traj: md.Trajectory, n_frames: int) -> md.Trajectory:
    """Evenly subsample a trajectory to a target number of frames.

    Args:
        traj: Input trajectory.
        n_frames: Desired number of output frames.

    Returns:
        A new trajectory with approximately ``n_frames`` evenly spaced frames.

    Raises:
        ValueError: If ``n_frames`` is less than 1 or exceeds the trajectory
            length.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1.")
    if n_frames > traj.n_frames:
        raise ValueError(f"n_frames ({n_frames}) exceeds trajectory length ({traj.n_frames}).")
    if n_frames == traj.n_frames:
        return traj

    indices = np.linspace(0, traj.n_frames - 1, n_frames, dtype=int)
    return traj[indices]
