"""Tests for trajectory loading helpers, especially n_frames partial loading."""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpp.core.trajectory import load_trajectories, load_trajectory


@pytest.fixture()
def traj_on_disk(tmp_path: Path) -> tuple[Path, Path, int]:
    """Write a small PDB + XTC to disk and return (xtc_path, pdb_path, n_frames).

    Creates a 50-frame trajectory with 2 CA atoms so tests run fast.
    Frame times are 0, 10, 20, ... , 490 ps.
    """
    topology = md.Topology()
    chain = topology.add_chain()
    res1 = topology.add_residue("ALA", chain, resSeq=1)
    res2 = topology.add_residue("GLY", chain, resSeq=2)
    a1 = topology.add_atom("CA", md.element.carbon, res1)
    a2 = topology.add_atom("CA", md.element.carbon, res2)
    topology.add_bond(a1, a2)

    n_frames = 50
    xyz = np.zeros((n_frames, 2, 3), dtype=np.float32)
    xyz[:, 1, 0] = np.linspace(0.1, 0.5, n_frames)
    time_ps = np.arange(n_frames, dtype=np.float64) * 10.0

    traj = md.Trajectory(xyz=xyz, topology=topology, time=time_ps)
    pdb_path = tmp_path / "topol.pdb"
    xtc_path = tmp_path / "traj.xtc"
    traj[0].save_pdb(str(pdb_path))
    traj.save_xtc(str(xtc_path))
    return xtc_path, pdb_path, n_frames


# --- Basic n_frames behavior ---


def test_n_frames_none_loads_all(traj_on_disk: tuple[Path, Path, int]) -> None:
    """n_frames=None should load the entire trajectory."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb)
    assert traj.n_frames == total


def test_n_frames_exact(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Requesting exactly 20 frames should return 20 frames."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, n_frames=20)
    assert traj.n_frames == 20


def test_n_frames_one(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Requesting 1 frame should return exactly 1 frame."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, n_frames=1)
    assert traj.n_frames == 1


def test_n_frames_exceeds_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Requesting more frames than available should return all frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, n_frames=total + 100)
    assert traj.n_frames == total


def test_n_frames_equals_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Requesting exactly the total frame count should return all frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, n_frames=total)
    assert traj.n_frames == total


# --- n_frames not a multiple of chunk size (1000) ---


def test_n_frames_not_multiple_of_chunk(traj_on_disk: tuple[Path, Path, int]) -> None:
    """n_frames=7 (not a multiple of chunk size) should still return exactly 7."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, n_frames=7)
    assert traj.n_frames == 7


# --- n_frames + stride interaction ---


def test_n_frames_with_stride(traj_on_disk: tuple[Path, Path, int]) -> None:
    """n_frames counts post-stride frames, not raw frames."""
    xtc, pdb, _total = traj_on_disk
    stride = 5
    traj = load_trajectory(xtc, topology_path=pdb, stride=stride, n_frames=3)
    assert traj.n_frames == 3


def test_n_frames_with_stride_exceeds_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """When stride reduces available frames below n_frames, return all strided frames."""
    xtc, pdb, total = traj_on_disk
    stride = 10
    max_strided = (total + stride - 1) // stride  # ceil division
    traj = load_trajectory(xtc, topology_path=pdb, stride=stride, n_frames=1000)
    assert traj.n_frames == max_strided


# --- n_frames + atom_selection ---


def test_n_frames_with_atom_selection(traj_on_disk: tuple[Path, Path, int]) -> None:
    """n_frames and atom_selection should both be applied."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(
        xtc, topology_path=pdb, n_frames=10, atom_selection="name CA and resSeq 1"
    )
    assert traj.n_frames == 10
    assert traj.n_atoms == 1


# --- Time values are preserved ---


def test_n_frames_preserves_time(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Loaded frames should have correct time values from the file."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, n_frames=5)
    expected_time = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(traj.time, expected_time, atol=1e-3)


def test_n_frames_with_stride_preserves_time(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Time values should reflect the stride."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stride=5, n_frames=3)
    expected_time = np.array([0.0, 50.0, 100.0])
    np.testing.assert_allclose(traj.time, expected_time, atol=1e-3)


# --- Coordinate consistency ---


def test_n_frames_matches_full_load(traj_on_disk: tuple[Path, Path, int]) -> None:
    """First N frames from partial load should match slicing the full trajectory."""
    xtc, pdb, _ = traj_on_disk
    n = 15
    partial = load_trajectory(xtc, topology_path=pdb, n_frames=n)
    full = load_trajectory(xtc, topology_path=pdb)
    np.testing.assert_allclose(partial.xyz, full[:n].xyz, atol=1e-6)
    np.testing.assert_allclose(partial.time, full[:n].time, atol=1e-3)


def test_n_frames_with_stride_matches_full_load(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Partial load with stride should match full load with same stride, then sliced."""
    xtc, pdb, _ = traj_on_disk
    stride = 5
    n = 3
    partial = load_trajectory(xtc, topology_path=pdb, stride=stride, n_frames=n)
    full = load_trajectory(xtc, topology_path=pdb, stride=stride)
    np.testing.assert_allclose(partial.xyz, full[:n].xyz, atol=1e-6)
    np.testing.assert_allclose(partial.time, full[:n].time, atol=1e-3)


# --- Validation errors ---


def test_n_frames_zero_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """n_frames=0 should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="n_frames must be >= 1"):
        load_trajectory(xtc, topology_path=pdb, n_frames=0)


def test_n_frames_negative_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Negative n_frames should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="n_frames must be >= 1"):
        load_trajectory(xtc, topology_path=pdb, n_frames=-5)


def test_stride_zero_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """stride=0 should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="stride must be >= 1"):
        load_trajectory(xtc, topology_path=pdb, stride=0)


# --- load_trajectories with n_frames ---


def test_load_trajectories_n_frames(traj_on_disk: tuple[Path, Path, int]) -> None:
    """load_trajectories should pass n_frames through to each trajectory."""
    xtc, pdb, _ = traj_on_disk
    trajs = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        n_frames=10,
    )
    assert len(trajs) == 2
    assert all(t.n_frames == 10 for t in trajs)


def test_load_trajectories_mismatched_lengths(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Mismatched path lengths should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="same length"):
        load_trajectories([xtc], topology_paths=[pdb, pdb])
