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


# --- skip parameter ---


def test_skip_frames(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skipping frames should start reading from the offset."""
    xtc, pdb, _ = traj_on_disk
    skipped = load_trajectory(xtc, topology_path=pdb, skip=10, n_frames=5)
    full = load_trajectory(xtc, topology_path=pdb)
    assert skipped.n_frames == 5
    np.testing.assert_allclose(skipped.xyz, full[10:15].xyz, atol=1e-6)
    np.testing.assert_allclose(skipped.time, full[10:15].time, atol=1e-3)


def test_skip_with_stride(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip is applied before stride."""
    xtc, pdb, _ = traj_on_disk
    # skip=10, stride=5, n_frames=3 -> raw frames 10, 15, 20
    skipped = load_trajectory(xtc, topology_path=pdb, skip=10, stride=5, n_frames=3)
    full = load_trajectory(xtc, topology_path=pdb)
    assert skipped.n_frames == 3
    np.testing.assert_allclose(skipped.xyz, full[10::5][:3].xyz, atol=1e-6)


def test_skip_without_n_frames(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip alone (no n_frames) should load from offset to end."""
    xtc, pdb, total = traj_on_disk
    skipped = load_trajectory(xtc, topology_path=pdb, skip=40)
    assert skipped.n_frames == total - 40


def test_skip_exceeds_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip past the end should return an empty or zero-frame trajectory."""
    xtc, pdb, total = traj_on_disk
    skipped = load_trajectory(xtc, topology_path=pdb, skip=total + 10, n_frames=5)
    assert skipped.n_frames == 0


def test_skip_zero_is_default(traj_on_disk: tuple[Path, Path, int]) -> None:
    """skip=0 should behave identically to omitting skip."""
    xtc, pdb, _ = traj_on_disk
    with_skip = load_trajectory(xtc, topology_path=pdb, skip=0, n_frames=10)
    without_skip = load_trajectory(xtc, topology_path=pdb, n_frames=10)
    np.testing.assert_allclose(with_skip.xyz, without_skip.xyz, atol=1e-6)


def test_skip_negative_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Negative skip should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="skip must be >= 0"):
        load_trajectory(xtc, topology_path=pdb, skip=-1)


def test_skip_preserves_time(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skipped frames should have time values from the file, not reset to zero."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, skip=20, n_frames=3)
    # Frame times are 0, 10, 20, ..., so frame 20 = 200 ps
    expected_time = np.array([200.0, 210.0, 220.0])
    np.testing.assert_allclose(traj.time, expected_time, atol=1e-3)


def test_skip_with_stride_preserves_time(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip + stride: time should reflect both the offset and the stride."""
    xtc, pdb, _ = traj_on_disk
    # skip=10, stride=5, n_frames=3 -> raw frames 10, 15, 20 -> times 100, 150, 200
    traj = load_trajectory(xtc, topology_path=pdb, skip=10, stride=5, n_frames=3)
    expected_time = np.array([100.0, 150.0, 200.0])
    np.testing.assert_allclose(traj.time, expected_time, atol=1e-3)


def test_skip_with_stride_matches_full_load(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip + stride + n_frames should match full[skip::stride][:n_frames]."""
    xtc, pdb, _ = traj_on_disk
    skip, stride, n = 5, 3, 7
    partial = load_trajectory(xtc, topology_path=pdb, skip=skip, stride=stride, n_frames=n)
    full = load_trajectory(xtc, topology_path=pdb)
    expected = full[skip::stride][:n]
    assert partial.n_frames == expected.n_frames
    np.testing.assert_allclose(partial.xyz, expected.xyz, atol=1e-6)
    np.testing.assert_allclose(partial.time, expected.time, atol=1e-3)


def test_skip_with_atom_selection(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip and atom_selection should both be applied."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(
        xtc, topology_path=pdb, skip=5, n_frames=10, atom_selection="name CA and resSeq 1"
    )
    assert traj.n_frames == 10
    assert traj.n_atoms == 1


def test_skip_n_frames_exceeds_remaining(traj_on_disk: tuple[Path, Path, int]) -> None:
    """When skip + n_frames exceeds total, return only available frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, skip=45, n_frames=100)
    assert traj.n_frames == total - 45


def test_skip_exact_boundary(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skipping exactly to the total should return zero frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, skip=total, n_frames=5)
    assert traj.n_frames == 0


def test_skip_last_frame(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skipping to total-1 should return exactly 1 frame."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, skip=total - 1, n_frames=10)
    assert traj.n_frames == 1
    full = load_trajectory(xtc, topology_path=pdb)
    np.testing.assert_allclose(traj.xyz, full[-1:].xyz, atol=1e-6)


def test_skip_without_n_frames_with_stride(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Skip + stride without n_frames should load from offset to end with stride."""
    xtc, pdb, _total = traj_on_disk
    skip, stride = 10, 5
    traj = load_trajectory(xtc, topology_path=pdb, skip=skip, stride=stride)
    full = load_trajectory(xtc, topology_path=pdb)
    expected = full[skip::stride]
    assert traj.n_frames == expected.n_frames
    np.testing.assert_allclose(traj.xyz, expected.xyz, atol=1e-6)


# --- load_trajectories with skip ---


def test_load_trajectories_skip(traj_on_disk: tuple[Path, Path, int]) -> None:
    """load_trajectories should pass skip through to each trajectory."""
    xtc, pdb, _ = traj_on_disk
    trajs = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        skip=10,
        n_frames=5,
    )
    ref = load_trajectory(xtc, topology_path=pdb, skip=10, n_frames=5)
    assert len(trajs) == 2
    for traj in trajs:
        assert traj.n_frames == 5
        np.testing.assert_allclose(traj.xyz, ref.xyz, atol=1e-6)


def test_load_trajectories_skip_parallel(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Parallel loading with skip should match sequential."""
    xtc, pdb, _ = traj_on_disk
    sequential = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        skip=15,
        n_frames=10,
        stride=2,
    )
    parallel = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        skip=15,
        n_frames=10,
        stride=2,
        max_workers=2,
    )
    for seq, par in zip(sequential, parallel, strict=True):
        assert seq.n_frames == par.n_frames
        np.testing.assert_allclose(seq.xyz, par.xyz, atol=1e-6)
        np.testing.assert_allclose(seq.time, par.time, atol=1e-3)


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


def test_load_trajectories_parallel(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Parallel loading should produce identical results to sequential."""
    xtc, pdb, _ = traj_on_disk
    sequential = load_trajectories(
        [xtc, xtc, xtc],
        topology_paths=[pdb, pdb, pdb],
        n_frames=10,
    )
    parallel = load_trajectories(
        [xtc, xtc, xtc],
        topology_paths=[pdb, pdb, pdb],
        n_frames=10,
        max_workers=3,
    )
    assert len(parallel) == len(sequential)
    for seq, par in zip(sequential, parallel, strict=True):
        assert seq.n_frames == par.n_frames
        np.testing.assert_allclose(seq.xyz, par.xyz, atol=1e-6)
        np.testing.assert_allclose(seq.time, par.time, atol=1e-3)


def test_load_trajectories_parallel_preserves_order(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Parallel loading must return trajectories in input order."""
    xtc, pdb, _total = traj_on_disk
    trajs = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        n_frames=5,
        max_workers=2,
    )
    ref = load_trajectory(xtc, topology_path=pdb, n_frames=5)
    for traj in trajs:
        np.testing.assert_allclose(traj.xyz, ref.xyz, atol=1e-6)


def test_load_trajectories_mismatched_lengths(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Mismatched path lengths should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="same length"):
        load_trajectories([xtc], topology_paths=[pdb, pdb])
