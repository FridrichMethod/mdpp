"""Tests for trajectory loading helpers with range-style start/stop/stride."""

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


# --- Default behavior (no start/stop) ---


def test_defaults_load_all(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Default start=0, stop=None should load the entire trajectory."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb)
    assert traj.n_frames == total


def test_stride_only(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Stride alone should subsample from the full trajectory."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stride=5)
    assert traj.n_frames == len(range(0, total, 5))


# --- stop (exclusive end) ---


def test_stop_exclusive(traj_on_disk: tuple[Path, Path, int]) -> None:
    """stop=20 should load frames 0..19 (20 excluded)."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stop=20)
    assert traj.n_frames == 20
    full = load_trajectory(xtc, topology_path=pdb)
    np.testing.assert_allclose(traj.xyz, full[:20].xyz, atol=1e-6)


def test_stop_one(traj_on_disk: tuple[Path, Path, int]) -> None:
    """stop=1 should load exactly frame 0."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stop=1)
    assert traj.n_frames == 1


def test_stop_exceeds_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Stop beyond the file length should return all available frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stop=total + 100)
    assert traj.n_frames == total


def test_stop_equals_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """stop=total should load the entire trajectory."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stop=total)
    assert traj.n_frames == total


# --- start (inclusive begin) ---


def test_start_offset(traj_on_disk: tuple[Path, Path, int]) -> None:
    """start=10 should skip the first 10 frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=10)
    assert traj.n_frames == total - 10
    full = load_trajectory(xtc, topology_path=pdb)
    np.testing.assert_allclose(traj.xyz, full[10:].xyz, atol=1e-6)


def test_start_zero_is_default(traj_on_disk: tuple[Path, Path, int]) -> None:
    """start=0 should be identical to omitting start."""
    xtc, pdb, _ = traj_on_disk
    with_start = load_trajectory(xtc, topology_path=pdb, start=0, stop=10)
    without_start = load_trajectory(xtc, topology_path=pdb, stop=10)
    np.testing.assert_allclose(with_start.xyz, without_start.xyz, atol=1e-6)


def test_start_exceeds_total(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Start past the end should return zero frames."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=total + 10, stop=total + 20)
    assert traj.n_frames == 0


def test_start_at_last_frame(traj_on_disk: tuple[Path, Path, int]) -> None:
    """start=total-1 with no stop should return exactly 1 frame."""
    xtc, pdb, total = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=total - 1)
    assert traj.n_frames == 1
    full = load_trajectory(xtc, topology_path=pdb)
    np.testing.assert_allclose(traj.xyz, full[-1:].xyz, atol=1e-6)


# --- start + stop (range-style window) ---


def test_start_stop_window(traj_on_disk: tuple[Path, Path, int]) -> None:
    """start=10, stop=25 should load frames [10, 25)."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=10, stop=25)
    full = load_trajectory(xtc, topology_path=pdb)
    assert traj.n_frames == 15
    np.testing.assert_allclose(traj.xyz, full[10:25].xyz, atol=1e-6)


def test_start_stop_preserves_time(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Loaded frames should have correct time values from the file."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=20, stop=23)
    # Frame times: 0, 10, 20, ..., so frame 20 = 200 ps
    expected_time = np.array([200.0, 210.0, 220.0])
    np.testing.assert_allclose(traj.time, expected_time, atol=1e-3)


# --- start + stop + stride ---


def test_start_stop_stride(traj_on_disk: tuple[Path, Path, int]) -> None:
    """range(10, 30, 5) -> frames 10, 15, 20, 25 -> 4 frames."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=10, stop=30, stride=5)
    full = load_trajectory(xtc, topology_path=pdb)
    expected = full[10:30:5]
    assert traj.n_frames == len(range(10, 30, 5))
    np.testing.assert_allclose(traj.xyz, expected.xyz, atol=1e-6)


def test_start_stop_stride_preserves_time(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Time values should reflect both offset and stride."""
    xtc, pdb, _ = traj_on_disk
    # range(10, 30, 5) -> raw frames 10, 15, 20, 25 -> times 100, 150, 200, 250
    traj = load_trajectory(xtc, topology_path=pdb, start=10, stop=30, stride=5)
    expected_time = np.array([100.0, 150.0, 200.0, 250.0])
    np.testing.assert_allclose(traj.time, expected_time, atol=1e-3)


def test_stride_with_stop(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Stride with stop should match range(0, stop, stride)."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, stop=30, stride=10)
    full = load_trajectory(xtc, topology_path=pdb)
    expected = full[:30:10]
    assert traj.n_frames == len(range(0, 30, 10))
    np.testing.assert_allclose(traj.xyz, expected.xyz, atol=1e-6)


def test_start_stride_no_stop(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Start + stride without stop should load from offset to end with stride."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(xtc, topology_path=pdb, start=10, stride=5)
    full = load_trajectory(xtc, topology_path=pdb)
    expected = full[10::5]
    assert traj.n_frames == expected.n_frames
    np.testing.assert_allclose(traj.xyz, expected.xyz, atol=1e-6)


def test_start_stop_stride_matches_full_slice(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Partial load should exactly match full[start:stop:stride]."""
    xtc, pdb, _ = traj_on_disk
    start, stop, stride = 5, 40, 3
    partial = load_trajectory(xtc, topology_path=pdb, start=start, stop=stop, stride=stride)
    full = load_trajectory(xtc, topology_path=pdb)
    expected = full[start:stop:stride]
    assert partial.n_frames == expected.n_frames
    np.testing.assert_allclose(partial.xyz, expected.xyz, atol=1e-6)
    np.testing.assert_allclose(partial.time, expected.time, atol=1e-3)


# --- atom_selection ---


def test_stop_with_atom_selection(traj_on_disk: tuple[Path, Path, int]) -> None:
    """start/stop and atom_selection should both be applied."""
    xtc, pdb, _ = traj_on_disk
    traj = load_trajectory(
        xtc, topology_path=pdb, start=5, stop=15, atom_selection="name CA and resSeq 1"
    )
    assert traj.n_frames == 10
    assert traj.n_atoms == 1


# --- Validation errors ---


def test_stride_zero_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """stride=0 should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="stride must be >= 1"):
        load_trajectory(xtc, topology_path=pdb, stride=0)


def test_start_negative_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Negative start should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="start must be >= 0"):
        load_trajectory(xtc, topology_path=pdb, start=-1)


def test_stop_equal_start_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Stop == start should raise ValueError (empty range)."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="stop must be greater than start"):
        load_trajectory(xtc, topology_path=pdb, start=10, stop=10)


def test_stop_less_than_start_raises(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Stop < start should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="stop must be greater than start"):
        load_trajectory(xtc, topology_path=pdb, start=20, stop=10)


# --- load_trajectories ---


def test_load_trajectories_start_stop(traj_on_disk: tuple[Path, Path, int]) -> None:
    """load_trajectories should pass start/stop through to each trajectory."""
    xtc, pdb, _ = traj_on_disk
    trajs = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        start=10,
        stop=15,
    )
    ref = load_trajectory(xtc, topology_path=pdb, start=10, stop=15)
    assert len(trajs) == 2
    for traj in trajs:
        assert traj.n_frames == 5
        np.testing.assert_allclose(traj.xyz, ref.xyz, atol=1e-6)


def test_load_trajectories_parallel(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Parallel loading should produce identical results to sequential."""
    xtc, pdb, _ = traj_on_disk
    sequential = load_trajectories(
        [xtc, xtc, xtc],
        topology_paths=[pdb, pdb, pdb],
        start=5,
        stop=30,
        stride=3,
    )
    parallel = load_trajectories(
        [xtc, xtc, xtc],
        topology_paths=[pdb, pdb, pdb],
        start=5,
        stop=30,
        stride=3,
        max_workers=3,
    )
    assert len(parallel) == len(sequential)
    for seq, par in zip(sequential, parallel, strict=True):
        assert seq.n_frames == par.n_frames
        np.testing.assert_allclose(seq.xyz, par.xyz, atol=1e-6)
        np.testing.assert_allclose(seq.time, par.time, atol=1e-3)


def test_load_trajectories_parallel_preserves_order(
    traj_on_disk: tuple[Path, Path, int],
) -> None:
    """Parallel loading must return trajectories in input order."""
    xtc, pdb, _ = traj_on_disk
    trajs = load_trajectories(
        [xtc, xtc],
        topology_paths=[pdb, pdb],
        stop=5,
        max_workers=2,
    )
    ref = load_trajectory(xtc, topology_path=pdb, stop=5)
    for traj in trajs:
        np.testing.assert_allclose(traj.xyz, ref.xyz, atol=1e-6)


def test_load_trajectories_mismatched_lengths(traj_on_disk: tuple[Path, Path, int]) -> None:
    """Mismatched path lengths should raise ValueError."""
    xtc, pdb, _ = traj_on_disk
    with pytest.raises(ValueError, match="same length"):
        load_trajectories([xtc], topology_paths=[pdb, pdb])
