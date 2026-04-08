"""Tests for trajectory loading and alignment helpers."""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdpp.core.trajectory import align_trajectory, load_trajectories, load_trajectory


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


# --- align_trajectory ---


@pytest.fixture()
def multi_atom_traj() -> md.Trajectory:
    """3-CA-atom, 5-frame trajectory with per-frame translations for alignment tests."""
    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(1, 4):
        res = topology.add_residue("ALA", chain, resSeq=i)
        topology.add_atom("CA", md.element.carbon, res)

    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(5, 3, 3)).astype(np.float32)
    for i in range(5):
        xyz[i] += i * 0.1
    time_ps = np.arange(5, dtype=np.float64) * 10.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


# -- Object identity and isolation (inplace=False) --


class TestAlignCopy:
    """Tests for the default non-inplace path."""

    def test_returns_new_object(self, multi_atom_traj: md.Trajectory) -> None:
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        assert aligned is not multi_atom_traj

    def test_original_xyz_untouched(self, multi_atom_traj: md.Trajectory) -> None:
        original_xyz = multi_atom_traj.xyz.copy()
        align_trajectory(multi_atom_traj, atom_selection="name CA")
        np.testing.assert_allclose(multi_atom_traj.xyz, original_xyz)

    def test_original_time_untouched(self, multi_atom_traj: md.Trajectory) -> None:
        original_time = multi_atom_traj.time.copy()
        align_trajectory(multi_atom_traj, atom_selection="name CA")
        np.testing.assert_allclose(multi_atom_traj.time, original_time)

    def test_shares_topology(self, multi_atom_traj: md.Trajectory) -> None:
        """Topology is shared (no deepcopy) for memory efficiency."""
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        assert aligned.topology is multi_atom_traj.topology

    def test_shares_time(self, multi_atom_traj: md.Trajectory) -> None:
        """Time array is shared (not copied)."""
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        assert np.shares_memory(aligned.time, multi_atom_traj.time)

    def test_xyz_not_shared(self, multi_atom_traj: md.Trajectory) -> None:
        """Xyz must be a separate copy so alignment doesn't corrupt the original."""
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        assert not np.shares_memory(aligned.xyz, multi_atom_traj.xyz)

    def test_n_frames_preserved(self, multi_atom_traj: md.Trajectory) -> None:
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        assert aligned.n_frames == multi_atom_traj.n_frames

    def test_n_atoms_preserved(self, multi_atom_traj: md.Trajectory) -> None:
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        assert aligned.n_atoms == multi_atom_traj.n_atoms

    def test_mutating_aligned_does_not_affect_original(
        self, multi_atom_traj: md.Trajectory
    ) -> None:
        """Writing to aligned.xyz must not change the original."""
        original_xyz = multi_atom_traj.xyz.copy()
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        aligned.xyz[:] = 999.0
        np.testing.assert_allclose(multi_atom_traj.xyz, original_xyz)

    def test_mutating_original_does_not_affect_aligned(
        self, multi_atom_traj: md.Trajectory
    ) -> None:
        """Writing to original.xyz must not change the aligned copy."""
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA")
        aligned_xyz = aligned.xyz.copy()
        multi_atom_traj.xyz[:] = 999.0
        np.testing.assert_allclose(aligned.xyz, aligned_xyz)


# -- In-place path --


class TestAlignInplace:
    """Tests for inplace=True."""

    def test_returns_same_object(self, multi_atom_traj: md.Trajectory) -> None:
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", inplace=True)
        assert aligned is multi_atom_traj

    def test_xyz_is_same_buffer(self, multi_atom_traj: md.Trajectory) -> None:
        """In-place alignment should modify the original xyz buffer, not create a new one."""
        xyz_id = id(multi_atom_traj.xyz)
        align_trajectory(multi_atom_traj, atom_selection="name CA", inplace=True)
        assert id(multi_atom_traj.xyz) == xyz_id

    def test_modifies_original_xyz(self, multi_atom_traj: md.Trajectory) -> None:
        original_xyz = multi_atom_traj.xyz.copy()
        align_trajectory(multi_atom_traj, atom_selection="name CA", inplace=True)
        assert not np.allclose(multi_atom_traj.xyz, original_xyz)

    def test_inplace_and_copy_produce_same_coordinates(
        self, multi_atom_traj: md.Trajectory
    ) -> None:
        """Both paths should produce identical aligned coordinates."""
        copy_aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", inplace=False)
        inplace_aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", inplace=True)
        np.testing.assert_allclose(inplace_aligned.xyz, copy_aligned.xyz, atol=1e-6)


# -- Alignment correctness --


class TestAlignCorrectness:
    """Tests that alignment actually does the right thing."""

    def test_reference_frame_unchanged(self, multi_atom_traj: md.Trajectory) -> None:
        ref_xyz = multi_atom_traj.xyz[0].copy()
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=0)
        np.testing.assert_allclose(aligned.xyz[0], ref_xyz, atol=1e-6)

    def test_reference_frame_last(self, multi_atom_traj: md.Trajectory) -> None:
        """Aligning to the last frame should work and leave that frame unchanged."""
        last = multi_atom_traj.n_frames - 1
        ref_xyz = multi_atom_traj.xyz[last].copy()
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=last)
        np.testing.assert_allclose(aligned.xyz[last], ref_xyz, atol=1e-6)

    def test_reduces_rmsd_to_reference(self, multi_atom_traj: md.Trajectory) -> None:
        """Alignment should reduce RMSD between each frame and the reference."""
        ref = 0
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=ref)
        for i in range(1, multi_atom_traj.n_frames):
            orig_rmsd = float(
                np.sqrt(np.mean((multi_atom_traj.xyz[i] - multi_atom_traj.xyz[ref]) ** 2))
            )
            new_rmsd = float(np.sqrt(np.mean((aligned.xyz[i] - aligned.xyz[ref]) ** 2)))
            assert new_rmsd <= orig_rmsd + 1e-6

    def test_pure_translation_removed(self) -> None:
        """A trajectory differing only by translation should have zero RMSD after alignment."""
        topology = md.Topology()
        chain = topology.add_chain()
        for i in range(1, 4):
            res = topology.add_residue("ALA", chain, resSeq=i)
            topology.add_atom("CA", md.element.carbon, res)

        base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        xyz = np.stack([base, base + 5.0, base - 3.0])
        traj = md.Trajectory(xyz=xyz, topology=topology)

        aligned = align_trajectory(traj, atom_selection="name CA", reference_frame=0)
        for i in range(1, 3):
            rmsd = float(np.sqrt(np.mean((aligned.xyz[i] - aligned.xyz[0]) ** 2)))
            assert rmsd == pytest.approx(0.0, abs=1e-5)

    def test_all_frames_identical_is_noop(self) -> None:
        """If all frames are identical, alignment should not change coordinates."""
        topology = md.Topology()
        chain = topology.add_chain()
        res = topology.add_residue("ALA", chain, resSeq=1)
        topology.add_atom("CA", md.element.carbon, res)

        xyz = np.tile([[1.0, 2.0, 3.0]], (5, 1, 1)).astype(np.float32)
        traj = md.Trajectory(xyz=xyz.copy(), topology=topology)

        aligned = align_trajectory(traj, atom_selection="name CA")
        np.testing.assert_allclose(aligned.xyz, xyz, atol=1e-6)


# -- Corner cases --


class TestAlignCornerCases:
    """Edge cases and error handling."""

    def test_single_frame(self) -> None:
        """A single-frame trajectory should align without error (trivial case)."""
        topology = md.Topology()
        chain = topology.add_chain()
        res = topology.add_residue("ALA", chain, resSeq=1)
        topology.add_atom("CA", md.element.carbon, res)

        xyz = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        traj = md.Trajectory(xyz=xyz.copy(), topology=topology)

        aligned = align_trajectory(traj, atom_selection="name CA", reference_frame=0)
        np.testing.assert_allclose(aligned.xyz, xyz, atol=1e-6)

    def test_two_frames(self) -> None:
        """Two-frame trajectory should align successfully."""
        topology = md.Topology()
        chain = topology.add_chain()
        for i in range(1, 3):
            res = topology.add_residue("ALA", chain, resSeq=i)
            topology.add_atom("CA", md.element.carbon, res)

        xyz = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
            dtype=np.float32,
        )
        traj = md.Trajectory(xyz=xyz, topology=topology)
        aligned = align_trajectory(traj, atom_selection="name CA")
        rmsd = float(np.sqrt(np.mean((aligned.xyz[1] - aligned.xyz[0]) ** 2)))
        assert rmsd == pytest.approx(0.0, abs=1e-5)

    def test_reference_frame_zero(self, multi_atom_traj: md.Trajectory) -> None:
        """reference_frame=0 (default) should not raise."""
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=0)
        assert aligned.n_frames == multi_atom_traj.n_frames

    def test_reference_frame_last_valid(self, multi_atom_traj: md.Trajectory) -> None:
        """reference_frame = n_frames - 1 (last valid) should not raise."""
        last = multi_atom_traj.n_frames - 1
        aligned = align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=last)
        assert aligned.n_frames == multi_atom_traj.n_frames

    def test_reference_frame_negative_raises(self, multi_atom_traj: md.Trajectory) -> None:
        with pytest.raises(ValueError, match="reference_frame must be in"):
            align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=-1)

    def test_reference_frame_equals_n_frames_raises(self, multi_atom_traj: md.Trajectory) -> None:
        """reference_frame = n_frames is out of bounds."""
        with pytest.raises(ValueError, match="reference_frame must be in"):
            align_trajectory(
                multi_atom_traj,
                atom_selection="name CA",
                reference_frame=multi_atom_traj.n_frames,
            )

    def test_reference_frame_far_out_of_range_raises(self, multi_atom_traj: md.Trajectory) -> None:
        with pytest.raises(ValueError, match="reference_frame must be in"):
            align_trajectory(multi_atom_traj, atom_selection="name CA", reference_frame=9999)

    def test_invalid_atom_selection_raises(self, multi_atom_traj: md.Trajectory) -> None:
        """An atom selection matching no atoms should raise ValueError."""
        with pytest.raises(ValueError, match="matched no atoms"):
            align_trajectory(multi_atom_traj, atom_selection="name ZZ")
