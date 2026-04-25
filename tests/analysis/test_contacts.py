"""Tests for contact analysis, focused on the two-pass native-contact rewrite.

The legacy ``compute_native_contacts`` invoked ``mdtraj.compute_contacts``
with ``contacts="all"`` over the full trajectory and discarded everything
that was not native, paying O(F * R^2) work to keep ~K << R^2 columns.
The current implementation finds the native pair list from the reference
frame in one O(R^2) call and only then computes distances for those K
pairs across all frames.  These tests pin the new path against the
legacy one-pass reference so the optimisation cannot silently drift.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest
from numpy.typing import NDArray

from mdpp.analysis.contacts import compute_native_contacts


def _legacy_native_contacts(
    traj: md.Trajectory,
    *,
    reference_frame: int,
    cutoff_nm: float,
    scheme: str,
    periodic: bool,
) -> NDArray[np.floating]:
    """Reference implementation: one-pass all-pair compute, then filter.

    This is the form ``compute_native_contacts`` had before the two-pass
    rewrite.  Kept here purely as a numerical oracle for the new code.
    """
    distances, _pairs = md.compute_contacts(traj, contacts="all", scheme=scheme, periodic=periodic)
    native_mask = distances[reference_frame] < cutoff_nm
    if not np.any(native_mask):
        raise ValueError("no native contacts in reference frame for the test")
    native_distances = distances[:, native_mask]
    return np.mean(native_distances < cutoff_nm, axis=1)


def _make_protein_trajectory(
    n_frames: int = 60,
    n_residues: int = 18,
    seed: int = 11,
) -> md.Trajectory:
    """Build a peptide-like trajectory rich enough to have real native contacts."""
    rng = np.random.default_rng(seed)
    topology = md.Topology()
    chain = topology.add_chain()
    template: list[tuple[int, str]] = []
    for resseq in range(1, n_residues + 1):
        residue = topology.add_residue("ALA", chain, resSeq=resseq)
        topology.add_atom("N", md.element.nitrogen, residue)
        topology.add_atom("CA", md.element.carbon, residue)
        topology.add_atom("C", md.element.carbon, residue)
        topology.add_atom("O", md.element.oxygen, residue)
        topology.add_atom("CB", md.element.carbon, residue)
        template.extend([
            (resseq, "N"),
            (resseq, "CA"),
            (resseq, "C"),
            (resseq, "O"),
            (resseq, "CB"),
        ])

    n_atoms = topology.n_atoms
    base = np.zeros((n_atoms, 3), dtype=np.float32)
    for atom_index, (resseq, name) in enumerate(template):
        offset = {"N": 0.00, "CA": 0.15, "C": 0.30, "O": 0.45, "CB": 0.20}[name]
        base[atom_index, 0] = (resseq - 1) * 0.38 + offset
        base[atom_index, 1] = 0.05 if name == "CB" else 0.0

    noise = rng.normal(scale=0.05, size=(n_frames, n_atoms, 3)).astype(np.float32)
    xyz = base[None, :, :] + noise
    time_ps = np.arange(n_frames, dtype=np.float64) * 10.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


@pytest.fixture()
def native_contacts_trajectory() -> md.Trajectory:
    return _make_protein_trajectory()


_REGRESSION_CUTOFF_NM = 1.0
"""Cutoff for the regression tests.

The synthetic extended-chain trajectory used by :func:`_make_protein_trajectory`
puts non-adjacent residues at ~0.69 nm minimum heavy-atom distance, so a
0.6 nm cutoff lands too close to that boundary and produces no native pairs
on some seeds.  1.0 nm sits comfortably in the regime where mdtraj's
default residue-separation filter still leaves a few hundred native pairs,
which is what we want to actually exercise the two-pass code.
"""


def test_native_contacts_two_pass_matches_legacy(native_contacts_trajectory) -> None:
    """The two-pass form must reproduce the legacy one-pass Q(t) exactly."""
    expected = _legacy_native_contacts(
        native_contacts_trajectory,
        reference_frame=0,
        cutoff_nm=_REGRESSION_CUTOFF_NM,
        scheme="closest-heavy",
        periodic=False,
    )
    actual = compute_native_contacts(
        native_contacts_trajectory,
        reference_frame=0,
        cutoff_nm=_REGRESSION_CUTOFF_NM,
        scheme="closest-heavy",
        periodic=False,
    )
    np.testing.assert_allclose(actual.fraction, expected, atol=1e-7)


def test_native_contacts_native_pairs_is_subset_of_all_pairs(
    native_contacts_trajectory,
) -> None:
    """The two-pass rewrite must keep the same native pair list.

    ``native_pairs`` should be exactly the rows of the all-pair table
    whose reference distance is below the cutoff, so the optimisation
    cannot silently change which pairs end up in the result.
    """
    distances, all_pairs = md.compute_contacts(
        native_contacts_trajectory[0:1],
        contacts="all",
        scheme="closest-heavy",
        periodic=False,
    )
    expected_pairs = all_pairs[distances[0] < _REGRESSION_CUTOFF_NM]
    result = compute_native_contacts(
        native_contacts_trajectory,
        cutoff_nm=_REGRESSION_CUTOFF_NM,
        scheme="closest-heavy",
        periodic=False,
    )
    np.testing.assert_array_equal(result.native_pairs, expected_pairs)


def test_native_contacts_invalid_reference_frame_raises(
    native_contacts_trajectory,
) -> None:
    with pytest.raises(ValueError, match="reference_frame must be in"):
        compute_native_contacts(
            native_contacts_trajectory,
            reference_frame=999,
            cutoff_nm=0.6,
            periodic=False,
        )


def test_native_contacts_no_native_at_reference_raises(
    native_contacts_trajectory,
) -> None:
    """An impossibly tight cutoff must surface the no-natives error.

    The function should raise rather than silently returning an empty
    Q(t) array, otherwise downstream plotting code would silently
    produce an all-zero or shape-mismatched series.
    """
    with pytest.raises(ValueError, match="No native contacts"):
        compute_native_contacts(
            native_contacts_trajectory,
            cutoff_nm=1e-6,
            periodic=False,
        )
