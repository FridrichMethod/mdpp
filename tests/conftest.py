"""Shared fixtures for mdpp analysis tests."""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest


@pytest.fixture()
def two_atom_trajectory() -> md.Trajectory:
    """Return a small two-atom trajectory with known fluctuations."""
    topology = md.Topology()
    chain = topology.add_chain()
    residue_1 = topology.add_residue("ALA", chain, resSeq=1)
    residue_2 = topology.add_residue("ALA", chain, resSeq=2)
    atom_1 = topology.add_atom("CA", md.element.carbon, residue_1)
    atom_2 = topology.add_atom("CA", md.element.carbon, residue_2)
    topology.add_bond(atom_1, atom_2)

    xyz = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    time_ps = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


@pytest.fixture()
def correlated_ca_trajectory() -> md.Trajectory:
    """Return a trajectory with strongly correlated and anti-correlated CA motion."""
    topology = md.Topology()
    chain = topology.add_chain()
    atoms = []
    for residue_index in range(1, 4):
        residue = topology.add_residue("ALA", chain, resSeq=residue_index)
        atom = topology.add_atom("CA", md.element.carbon, residue)
        atoms.append(atom)
        if residue_index > 1:
            topology.add_bond(atoms[residue_index - 2], atom)

    xyz = np.array(
        [
            [[0.00, 0.0, 0.0], [0.20, 0.0, 0.0], [0.40, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [0.22, 0.0, 0.0], [0.38, 0.0, 0.0]],
            [[-0.02, 0.0, 0.0], [0.18, 0.0, 0.0], [0.42, 0.0, 0.0]],
            [[0.04, 0.0, 0.0], [0.24, 0.0, 0.0], [0.36, 0.0, 0.0]],
            [[-0.04, 0.0, 0.0], [0.16, 0.0, 0.0], [0.44, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    time_ps = np.arange(xyz.shape[0], dtype=np.float64) * 100.0
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)


@pytest.fixture()
def hbond_trajectory() -> md.Trajectory:
    """Return a trajectory with a hydrogen bond present in selected frames."""
    topology = md.Topology()
    chain = topology.add_chain()
    donor_residue = topology.add_residue("DON", chain, resSeq=1)
    acceptor_residue = topology.add_residue("ACC", chain, resSeq=2)

    donor_n = topology.add_atom("N", md.element.nitrogen, donor_residue)
    donor_h = topology.add_atom("H", md.element.hydrogen, donor_residue)
    topology.add_atom("O", md.element.oxygen, acceptor_residue)
    topology.add_bond(donor_n, donor_h)

    xyz = np.array(
        [
            [[0.00, 0.0, 0.0], [0.10, 0.0, 0.0], [0.25, 0.0, 0.0]],
            [[0.00, 0.0, 0.0], [0.10, 0.0, 0.0], [0.45, 0.0, 0.0]],
            [[0.00, 0.0, 0.0], [0.10, 0.0, 0.0], [0.24, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    time_ps = np.array([0.0, 20.0, 40.0], dtype=np.float64)
    return md.Trajectory(xyz=xyz, topology=topology, time=time_ps)
