"""Tests for decomposition and projection utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mdpp.analysis.decomposition import compute_pca, compute_tica


def test_compute_pca_identifies_dominant_axis() -> None:
    """PCA should place most variance on the first component for anisotropic data."""
    rng = np.random.default_rng(0)
    x_axis = np.linspace(-5.0, 5.0, 500)
    y_axis = 0.02 * rng.normal(size=x_axis.shape[0])
    features = np.column_stack([x_axis, y_axis])

    result = compute_pca(features, n_components=2, standardize=False)

    assert result.projections.shape == (features.shape[0], 2)
    assert result.components.shape == (2, 2)
    assert result.explained_variance_ratio[0] > 0.999


def test_compute_tica_returns_valid_projection() -> None:
    """TICA should produce finite projections with the requested dimensionality."""
    pytest.importorskip("deeptime")

    rng = np.random.default_rng(1)
    n_samples = 400
    slow = np.zeros(n_samples, dtype=np.float64)
    for index in range(1, n_samples):
        slow[index] = 0.95 * slow[index - 1] + 0.1 * rng.normal()
    fast = rng.normal(scale=0.3, size=n_samples)
    features = np.column_stack([slow, fast])

    result = compute_tica(features, lagtime=5, n_components=2)

    assert result.projections.shape == (n_samples, 2)
    assert np.all(np.isfinite(result.projections))


@pytest.mark.parametrize("atom_selection", [None, "protein"])
@pytest.mark.parametrize("sincos_embedding", [False, True])
def test_torsion_labels_identify_original_residues(atom_selection, sincos_embedding) -> None:
    """Phi/psi labels must survive termini, repeated chain numbering and slicing."""
    import mdtraj as md

    from mdpp.analysis.decomposition import featurize_backbone_torsions

    top = md.Topology()
    water_chain = top.add_chain()
    water = top.add_residue("HOH", water_chain, resSeq=1)
    top.add_atom("O", md.element.oxygen, water)
    for _ in range(2):
        chain = top.add_chain()
        for resseq in range(1, 4):
            residue = top.add_residue("ALA", chain, resSeq=resseq)
            for name, element in [
                ("N", md.element.nitrogen),
                ("CA", md.element.carbon),
                ("C", md.element.carbon),
                ("O", md.element.oxygen),
            ]:
                top.add_atom(name, element, residue)
    top.create_standard_bonds()
    rng = np.random.default_rng(17)
    traj = md.Trajectory(rng.normal(size=(4, top.n_atoms, 3)), top)
    result = featurize_backbone_torsions(
        traj, atom_selection=atom_selection, sincos_embedding=sincos_embedding
    )
    phi_names = ["phi_2", "phi_3", "phi_5", "phi_6"]
    psi_names = ["psi_1", "psi_2", "psi_4", "psi_5"]
    phi = md.compute_phi(traj)[1]
    psi = md.compute_psi(traj)[1]
    if sincos_embedding:
        expected_labels = [
            f"{fn}({label})"
            for labels in [phi_names, psi_names]
            for fn in ["cos", "sin"]
            for label in labels
        ]
        expected_values = np.hstack([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)])
    else:
        expected_labels = phi_names + psi_names
        expected_values = np.hstack([phi, psi])
    assert result.labels == expected_labels
    np.testing.assert_allclose(result.values, expected_values, atol=1e-6)
