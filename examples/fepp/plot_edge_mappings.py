#!/usr/bin/env python3
"""Render per-edge FEP+ atom mappings from the JSON written by the extractor.

For every perturbation edge this draws ligand A and ligand B side by side using
the *actual* mapping computed by ``fep_mapper`` (read from the .fmp by
``extract_edge_mappings.py``), the analog of OpenFE's per-edge ``mapping`` view.
Atoms that disappear going A -> B are highlighted red on A; atoms that appear
are highlighted green on B. Mapped-core atoms whose element, isotope, formal
charge, or aromaticity changes are orange, and changed mapped-core bonds are
blue. B's depiction is aligned to A over the shared core.

Run under the mdpp env (its RDKit has the Cairo drawer):

    $SCHRODINGER/run python3 extract_edge_mappings.py            # -> tmp/open_map_mappings.json
    conda run -n mdpp python3 plot_edge_mappings.py              # -> PDF + PNG
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

# Highlight colors for dummy atoms and otherwise hidden mapped-core changes.
DELETED_RGB = (0.96, 0.55, 0.55)
ADDED_RGB = (0.55, 0.86, 0.55)
CORE_ATOM_RGB = (1.0, 0.76, 0.30)
CORE_BOND_RGB = (0.40, 0.66, 0.96)


def heavy_mol(molblock: str) -> tuple[Chem.Mol, dict[int, int]]:
    """Parse a MOL block to a heavy-atom RDKit mol and an index remap.

    Explicit hydrogens clutter these large ligands, so they are removed for
    depiction. Heavy atoms keep their relative order through ``RemoveHs``, so
    the remap is just a running count of non-hydrogen atoms.

    Args:
        molblock: A MOL block (atoms 0-based once parsed).

    Returns:
        ``(mol, old_to_new)`` where ``old_to_new`` maps a 0-based index in the
        all-atom mol to its 0-based index in the heavy-atom mol.

    Raises:
        ValueError: If the MOL block cannot be parsed, or if ``RemoveHs`` kept
            a hydrogen (which would silently shift every highlight index).
    """
    full = Chem.MolFromMolBlock(molblock, removeHs=False)
    if full is None:
        raise ValueError("failed to parse MOL block")
    old_to_new: dict[int, int] = {}
    count = 0
    for i, atom in enumerate(full.GetAtoms()):
        if atom.GetAtomicNum() != 1:
            old_to_new[i] = count
            count += 1
    heavy = Chem.RemoveHs(full)
    # RemoveHs keeps hydrogens in some cases (charged, isotopic, stereo-defining).
    # The running-count remap above assumes none survive, so verify rather than
    # silently mis-highlighting atoms.
    if heavy.GetNumAtoms() != count:
        raise ValueError(
            f"RemoveHs kept {heavy.GetNumAtoms() - count} hydrogen(s); "
            "the heavy-atom index remap would be invalid"
        )
    return heavy, old_to_new


def rigid_align_2d(mol_b: Chem.Mol, mol_a: Chem.Mol, pairs: list[tuple[int, int]]) -> None:
    """Rigidly orient mol_b's 2D conformer onto mol_a over the core pairs.

    Unlike ``GenerateDepictionMatching2DStructure`` (which pins B's core atoms
    onto A's exact coordinates and distorts B when the cores differ), this keeps
    B's own clean depiction and only applies the optimal rotation + translation
    (reflection allowed, which is fine for a 2D drawing) that maps B's core onto
    A's. mol_b's conformer is modified in place.

    Args:
        mol_b: Heavy-atom mol to reorient (has a 2D conformer).
        mol_a: Reference heavy-atom mol (has a 2D conformer).
        pairs: ``(b_idx, a_idx)`` heavy-atom index pairs for the shared core.
    """
    if len(pairs) < 2:
        return
    conf_a, conf_b = mol_a.GetConformer(), mol_b.GetConformer()
    pts_a = np.array([[conf_a.GetAtomPosition(a).x, conf_a.GetAtomPosition(a).y] for _, a in pairs])
    pts_b = np.array([[conf_b.GetAtomPosition(b).x, conf_b.GetAtomPosition(b).y] for b, _ in pairs])
    center_a, center_b = pts_a.mean(0), pts_b.mean(0)
    # Orthogonal Procrustes: rot maps (B - center_b) onto (A - center_a).
    u, _, vt = np.linalg.svd((pts_b - center_b).T @ (pts_a - center_a))
    rot = u @ vt
    for i in range(mol_b.GetNumAtoms()):
        pos = conf_b.GetAtomPosition(i)
        moved = (np.array([pos.x, pos.y]) - center_b) @ rot + center_a
        conf_b.SetAtomPosition(i, Point3D(float(moved[0]), float(moved[1]), 0.0))


def aligned_mols(record: dict) -> tuple[Chem.Mol, Chem.Mol, dict[int, int], dict[int, int]]:
    """Build heavy-atom 2D mols for an edge, B rigidly oriented onto A's core.

    Args:
        record: One edge dict from the extractor JSON.

    Returns:
        ``(mol_a, mol_b, map_a, map_b)`` heavy-atom mols with 2D coordinates
        and their all-atom -> heavy-atom index remaps.
    """
    mol_a, map_a = heavy_mol(record["molblock_a"])
    mol_b, map_b = heavy_mol(record["molblock_b"])
    rdDepictor.Compute2DCoords(mol_a)
    rdDepictor.Compute2DCoords(mol_b)
    # Core pairs (1-based, all-atom) -> heavy 0-based, kept only when both atoms
    # survive RemoveHs. Pairs are (b_idx, a_idx).
    pairs = [
        (map_b[b - 1], map_a[a - 1])
        for a, b in zip(record["core_a"], record["core_b"])
        if (a - 1) in map_a and (b - 1) in map_b
    ]
    rigid_align_2d(mol_b, mol_a, pairs)
    return mol_a, mol_b, map_a, map_b


def core_changes(  # noqa: C901
    record: dict,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    map_a: dict[int, int],
    map_b: dict[int, int],
) -> dict[str, list[int] | int]:
    """Classify mapped-core atom and bond changes for both ligands.

    Args:
        record: One edge record from the extractor JSON.
        mol_a: Heavy-atom ligand A.
        mol_b: Heavy-atom ligand B.
        map_a: A all-atom to heavy-atom index map.
        map_b: B all-atom to heavy-atom index map.

    Returns:
        Changed atom and bond indices on A/B plus the number of mapped bond
        correspondences whose presence or attributes differ.

    Raises:
        ValueError: If the stored core mapping has mismatched, duplicated, or
            out-of-range indices.
    """
    core_a = record["core_a"]
    core_b = record["core_b"]
    if len(core_a) != len(core_b):
        raise ValueError("core_a and core_b must have equal length")
    if len(set(core_a)) != len(core_a) or len(set(core_b)) != len(core_b):
        raise ValueError("core atom indices must be unique on each ligand")
    full_a = Chem.MolFromMolBlock(record["molblock_a"], removeHs=False)
    full_b = Chem.MolFromMolBlock(record["molblock_b"], removeHs=False)
    if full_a is None or full_b is None:
        raise ValueError("failed to parse a core-mapping MOL block")
    if any(index < 1 or index > full_a.GetNumAtoms() for index in core_a):
        raise ValueError("core_a contains an out-of-range atom index")
    if any(index < 1 or index > full_b.GetNumAtoms() for index in core_b):
        raise ValueError("core_b contains an out-of-range atom index")

    heavy_pairs = [
        (map_a[a - 1], map_b[b - 1])
        for a, b in zip(core_a, core_b)
        if (a - 1) in map_a and (b - 1) in map_b
    ]

    def atom_signature(atom: Chem.Atom) -> tuple[int, int, int, bool]:
        return (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
        )

    changed_atoms_a: list[int] = []
    changed_atoms_b: list[int] = []
    for atom_a, atom_b in heavy_pairs:
        if atom_signature(mol_a.GetAtomWithIdx(atom_a)) != atom_signature(
            mol_b.GetAtomWithIdx(atom_b)
        ):
            changed_atoms_a.append(atom_a)
            changed_atoms_b.append(atom_b)

    def bond_signature(bond: Chem.Bond | None) -> tuple[float, bool, str] | None:
        if bond is None:
            return None
        return (bond.GetBondTypeAsDouble(), bond.GetIsAromatic(), str(bond.GetStereo()))

    changed_bonds_a: list[int] = []
    changed_bonds_b: list[int] = []
    changed_bond_count = 0
    for pair_index, (a1, b1) in enumerate(heavy_pairs):
        for a2, b2 in heavy_pairs[pair_index + 1 :]:
            bond_a = mol_a.GetBondBetweenAtoms(a1, a2)
            bond_b = mol_b.GetBondBetweenAtoms(b1, b2)
            if bond_signature(bond_a) == bond_signature(bond_b):
                continue
            changed_bond_count += 1
            if bond_a is not None:
                changed_bonds_a.append(bond_a.GetIdx())
            if bond_b is not None:
                changed_bonds_b.append(bond_b.GetIdx())
    return {
        "atoms_a": changed_atoms_a,
        "atoms_b": changed_atoms_b,
        "bonds_a": changed_bonds_a,
        "bonds_b": changed_bonds_b,
        "bond_change_count": changed_bond_count,
    }


def edge_panel(record: dict, *, size: int = 480):
    """Render one A|B panel with dummy and mapped-core changes highlighted.

    Args:
        record: One edge dict from the extractor JSON.
        size: Per-panel pixel size (square).

    Returns:
        RGBA drawing and counts for deleted atoms, added atoms, changed core
        atoms, and changed core bonds.
    """
    mol_a, mol_b, map_a, map_b = aligned_mols(record)
    hl_a = [map_a[i - 1] for i in record["dummy_a"] if (i - 1) in map_a]
    hl_b = [map_b[i - 1] for i in record["dummy_b"] if (i - 1) in map_b]
    changes = core_changes(record, mol_a, mol_b, map_a, map_b)
    core_atoms_a = changes["atoms_a"]
    core_atoms_b = changes["atoms_b"]
    core_bonds_a = changes["bonds_a"]
    core_bonds_b = changes["bonds_b"]
    assert isinstance(core_atoms_a, list)
    assert isinstance(core_atoms_b, list)
    assert isinstance(core_bonds_a, list)
    assert isinstance(core_bonds_b, list)
    drawer = rdMolDraw2D.MolDraw2DCairo(2 * size, size, size, size)
    drawer.drawOptions().legendFontSize = 20  # type: ignore[assignment]
    drawer.DrawMolecules(
        [mol_a, mol_b],
        legends=[record["name_a"], record["name_b"]],
        highlightAtoms=[hl_a + core_atoms_a, hl_b + core_atoms_b],
        highlightAtomColors=[
            {**dict.fromkeys(core_atoms_a, CORE_ATOM_RGB), **dict.fromkeys(hl_a, DELETED_RGB)},
            {**dict.fromkeys(core_atoms_b, CORE_ATOM_RGB), **dict.fromkeys(hl_b, ADDED_RGB)},
        ],
        highlightBonds=[core_bonds_a, core_bonds_b],
        highlightBondColors=[
            dict.fromkeys(core_bonds_a, CORE_BOND_RGB),
            dict.fromkeys(core_bonds_b, CORE_BOND_RGB),
        ],
    )
    drawer.FinishDrawing()
    img = mpimg.imread(io.BytesIO(drawer.GetDrawingText()), format="png")
    return (
        img,
        len(hl_a),
        len(hl_b),
        len(core_atoms_a),
        changes["bond_change_count"],
    )


def render(json_path: Path, pdf_path: Path, png_path: Path) -> int:
    """Render all edge mappings to a multi-page PDF and an overview PNG.

    Args:
        json_path: Extractor JSON (from ``extract_edge_mappings.py``).
        pdf_path: Output multi-page PDF (one edge per page).
        png_path: Output overview PNG (grid of all edges).

    Returns:
        Number of edges rendered.
    """
    data = json.loads(json_path.read_text())
    records = data["edges"]
    stem = data.get("fmp", json_path.stem)
    if not records:
        raise ValueError(f"no edges in {json_path}")

    panels = []
    for record in records:
        img, n_del, n_add, n_core_atoms, n_core_bonds = edge_panel(record)
        header = (
            f"{record['name_a']} -> {record['name_b']}   "
            f"sim={record['similarity']:.2f}   "
            f"-{n_del}/+{n_add} dummy; "
            f"{n_core_atoms} core atoms; {n_core_bonds} core bonds"
        )
        panels.append((header, img))

    with PdfPages(pdf_path) as pdf:
        for header, img in panels:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(header, fontsize=13)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.1 * nrows))
    for ax, (header, img) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(header, fontsize=11)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        f"FEP+ per-edge atom mappings: {stem} ({len(panels)} edges)",
        fontsize=15,
        y=0.999,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return len(panels)


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-j",
        "--json",
        type=Path,
        default=here / "tmp" / "open_map_mappings.json",
        help="Extractor JSON (default: tmp/open_map_mappings.json).",
    )
    parser.add_argument(
        "-o",
        "--pdf",
        type=Path,
        default=None,
        help="Output PDF (default: <json-stem>.pdf).",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Output overview PNG (default: <json-stem>.png).",
    )
    args = parser.parse_args()
    pdf_path = args.pdf or args.json.with_suffix(".pdf")
    png_path = args.png or args.json.with_suffix(".png")
    n = render(args.json, pdf_path, png_path)
    print(f"rendered {n} edge mappings\n  PDF: {pdf_path}\n  PNG: {png_path}")


if __name__ == "__main__":
    main()
