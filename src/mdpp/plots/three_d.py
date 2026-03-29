"""Interactive 3D visualization helpers for molecules and trajectories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import mdtraj as md
import nglview
import py3Dmol
from rdkit import Chem

_PY3DMOL_POSITION_KEYS = ("x", "y", "z")
_DEFAULT_TRAJ_REPRESENTATIONS: tuple[dict[str, object], ...] = (
    {"type": "cartoon", "selection": "protein"},
    {"type": "ball+stick", "selection": "not protein"},
)


def _get_selected_conformer(
    mol: Chem.rdchem.Mol,
    conformer_id: int,
) -> tuple[Chem.rdchem.Conformer, int]:
    """Return the selected conformer object and its RDKit conformer ID."""
    if mol.GetNumConformers() == 0:
        raise ValueError("view_mol_3d requires a molecule with at least one conformer.")

    conformer = mol.GetConformer() if conformer_id == -1 else mol.GetConformer(conformer_id)
    return conformer, conformer.GetId()


def _label_position_from_atom(
    conformer: Chem.rdchem.Conformer,
    atom_index: int,
) -> dict[str, float]:
    """Return a py3Dmol-compatible position dictionary for an atom index."""
    position = conformer.GetAtomPosition(atom_index)
    return {"x": float(position.x), "y": float(position.y), "z": float(position.z)}


def _normalize_label_spec(
    label: Mapping[str, object],
    conformer: Chem.rdchem.Conformer,
) -> tuple[str, dict[str, object]]:
    """Validate and normalize one py3Dmol label specification."""
    if "text" not in label:
        raise ValueError("Each label must include a 'text' field.")

    has_atom_index = "atom_index" in label
    has_position = "position" in label
    if has_atom_index == has_position:
        raise ValueError("Each label must include exactly one of 'atom_index' or 'position'.")

    text = str(label["text"])
    style = {key: value for key, value in label.items() if key not in {"text", "atom_index"}}

    if has_atom_index:
        atom_index_value = label["atom_index"]
        if isinstance(atom_index_value, bool) or not isinstance(atom_index_value, int | str):
            raise ValueError("Label atom_index must be an integer.")
        atom_index = int(atom_index_value)
        style["position"] = _label_position_from_atom(conformer, atom_index)
    else:
        position = label["position"]
        if not isinstance(position, Mapping) or any(
            key not in position for key in _PY3DMOL_POSITION_KEYS
        ):
            raise ValueError("Label positions must provide mapping keys 'x', 'y', and 'z'.")
        style["position"] = {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        }

    return text, style


def make_atom_labels_3d(
    mol: Chem.rdchem.Mol,
    *,
    conformer_id: int = -1,
    atom_indices: Sequence[int] | None = None,
    text_fn: Callable[[Chem.rdchem.Atom], str] | None = None,
    color_fn: Callable[[Chem.rdchem.Atom], str | None] | None = None,
    base_style: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build generic per-atom label specifications for ``view_mol_3d``.

    Args:
        mol: Molecule to label.
        conformer_id: RDKit conformer ID to use. ``-1`` selects the default conformer.
        atom_indices: Optional subset of atom indices to label.
        text_fn: Function that returns the label text for an atom.
        color_fn: Optional function that returns a font color for an atom.
        base_style: Optional py3Dmol label-style keys to apply to every label.

    Returns:
        Label dictionaries consumable by ``view_mol_3d``.

    Raises:
        ValueError: If the molecule has no conformer or ``text_fn`` is not provided.
    """
    if text_fn is None:
        raise ValueError("make_atom_labels_3d requires a text_fn.")

    conformer, _ = _get_selected_conformer(mol, conformer_id)
    if atom_indices is None:
        atom_indices = tuple(range(mol.GetNumAtoms()))

    labels: list[dict[str, object]] = []
    for atom_index in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_index))
        label: dict[str, object] = dict(base_style or {})
        label["text"] = text_fn(atom)
        label["position"] = _label_position_from_atom(conformer, atom.GetIdx())
        color = color_fn(atom) if color_fn is not None else None
        if color is not None:
            label["fontColor"] = color
        labels.append(label)
    return labels


def view_mol_3d(
    mol: Chem.rdchem.Mol,
    *,
    conformer_id: int = -1,
    width: int = 800,
    height: int = 400,
    style: Mapping[str, object] | None = None,
    labels: Sequence[Mapping[str, object]] | None = None,
    background_color: str = "white",
    zoom_to: bool = True,
    show: bool = False,
) -> py3Dmol.view:
    """Create an interactive py3Dmol viewer for an RDKit molecule.

    Args:
        mol: Molecule to visualize.
        conformer_id: RDKit conformer ID to use. ``-1`` selects the default conformer.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        style: py3Dmol style dictionary passed to ``setStyle``.
        labels: Sequence of label dictionaries. Each must contain ``text`` and
            exactly one of ``atom_index`` or ``position``.
        background_color: Viewer background color.
        zoom_to: Whether to call ``zoomTo()`` after loading content.
        show: Whether to call ``show()`` before returning the viewer.

    Returns:
        A configured py3Dmol viewer.

    Raises:
        ValueError: If the molecule has no conformer or any label spec is invalid.
    """
    conformer, selected_conformer_id = _get_selected_conformer(mol, conformer_id)
    del conformer  # Only needed for label coordinate lookup below.

    viewer = py3Dmol.view(width=width, height=height)
    viewer.setBackgroundColor(background_color)
    viewer.addModel(Chem.MolToMolBlock(mol, confId=selected_conformer_id), "sdf")
    viewer.setStyle(dict(style) if style is not None else {"stick": {}})

    if labels:
        selected_conformer = mol.GetConformer(selected_conformer_id)
        for label in labels:
            text, label_style = _normalize_label_spec(label, selected_conformer)
            viewer.addLabel(text, label_style)

    if zoom_to:
        viewer.zoomTo()
    if show:
        viewer.show()
    return viewer


def view_traj_3d(
    traj: md.Trajectory,
    *,
    representations: Sequence[Mapping[str, object]] | None = None,
) -> nglview.NGLWidget:
    """Create an interactive nglview widget for a trajectory or structure.

    Args:
        traj: MDTraj trajectory to visualize.
        representations: Sequence of representation dictionaries. Each must
            include ``type`` and may include ``selection`` plus any additional
            nglview representation keyword arguments.

    Returns:
        A configured nglview widget.

    Raises:
        ValueError: If any representation dictionary is missing ``type``.
    """
    widget = nglview.show_mdtraj(traj)
    widget.clear_representations()

    for representation in representations or _DEFAULT_TRAJ_REPRESENTATIONS:
        if "type" not in representation:
            raise ValueError("Each trajectory representation must include a 'type' field.")

        rep_kwargs = dict(representation)
        rep_type = str(rep_kwargs.pop("type"))
        selection = rep_kwargs.pop("selection", "all")
        widget.add_representation(rep_type, selection=selection, **rep_kwargs)

    return widget
