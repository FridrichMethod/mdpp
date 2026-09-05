"""FEP+ network and atom-mapping depictions with reusable Matplotlib axes.

Mapping records retain Suite's 1-based atom indices. Importing this module does
not select a Matplotlib backend or require Schrödinger's Python environment.
"""

from __future__ import annotations

import io
import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import image as mpimg
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

from mdpp._types import StrPath

FIG_SIZE_IN = (16.0, 14.0)
AXIS_MARGIN = 0.25
THUMB_PX = 320
MAX_ZOOM = 0.27
DELETED_RGB = (0.96, 0.55, 0.55)
ADDED_RGB = (0.55, 0.86, 0.55)
CORE_ATOM_RGB = (1.0, 0.76, 0.30)
CORE_BOND_RGB = (0.40, 0.66, 0.96)


def _normalize_layout(pos: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Rescale a networkx layout into the fixed [-1, 1] box on both axes.

    Args:
        pos: Layout mapping node -> (x, y), as returned by a networkx layout
            (values are numpy arrays, hence the untyped value).

    Returns:
        The same mapping rescaled so each axis spans exactly [-1, 1].
    """
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    return {
        n: ((p[0] - min(xs)) / span_x * 2 - 1, (p[1] - min(ys)) / span_y * 2 - 1)
        for n, p in pos.items()
    }


def _min_separation(pos: dict[str, tuple[float, float]]) -> float:
    """Return the smallest pairwise node distance in a normalized layout.

    Args:
        pos: Normalized layout mapping node -> (x, y).

    Returns:
        The minimum Euclidean distance between any two nodes, or ``inf`` for
        fewer than two nodes.
    """
    return min(
        (math.dist(a, b) for a, b in itertools.combinations(pos.values(), 2)),
        default=math.inf,
    )


def _spread_nodes(
    pos: dict[str, tuple[float, float]],
    target: float,
    *,
    iterations: int = 500,
) -> dict[str, tuple[float, float]]:
    """Push apart node pairs closer than ``target``, preserving overall layout.

    Kamada-Kawai lays these sparse ligand graphs out well globally but happily
    puts two ligands almost on the same point, whose thumbnails then overlap.
    Rather than discarding the layout, this nudges only the offending pairs
    apart. Pairs are visited in a fixed order, so the result is deterministic.

    Args:
        pos: Normalized layout mapping node -> (x, y).
        target: Minimum separation to enforce, in normalized layout units.
        iterations: Maximum relaxation sweeps.

    Returns:
        A new layout with the same node keys, renormalized to the [-1, 1] box.
    """
    pts = {n: [float(p[0]), float(p[1])] for n, p in pos.items()}
    names = sorted(pts)
    for _ in range(iterations):
        collided = False
        for a, b in itertools.combinations(names, 2):
            pa, pb = pts[a], pts[b]
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            dist = math.hypot(dx, dy)
            if dist >= target:
                continue
            collided = True
            if dist < 1e-9:
                # Exactly coincident: separate along a fixed axis so the
                # outcome stays reproducible.
                dx, dy, dist = 1.0, 0.0, 1.0
            shift = (target - dist) / 2.0
            ux, uy = dx / dist, dy / dist
            pa[0] -= ux * shift
            pa[1] -= uy * shift
            pb[0] += ux * shift
            pb[1] += uy * shift
        if not collided:
            break
    return _normalize_layout({n: (p[0], p[1]) for n, p in pts.items()})


def _ligand_image(sdf_path: Path, size: int = THUMB_PX, zoom: float = MAX_ZOOM) -> OffsetImage:
    """Render a ligand SDF to a 2D-depiction OffsetImage for use as a graph node.

    Args:
        sdf_path: Path to a single-molecule SDF file.
        size: Pixel width/height of the square depiction.
        zoom: OffsetImage scale factor (points per pixel of the depiction).

    Returns:
        A matplotlib OffsetImage of the ligand's 2D structure.

    Raises:
        ValueError: If the SDF cannot be parsed or its title does not match its
            filename.
    """
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
    if mol is None:
        raise ValueError(f"failed to parse {sdf_path}")
    title = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    if title != sdf_path.stem:
        raise ValueError(
            f"{sdf_path}: SDF title {title!r} does not match filename {sdf_path.stem!r}"
        )
    mol = Draw.PrepareMolForDrawing(mol)  # 2D coords + wedging
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.drawOptions().clearBackground = False  # type: ignore[assignment]
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    arr = mpimg.imread(io.BytesIO(png), format="png")
    return OffsetImage(arr, zoom=zoom)


def plot_fep_map(
    edges: Sequence[tuple[str, str]],
    *,
    ligand_dir: StrPath,
    title: str | None = None,
    seed: int = 42,
    ax: Axes | None = None,
) -> Axes:
    """Draw a directed ligand perturbation graph with molecular thumbnails.

    Args:
        edges: Directed ligand-name pairs, for example from
            :func:`mdpp.core.fepp.read_fep_edges`.
        ligand_dir: Directory of ``<ligand>.sdf`` files with matching titles.
        title: Optional figure title.
        seed: Seed for the spring-layout fallback.
        ax: Axes to draw on; creates a figure when omitted.

    Returns:
        Axes containing the graph. The caller owns the figure.

    Raises:
        ValueError: If the graph is empty or disconnected, or ligand SDFs are
            missing, invalid, or have mismatched titles.
    """
    ligand_dir = Path(ligand_dir)
    if not edges:
        raise ValueError("at least one perturbation edge is required")
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    if not nx.is_weakly_connected(graph):
        raise ValueError("perturbation graph is disconnected")
    missing_sdfs = sorted(name for name in graph if not (ligand_dir / f"{name}.sdf").is_file())
    if missing_sdfs:
        raise ValueError(f"missing ligand SDFs for map nodes: {missing_sdfs}")

    # Validate all depictions before allocating a caller-invisible figure.
    images = {name: _ligand_image(ligand_dir / f"{name}.sdf") for name in graph}

    # A thumbnail spans THUMB_PX*zoom points; the axes span (2 + 2*AXIS_MARGIN)
    # data units over the *smaller* figure dimension (the square thumbnails are
    # limited by the tighter of the two axis scales).
    if ax is None:
        _, ax = plt.subplots(figsize=FIG_SIZE_IN)
    fig = ax.get_figure()
    assert fig is not None
    bounds = ax.get_window_extent()
    points_per_unit = min(bounds.width, bounds.height) * 72.0 / fig.dpi / (2.0 + 2.0 * AXIS_MARGIN)
    full_size_gap = THUMB_PX * MAX_ZOOM / points_per_unit

    # Kamada-Kawai spreads sparse graphs more evenly than spring, but it can put
    # two ligands almost on the same point; _spread_nodes fixes just those pairs.
    # Layout and relaxation are both deterministic, so `seed` only matters for
    # the spring fallback used when Kamada-Kawai degenerates entirely.
    try:
        pos = _normalize_layout(nx.kamada_kawai_layout(graph))
    except (ValueError, nx.NetworkXError):
        pos = _normalize_layout(nx.spring_layout(graph, seed=seed, k=2.5, iterations=500))
    pos = _spread_nodes(pos, full_size_gap)
    separation = _min_separation(pos)

    # Shrink the thumbnails if the relaxation could not reach the target gap
    # (dense graphs), so nodes never overlap regardless of the ligand set.
    zoom = min(MAX_ZOOM, 0.95 * separation * points_per_unit / THUMB_PX)

    # Pull edge ends clear of the thumbnails so the arrowheads stay visible.
    node_margin_pt = THUMB_PX * zoom / 2.0 + 4.0
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="0.5",
        width=1.6,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        min_source_margin=node_margin_pt,
        min_target_margin=node_margin_pt,
        connectionstyle="arc3,rad=0.04",
    )

    label_offset = THUMB_PX * zoom / 2.0 / points_per_unit + 0.02
    for name, (x, y) in pos.items():
        images[name].set_zoom(zoom)
        box = AnnotationBbox(
            images[name],
            (x, y),
            frameon=True,
            pad=0.1,
            bboxprops={"edgecolor": "0.3", "boxstyle": "round"},
        )
        ax.add_artist(box)
        ax.text(
            x,
            y - label_offset,
            name,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
        )

    ax.set_title(
        title
        or "FEP+ perturbation map: "
        f"({graph.number_of_nodes()} ligands, {graph.number_of_edges()} edges)",
        fontsize=13,
    )
    ax.axis("off")
    # Must match the span assumed by points_per_unit above, or the thumbnail
    # sizing that guarantees non-overlap no longer holds.
    ax.set_xlim(-1.0 - AXIS_MARGIN, 1.0 + AXIS_MARGIN)
    ax.set_ylim(-1.0 - AXIS_MARGIN, 1.0 + AXIS_MARGIN)
    return ax


def _heavy_mol(molblock: str) -> tuple[Chem.Mol, dict[int, int]]:
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


def _rigid_align_2d(mol_b: Chem.Mol, mol_a: Chem.Mol, pairs: list[tuple[int, int]]) -> None:
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


def _aligned_mols(
    record: Mapping[str, Any],
) -> tuple[Chem.Mol, Chem.Mol, dict[int, int], dict[int, int]]:
    """Build heavy-atom 2D mols for an edge, B rigidly oriented onto A's core.

    Args:
        record: One edge dict from the extractor JSON.

    Returns:
        ``(mol_a, mol_b, map_a, map_b)`` heavy-atom mols with 2D coordinates
        and their all-atom -> heavy-atom index remaps.
    """
    mol_a, map_a = _heavy_mol(record["molblock_a"])
    mol_b, map_b = _heavy_mol(record["molblock_b"])
    rdDepictor.Compute2DCoords(mol_a)
    rdDepictor.Compute2DCoords(mol_b)
    # Core pairs (1-based, all-atom) -> heavy 0-based, kept only when both atoms
    # survive RemoveHs. Pairs are (b_idx, a_idx).
    pairs = [
        (map_b[b - 1], map_a[a - 1])
        for a, b in zip(record["core_a"], record["core_b"])
        if (a - 1) in map_a and (b - 1) in map_b
    ]
    _rigid_align_2d(mol_b, mol_a, pairs)
    return mol_a, mol_b, map_a, map_b


def _core_changes(  # noqa: C901
    record: Mapping[str, Any],
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


def _edge_panel(
    record: Mapping[str, Any], *, size: int = 480
) -> tuple[NDArray[np.floating], int, int, int, int]:
    """Render one A|B panel with dummy and mapped-core changes highlighted.

    Args:
        record: One edge dict from the extractor JSON.
        size: Per-panel pixel size (square).

    Returns:
        RGB or RGBA drawing and counts for deleted atoms, added atoms, changed core
        atoms, and changed core bonds.
    """
    mol_a, mol_b, map_a, map_b = _aligned_mols(record)
    hl_a = [map_a[i - 1] for i in record["dummy_a"] if (i - 1) in map_a]
    hl_b = [map_b[i - 1] for i in record["dummy_b"] if (i - 1) in map_b]
    changes = _core_changes(record, mol_a, mol_b, map_a, map_b)
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
    bond_count = changes["bond_change_count"]
    assert isinstance(bond_count, int)
    return (
        img,
        len(hl_a),
        len(hl_b),
        len(core_atoms_a),
        bond_count,
    )


def plot_fep_mapping(
    record: Mapping[str, Any],
    *,
    size: int = 480,
    ax: Axes | None = None,
) -> Axes:
    """Draw one stored atom mapping with dummy and mapped-core changes.

    Args:
        record: Extracted edge with ligand names, MOL blocks, similarity, and
            1-based ``core_a``, ``core_b``, ``dummy_a``, ``dummy_b`` indices.
        size: Pixel size of each ligand depiction.
        ax: Axes to draw on; creates a figure when omitted.

    Returns:
        Axes showing aligned ligand depictions and a change-count title.

    Raises:
        ValueError: If MOL blocks or the stored atom correspondence are invalid.
    """
    img, n_del, n_add, n_core_atoms, n_core_bonds = _edge_panel(record, size=size)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"{record['name_a']} -> {record['name_b']}   "
        f"sim={record['similarity']:.2f}   "
        f"-{n_del}/+{n_add} dummy; "
        f"{n_core_atoms} core atoms; {n_core_bonds} core bonds",
        fontsize=13,
    )
    return ax


def save_fep_mappings(
    data: Mapping[str, Any],
    *,
    pdf_path: StrPath,
    png_path: StrPath,
) -> int:
    """Write a multi-page mapping PDF and a PNG overview from extracted data.

    Args:
        data: Extraction document containing ``edges`` and optionally an
            ``fmp`` label. Atom indices follow the 1-based Suite convention.
        pdf_path: Destination PDF, one edge per page.
        png_path: Destination PNG showing all edge panels.

    Returns:
        Number of rendered edges. Figures created here are closed after saving.

    Raises:
        ValueError: If no edges exist or a stored mapping is invalid.
        OSError: If an output file cannot be written.
    """
    records = data["edges"]
    if not records:
        raise ValueError("no edges in mapping data")
    pdf_path, png_path = Path(pdf_path), Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    with PdfPages(pdf_path) as pdf:
        for record in records:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            try:
                plot_fep_mapping(record, ax=ax)
                panels.append((ax.get_title(), ax.images[0].get_array()))
                fig.tight_layout()
                pdf.savefig(fig)
            finally:
                plt.close(fig)

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.1 * nrows), squeeze=False)
    try:
        for ax, (header, img) in zip(axes.flat, panels):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(header, fontsize=11)
        for ax in axes.flat[len(panels) :]:
            ax.axis("off")
        fig.suptitle(
            f"FEP+ per-edge atom mappings: {data.get('fmp', 'network')} ({len(panels)} edges)",
            fontsize=15,
            y=0.999,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
    finally:
        plt.close(fig)
    return len(panels)
