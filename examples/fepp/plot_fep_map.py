#!/usr/bin/env python3
"""Render a FEP+ ligand perturbation map as a network graph.

This is the static, run-time-independent analog of OpenFE's
``plot_atommapping_network``: nodes are the ligand 2D structures and edges are
the perturbations chosen by ``fep_mapper``. The topology is read from the
``<conf>_map.edge`` file written by ``build_fepp_inputs.sh`` (each line is
``<hexA>:<hexB>  # <ligA> -> <ligB>``), and the 2D depictions come from the
input ligand SDFs.

Usage:
    conda run -n mdpp python3 plot_fep_map.py                  # open, default paths
    conda run -n mdpp python3 plot_fep_map.py -c closed
    conda run -n mdpp python3 plot_fep_map.py -e tmp/open_map.edge -o tmp/open_map.png
"""

from __future__ import annotations

import argparse
import io
import itertools
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

# "<hexA>:<hexB>  # <ligA> -> <ligB>"; names are the source of truth for labels.
EDGE_LINE = re.compile(r"^\s*\S+:\S+\s*#\s*(?P<a>\S+)\s*->\s*(?P<b>\S+)\s*$")

# Figure geometry. The thumbnail zoom is derived from these plus the achieved
# node separation, so nodes never overlap regardless of the ligand set.
FIG_SIZE_IN = (16.0, 14.0)
AXIS_MARGIN = 0.25  # padding beyond the normalized [-1, 1] layout box
THUMB_PX = 320  # rendered ligand depiction size, pixels
MAX_ZOOM = 0.27  # upper bound on OffsetImage zoom (largest readable thumbnail)
# Fraction of the figure the axes actually occupy once the title and
# tight_layout have taken their share; used to convert points to data units.
USABLE_FRAC = 0.88


def parse_edges(edge_file: Path) -> list[tuple[str, str]]:
    """Parse ligand-name edge pairs from a fep_mapper ``.edge`` file.

    Args:
        edge_file: Path to a ``<conf>_map.edge`` file.

    Returns:
        List of ``(ligA, ligB)`` ligand-name pairs, one per perturbation edge.

    Raises:
        ValueError: If the file contains no parseable edges.
    """
    edges: list[tuple[str, str]] = []
    for line in edge_file.read_text().splitlines():
        match = EDGE_LINE.match(line)
        if match:
            edges.append((match["a"], match["b"]))
    if not edges:
        raise ValueError(f"no edges parsed from {edge_file}")
    return edges


def normalize_layout(pos: dict[str, Any]) -> dict[str, tuple[float, float]]:
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


def min_separation(pos: dict[str, tuple[float, float]]) -> float:
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


def spread_nodes(
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
    return normalize_layout({n: (p[0], p[1]) for n, p in pts.items()})


def ligand_image(sdf_path: Path, size: int = THUMB_PX, zoom: float = MAX_ZOOM) -> OffsetImage:
    """Render a ligand SDF to a 2D-depiction OffsetImage for use as a graph node.

    Args:
        sdf_path: Path to a single-molecule SDF file.
        size: Pixel width/height of the square depiction.
        zoom: OffsetImage scale factor (points per pixel of the depiction).

    Returns:
        A matplotlib OffsetImage of the ligand's 2D structure.

    Raises:
        ValueError: If the SDF cannot be parsed.
    """
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=True)
    if mol is None:
        raise ValueError(f"failed to parse {sdf_path}")
    mol = Draw.PrepareMolForDrawing(mol)  # 2D coords + wedging
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    arr = mpimg.imread(io.BytesIO(png), format="png")
    return OffsetImage(arr, zoom=zoom)


def plot_map(
    edge_file: Path,
    ligand_dir: Path,
    output: Path,
    *,
    title: str | None = None,
    seed: int = 42,
) -> Path:
    """Draw the perturbation graph with ligand structures on the nodes.

    Args:
        edge_file: ``<conf>_map.edge`` file describing the perturbation edges.
        ligand_dir: Directory of ``<ligand>.sdf`` 2D-depiction sources.
        output: Output image path (extension picks the format, e.g. .png/.pdf).
        title: Optional figure title.
        seed: Layout seed for a deterministic node arrangement.

    Returns:
        The output path that was written.
    """
    edges = parse_edges(edge_file)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    # A thumbnail spans THUMB_PX*zoom points; the axes span (2 + 2*AXIS_MARGIN)
    # data units over the *smaller* figure dimension (the square thumbnails are
    # limited by the tighter of the two axis scales).
    points_per_unit = min(FIG_SIZE_IN) * 72.0 * USABLE_FRAC / (2.0 + 2.0 * AXIS_MARGIN)
    full_size_gap = THUMB_PX * MAX_ZOOM / points_per_unit

    # Kamada-Kawai spreads sparse graphs more evenly than spring, but it can put
    # two ligands almost on the same point; spread_nodes fixes just those pairs.
    # Layout and relaxation are both deterministic, so `seed` only matters for
    # the spring fallback used when Kamada-Kawai degenerates entirely.
    try:
        pos = normalize_layout(nx.kamada_kawai_layout(graph))
    except (ValueError, nx.NetworkXError):
        pos = normalize_layout(nx.spring_layout(graph, seed=seed, k=2.5, iterations=500))
    pos = spread_nodes(pos, full_size_gap)
    separation = min_separation(pos)

    # Shrink the thumbnails if the relaxation could not reach the target gap
    # (dense graphs), so nodes never overlap regardless of the ligand set.
    zoom = min(MAX_ZOOM, 0.95 * separation * points_per_unit / THUMB_PX)

    fig, ax = plt.subplots(figsize=FIG_SIZE_IN)
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
        sdf = ligand_dir / f"{name}.sdf"
        if sdf.exists():
            box = AnnotationBbox(
                ligand_image(sdf, zoom=zoom),
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
        or f"FEP+ perturbation map: {edge_file.stem} "
        f"({graph.number_of_nodes()} ligands, {graph.number_of_edges()} edges)",
        fontsize=13,
    )
    ax.axis("off")
    # Must match the span assumed by points_per_unit above, or the thumbnail
    # sizing that guarantees non-overlap no longer holds.
    ax.set_xlim(-1.0 - AXIS_MARGIN, 1.0 + AXIS_MARGIN)
    ax.set_ylim(-1.0 - AXIS_MARGIN, 1.0 + AXIS_MARGIN)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    """CLI entry point."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--conformation",
        default="open",
        choices=("open", "closed"),
        help="Conformation whose map to plot (default: open).",
    )
    parser.add_argument(
        "-e",
        "--edge-file",
        type=Path,
        default=None,
        help="Override path to the .edge file (default: tmp/<conf>_map.edge).",
    )
    parser.add_argument(
        "-l",
        "--ligand-dir",
        type=Path,
        default=here / "inputs" / "ligands",
        help="Directory of <ligand>.sdf files (default: inputs/ligands).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: tmp/<conf>_map.png).",
    )
    args = parser.parse_args()

    edge_file = args.edge_file or (here / "tmp" / f"{args.conformation}_map.edge")
    output = args.output or (here / "tmp" / f"{args.conformation}_map.png")
    written = plot_map(edge_file, args.ligand_dir, output)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
