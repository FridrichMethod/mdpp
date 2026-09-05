#!/usr/bin/env python3
"""Render this example's FEP+ perturbation map using mdpp.plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mdpp.core.fepp import read_fep_edges
from mdpp.plots.fepp import plot_fep_map


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
    edges = read_fep_edges(edge_file)
    node_count = len({name for edge in edges for name in edge})
    title = f"FEP+ perturbation map: {edge_file.stem} ({node_count} ligands, {len(edges)} edges)"
    ax = plot_fep_map(edges, ligand_dir=args.ligand_dir, title=title)
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    try:
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
