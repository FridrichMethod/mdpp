"""Extract per-edge atom mappings from a FEP+ map (.fmp) to JSON.

Reads the mapping that ``fep_mapper`` stored in the .fmp -- for each edge the
two ligand structures plus the core (mapped) and dummy (changing) atom indices
-- and writes them to a JSON that ``plot_edge_mappings.py`` renders. This split
exists because the .fmp can only be read with Schrodinger's Python (the fep
graph API), while Schrodinger's bundled RDKit has no Cairo drawer; the drawing
is done separately in the mdpp env.

Atom indices are kept 1-based (Schrodinger convention). Structures are emitted
as MOL blocks, which preserve atom order across the SDF round-trip so the
indices stay valid in the drawing step.

Run under Schrodinger's Python:

    $SCHRODINGER/run python3 /path/to/_fepp_mapping_worker.py -f map.fmp

This module deliberately has no mdpp imports and remains Python 3.11 compatible
so older Suite releases can execute its installed file directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def mapping_fingerprint(records: list[dict[str, Any]]) -> str:
    """Hash the graph's atom mappings without coordinates or FMP filename.

    Args:
        records: Deterministically ordered mapping records.

    Returns:
        Lowercase SHA-256 of ligand names, mapped/dummy atom indices, and
        mapper similarity values.

    Raises:
        KeyError: If a record lacks a required mapping field.
        TypeError: If a mapping value is not JSON serializable.
    """
    payload = [
        {
            "name_a": record["name_a"],
            "name_b": record["name_b"],
            "similarity": record["similarity"],
            "core_a": record["core_a"],
            "core_b": record["core_b"],
            "dummy_a": record["dummy_a"],
            "dummy_b": record["dummy_b"],
        }
        for record in records
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def environment_fingerprint(environment_structures: list[Any]) -> str:
    """Hash the ordered receptor, membrane, and solvent structures.

    Args:
        environment_structures: Ordered FEP graph environment structures;
            entries may be ``None``.

    Returns:
        Lowercase SHA-256 of length-prefixed Maestro CT serializations and
        explicit positional markers.

    Raises:
        ImportError: If the Schrodinger structure API is unavailable.
    """
    from schrodinger import structure

    digest = hashlib.sha256(b"fepp-fmp-environment-v1\0")
    digest.update(len(environment_structures).to_bytes(8, "big"))
    for environment_structure in environment_structures:
        if environment_structure is None:
            digest.update(b"N")
            continue
        payload = structure.write_ct_to_string(environment_structure).encode()
        digest.update(b"S")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def extract_edge_mappings(fmp: Path, out_json: Path) -> int:
    """Write per-edge mapping records of a .fmp to JSON.

    Args:
        fmp: Path to a FEP+ map file.
        out_json: Output JSON path.

    Returns:
        Number of edges written.

    Raises:
        ImportError: If Suite or its RDKit adapter is unavailable.
        OSError: If the input cannot be read or the output cannot be written.
    """
    from rdkit import Chem
    from schrodinger.application.scisol.packages.fep import graph as fepgraph
    from schrodinger.rdkit.rdkit_adapter import to_rdkit

    graph = fepgraph.Graph.deserialize(str(fmp))
    edges = sorted(
        graph.edges_iter(),
        key=lambda e: (e.nodes[0].struc.title, e.nodes[1].struc.title),
    )

    records = []
    for edge in edges:
        node_a, node_b = edge.nodes
        core_a, core_b = edge.core_atoms
        dummy_a, dummy_b = edge.dummy_atoms
        # core_a[i] pairs with core_b[i]; the API returns the two lists jointly
        # permuted in an unstable order, so sort the *pairs* to keep the JSON
        # byte-reproducible without disturbing the correspondence.
        core_pairs = sorted((int(a), int(b)) for a, b in zip(core_a, core_b))
        records.append({
            "name_a": node_a.struc.title,
            "name_b": node_b.struc.title,
            "similarity": float(edge.similarity),
            "molblock_a": Chem.MolToMolBlock(to_rdkit(node_a.struc)),
            "molblock_b": Chem.MolToMolBlock(to_rdkit(node_b.struc)),
            "core_a": [a for a, _ in core_pairs],
            "core_b": [b for _, b in core_pairs],
            "dummy_a": sorted(int(i) for i in dummy_a),
            "dummy_b": sorted(int(i) for i in dummy_b),
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fmp": fmp.stem,
                "mapping_fingerprint": mapping_fingerprint(records),
                "environment_fingerprint": environment_fingerprint(graph.environment_struc),
                "edges": records,
            },
            indent=1,
        )
        + "\n"
    )
    return len(records)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--fmp",
        type=Path,
        required=True,
        help="FEP+ map file to read.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <fmp-stem>_mappings.json).",
    )
    args = parser.parse_args()
    out_json = args.output or args.fmp.with_name(f"{args.fmp.stem}_mappings.json")
    n = extract_edge_mappings(args.fmp, out_json)
    print(f"extracted {n} edge mappings -> {out_json}")


if __name__ == "__main__":
    main()
