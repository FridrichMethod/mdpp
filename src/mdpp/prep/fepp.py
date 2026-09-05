"""Prepared-receptor chemistry checks for FEP+ workflows.

Only file reading requires Schrodinger. Comparison of previously generated
summaries and policy checks are available in the regular mdpp environment.
The Suite reader runs in a separate process because its bundled Python may be
older than mdpp's supported Python version.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mdpp._types import StrPath
from mdpp.prep._fepp_receptor_worker import (
    compare_receptor_summaries,
    enforce_receptor_microstate_policy,
)

__all__ = [
    "compare_receptor_microstates",
    "compare_receptor_summaries",
    "enforce_receptor_microstate_policy",
    "summarize_receptor",
]


def summarize_receptor(path: StrPath, *, schrodinger_root: StrPath) -> dict[str, Any]:
    """Summarize a prepared receptor with the chosen Suite installation.

    Formal charges and hydrogen connectivity define the protonation signature;
    residue names alone are insufficient. Heavy-atom neighbors retain residue
    identifiers so changed cross-residue connectivity remains detectable.

    Args:
        path: Prepared receptor structure file containing exactly one structure.
        schrodinger_root: Explicit Suite installation root containing ``run``.

    Returns:
        JSON-serializable atom, charge, sequence, and microstate summary.

    Raises:
        OSError: If the Suite launcher or a required file is inaccessible.
        subprocess.CalledProcessError: If the Suite reader rejects the input or
            cannot run in the selected environment.
        ValueError: If the worker does not produce a JSON object.
    """
    worker = Path(__file__).with_name("_fepp_receptor_worker.py")
    with tempfile.TemporaryDirectory(prefix="mdpp-fepp-receptor-") as directory:
        output = Path(directory) / "summary.json"
        subprocess.run(
            [
                str(Path(schrodinger_root).resolve() / "run"),
                "python3",
                str(worker.resolve()),
                "--summary",
                str(path),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(output.read_text())
    if not isinstance(summary, dict):
        raise ValueError("receptor summary must be a JSON object")
    return summary


def compare_receptor_microstates(
    open_path: StrPath,
    closed_path: StrPath,
    *,
    schrodinger_root: StrPath,
) -> dict[str, Any]:
    """Compare prepared receptors using an explicitly selected Suite reader.

    The returned report records all differences without rejecting them. Apply
    :func:`enforce_receptor_microstate_policy` when a workflow requires matching
    receptor chemistry before proceeding with two conformational basins.

    Args:
        open_path: Prepared receptor for the open conformation.
        closed_path: Prepared receptor for the closed conformation.
        schrodinger_root: Explicit Suite installation root containing ``run``.

    Returns:
        Schema-version-1 comparison report with sequence, heavy-atom connectivity,
        protonation, and formal-charge differences.

    Raises:
        OSError: If the Suite launcher or a required file is inaccessible.
        subprocess.CalledProcessError: If a receptor cannot be summarized.
        ValueError: If a worker result is invalid JSON or not a JSON object.
        KeyError: If a required receptor summary field is absent.
    """
    return compare_receptor_summaries(
        summarize_receptor(open_path, schrodinger_root=schrodinger_root),
        summarize_receptor(closed_path, schrodinger_root=schrodinger_root),
    )
