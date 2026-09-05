"""End-to-end coverage of the single-state RBFE CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

FEPP_DIR = Path(__file__).parents[3] / "examples" / "fepp"


def test_rbfe_cli_preserves_outputs_from_unrelated_workdir(
    tmp_path: Path, write_single_edges: Callable[..., None]
) -> None:
    input_path = tmp_path / "open.csv"
    write_single_edges(input_path, "open", [("A", "B", 1.25, 0.2)])
    output = tmp_path / "analysis"
    result = subprocess.run(
        [
            sys.executable,
            str(FEPP_DIR / "analyze_rbfe_results.py"),
            "--input",
            str(input_path),
            "--state",
            "open",
            "--reference",
            "A",
            "--unit",
            "kJ/mol",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=str(FEPP_DIR.parents[1] / "src")),
    )
    assert result.returncode == 0, result.stderr
    assert "wrote FEP+ open RBFE analysis" in result.stdout
    summary = json.loads((output / "analysis.json").read_text())
    np.testing.assert_allclose(summary["relative_binding_free_energy"]["value"], [0.0, 5.23])
    assert (output / "analysis.provenance.json").is_file()
