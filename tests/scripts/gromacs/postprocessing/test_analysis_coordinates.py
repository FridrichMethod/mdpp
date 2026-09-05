"""Check that periodic analysis scripts receive unfitted coordinate artifacts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts" / "gromacs"


@pytest.fixture()
def gmx_stub(tmp_path: Path) -> dict[str, str]:
    """Record command arguments and materialize declared GROMACS outputs."""
    executable = tmp_path / "gmx"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "cat > /dev/null\n"
        'printf "%s\\n" "$*" >> "$GMX_REVIEW_LOG"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        '    -o|-on|-ol) touch "$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    executable.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GMX_REVIEW_LOG": str(tmp_path / "gmx.log"),
    }


@pytest.mark.parametrize("script", ["gmx_postprocessing.sh", "gmx_postprocessing_fast.sh"])
def test_preprocessing_preserves_periodic_artifact(tmp_path, gmx_stub, script) -> None:
    """Both pipelines retain whole unfitted solute coordinates for analysis."""
    for suffix in ["gro", "edr", "tpr", "xtc"]:
        (tmp_path / f"step5_production.{suffix}").write_text("input\n")
    (tmp_path / "index.ndx").write_text("[ SOLU ]\n1 2 3\n")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "postprocessing" / script)],
        cwd=tmp_path,
        env=gmx_stub,
        capture_output=True,
        text=True,
        input="",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "tmp/step5_production_complex_center.xtc").exists()
    assert (tmp_path / "tmp/step5_production_complex_fit.xtc").exists()
    calls = Path(gmx_stub["GMX_REVIEW_LOG"]).read_text().splitlines()
    center_call = next(c for c in calls if "-o step5_production_complex_center.xtc" in c)
    assert "_fit.xtc" not in center_call
    assert "-fit" not in center_call


@pytest.mark.parametrize("script", ["gmx_hbond.sh", "gmx_sasa.sh", "gmx_dssp.sh"])
def test_periodic_analysis_uses_unfitted_coordinates(tmp_path, gmx_stub, script) -> None:
    """Periodic kernels must not consume rotated coordinates with an unrotated box."""
    result = subprocess.run(
        ["bash", str(SCRIPTS / "analysis" / script)],
        cwd=tmp_path,
        env=gmx_stub,
        capture_output=True,
        text=True,
        input="",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = Path(gmx_stub["GMX_REVIEW_LOG"]).read_text()
    assert "-f step5_production_complex_center.xtc" in calls
    assert "-f step5_production_complex_fit.xtc" not in calls
