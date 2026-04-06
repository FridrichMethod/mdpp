"""Shared fixtures for shell script tests.

Creates a fake Slurm environment by placing shim executables for squeue,
scontrol, sacct, and sbatch on PATH ahead of any real installations.
"""

from __future__ import annotations

import stat
import textwrap
from pathlib import Path

import pytest

_SHIM_TEMPLATE = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    resp="{response_file}"
    if [[ -f "$resp" ]]; then cat "$resp"; fi
    exit 0
    """
)

_SBATCH_SHIM = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    log="{log_file}"
    echo "$@" >> "$log"
    echo "Submitted batch job 99999"
    exit 0
    """
)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture()
def slurm_env(tmp_path: Path) -> dict[str, Path]:
    """Fake Slurm environment with shim executables.

    Returns a dict with paths keyed by command name:
    - ``squeue``, ``scontrol``, ``sacct``: write canned responses to these files.
    - ``sbatch``: read this file to see logged invocations.
    - ``_bin_dir``: path to prepend to PATH.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    files: dict[str, Path] = {}

    for cmd in ("squeue", "scontrol", "sacct"):
        resp = tmp_path / f"{cmd}_response.txt"
        resp.write_text("")
        shim = bin_dir / cmd
        shim.write_text(_SHIM_TEMPLATE.format(response_file=resp))
        _make_executable(shim)
        files[cmd] = resp

    sbatch_log = tmp_path / "sbatch_log.txt"
    sbatch_log.write_text("")
    sbatch_shim = bin_dir / "sbatch"
    sbatch_shim.write_text(_SBATCH_SHIM.format(log_file=sbatch_log))
    _make_executable(sbatch_shim)
    files["sbatch"] = sbatch_log

    files["_bin_dir"] = bin_dir
    return files


@pytest.fixture()
def openfe_workspace(tmp_path: Path) -> dict[str, Path]:
    """Minimal OpenFE working directory with one transformation.

    Layout::

        workspace/
            transformations/rbfe_A_complex_B_complex.json
            results/
    """
    ws = tmp_path / "workspace"
    transforms = ws / "transformations"
    results = ws / "results"
    transforms.mkdir(parents=True)
    results.mkdir(parents=True)

    (transforms / "rbfe_A_complex_B_complex.json").write_text("{}")

    return {"root": ws, "transforms_dir": transforms, "results_dir": results}
